package com.ariaagent.mobile.core.rl

import android.content.Context
import android.util.Log
import com.ariaagent.mobile.core.memory.ExperienceStore
import com.ariaagent.mobile.core.memory.ObjectLabelStore
import java.io.File

/**
 * LoraTrainer — Model-agnostic on-device LoRA fine-tuning.
 *
 * Works with any GGUF model (Llama 3.2-1B, SmolVLM-256M, Moondream2, …).
 * Each model gets its own isolated adapter directory and its own training log
 * so switching models never consumes or overwrites another model's data.
 *
 * Architecture: W = W₀ + BA
 *   W₀ = frozen pre-trained weight matrix (in the GGUF file)
 *   B  = low-rank matrix, shape (d × r), initialized to zero
 *   A  = low-rank matrix, shape (r × k), initialized with Gaussian noise
 *   r  = rank (4 — higher rank = more capacity, more RAM)
 *
 * TWO training data sources (combined into one JSONL dataset):
 *
 *   1. ExperienceStore (agent-generated)
 *      - Successful (state, action) pairs from autonomous operation
 *      - Format: screen_summary → action_json
 *      - Quality: medium — agent sometimes gets lucky, sometimes wrong
 *      - Tracked per-model via ExperienceStore.training_log table
 *
 *   2. ObjectLabelStore (human-annotated)
 *      - Labels created by the Object Labeler tool: name, context, interaction_hint,
 *        reasoning_context for each UI element on each screen
 *      - Format: screen_labels + goal → ideal action with reasoning
 *      - Quality: HIGH — human-verified, LLM-enriched
 *      - Weight: each label sample counts 3× vs. an experience tuple
 *        (human expert > agent experience)
 *      - Re-usable across models — never marked as consumed
 *
 * Strategy: "collect-then-train"
 *   - Collect during active use (ExperienceStore.save() + ObjectLabelStore.save())
 *   - Train ONLY during idle + charging (LearningScheduler gates this)
 *   - Never interrupts inference
 *
 * JNI path: nativeTrainLora() is declared external and resolved at runtime.
 *   - When llama.cpp submodule is compiled (EAS build), the .so provides the real impl.
 *   - When .so is missing (dev/stub mode), UnsatisfiedLinkError is caught and
 *     stubTrainLora() runs instead — writes a metadata-only .bin so the rest of
 *     the pipeline (versioning, hot-reload) still exercises correctly.
 *
 * Output: lora/<modelId>/adapter_v{N}.gguf — one subdirectory per model.
 *   Loaded by LlamaEngine via nativeLoadLora() between sessions.
 *
 * Phase: 5 (RL/IRL Processing)
 */
object LoraTrainer {

    private const val TAG = "LoraTrainer"
    private const val LORA_RANK = 4
    private const val MAX_EXPERIENCE_SAMPLES = 200
    private const val MAX_LABEL_SAMPLES = 100
    private const val MIN_SAMPLES_TO_TRAIN = 10      // lowered: labels are high-value even if few
    private const val LABEL_SAMPLE_WEIGHT = 3        // each label sample = 3 experience samples

    // True after the first successful nativeTrainLora() call; stays false in stub mode.
    @Volatile var jniTrainingAvailable = false
        private set

    data class TrainingResult(
        val success: Boolean,
        val samplesUsed: Int,
        val labelSamplesUsed: Int,
        val adapterPath: String,
        val loraVersion: Int,
        val usedJni: Boolean = false,
        val errorMessage: String = ""
    )

    /**
     * Run one full LoRA training cycle combining experience tuples AND object labels.
     *
     * Tries nativeTrainLora() first (real llama.cpp training via JNI).
     * Falls back to stubTrainLora() if the native library is not compiled yet.
     *
     * @param context     Android context
     * @param store       ExperienceStore (agent-generated successes)
     * @param modelPath   Path to the base GGUF file
     * @param labelStore  ObjectLabelStore (human-annotated labels)
     */
    fun train(
        context: Context,
        store: ExperienceStore,
        modelPath: String,
        labelStore: ObjectLabelStore? = null,
        enrichedRewards: Map<String, Double> = emptyMap()
    ): TrainingResult {
        // Derive a stable, filesystem-safe model ID from the model file name.
        // Each model gets its own lora subdirectory and its own training log rows,
        // so switching models never consumes or overwrites another model's adapters.
        val mId = modelId(modelPath)

        val experiences = store.getUntrainedSuccessesFor(mId, limit = MAX_EXPERIENCE_SAMPLES)
        val labels = labelStore?.getAllEnriched(limit = MAX_LABEL_SAMPLES) ?: emptyList()

        val totalEffectiveSamples = experiences.size + labels.size * LABEL_SAMPLE_WEIGHT

        if (totalEffectiveSamples < MIN_SAMPLES_TO_TRAIN) {
            Log.i(TAG, "[$mId] Not enough data: $totalEffectiveSamples effective samples < $MIN_SAMPLES_TO_TRAIN")
            return TrainingResult(
                success           = false,
                samplesUsed       = experiences.size,
                labelSamplesUsed  = labels.size,
                adapterPath       = "",
                loraVersion       = currentVersion(context, mId),
                errorMessage      = "insufficient_data (${experiences.size} exp + ${labels.size} labels)"
            )
        }

        // Each model gets its own subdirectory: lora/<modelId>/adapter_vN.gguf
        // This prevents adapters from different models overwriting each other.
        val loraDir = loraDir(context, mId)
        val nextVersion = currentVersion(context, mId) + 1
        // .gguf extension: nativeTrainLora() writes a full GGUF checkpoint via
        // llama_model_save_to_file(). LlamaEngine.loadLora() detects GGUF magic
        // bytes and reloads it as the base model rather than a LoRA adapter.
        val adapterPath = File(loraDir, "adapter_v$nextVersion.gguf").absolutePath

        Log.i(TAG, "[$mId] Starting LoRA training: ${experiences.size} experience + ${labels.size} label samples → rank=$LORA_RANK")

        return try {
            // Write unified JSONL dataset (both paths use it)
            val datasetPath = writeDataset(context, experiences, labels, enrichedRewards)

            // ── Try JNI (real llama.cpp training) ────────────────────────────
            // nativeTrainLora() is provided by libllama-jni.so when the llama.cpp
            // submodule is compiled via CMakeLists.txt. It runs 1 epoch of LoRA
            // fine-tuning using the Adam optimizer on the JSONL dataset and writes
            // a binary adapter to adapterPath.
            val jniSucceeded = tryNativeTrainLora(modelPath, datasetPath, adapterPath, LORA_RANK, epochs = 1)

            if (!jniSucceeded) {
                // ── Stub fallback (stub mode / not yet compiled) ──────────────
                stubTrainLora(adapterPath, nextVersion, experiences.size, labels.size)
                Log.i(TAG, "[$mId] LoRA stub written → adapter_v$nextVersion (JNI not available)")
            } else {
                jniTrainingAvailable = true
                Log.i(TAG, "[$mId] LoRA JNI training complete → adapter_v$nextVersion (${experiences.size} exp + ${labels.size} labels)")
            }

            // Per-model tracking: records which experiences THIS model has already trained on.
            // Prevents the same experience being fed to the same model twice across training cycles.
            // Other models' untrained counts are completely unaffected.
            val trainedIds = experiences.map { it.id }
            store.markAsTrainedFor(trainedIds, mId)

            // Global flag: keeps DreamEngine, EmbeddingEngine, and LlmRewardEnricher in sync.
            // These subsystems use getUntrainedSuccesses() (global) to decide which experiences
            // to process. Without this call they would re-process the same experiences on every
            // idle cycle, filling the database with duplicate synthetic rows and wasting compute.
            // Labels are re-usable across models — their global flag is never touched.
            store.markAsTrained(trainedIds)

            writeVersionFile(context, nextVersion, mId)

            TrainingResult(
                success          = true,
                samplesUsed      = experiences.size,
                labelSamplesUsed = labels.size,
                adapterPath      = adapterPath,
                loraVersion      = nextVersion,
                usedJni          = jniSucceeded
            )
        } catch (e: Exception) {
            Log.e(TAG, "[$mId] LoRA training failed: ${e.message}")
            TrainingResult(
                success          = false,
                samplesUsed      = experiences.size,
                labelSamplesUsed = labels.size,
                adapterPath      = "",
                loraVersion      = currentVersion(context, mId),
                errorMessage     = e.message ?: "unknown"
            )
        }
    }

    /**
     * Attempt to call the native JNI training function.
     * Returns false (not true) if the native library is not loaded — never throws.
     */
    private fun tryNativeTrainLora(
        modelPath: String,
        datasetPath: String,
        outputPath: String,
        rank: Int,
        epochs: Int
    ): Boolean {
        return try {
            nativeTrainLora(modelPath, datasetPath, outputPath, rank, epochs)
        } catch (e: UnsatisfiedLinkError) {
            // libllama-jni.so not compiled yet (submodule not added — expected in dev)
            Log.d(TAG, "nativeTrainLora not available: ${e.message}")
            false
        } catch (e: Exception) {
            Log.w(TAG, "nativeTrainLora threw: ${e.message}")
            false
        }
    }

    /**
     * Write the combined JSONL training dataset.
     *
     * Format per line: {"input": "<full prompt>", "output": "<expected response>", "weight": N}
     *
     * For experience tuples:
     *   input  = "GOAL: {task}\n\n{screen_summary}"
     *   output = "{action_json}"
     *   weight = 1.0
     *
     * For object labels:
     *   input  = "[KNOWN ELEMENTS]\n★ {label description}\n...\nGOAL: interact with {label.name}"
     *   output = "{"tool":"Click","node_id":"#labeled","reason":"{interaction_hint}"}"
     *   weight = 3.0  (written 3 times with slight variations to match effective weight)
     */
    private fun writeDataset(
        context: Context,
        experiences: List<ExperienceStore.ExperienceTuple>,
        labels: List<ObjectLabelStore.ObjectLabel>,
        enrichedRewards: Map<String, Double> = emptyMap()
    ): String {
        val datasetFile = File(context.cacheDir, "lora_dataset.jsonl")

        datasetFile.bufferedWriter().use { writer ->

            // ── Experience-based training pairs ──────────────────────────────
            // Weight formula: 0.5 + enrichedReward × 1.5  → range [0.5, 2.0]
            // If no enriched reward exists for a tuple, use weight = 1.0 (neutral).
            experiences.forEach { tuple ->
                val prompt = "GOAL: ${tuple.taskType.take(60)}\n\n${tuple.screenSummary.take(400)}"
                val weight = enrichedRewards[tuple.id]
                    ?.let { r -> (0.5 + r * 1.5).toFloat().coerceIn(0.5f, 2.0f) }
                    ?: 1.0f
                writer.appendLine(trainingLine(prompt, tuple.actionJson, weight = weight))
            }

            // ── Label-based training pairs (3× weight — written 3 times) ─────
            labels.forEach { label ->
                // Variation 1: direct interaction goal
                val prompt1 = buildLabelPrompt(label, "Interact with ${label.name}")
                val output1 = label.toLoraTargetAction("Interact with ${label.name}")
                writer.appendLine(trainingLine(prompt1, output1, weight = 1.0f))

                // Variation 2: task-oriented goal using the label's context
                val prompt2 = buildLabelPrompt(label, label.context.take(60))
                val output2 = label.toLoraTargetAction(label.context)
                writer.appendLine(trainingLine(prompt2, output2, weight = 1.0f))

                // Variation 3: reasoning-context goal (teaches the agent the "why")
                if (label.reasoningContext.isNotBlank()) {
                    val prompt3 = buildLabelPrompt(label, "Complete: ${label.reasoningContext.take(50)}")
                    val output3 = """{"tool":"Click","node_id":"#labeled","reason":"${escapeJson(label.interactionHint.take(60))}"}"""
                    writer.appendLine(trainingLine(prompt3, output3, weight = 1.0f))
                }
            }
        }

        val lineCount = datasetFile.readLines().size
        Log.i(TAG, "Dataset written: $lineCount training lines → ${datasetFile.absolutePath}")
        return datasetFile.absolutePath
    }

    private fun buildLabelPrompt(label: ObjectLabelStore.ObjectLabel, goal: String): String {
        val knownElements = "[KNOWN ELEMENTS]\n${label.toPromptLine()}\n"
        val screenInfo = "APP: ${label.appPackage}  SCREEN: (labeled screen)"
        return "$knownElements\n$screenInfo\n\nGOAL: $goal"
    }

    private fun trainingLine(prompt: String, output: String, weight: Float): String {
        return """{"input":${escapeJson(prompt)},"output":${escapeJson(output)},"weight":$weight}"""
    }

    @Suppress("unused")
    private fun buildTrainingPrompt(goal: String, screenSummary: String): String =
        "GOAL: $goal\n\n$screenSummary"

    private fun escapeJson(s: String): String =
        "\"${s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "")}\""

    /**
     * Stub: writes metadata file so versioning and hot-reload paths can be tested
     * before the real nativeTrainLora() JNI is available.
     */
    private fun stubTrainLora(adapterPath: String, version: Int, expCount: Int, labelCount: Int) {
        File(adapterPath).writeText(
            """{"lora_stub":true,"rank":$LORA_RANK,"version":$version,"experience_samples":$expCount,"label_samples":$labelCount}"""
        )
    }

    // ─── Model ID helpers ─────────────────────────────────────────────────────

    /**
     * Derive a stable, filesystem-safe ID from a GGUF model path.
     * Example: "/files/models/llama-3.2-1b.Q4_K_M.gguf" → "llama-3.2-1b.Q4_K_M"
     * Used as the subdirectory name under lora/ and as the key in training_log.
     */
    fun modelId(modelPath: String): String =
        java.io.File(modelPath).nameWithoutExtension
            .replace(Regex("[^a-zA-Z0-9._-]"), "_")
            .take(64)
            .ifEmpty { "default" }

    // ─── Directory helpers ────────────────────────────────────────────────────

    /** Base lora directory (internal storage, writable fallback to external). */
    private fun loraDir(context: Context): File {
        val internal = File(context.filesDir, "lora").also { it.mkdirs() }
        if (internal.canWrite()) return internal
        return (context.getExternalFilesDir("lora") ?: internal).also { it.mkdirs() }
    }

    /**
     * Model-specific lora directory: lora/<modelId>/
     * Each model's adapters live here so they never overwrite each other.
     */
    private fun loraDir(context: Context, modelId: String): File =
        File(loraDir(context), modelId).also { it.mkdirs() }

    // ─── Version helpers ──────────────────────────────────────────────────────

    /**
     * Version for a specific model (reads lora/<modelId>/version.txt).
     * This is what LoRA training uses internally.
     */
    fun currentVersion(context: Context, modelId: String): Int {
        val versionFile = File(loraDir(context, modelId), "version.txt")
        return if (versionFile.exists()) versionFile.readText().trim().toIntOrNull() ?: 0 else 0
    }

    /**
     * Global version for display / monitoring — returns the highest version
     * across all model subdirs, falling back to the legacy flat lora/version.txt.
     * Callers that don't know the active model (MonitoringPusher, UI stats) use this.
     */
    fun currentVersion(context: Context): Int {
        // Legacy flat version.txt (backward compat for existing installs)
        val flatVersion = File(loraDir(context), "version.txt").let { f ->
            if (f.exists()) f.readText().trim().toIntOrNull() ?: 0 else 0
        }
        // Highest version across all model subdirectories
        val subDirMax = loraDir(context).listFiles()
            ?.filter { it.isDirectory }
            ?.mapNotNull { dir ->
                val vf = File(dir, "version.txt")
                if (vf.exists()) vf.readText().trim().toIntOrNull() else null
            }
            ?.maxOrNull() ?: 0
        return maxOf(flatVersion, subDirMax)
    }

    private fun writeVersionFile(context: Context, version: Int, modelId: String) {
        File(loraDir(context, modelId), "version.txt").writeText(version.toString())
    }

    // ─── Adapter path helpers ─────────────────────────────────────────────────

    /**
     * Latest adapter path for a specific model (lora/<modelId>/adapter_vN.gguf).
     * Use this when the active model is known (e.g. when loading into LlamaEngine).
     */
    fun latestAdapterPath(context: Context, modelId: String): String? {
        val version = currentVersion(context, modelId)
        if (version == 0) return null
        val dir = loraDir(context, modelId)
        val gguf = File(dir, "adapter_v$version.gguf")
        if (gguf.exists()) return gguf.absolutePath
        val bin = File(dir, "adapter_v$version.bin")
        return if (bin.exists()) bin.absolutePath else null
    }

    /**
     * Latest adapter path without a known model — scans all model subdirs and
     * returns the adapter with the highest version number. Also falls back to the
     * legacy flat lora/adapter_vN.gguf structure for existing installs.
     * Used by ConfigStore default population and MonitoringPusher reporting.
     */
    fun latestAdapterPath(context: Context): String? {
        // Scan model subdirectories for the highest versioned adapter
        val bestSubDir = loraDir(context).listFiles()
            ?.filter { it.isDirectory }
            ?.mapNotNull { dir ->
                val vf = File(dir, "version.txt")
                val ver = if (vf.exists()) vf.readText().trim().toIntOrNull() ?: 0 else 0
                if (ver > 0) {
                    val gguf = File(dir, "adapter_v$ver.gguf")
                    if (gguf.exists()) return@mapNotNull Pair(ver, gguf.absolutePath)
                    val bin = File(dir, "adapter_v$ver.bin")
                    if (bin.exists()) return@mapNotNull Pair(ver, bin.absolutePath)
                }
                null
            }
            ?.maxByOrNull { it.first }

        if (bestSubDir != null) return bestSubDir.second

        // Legacy fallback: flat lora/adapter_vN.gguf (pre-model-aware builds)
        val flatVer = File(loraDir(context), "version.txt").let { f ->
            if (f.exists()) f.readText().trim().toIntOrNull() ?: 0 else 0
        }
        if (flatVer == 0) return null
        val gguf = File(loraDir(context), "adapter_v$flatVer.gguf")
        if (gguf.exists()) return gguf.absolutePath
        val bin = File(loraDir(context), "adapter_v$flatVer.bin")
        return if (bin.exists()) bin.absolutePath else null
    }

    // ─── JNI declaration ──────────────────────────────────────────────────────
    // Resolved at runtime from libllama-jni.so (compiled via CMakeLists.txt + NDK).
    // When the .so is absent (dev/stub mode), tryNativeTrainLora() catches
    // UnsatisfiedLinkError and falls back to stubTrainLora().
    //
    // C++ implementation: llama_jni.cpp → Java_com_ariaagent_mobile_core_rl_LoraTrainer_nativeTrainLora
    // Uses llama.cpp Adam optimizer via common/train.h + ggml gradient API.
    // Trains rank-${LORA_RANK} LoRA matrices for 1 epoch on the JSONL dataset.
    // Returns true on success, false if training failed or was skipped.
    private external fun nativeTrainLora(
        modelPath: String,
        datasetPath: String,
        outputPath: String,
        rank: Int,
        epochs: Int
    ): Boolean
}
