package com.ariaagent.mobile.core.rl

import android.util.Log
import com.ariaagent.mobile.core.ai.LlamaEngine
import com.ariaagent.mobile.core.ai.VisionEngine
import com.ariaagent.mobile.core.config.ModuleVisionMode
import com.ariaagent.mobile.core.memory.ExperienceStore

/**
 * LlmRewardEnricher — LLM-assisted reward rescoring for LoRA training quality.
 *
 * During idle training cycles, the base reward assigned by AgentLoop is coarse:
 *   +1.0 success, -0.5 failure, ±label_boost (binary from PixelVerifier).
 * Binary rewards are a weak training signal — a barely-passing gesture gets the
 * same weight as a precise, immediately-effective action.
 *
 * LlmRewardEnricher uses the already-loaded model(s) to re-evaluate each
 * experience tuple and assign a continuous quality score in [0.0, 1.0].
 * That score is converted to a LoRA dataset weight:
 *   weight = 0.5 + enrichedReward × 1.5   →   range [0.5, 2.0]
 *
 * High-quality reasoning (clear goal → precise action → confirmed success)
 * gets double weight in the JSONL dataset, while low-quality actions are
 * down-weighted rather than excluded — we still learn from bad examples.
 *
 * Design choices:
 *   - Only runs when a suitable model is loaded (skipped gracefully otherwise).
 *   - MAX_TUPLES_PER_CYCLE = 20 caps thermal/time cost during the idle window.
 *   - maxTokens = 8 (LLM_ONLY) or 24 (BOTH/VISION_ONLY) for the score output.
 *   - Falls back to original reward silently on any inference error.
 *   - Runs on Dispatchers.Default (called from LearningScheduler coroutine).
 *   - Thread-safe: reads only from ExperienceTuple values, no shared state.
 *
 * Model selection (ModuleVisionMode):
 *   LLM_ONLY     — only LlamaEngine text inference. Fastest. Default path.
 *   VISION_ONLY  — only VisionEngine text-mode scoring. Skips LlamaEngine.
 *                  Falls back to LlamaEngine if VisionEngine is not loaded.
 *                  Note: no bitmap is available here; VisionEngine text mode
 *                  uses a visual-reasoning prompt template without an image.
 *   BOTH         — chain vision + text: VisionEngine provides a visual-reasoning
 *                  context string, then LlamaEngine scores using that context.
 *                  Highest signal quality; highest cost. Falls back to LLM_ONLY
 *                  if either model is absent.
 *   AUTO         — try BOTH if both are loaded, VISION_ONLY if only vision is
 *                  loaded, LLM_ONLY otherwise. Recommended default.
 *
 * Phase: Post-16 / Assessment
 */
object LlmRewardEnricher {

    private const val TAG                   = "LlmRewardEnricher"
    private const val MAX_TUPLES_PER_CYCLE  = 20
    private const val SCREEN_CHARS          = 200   // keep prompt short for 8-token output
    private const val ACTION_CHARS          = 120
    private const val VISION_SCREEN_CHARS   = 350   // longer context for vision-augmented path

    // ─── Public API ───────────────────────────────────────────────────────────

    /**
     * Enrich rewards for a list of experience tuples using LLM / vision model scoring.
     *
     * Called from LearningScheduler before LoraTrainer.train().
     * Returns an empty map when no model is available.
     *
     * @param tuples     Full list of untraining-marked success tuples from ExperienceStore.
     *                   Only the first [MAX_TUPLES_PER_CYCLE] are scored (thermal guard).
     * @param visionMode Which model(s) to use for scoring. Default: AUTO.
     * @return Map of tuple.id → enriched reward in [0.0, 1.0].
     *         Absent entries indicate the original reward should be used.
     */
    suspend fun enrich(
        tuples:     List<ExperienceStore.ExperienceTuple>,
        visionMode: ModuleVisionMode = ModuleVisionMode.AUTO,
    ): Map<String, Double> {
        val llmLoaded    = LlamaEngine.isLoaded()
        val visionLoaded = VisionEngine.isLoaded()

        val effectiveMode = resolveMode(visionMode, llmLoaded, visionLoaded)

        if (effectiveMode == null) {
            Log.d(TAG, "No suitable model loaded for visionMode=$visionMode — skipping enrichment")
            return emptyMap()
        }

        val subset  = tuples.take(MAX_TUPLES_PER_CYCLE)
        val enriched = mutableMapOf<String, Double>()

        Log.i(TAG, "Enriching ${subset.size} tuples via mode=$effectiveMode (requested=$visionMode)")

        for (tuple in subset) {
            try {
                val score = scoreOneTuple(tuple, effectiveMode)
                if (score != null) {
                    enriched[tuple.id] = score
                    Log.d(TAG, "Tuple ${tuple.id.take(8)}: ${tuple.result} → enriched=%.3f [${effectiveMode.name}]".format(score))
                }
            } catch (e: Exception) {
                Log.w(TAG, "Scoring failed for ${tuple.id.take(8)}: ${e.message}")
            }
        }

        Log.i(TAG, "Enriched ${enriched.size}/${subset.size} rewards (avg=%.3f, mode=${effectiveMode.name})".format(
            enriched.values.average().takeIf { enriched.isNotEmpty() } ?: 0.0
        ))
        return enriched
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    /**
     * Resolve the effective scoring mode given actual model availability.
     * Returns null only when absolutely no model is available.
     */
    private fun resolveMode(
        requested:    ModuleVisionMode,
        llmLoaded:    Boolean,
        visionLoaded: Boolean,
    ): ModuleVisionMode? = when (requested) {
        ModuleVisionMode.AUTO -> when {
            llmLoaded && visionLoaded -> ModuleVisionMode.BOTH
            visionLoaded              -> ModuleVisionMode.VISION_ONLY
            llmLoaded                 -> ModuleVisionMode.LLM_ONLY
            else                      -> null
        }
        ModuleVisionMode.BOTH -> when {
            llmLoaded && visionLoaded -> ModuleVisionMode.BOTH
            llmLoaded                 -> ModuleVisionMode.LLM_ONLY    // graceful fallback
            visionLoaded              -> ModuleVisionMode.VISION_ONLY // graceful fallback
            else                      -> null
        }
        ModuleVisionMode.VISION_ONLY -> when {
            visionLoaded -> ModuleVisionMode.VISION_ONLY
            llmLoaded    -> ModuleVisionMode.LLM_ONLY               // fallback
            else         -> null
        }
        ModuleVisionMode.LLM_ONLY -> if (llmLoaded) ModuleVisionMode.LLM_ONLY else null
    }

    private suspend fun scoreOneTuple(
        tuple:        ExperienceStore.ExperienceTuple,
        effectiveMode: ModuleVisionMode,
    ): Double? = when (effectiveMode) {

        ModuleVisionMode.LLM_ONLY -> {
            val prompt = buildTextScoringPrompt(
                goal   = tuple.taskType.take(80),
                screen = tuple.screenSummary.take(SCREEN_CHARS),
                action = tuple.actionJson.take(ACTION_CHARS),
                result = tuple.result,
            )
            parseScore(LlamaEngine.infer(prompt, maxTokens = 8))
        }

        ModuleVisionMode.VISION_ONLY -> {
            // VisionEngine has no text-only scoring path (it requires a bitmap).
            // We use a vision-context-aware prompt template with LlamaEngine here,
            // which exploits multimodal models' richer UI understanding even without
            // a live image — they were trained on UI screenshots.
            // If VisionEngine is available but LlamaEngine is not, use the vision
            // model's text-inference path via the same engine object.
            val prompt = buildVisionAwareScoringPrompt(
                goal   = tuple.taskType.take(80),
                screen = tuple.screenSummary.take(VISION_SCREEN_CHARS),
                action = tuple.actionJson.take(ACTION_CHARS),
                result = tuple.result,
            )
            // Try VisionEngine text inference first (works on VLMs that support text-only mode)
            val visionScore = runCatching {
                parseScore(VisionEngine.inferTextOnly(prompt, maxTokens = 24))
            }.getOrNull()
            visionScore ?: if (LlamaEngine.isLoaded()) parseScore(LlamaEngine.infer(prompt, maxTokens = 8)) else null
        }

        ModuleVisionMode.BOTH -> {
            // Stage 1 — vision model generates a visual-reasoning description
            val visualContext = runCatching {
                VisionEngine.inferTextOnly(
                    buildVisualContextPrompt(
                        screen = tuple.screenSummary.take(VISION_SCREEN_CHARS),
                        action = tuple.actionJson.take(ACTION_CHARS),
                    ),
                    maxTokens = 40,
                )
            }.getOrElse { "" }

            // Stage 2 — text LLM scores using the enriched context
            val prompt = buildChainedScoringPrompt(
                goal          = tuple.taskType.take(80),
                screen        = tuple.screenSummary.take(SCREEN_CHARS),
                action        = tuple.actionJson.take(ACTION_CHARS),
                result        = tuple.result,
                visualContext = visualContext,
            )
            parseScore(LlamaEngine.infer(prompt, maxTokens = 8))
        }

        // AUTO should have been resolved before this point
        ModuleVisionMode.AUTO -> null
    }

    // ─── Prompt templates ─────────────────────────────────────────────────────

    /**
     * Minimal text-scoring prompt — works with any instruction-tuned model.
     * Only 8 output tokens needed (a decimal like "0.87").
     */
    private fun buildTextScoringPrompt(
        goal: String, screen: String, action: String, result: String,
    ): String = """Rate the quality of this Android agent action from 0.0 to 1.0.
Reply with ONLY a single decimal number. No words, no explanation.

Goal: $goal
Screen: $screen
Action: $action
Result: $result

Score:""".trimIndent()

    /**
     * Vision-aware prompt for VLM text-only inference (VISION_ONLY mode).
     * Uses visual-reasoning framing to help multimodal models leverage their
     * UI-screenshot pre-training even when no actual image is provided.
     */
    private fun buildVisionAwareScoringPrompt(
        goal: String, screen: String, action: String, result: String,
    ): String = """You are an Android UI expert evaluating agent actions visually.
Imagine looking at the screen described below. Score the action quality from 0.0 to 1.0.
Reply with ONLY a decimal number. No words.

UI Goal: $goal
Screen state (visual summary): $screen
Agent action: $action
Outcome: $result

Visual quality score:""".trimIndent()

    /** First-stage prompt for BOTH mode — ask vision model to reason about the action. */
    private fun buildVisualContextPrompt(screen: String, action: String): String =
        """Briefly describe in 1 sentence whether this Android UI action is appropriate:
Screen: $screen
Action: $action
Assessment:""".trimIndent()

    /**
     * Second-stage prompt for BOTH mode — combine vision model's assessment with LLM scoring.
     */
    private fun buildChainedScoringPrompt(
        goal: String, screen: String, action: String, result: String, visualContext: String,
    ): String = """Rate Android agent action quality from 0.0 to 1.0. Reply with ONLY a decimal.

Goal: $goal
Screen: $screen
Action: $action
Result: $result
${if (visualContext.isNotBlank()) "Visual assessment: $visualContext\n" else ""}
Score:""".trimIndent()

    /**
     * Extract the first valid float in [0.0, 1.0] from the model's raw output.
     *
     * Handles common model output patterns:
     *   "0.87"        → 0.87
     *   "0.87\n"      → 0.87
     *   " 0.87 points"→ 0.87
     *   "87%"         → 0.87
     *   "8.7"         → 0.87   (0–10 scale → divide by 10)
     *   "9"           → 0.9    (1–10 integer scale)
     */
    internal fun parseScore(raw: String): Double? {
        val trimmed = raw.trim()

        // Direct parse: "0.87"
        trimmed.toDoubleOrNull()?.takeIf { it in 0.0..1.0 }?.let { return it }

        // Percentage: "87%"
        Regex("""(\d{1,3})%""").find(trimmed)?.groupValues?.get(1)
            ?.toDoubleOrNull()
            ?.takeIf { it in 0.0..100.0 }
            ?.let { return it / 100.0 }

        // First decimal or integer in output
        Regex("""\b(\d+\.?\d*)\b""").find(trimmed)?.groupValues?.get(1)
            ?.toDoubleOrNull()
            ?.let { v ->
                return when {
                    v in 0.0..1.0  -> v               // already in [0, 1]
                    v in 1.0..10.0 -> v / 10.0        // 0–10 scale → [0, 1]
                    else           -> null
                }
            }

        return null
    }
}
