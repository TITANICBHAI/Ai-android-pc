package com.ariaagent.mobile.core.rl

import android.content.Context
import android.util.Log
import com.ariaagent.mobile.core.ai.LlamaEngine
import com.ariaagent.mobile.core.memory.ExperienceStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * DreamEngine — Passive Dream Mode.
 *
 * While the device is charging and idle (called from LearningScheduler after
 * LoRA training), DreamEngine takes the top remembered experiences and asks
 * LlamaEngine to imagine what a *better* next action could have been for each.
 *
 * Each synthetic pair is stored back into ExperienceStore as a synthetic tuple
 * (is_synthetic = 1). These feed into the next LoRA training cycle alongside
 * real experiences, effectively letting ARIA learn from hypothetical scenarios
 * it has never directly executed.
 *
 * Analogy: the agent "dreams" about its memories and rehearses better strategies,
 * similar to how biological memory consolidation during sleep strengthens learning.
 *
 * Input:  Top-N recent successful ExperienceTuples (seeds)
 * Output: N synthetic ExperienceTuples stored with is_synthetic=1
 *
 * Safety: skipped entirely if LlamaEngine is not loaded (avoids OOM competition
 * with concurrent LoRA training). Max 20 dreams per cycle to bound RAM usage.
 */
object DreamEngine {

    private const val TAG        = "DreamEngine"
    private const val MAX_DREAMS = 20

    /**
     * Run one dream cycle. Generates synthetic experience pairs from memories.
     * Suspend function — runs on Dispatchers.Default inside LearningScheduler's coroutine.
     * Returns the number of synthetic tuples generated.
     */
    suspend fun dream(context: Context): Int = withContext(Dispatchers.Default) {
        if (!LlamaEngine.isLoaded()) {
            Log.i(TAG, "Skipping dream — LlamaEngine not loaded")
            return@withContext 0
        }

        val store = ExperienceStore.getInstance(context)
        // Seeds come from the GLOBAL used_for_training flag (not per-model).
        // This is intentional: DreamEngine processes experiences for augmentation,
        // independent of which model they were LoRA-trained by. LoraTrainer always
        // calls markAsTrained() (global) immediately after markAsTrainedFor() (per-model),
        // so these seeds are guaranteed to be fresh and not re-processed every cycle.
        val seeds = store.getUntrainedSuccesses(limit = MAX_DREAMS)
        if (seeds.isEmpty()) {
            Log.i(TAG, "No seeds for dreaming — all experiences already processed")
            return@withContext 0
        }

        var count = 0
        for (seed in seeds) {
            try {
                val synthetic = generateDream(seed) ?: continue
                store.save(synthetic)
                count++
            } catch (e: Exception) {
                Log.w(TAG, "Dream failed for seed ${seed.id}: ${e.message}")
            }
        }

        Log.i(TAG, "Dream cycle complete: $count synthetic tuples generated")
        count
    }

    /**
     * Generate one synthetic experience from a real experience seed.
     *
     * Prompt design: show the seed's screen summary and ask the LLM to propose
     * an *alternative* action that might have been even more efficient. This
     * creates counterfactual training data — plausible paths ARIA didn't take.
     *
     * The synthetic action is stored with:
     *   - reward = 0.7 (slightly below confirmed real-world successes at ~0.9)
     *   - is_synthetic = true
     *   - edgeCaseNotes = "dream:v1" for traceability
     */
    private suspend fun generateDream(seed: ExperienceStore.ExperienceTuple): ExperienceStore.ExperienceTuple? {
        val prompt = buildDreamPrompt(seed)
        val raw = runCatching {
            LlamaEngine.infer(prompt, maxTokens = 80)
        }.getOrNull() ?: return null

        val actionJson = extractJson(raw) ?: return null

        return ExperienceStore.ExperienceTuple(
            appPackage    = seed.appPackage,
            taskType      = seed.taskType,
            screenSummary = seed.screenSummary,
            actionJson    = actionJson,
            result        = "success",
            reward        = 0.7,
            isSynthetic   = true,
            edgeCaseNotes = "dream:v1 seed=${seed.id.take(8)}"
        )
    }

    private fun buildDreamPrompt(seed: ExperienceStore.ExperienceTuple): String = """
You are helping an Android AI agent improve its strategy through reflection.

Past experience:
  App: ${seed.appPackage}
  Task: ${seed.taskType}
  Screen: ${seed.screenSummary.take(200)}
  Action taken: ${seed.actionJson.take(150)}
  Result: ${seed.result} (reward=${seed.reward})

Propose ONE alternative action JSON that could have been more efficient.
Output only valid JSON in this format (no explanation):
{"tool":"TapNode","nodeId":"#N","reason":"brief reason"}
""".trimIndent()

    private fun extractJson(raw: String): String? {
        val start = raw.indexOf('{')
        val end   = raw.lastIndexOf('}')
        if (start < 0 || end <= start) return null
        return raw.substring(start, end + 1).take(300)
    }
}
