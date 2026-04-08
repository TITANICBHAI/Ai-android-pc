package com.ariaagent.mobile.core.agent

import android.content.Context
import android.util.Log
import com.ariaagent.mobile.core.ai.LlamaEngine

/**
 * TaskDecomposer — LLM pre-pass that breaks a natural-language goal into
 * an ordered list of concrete sub-tasks before AgentLoop starts executing.
 *
 * Why this matters:
 *   Without decomposition a 4-step task like "Open YouTube, search for coding
 *   tutorials, play the first result, enable captions" is treated as one opaque
 *   goal. The agent has no plan — it discovers sub-steps through trial and error.
 *   With decomposition the agent knows ALL steps upfront, marks each one done
 *   as it progresses, and can recover from interruptions by resuming at the
 *   correct sub-task.
 *
 * Output contract:
 *   - Returns a non-empty List<String> — always at least [goal] as fallback.
 *   - Steps are ordered, concrete, and each ≤ 10 words.
 *   - Maximum 7 steps (prevents over-decomposition on simple goals).
 *
 * Graceful fallbacks:
 *   - LLM not loaded → returns listOf(goal) immediately (0 ms overhead).
 *   - LLM output unparseable → same single-step fallback.
 *   - LLM in stub mode → stub output is detectable and falls back.
 *
 * Integration:
 *   Called by AgentLoop.start() before the main while loop.
 *   Results written to ProgressPersistence.initGoals() for persistence + resumption.
 *
 * Phase: 19 — Task Plan Decomposition.
 */
object TaskDecomposer {

    private const val TAG          = "TaskDecomposer"
    private const val MAX_SUBTASKS = 7
    private const val MAX_TOKENS   = 160   // enough for 7 numbered lines

    /**
     * Decompose [goal] into an ordered list of concrete sub-steps.
     *
     * When a dedicated PLANNING-role model is registered and its file exists on
     * disk, [LlamaEngine.withModel] temporarily swaps it in, runs decomposition,
     * then restores the previously loaded model. This happens only once per task
     * start so the swap overhead (~10–30 s) is acceptable.
     *
     * @param context         Android context — needed for model-file resolution.
     * @param goal            The user's full natural-language task goal.
     * @param planningModelId Catalog model-ID of the PLANNING-role model, or null
     *                        to use whichever model is currently loaded.
     * @return                Ordered list of step strings. Never empty — falls back to [goal].
     */
    suspend fun decompose(
        context: Context? = null,
        goal: String,
        planningModelId: String? = null
    ): List<String> {
        if (!LlamaEngine.isLoaded()) {
            Log.d(TAG, "LLM not loaded — single-step plan for: \"$goal\"")
            return listOf(goal)
        }

        val prompt = buildPrompt(goal)

        // Route to PLANNING specialist if one is registered and its file is on disk.
        // LlamaEngine.withModel() is a no-op when the model is already loaded.
        suspend fun runDecomposition(): List<String> {
            return try {
                val raw = "1." + LlamaEngine.infer(prompt, maxTokens = MAX_TOKENS, onToken = null)
                val steps = parseSteps(raw)
                if (steps.isEmpty()) {
                    Log.d(TAG, "No parseable steps — falling back to single-step plan")
                    listOf(goal)
                } else {
                    Log.i(TAG, "Decomposed \"$goal\" → ${steps.size} steps: ${steps.joinToString(" | ")}")
                    steps
                }
            } catch (e: Exception) {
                Log.w(TAG, "Decomposition failed (${e.message}) — single-step fallback")
                listOf(goal)
            }
        }

        return if (planningModelId != null && context != null &&
                   planningModelId != LlamaEngine.currentModelId) {
            Log.i(TAG, "Using PLANNING specialist: $planningModelId (swap in)")
            val result = LlamaEngine.withModel(planningModelId, context) { runDecomposition() }
            result ?: runDecomposition()   // fallback if model file missing
        } else {
            runDecomposition()
        }
    }

    // ─── Prompt ──────────────────────────────────────────────────────────────

    private fun buildPrompt(goal: String): String = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a step planner for an Android UI automation agent.
Break the goal into 2–$MAX_SUBTASKS ordered steps.
Rules:
- Each step is a short, concrete UI action (≤ 10 words).
- Output ONLY a numbered list. No explanation, no preamble.
- Steps must be doable one at a time on Android.
- If the goal is already a single action, output 1 step.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Goal: $goal
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".trimIndent()

    // ─── Parser ───────────────────────────────────────────────────────────────

    /**
     * Extract step strings from a numbered list in raw LLM output.
     *
     * Accepts formats: "1. foo", "1) foo", "1 - foo"
     * Filters lines that are too short (< 4 chars) or look like LLM boilerplate.
     */
    private fun parseSteps(raw: String): List<String> {
        val stepRegex = Regex("""^\s*\d+[\.\)\-]\s*(.+)""")
        return raw
            .lines()
            .mapNotNull { line -> stepRegex.find(line)?.groupValues?.get(1)?.trim() }
            .filter { step ->
                step.length >= 4 &&
                !step.startsWith("Note", ignoreCase = true) &&
                !step.startsWith("Remember", ignoreCase = true) &&
                !step.contains("stub inference", ignoreCase = true)
            }
            .take(MAX_SUBTASKS)
    }
}
