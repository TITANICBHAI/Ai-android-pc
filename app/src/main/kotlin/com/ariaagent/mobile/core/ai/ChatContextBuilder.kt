package com.ariaagent.mobile.core.ai

import android.content.Context
import android.util.Log
import com.ariaagent.mobile.core.agent.AgentLoop
import com.ariaagent.mobile.core.agent.AppSkillRegistry
import com.ariaagent.mobile.core.agent.TaskQueueManager
import com.ariaagent.mobile.core.memory.ExperienceStore

/**
 * ChatContextBuilder — Kotlin-side system prompt builder for the chat interface.
 *
 * Previously, chat.tsx called four separate bridge methods (getAgentState,
 * getMemoryEntries, getTaskQueue, getAllAppSkills) and assembled the system prompt
 * in JavaScript. This violated the Kotlin-owns-logic rule and caused 4× bridge
 * round-trips per message. Now a single bridge call returns the fully-formed prompt.
 *
 * Called by: AgentViewModel.sendChatMessage()
 * Consumed by: ChatScreen via AgentViewModel.
 *
 * Assessment gap fix: non-UI logic migrated to pure Kotlin (JS-Independence principle).
 * Phase: Post-16 / Assessment
 */
object ChatContextBuilder {

    private const val TAG = "ChatContextBuilder"

    /**
     * Build the full system prompt for the ARIA chat interface.
     *
     * Reads directly from:
     *   - AgentLoop.state          (agent status, task, app, step count)
     *   - LlamaEngine              (loaded flag, tok/s, RAM)
     *   - ExperienceStore          (last 5 experience tuples as memory context)
     *   - TaskQueueManager         (queued goals list)
     *   - AppSkillRegistry         (per-app knowledge accumulated over sessions)
     *
     * @param context     Android context (needed for DB singletons)
     * @param userMessage The user's incoming message (reserved for future retrieval)
     * @param historyJson JSON array string of past messages [{role, text}] — reserved
     *                    for future retrieval-augmented memory lookup
     * @return            Fully-formatted system prompt string, ready to prepend before
     *                    the user's message when calling LlamaEngine.infer()
     */
    fun build(context: Context, userMessage: String, @Suppress("UNUSED_PARAMETER") historyJson: String): String {
        val lines = mutableListOf<String>()

        // ── System identity ──────────────────────────────────────────────────
        lines += "You are ARIA (Adaptive Reasoning Intelligence Agent), an on-device Android AI agent"
        lines += "running locally on a Samsung Galaxy M31 (Exynos 9611, 6 GB LPDDR4X RAM)."
        lines += "You reason and act entirely on-device. No cloud. No internet. All logic runs in Kotlin."
        lines += "You have: llama.cpp LLM inference · ML Kit OCR · AccessibilityService gestures ·"
        lines += "REINFORCE RL · LoRA fine-tuning · SQLite memory store · per-app skill registry."
        lines += "Answer concisely and helpfully. When the user asks you to do something on the device,"
        lines += "explain that they should use the Control tab to start a task — you cannot act directly"
        lines += "from this chat interface."
        lines += ""

        // ── Device / agent state ─────────────────────────────────────────────
        lines += "[DEVICE STATE]"
        val loop = AgentLoop.state
        lines += "Agent status    : ${loop.status.name.lowercase()}"
        lines += "Current task    : ${loop.goal.ifBlank { "none" }}"
        lines += "Current app     : ${loop.appPackage.ifBlank { "none" }}"
        lines += "Step count      : ${loop.stepCount}"
        lines += "LLM loaded      : ${if (LlamaEngine.isLoaded()) "yes" else "no — load it in Control first"}"
        lines += "Token rate      : ${if (LlamaEngine.lastToksPerSec > 0.0) "${LlamaEngine.lastToksPerSec.toInt()} tok/s" else "—"}"
        lines += "RAM used        : ${if (LlamaEngine.memoryMb > 0.0) "${LlamaEngine.memoryMb.toInt()} MB" else "—"}"
        lines += ""

        // ── Recent memory (last 5 experience tuples) ─────────────────────────
        try {
            val store = ExperienceStore.getInstance(context)
            val recent = store.getRecent(limit = 5)
            if (recent.isNotEmpty()) {
                lines += "[RECENT MEMORY] (last ${recent.size} entries from SQLite)"
                recent.forEachIndexed { i, entry ->
                    val app = entry.appPackage.substringAfterLast('.').ifBlank { "global" }
                    val reward = if (entry.reward >= 0) "+%.2f".format(entry.reward)
                                 else "%.2f".format(entry.reward)
                    val outcome = if (entry.result == "success") "✓" else "✗"
                    lines += "${i + 1}. [$app] $outcome (reward: $reward) ${entry.taskType}"
                }
                lines += ""
            }
        } catch (e: Exception) {
            Log.w(TAG, "Memory entries unavailable: ${e.message}")
        }

        // ── Task queue ───────────────────────────────────────────────────────
        try {
            val queue = TaskQueueManager.getAll(context)
            if (queue.isNotEmpty()) {
                lines += "[TASK QUEUE] (${queue.size} pending)"
                queue.take(3).forEachIndexed { i, task ->
                    val pkg = if (task.appPackage.isNotBlank()) " → ${task.appPackage}" else ""
                    lines += "${i + 1}. ${task.goal}$pkg"
                }
                if (queue.size > 3) lines += "   …and ${queue.size - 3} more"
                lines += ""
            }
        } catch (e: Exception) {
            Log.w(TAG, "Task queue unavailable: ${e.message}")
        }

        // ── Learned app skills (capped at 5 most-recent to keep context small) ──
        try {
            val skills = AppSkillRegistry.getInstance(context).getAll().take(5)
            if (skills.isNotEmpty()) {
                lines += "[LEARNED APP SKILLS] (${skills.size} recent)"
                skills.forEach { s ->
                    val name = s.appName.ifBlank {
                        s.appPackage.substringAfterLast('.').ifBlank { s.appPackage }
                    }
                    val rate = if (s.taskSuccess + s.taskFailure > 0) {
                        "${(s.successRate * 100).toInt()}% success"
                    } else "no runs yet"
                    lines += "• $name: ${s.taskSuccess}✓ / ${s.taskFailure}✗, $rate"
                }
                lines += ""
            }
        } catch (e: Exception) {
            Log.w(TAG, "App skills unavailable: ${e.message}")
        }

        return lines.joinToString("\n")
    }
}
