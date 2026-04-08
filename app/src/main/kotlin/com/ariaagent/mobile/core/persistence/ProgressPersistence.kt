package com.ariaagent.mobile.core.persistence

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * ProgressPersistence — The "Ralph Loop": self-referential progress logging.
 *
 * Problem: Without persistent state, the agent "forgets" what it has tried every time
 * the app restarts or the background process is killed by the Android LMK. It loops
 * forever on the same failed approaches or restarts a 20-step task from scratch.
 *
 * Solution: Two complementary files maintained on internal storage:
 *
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │ progress.txt  — append-only action log                                   │
 * │                                                                          │
 * │ Format: [2026-04-02 14:23:01] STEP 12 | Click #4 | result=success       │
 * │         [2026-04-02 14:23:05] STEP 13 | Swipe up | result=failure       │
 * │         [2026-04-02 14:23:07] NOTE: popup blocked navigation             │
 * │                                                                          │
 * │ The agent reads the last N lines at the start of every new task so Llama │
 * │ 3.2 can "sync" its context — it knows what was already tried.            │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │ goals.json  — structured sub-task state                                  │
 * │                                                                          │
 * │ Format: {                                                                │
 * │   "goal": "Open YouTube and play trending video",                        │
 * │   "subTasks": [                                                          │
 * │     {"id": "1", "label": "Open YouTube", "passed": true},               │
 * │     {"id": "2", "label": "Navigate to Trending", "passed": false}       │
 * │   ],                                                                     │
 * │   "updatedAt": "2026-04-02T14:23:07Z"                                   │
 * │ }                                                                        │
 * │                                                                          │
 * │ If the process is killed mid-task, on next launch the agent reads this  │
 * │ file and resumes from the first sub-task where passed=false.             │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 * Both files live in the app's filesDir — no cloud, no external storage permission needed.
 *
 * Phase: 14.4 (Advanced Architecture — Self-Referential Progress Persistence)
 */
object ProgressPersistence {

    private const val TAG = "ProgressPersistence"
    private const val PROGRESS_FILE = "aria_progress.txt"
    private const val GOALS_FILE    = "aria_goals.json"

    // Max lines returned by readContext() to keep LLM prompt within 4096 token budget
    private const val CONTEXT_LINES = 20

    private val dateFmt = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
    private val isoFmt  = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US)

    // ─── Data types ───────────────────────────────────────────────────────────

    data class SubTask(
        val id: String,
        val label: String,
        val passed: Boolean = false
    )

    data class GoalState(
        val goal: String,
        val subTasks: List<SubTask>,
        val updatedAt: String = ""
    )

    // ─── progress.txt ─────────────────────────────────────────────────────────

    /**
     * Append a step result to progress.txt.
     *
     * Call after every AgentLoop step — even failures.
     * Thread-safe: synchronized on the file path.
     *
     * @param context   Android context (used to resolve filesDir)
     * @param stepNum   Current step number in this session
     * @param actionJson The JSON action the agent decided to take
     * @param result    "success" or "failure"
     */
    fun logStep(context: Context, stepNum: Int, actionJson: String, result: String) {
        val tool = extractField(actionJson, "tool") ?: "unknown"
        val nodeId = extractField(actionJson, "node_id") ?: ""
        val label = if (nodeId.isNotEmpty()) "$tool $nodeId" else tool
        val line = "[${dateFmt.format(Date())}] STEP $stepNum | $label | result=$result"
        appendLine(context, line)
    }

    /**
     * Append a free-text observation note to progress.txt.
     *
     * Use for significant agent observations: "popup appeared", "wrong app opened", etc.
     */
    fun logNote(context: Context, note: String) {
        val line = "[${dateFmt.format(Date())}] NOTE: $note"
        appendLine(context, line)
    }

    /**
     * Append a task-boundary marker so the log is easier to scan.
     */
    fun logTaskStart(context: Context, goal: String) {
        appendLine(context, "")
        appendLine(context, "══ TASK START [${dateFmt.format(Date())}]: $goal ══")
    }

    /**
     * Append a task completion marker.
     */
    fun logTaskEnd(context: Context, goal: String, succeeded: Boolean) {
        val marker = if (succeeded) "SUCCESS" else "ABANDONED"
        appendLine(context, "══ TASK END [$marker] [${dateFmt.format(Date())}]: $goal ══")
        appendLine(context, "")
    }

    /**
     * Read the last [CONTEXT_LINES] lines of progress.txt.
     *
     * Returned as a single compact string for injection into the LLM's context block.
     * Returns an empty string if no log exists yet.
     */
    fun readContext(context: Context): String {
        val file = File(context.filesDir, PROGRESS_FILE)
        if (!file.exists()) return ""
        return runCatching {
            val lines = file.readLines()
            lines.takeLast(CONTEXT_LINES).joinToString("\n")
        }.getOrDefault("")
    }

    private fun appendLine(context: Context, line: String) {
        val file = File(context.filesDir, PROGRESS_FILE)
        synchronized(PROGRESS_FILE) {
            runCatching {
                PrintWriter(FileWriter(file, true)).use { it.println(line) }
            }.onFailure { Log.e(TAG, "appendLine failed: ${it.message}") }
        }
    }

    // ─── goals.json ───────────────────────────────────────────────────────────

    /**
     * Write (or overwrite) the goals.json file with a fresh goal and sub-task list.
     *
     * Call when the user starts a new task. The LLM should generate the sub-task
     * breakdown as part of its planning phase; pass those labels here.
     *
     * @param context    Android context
     * @param goal       High-level goal string
     * @param subTasks   Ordered list of sub-tasks (all passed=false initially)
     */
    fun initGoals(context: Context, goal: String, subTasks: List<String>) {
        val state = GoalState(
            goal = goal,
            subTasks = subTasks.mapIndexed { i, label ->
                SubTask(id = (i + 1).toString(), label = label)
            },
            updatedAt = isoFmt.format(Date())
        )
        writeGoalState(context, state)
        Log.i(TAG, "Goals initialised: '${goal}' with ${subTasks.size} sub-tasks")
    }

    /**
     * Mark a sub-task as passed.
     *
     * @param context  Android context
     * @param subTaskId  The id of the SubTask to mark (1-indexed string)
     *
     * If the id is not found, the call is a no-op (safe to call with stale IDs).
     */
    fun markSubTaskPassed(context: Context, subTaskId: String) {
        val state = readGoalState(context) ?: return
        val updated = state.copy(
            subTasks = state.subTasks.map { st ->
                if (st.id == subTaskId) st.copy(passed = true) else st
            },
            updatedAt = isoFmt.format(Date())
        )
        writeGoalState(context, updated)
        Log.i(TAG, "Sub-task $subTaskId marked passed in goals.json")
    }

    /**
     * Read the current goal state from goals.json.
     *
     * Returns null if no file exists or parsing fails.
     */
    fun readGoalState(context: Context): GoalState? {
        val file = File(context.filesDir, GOALS_FILE)
        if (!file.exists()) return null
        return runCatching {
            val json = JSONObject(file.readText())
            val goal = json.getString("goal")
            val arr  = json.getJSONArray("subTasks")
            val subTasks = (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                SubTask(
                    id     = obj.getString("id"),
                    label  = obj.getString("label"),
                    passed = obj.optBoolean("passed", false)
                )
            }
            GoalState(
                goal      = goal,
                subTasks  = subTasks,
                updatedAt = json.optString("updatedAt", "")
            )
        }.getOrNull()
    }

    /**
     * Return the first sub-task that has not yet passed, or null if all are done.
     *
     * Used by AgentLoop on resume to skip already-completed steps.
     */
    fun nextPendingSubTask(context: Context): SubTask? {
        return readGoalState(context)?.subTasks?.firstOrNull { !it.passed }
    }

    /**
     * Build a compact summary of the goal state for LLM injection.
     *
     * Example output:
     *   Goal: Open YouTube and play trending video
     *   [x] Open YouTube
     *   [x] Navigate to Trending
     *   [ ] Tap first video
     *
     * Returns empty string if no goals file exists.
     */
    fun goalSummary(context: Context): String {
        val state = readGoalState(context) ?: return ""
        return buildString {
            append("Goal: ${state.goal}\n")
            state.subTasks.forEach { st ->
                val check = if (st.passed) "[x]" else "[ ]"
                append("$check ${st.label}\n")
            }
        }.trimEnd()
    }

    /**
     * Clear both log files. Call when the user explicitly resets the agent.
     */
    fun clear(context: Context) {
        File(context.filesDir, PROGRESS_FILE).delete()
        File(context.filesDir, GOALS_FILE).delete()
        Log.i(TAG, "Progress and goals cleared")
    }

    // ─── Internal helpers ─────────────────────────────────────────────────────

    private fun writeGoalState(context: Context, state: GoalState) {
        val json = JSONObject().apply {
            put("goal", state.goal)
            put("updatedAt", state.updatedAt)
            put("subTasks", JSONArray().apply {
                state.subTasks.forEach { st ->
                    put(JSONObject().apply {
                        put("id",     st.id)
                        put("label",  st.label)
                        put("passed", st.passed)
                    })
                }
            })
        }
        synchronized(GOALS_FILE) {
            runCatching {
                File(context.filesDir, GOALS_FILE).writeText(json.toString(2))
            }.onFailure { Log.e(TAG, "writeGoalState failed: ${it.message}") }
        }
    }

    private fun extractField(json: String, field: String): String? =
        Regex("\"$field\"\\s*:\\s*\"([^\"]+)\"").find(json)?.groupValues?.get(1)
}
