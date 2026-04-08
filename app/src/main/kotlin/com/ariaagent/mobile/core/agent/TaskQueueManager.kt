package com.ariaagent.mobile.core.agent

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * TaskQueueManager — JSON-file-backed sequential task queue.
 *
 * Allows the user to pre-queue multiple goals. When the running task completes
 * (success, failure, or max-steps), AgentLoop automatically dequeues the next
 * task and starts it — zero user intervention required.
 *
 * Persistence: `filesDir/aria_task_queue.json` — survives app restarts and LMK kills.
 *
 * Ordering: tasks are sorted by priority (lower = higher priority), then by
 * enqueue time (FIFO within the same priority level). Default priority = 0.
 *
 * Task JSON shape:
 * ```json
 * {
 *   "id":          "uuid",
 *   "goal":        "Open YouTube and find cat videos",
 *   "appPackage":  "com.google.android.youtube",
 *   "priority":    0,
 *   "enqueuedAt":  1234567890000
 * }
 * ```
 *
 * ControlScreen operations (via AgentViewModel):
 *   enqueueTask(goal, appPackage, priority) → QueuedTask
 *   dequeueTask()                           → QueuedTask?
 *   getTaskQueue()                          → List<QueuedTask>
 *   removeQueuedTask(taskId)                → Boolean
 *   clearTaskQueue()                        → Unit
 *
 * Phase: 15 — App Skill Registry & Task Chaining.
 */
object TaskQueueManager {

    private const val TAG        = "TaskQueueManager"
    private const val QUEUE_FILE = "aria_task_queue.json"

    // ─── Data model ───────────────────────────────────────────────────────────

    data class QueuedTask(
        val id: String,
        val goal: String,
        val appPackage: String,
        val priority: Int  = 0,
        val enqueuedAt: Long = System.currentTimeMillis(),
    )

    // ─── Read ─────────────────────────────────────────────────────────────────

    @Synchronized
    fun getAll(context: Context): List<QueuedTask> = readQueue(context)

    @Synchronized
    fun peek(context: Context): QueuedTask? = readQueue(context).firstOrNull()

    @Synchronized
    fun size(context: Context): Int = readQueue(context).size

    @Synchronized
    fun isEmpty(context: Context): Boolean = readQueue(context).isEmpty()

    // ─── Write ────────────────────────────────────────────────────────────────

    /**
     * Add a task to the queue. Tasks are sorted by priority (asc) then enqueuedAt (asc).
     * Returns the newly created [QueuedTask].
     */
    @Synchronized
    fun enqueue(context: Context, goal: String, appPackage: String, priority: Int = 0): QueuedTask {
        val task = QueuedTask(
            id         = java.util.UUID.randomUUID().toString(),
            goal       = goal.trim(),
            appPackage = appPackage.trim(),
            priority   = priority,
            enqueuedAt = System.currentTimeMillis(),
        )
        val current = readQueue(context).toMutableList()
        current.add(task)
        current.sortWith(compareBy({ it.priority }, { it.enqueuedAt }))
        writeQueue(context, current)
        Log.i(TAG, "Enqueued task ${task.id}: \"${task.goal}\" (priority=$priority, queueSize=${current.size})")
        return task
    }

    /**
     * Remove and return the head of the queue (highest priority, oldest within priority).
     * Returns `null` if the queue is empty.
     */
    @Synchronized
    fun dequeue(context: Context): QueuedTask? {
        val current = readQueue(context).toMutableList()
        if (current.isEmpty()) return null
        val head = current.removeAt(0)
        writeQueue(context, current)
        Log.i(TAG, "Dequeued task ${head.id}: \"${head.goal}\"")
        return head
    }

    /**
     * Remove a specific task by its UUID without dequeuing the head.
     * Returns `true` if a task was removed, `false` if the ID was not found.
     */
    @Synchronized
    fun remove(context: Context, taskId: String): Boolean {
        val current = readQueue(context).toMutableList()
        val before  = current.size
        current.removeAll { it.id == taskId }
        writeQueue(context, current)
        val removed = current.size < before
        if (removed) Log.i(TAG, "Removed task $taskId from queue (remaining=${current.size})")
        return removed
    }

    @Synchronized
    fun clear(context: Context) {
        writeQueue(context, emptyList())
        Log.i(TAG, "Task queue cleared")
    }

    // ─── JSON ─────────────────────────────────────────────────────────────────

    fun toJson(task: QueuedTask): JSONObject = JSONObject().apply {
        put("id",         task.id)
        put("goal",       task.goal)
        put("appPackage", task.appPackage)
        put("priority",   task.priority)
        put("enqueuedAt", task.enqueuedAt)
    }

    fun listToJson(tasks: List<QueuedTask>): JSONArray {
        val arr = JSONArray()
        tasks.forEach { arr.put(toJson(it)) }
        return arr
    }

    // ─── Serialization ────────────────────────────────────────────────────────

    private fun queueFile(context: Context) = File(context.filesDir, QUEUE_FILE)

    private fun readQueue(context: Context): List<QueuedTask> = runCatching {
        val file = queueFile(context)
        if (!file.exists()) return emptyList()
        val arr = JSONArray(file.readText())
        (0 until arr.length()).map { i ->
            val obj = arr.getJSONObject(i)
            QueuedTask(
                id         = obj.getString("id"),
                goal       = obj.getString("goal"),
                appPackage = obj.optString("appPackage", ""),
                priority   = obj.optInt("priority", 0),
                enqueuedAt = obj.optLong("enqueuedAt", 0L),
            )
        }
    }.getOrElse {
        Log.w(TAG, "Failed to read task queue: ${it.message}")
        emptyList()
    }

    private fun writeQueue(context: Context, tasks: List<QueuedTask>) = runCatching {
        queueFile(context).writeText(listToJson(tasks).toString())
    }.onFailure {
        Log.e(TAG, "Failed to write task queue: ${it.message}")
    }
}
