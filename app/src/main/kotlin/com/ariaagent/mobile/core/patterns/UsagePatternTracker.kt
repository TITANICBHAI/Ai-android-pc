package com.ariaagent.mobile.core.patterns

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.util.Log

/**
 * UsagePatternTracker — detects repeated task routines and surfaces automation suggestions.
 *
 * Every time AgentLoop completes a task, it records:
 *   - the app package
 *   - a fingerprint of the action sequence (SHA hash of tools used in order)
 *   - the goal text
 *   - timestamp
 *
 * Pattern detection: if the SAME (appPackage + sequenceHash) appears 3+ times
 * within the last 7 days, the task is "recurring" and a suggestion is generated.
 *
 * Suggestions are stored in SuggestionStore and surfaced on DashboardScreen
 * as a dismissible banner: "You've done this 4 times — want me to automate it?"
 *
 * Design principles:
 *   - Zero LLM calls during recording (fast, always-on)
 *   - LLM used only once to generate the suggestion text (deferred to idle)
 *   - Suggestions expire after 7 days if not acted on
 */
class UsagePatternTracker private constructor(context: Context) :
    SQLiteOpenHelper(context, "aria_patterns.db", null, 1) {

    companion object {
        @Volatile private var instance: UsagePatternTracker? = null
        fun getInstance(ctx: Context): UsagePatternTracker =
            instance ?: synchronized(this) {
                instance ?: UsagePatternTracker(ctx.applicationContext).also { instance = it }
            }

        private const val TAG               = "UsagePatternTracker"
        private const val REPEAT_THRESHOLD  = 3           // how many times before surfacing
        private const val WINDOW_DAYS       = 7L
        private const val WINDOW_MS         = WINDOW_DAYS * 24 * 60 * 60 * 1000L
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS task_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                app_package     TEXT    NOT NULL,
                sequence_hash   TEXT    NOT NULL,
                goal_sample     TEXT    NOT NULL,
                action_sequence TEXT    NOT NULL,
                timestamp       INTEGER NOT NULL,
                step_count      INTEGER NOT NULL DEFAULT 0
            )
        """.trimIndent())
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_seq ON task_runs(sequence_hash, timestamp DESC)")
    }

    override fun onUpgrade(db: SQLiteDatabase, old: Int, new: Int) {
        db.execSQL("DROP TABLE IF EXISTS task_runs")
        onCreate(db)
    }

    /**
     * Record a completed task run.
     * @param actionSequence  Ordered list of tool names used (e.g. ["TapNode","TypeText","TapNode"])
     * Returns a PatternHit if this run pushes the pattern past the repeat threshold.
     */
    fun record(
        context: Context,
        appPackage: String,
        goal: String,
        actionSequence: List<String>,
        stepCount: Int
    ): PatternHit? {
        val seqHash  = sequenceHash(appPackage, actionSequence)
        val now      = System.currentTimeMillis()

        writableDatabase.insert("task_runs", null, ContentValues().apply {
            put("app_package",     appPackage)
            put("sequence_hash",   seqHash)
            put("goal_sample",     goal.take(120))
            put("action_sequence", actionSequence.joinToString(",").take(200))
            put("timestamp",       now)
            put("step_count",      stepCount)
        })

        pruneOld()

        val windowStart = now - WINDOW_MS
        val count = readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM task_runs WHERE sequence_hash=? AND timestamp>=?",
            arrayOf(seqHash, windowStart.toString())
        ).use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }

        return if (count >= REPEAT_THRESHOLD) {
            Log.i(TAG, "Pattern detected: $appPackage seq=$seqHash count=$count")
            PatternHit(
                appPackage     = appPackage,
                sequenceHash   = seqHash,
                goal           = goal,
                repeatCount    = count,
                actionSequence = actionSequence
            )
        } else null
    }

    fun getTopPatterns(limit: Int = 10): List<PatternStat> {
        val now         = System.currentTimeMillis()
        val windowStart = now - WINDOW_MS
        val list        = mutableListOf<PatternStat>()

        readableDatabase.rawQuery("""
            SELECT sequence_hash, app_package, goal_sample, action_sequence,
                   COUNT(*) as cnt, MAX(timestamp) as last_seen
            FROM task_runs
            WHERE timestamp >= ?
            GROUP BY sequence_hash
            ORDER BY cnt DESC
            LIMIT ?
        """.trimIndent(), arrayOf(windowStart.toString(), limit.toString())
        ).use { c ->
            while (c.moveToNext()) {
                list += PatternStat(
                    sequenceHash   = c.getString(0),
                    appPackage     = c.getString(1),
                    goalSample     = c.getString(2),
                    actionSequence = c.getString(3).split(","),
                    count          = c.getInt(4),
                    lastSeen       = c.getLong(5)
                )
            }
        }
        return list
    }

    private fun pruneOld() {
        val cutoff = System.currentTimeMillis() - WINDOW_MS
        writableDatabase.execSQL(
            "DELETE FROM task_runs WHERE timestamp<?",
            arrayOf(cutoff)
        )
    }

    private fun sequenceHash(appPackage: String, sequence: List<String>): String {
        val raw = "$appPackage|${sequence.joinToString(",")}"
        return try {
            val digest = java.security.MessageDigest.getInstance("SHA-256")
            val bytes  = digest.digest(raw.toByteArray())
            bytes.take(6).joinToString("") { "%02x".format(it) }
        } catch (_: Exception) {
            raw.hashCode().toString(16)
        }
    }

    data class PatternHit(
        val appPackage:     String,
        val sequenceHash:   String,
        val goal:           String,
        val repeatCount:    Int,
        val actionSequence: List<String>
    )

    data class PatternStat(
        val sequenceHash:   String,
        val appPackage:     String,
        val goalSample:     String,
        val actionSequence: List<String>,
        val count:          Int,
        val lastSeen:       Long
    )
}
