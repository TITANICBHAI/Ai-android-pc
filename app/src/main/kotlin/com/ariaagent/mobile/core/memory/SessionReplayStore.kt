package com.ariaagent.mobile.core.memory

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * SessionReplayStore — per-step DVR for every agent session.
 *
 * Every AgentLoop step writes one ReplayEntry here.
 * The ActivityScreen "Replay" tab reads these to render the timeline:
 *   - Horizontal scrollable strip of colored blocks (green/red/yellow)
 *   - Tap any block → detail sheet (action JSON + LLM reason)
 *
 * Retention: last 50 sessions (older sessions auto-pruned on insert).
 * Max steps per session: 200 (AgentLoop.MAX_STEPS cap).
 *
 * Schema keeps blocks small (<500 bytes each) so the full 200-step
 * session history for 50 sessions fits comfortably in <5 MB.
 */
class SessionReplayStore private constructor(context: Context) :
    SQLiteOpenHelper(context, "aria_replay.db", null, 1) {

    companion object {
        @Volatile private var instance: SessionReplayStore? = null
        fun getInstance(ctx: Context): SessionReplayStore =
            instance ?: synchronized(this) {
                instance ?: SessionReplayStore(ctx.applicationContext).also { instance = it }
            }

        const val RESULT_SUCCESS = "success"
        const val RESULT_FAIL    = "fail"
        const val RESULT_STUCK   = "stuck"
        const val RESULT_WAIT    = "wait"
    }

    data class ReplayEntry(
        val sessionId:  String,
        val stepIdx:    Int,
        val screenHash: String,
        val actionJson: String,
        val reason:     String,
        val result:     String,
        val appPackage: String,
        val timestamp:  Long = System.currentTimeMillis()
    )

    data class SessionSummary(
        val sessionId:  String,
        val goal:       String,
        val stepCount:  Int,
        val succeeded:  Int,
        val failed:     Int,
        val startTime:  Long,
        val endTime:    Long
    )

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS replay_steps (
                session_id  TEXT    NOT NULL,
                step_idx    INTEGER NOT NULL,
                screen_hash TEXT    NOT NULL,
                action_json TEXT    NOT NULL,
                reason      TEXT    NOT NULL,
                result      TEXT    NOT NULL,
                app_package TEXT    NOT NULL,
                timestamp   INTEGER NOT NULL,
                PRIMARY KEY (session_id, step_idx)
            )
        """.trimIndent())
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS replay_sessions (
                session_id TEXT PRIMARY KEY,
                goal       TEXT    NOT NULL,
                start_time INTEGER NOT NULL,
                end_time   INTEGER NOT NULL DEFAULT 0,
                step_count INTEGER NOT NULL DEFAULT 0,
                succeeded  INTEGER NOT NULL DEFAULT 0,
                failed     INTEGER NOT NULL DEFAULT 0
            )
        """.trimIndent())
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_session_ts ON replay_sessions(start_time DESC)")
    }

    override fun onUpgrade(db: SQLiteDatabase, old: Int, new: Int) {
        db.execSQL("DROP TABLE IF EXISTS replay_steps")
        db.execSQL("DROP TABLE IF EXISTS replay_sessions")
        onCreate(db)
    }

    fun startSession(sessionId: String, goal: String) {
        writableDatabase.insertWithOnConflict(
            "replay_sessions", null,
            ContentValues().apply {
                put("session_id", sessionId)
                put("goal",       goal)
                put("start_time", System.currentTimeMillis())
            },
            SQLiteDatabase.CONFLICT_IGNORE
        )
        pruneOldSessions()
    }

    fun recordStep(entry: ReplayEntry) {
        writableDatabase.insertWithOnConflict(
            "replay_steps", null,
            ContentValues().apply {
                put("session_id",  entry.sessionId)
                put("step_idx",    entry.stepIdx)
                put("screen_hash", entry.screenHash)
                put("action_json", entry.actionJson.take(400))
                put("reason",      entry.reason.take(300))
                put("result",      entry.result)
                put("app_package", entry.appPackage)
                put("timestamp",   entry.timestamp)
            },
            SQLiteDatabase.CONFLICT_REPLACE
        )
        writableDatabase.execSQL("""
            UPDATE replay_sessions SET
                step_count = step_count + 1,
                succeeded  = succeeded  + CASE WHEN '${entry.result}' = 'success' THEN 1 ELSE 0 END,
                failed     = failed     + CASE WHEN '${entry.result}' = 'fail'    THEN 1 ELSE 0 END
            WHERE session_id = '${entry.sessionId}'
        """.trimIndent())
    }

    fun endSession(sessionId: String) {
        writableDatabase.execSQL(
            "UPDATE replay_sessions SET end_time=? WHERE session_id=?",
            arrayOf(System.currentTimeMillis(), sessionId)
        )
    }

    fun getRecentSessions(limit: Int = 20): List<SessionSummary> {
        val list = mutableListOf<SessionSummary>()
        readableDatabase.rawQuery(
            "SELECT * FROM replay_sessions ORDER BY start_time DESC LIMIT ?",
            arrayOf(limit.toString())
        ).use { c ->
            while (c.moveToNext()) {
                list += SessionSummary(
                    sessionId = c.getString(c.getColumnIndexOrThrow("session_id")),
                    goal      = c.getString(c.getColumnIndexOrThrow("goal")),
                    stepCount = c.getInt(c.getColumnIndexOrThrow("step_count")),
                    succeeded = c.getInt(c.getColumnIndexOrThrow("succeeded")),
                    failed    = c.getInt(c.getColumnIndexOrThrow("failed")),
                    startTime = c.getLong(c.getColumnIndexOrThrow("start_time")),
                    endTime   = c.getLong(c.getColumnIndexOrThrow("end_time"))
                )
            }
        }
        return list
    }

    fun getSteps(sessionId: String): List<ReplayEntry> {
        val list = mutableListOf<ReplayEntry>()
        readableDatabase.rawQuery(
            "SELECT * FROM replay_steps WHERE session_id=? ORDER BY step_idx ASC",
            arrayOf(sessionId)
        ).use { c ->
            while (c.moveToNext()) {
                list += ReplayEntry(
                    sessionId  = c.getString(c.getColumnIndexOrThrow("session_id")),
                    stepIdx    = c.getInt(c.getColumnIndexOrThrow("step_idx")),
                    screenHash = c.getString(c.getColumnIndexOrThrow("screen_hash")),
                    actionJson = c.getString(c.getColumnIndexOrThrow("action_json")),
                    reason     = c.getString(c.getColumnIndexOrThrow("reason")),
                    result     = c.getString(c.getColumnIndexOrThrow("result")),
                    appPackage = c.getString(c.getColumnIndexOrThrow("app_package")),
                    timestamp  = c.getLong(c.getColumnIndexOrThrow("timestamp"))
                )
            }
        }
        return list
    }

    private fun pruneOldSessions() {
        readableDatabase.rawQuery(
            "SELECT session_id FROM replay_sessions ORDER BY start_time DESC LIMIT -1 OFFSET 50",
            null
        ).use { c ->
            val old = mutableListOf<String>()
            while (c.moveToNext()) old += c.getString(0)
            old.forEach { id ->
                writableDatabase.execSQL("DELETE FROM replay_steps WHERE session_id=?", arrayOf(id))
                writableDatabase.execSQL("DELETE FROM replay_sessions WHERE session_id=?", arrayOf(id))
            }
        }
    }
}
