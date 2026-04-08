package com.ariaagent.mobile.core.memory

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.content.ContentValues
import java.util.UUID

/**
 * ExperienceStore — SQLite database for the agent's learning pipeline.
 *
 * Every observe→reason→act loop produces one ExperienceTuple.
 * These tuples are the raw material for:
 *   - RL training (REINFORCE policy gradient update)
 *   - LoRA fine-tuning (successful traces → adapter weights)
 *   - Edge case memory (failures that eventually succeeded)
 *
 * The learning loop ONLY runs during idle + charging.
 * Collection happens continuously during active use.
 *
 * Phase: 4 (Data Collection)
 */
class ExperienceStore private constructor(context: Context) :
    SQLiteOpenHelper(context, "aria_experience.db", null, 3) {

    companion object {
        @Volatile
        private var instance: ExperienceStore? = null

        fun getInstance(context: Context): ExperienceStore =
            instance ?: synchronized(this) {
                instance ?: ExperienceStore(context.applicationContext).also { instance = it }
            }
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS experience (
                id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                app_package TEXT NOT NULL,
                task_type TEXT NOT NULL,
                screen_summary TEXT NOT NULL,
                action_json TEXT NOT NULL,
                result TEXT NOT NULL,
                reward REAL NOT NULL,
                is_edge_case INTEGER DEFAULT 0,
                edge_case_notes TEXT,
                session_id TEXT,
                used_for_training INTEGER DEFAULT 0,
                is_synthetic      INTEGER DEFAULT 0
            )
        """.trimIndent())

        db.execSQL("""
            CREATE TABLE IF NOT EXISTS edge_cases (
                id TEXT PRIMARY KEY,
                screen_pattern TEXT NOT NULL,
                resolution TEXT NOT NULL,
                app_package TEXT NOT NULL,
                recall_count INTEGER DEFAULT 0,
                last_seen INTEGER NOT NULL
            )
        """.trimIndent())

        db.execSQL("CREATE INDEX IF NOT EXISTS idx_app ON experience(app_package)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_result ON experience(result)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_timestamp ON experience(timestamp DESC)")

        // v3: per-model training log so every model trains on ALL collected data,
        // not just experiences that haven't been consumed by a previous model.
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS training_log (
                experience_id TEXT NOT NULL,
                model_id      TEXT NOT NULL,
                trained_at    INTEGER NOT NULL,
                PRIMARY KEY (experience_id, model_id)
            )
        """.trimIndent())
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_tlog_model ON training_log(model_id)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            runCatching {
                db.execSQL("ALTER TABLE experience ADD COLUMN is_synthetic INTEGER DEFAULT 0")
            }
        }
        if (oldVersion < 3) {
            // Add training_log — non-destructive, no existing data lost
            runCatching {
                db.execSQL("""
                    CREATE TABLE IF NOT EXISTS training_log (
                        experience_id TEXT NOT NULL,
                        model_id      TEXT NOT NULL,
                        trained_at    INTEGER NOT NULL,
                        PRIMARY KEY (experience_id, model_id)
                    )
                """.trimIndent())
                db.execSQL("CREATE INDEX IF NOT EXISTS idx_tlog_model ON training_log(model_id)")
            }
        }
    }

    data class ExperienceTuple(
        val id: String = UUID.randomUUID().toString(),
        val timestamp: Long = System.currentTimeMillis(),
        val appPackage: String,
        val taskType: String,
        val screenSummary: String,
        val actionJson: String,
        val result: String,
        val reward: Double,
        val isEdgeCase: Boolean = false,
        val edgeCaseNotes: String? = null,
        val sessionId: String? = null,
        val isSynthetic: Boolean = false
    )

    fun save(tuple: ExperienceTuple) {
        writableDatabase.insert("experience", null, ContentValues().apply {
            put("id", tuple.id)
            put("timestamp", tuple.timestamp)
            put("app_package", tuple.appPackage)
            put("task_type", tuple.taskType)
            put("screen_summary", tuple.screenSummary)
            put("action_json", tuple.actionJson)
            put("result", tuple.result)
            put("reward", tuple.reward)
            put("is_edge_case", if (tuple.isEdgeCase) 1 else 0)
            put("edge_case_notes", tuple.edgeCaseNotes)
            put("session_id",   tuple.sessionId)
            put("is_synthetic", if (tuple.isSynthetic) 1 else 0)
        })
    }

    fun count(): Int =
        readableDatabase.rawQuery("SELECT COUNT(*) FROM experience", null).use {
            if (it.moveToFirst()) it.getInt(0) else 0
        }

    fun countByResult(result: String): Int =
        readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM experience WHERE result=?", arrayOf(result)
        ).use { if (it.moveToFirst()) it.getInt(0) else 0 }

    fun edgeCaseCount(): Int =
        readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM experience WHERE is_edge_case=1", null
        ).use { if (it.moveToFirst()) it.getInt(0) else 0 }

    fun getRecent(limit: Int): List<ExperienceTuple> {
        val list = mutableListOf<ExperienceTuple>()
        readableDatabase.rawQuery(
            "SELECT * FROM experience ORDER BY timestamp DESC LIMIT ?",
            arrayOf(limit.toString())
        ).use { c ->
            while (c.moveToNext()) {
                list.add(ExperienceTuple(
                    id = c.getString(c.getColumnIndexOrThrow("id")),
                    timestamp = c.getLong(c.getColumnIndexOrThrow("timestamp")),
                    appPackage = c.getString(c.getColumnIndexOrThrow("app_package")),
                    taskType = c.getString(c.getColumnIndexOrThrow("task_type")),
                    screenSummary = c.getString(c.getColumnIndexOrThrow("screen_summary")),
                    actionJson = c.getString(c.getColumnIndexOrThrow("action_json")),
                    result = c.getString(c.getColumnIndexOrThrow("result")),
                    reward = c.getDouble(c.getColumnIndexOrThrow("reward")),
                    isEdgeCase = c.getInt(c.getColumnIndexOrThrow("is_edge_case")) == 1
                ))
            }
        }
        return list
    }

    fun getUntrainedSuccesses(limit: Int = 100): List<ExperienceTuple> {
        val list = mutableListOf<ExperienceTuple>()
        readableDatabase.rawQuery(
            "SELECT * FROM experience WHERE result='success' AND used_for_training=0 LIMIT ?",
            arrayOf(limit.toString())
        ).use { c ->
            while (c.moveToNext()) {
                list.add(ExperienceTuple(
                    id = c.getString(c.getColumnIndexOrThrow("id")),
                    timestamp = c.getLong(c.getColumnIndexOrThrow("timestamp")),
                    appPackage = c.getString(c.getColumnIndexOrThrow("app_package")),
                    taskType = c.getString(c.getColumnIndexOrThrow("task_type")),
                    screenSummary = c.getString(c.getColumnIndexOrThrow("screen_summary")),
                    actionJson = c.getString(c.getColumnIndexOrThrow("action_json")),
                    result = c.getString(c.getColumnIndexOrThrow("result")),
                    reward = c.getDouble(c.getColumnIndexOrThrow("reward"))
                ))
            }
        }
        return list
    }

    fun markAsTrained(ids: List<String>) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            ids.forEach { id ->
                db.execSQL("UPDATE experience SET used_for_training=1 WHERE id=?", arrayOf(id))
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    // ── Per-model training tracking ───────────────────────────────────────────
    //
    // These methods replace the global used_for_training flag for LoRA training.
    // Every model gets its own "trained" set via the training_log table so that:
    //   - Switching from SmolVLM-256M to Moondream2 does NOT discard prior experiences
    //   - Moondream2 can train on ALL accumulated data from day one
    //   - SmolVLM-256M's trained records do NOT interfere with Moondream2's pipeline
    //
    // Callers outside of LoraTrainer (EmbeddingEngine, DreamEngine, LlmRewardEnricher)
    // continue using getUntrainedSuccesses() / markAsTrained() — those serve different
    // purposes (embedding, dream generation, reward scoring) and use the global flag.

    /**
     * Return successful experiences this specific [modelId] has NOT yet trained on.
     * Different models each see the full pool of successes until they train on them.
     */
    fun getUntrainedSuccessesFor(modelId: String, limit: Int = 100): List<ExperienceTuple> {
        val list = mutableListOf<ExperienceTuple>()
        readableDatabase.rawQuery(
            """SELECT * FROM experience
               WHERE result='success'
               AND id NOT IN (
                   SELECT experience_id FROM training_log WHERE model_id=?
               )
               ORDER BY timestamp DESC LIMIT ?""",
            arrayOf(modelId, limit.toString())
        ).use { c ->
            while (c.moveToNext()) {
                list.add(ExperienceTuple(
                    id            = c.getString(c.getColumnIndexOrThrow("id")),
                    timestamp     = c.getLong(c.getColumnIndexOrThrow("timestamp")),
                    appPackage    = c.getString(c.getColumnIndexOrThrow("app_package")),
                    taskType      = c.getString(c.getColumnIndexOrThrow("task_type")),
                    screenSummary = c.getString(c.getColumnIndexOrThrow("screen_summary")),
                    actionJson    = c.getString(c.getColumnIndexOrThrow("action_json")),
                    result        = c.getString(c.getColumnIndexOrThrow("result")),
                    reward        = c.getDouble(c.getColumnIndexOrThrow("reward"))
                ))
            }
        }
        return list
    }

    /**
     * Record that [modelId] has trained on [ids] so they won't be returned again
     * by [getUntrainedSuccessesFor] for this model. Other models are unaffected.
     */
    fun markAsTrainedFor(ids: List<String>, modelId: String) {
        if (ids.isEmpty()) return
        val db  = writableDatabase
        val now = System.currentTimeMillis()
        db.beginTransaction()
        try {
            ids.forEach { expId ->
                db.execSQL(
                    "INSERT OR IGNORE INTO training_log (experience_id, model_id, trained_at) VALUES (?,?,?)",
                    arrayOf(expId, modelId, now)
                )
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    /**
     * Count of successes this [modelId] has not yet trained on.
     * Shown in the UI alongside the active model name.
     */
    fun untrainedCountFor(modelId: String): Int =
        readableDatabase.rawQuery(
            """SELECT COUNT(*) FROM experience
               WHERE result='success'
               AND id NOT IN (SELECT experience_id FROM training_log WHERE model_id=?)""",
            arrayOf(modelId)
        ).use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }

    fun clearAll() {
        writableDatabase.execSQL("DELETE FROM experience")
        writableDatabase.execSQL("DELETE FROM edge_cases")
    }

    // ── Gradient reward helper ────────────────────────────────────────────────

    /**
     * Compute a standardised gradient reward signal for a completed step.
     *
     * The reward is used as the [ExperienceTuple.reward] value when storing
     * experiences, and as the RL training signal in LoraTrainer.
     *
     * Reward table (inspired by RLHF practice for agent tasks):
     *
     *   +1.0  task_done        — full task completed successfully
     *   +0.3  right_screen     — navigated to the correct app/screen
     *   +0.1  action_success   — action executed without error (partial progress)
     *    0.0  no_change        — neutral / wait / no visible result
     *   −0.5  wrong_app        — opened the wrong app / navigated away unexpectedly
     *   −1.0  destructive      — deleted data, sent wrong message, purchased accidentally
     *   −0.8  task_failed      — task aborted (stuck, error, timeout)
     *
     * @param outcome  One of: "task_done", "right_screen", "action_success",
     *                 "no_change", "wrong_app", "destructive", "task_failed"
     * @return Reward in [−1.0, +1.0].
     */
    fun gradientReward(outcome: String): Double = when (outcome) {
        "task_done"      ->  1.0
        "right_screen"   ->  0.3
        "action_success" ->  0.1
        "no_change"      ->  0.0
        "wrong_app"      -> -0.5
        "task_failed"    -> -0.8
        "destructive"    -> -1.0
        else             ->  0.0
    }
}
