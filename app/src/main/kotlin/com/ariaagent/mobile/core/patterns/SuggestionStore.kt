package com.ariaagent.mobile.core.patterns

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper

/**
 * SuggestionStore — persists proactive automation suggestions.
 *
 * A suggestion is created when UsagePatternTracker.record() returns a PatternHit.
 * DashboardScreen reads pending suggestions and shows a dismissible banner.
 *
 * Suggestion lifecycle:
 *   PENDING   → shown as banner on DashboardScreen
 *   ACCEPTED  → user tapped "Automate it" → goal launched via AgentLoop
 *   DISMISSED → user tapped "Not now" → hidden for 7 days before resurfacing
 *   EXPIRED   → 14 days old and never accepted
 *
 * At most 5 pending suggestions shown at once (oldest auto-dismissed if overflow).
 */
class SuggestionStore private constructor(context: Context) :
    SQLiteOpenHelper(context, "aria_suggestions.db", null, 1) {

    companion object {
        @Volatile private var instance: SuggestionStore? = null
        fun getInstance(ctx: Context): SuggestionStore =
            instance ?: synchronized(this) {
                instance ?: SuggestionStore(ctx.applicationContext).also { instance = it }
            }

        const val STATE_PENDING   = "pending"
        const val STATE_ACCEPTED  = "accepted"
        const val STATE_DISMISSED = "dismissed"
        const val STATE_EXPIRED   = "expired"

        private const val MAX_PENDING  = 5
        private const val EXPIRE_MS    = 14L * 24 * 60 * 60 * 1000L
        private const val SNOOZE_MS    = 7L  * 24 * 60 * 60 * 1000L
    }

    data class Suggestion(
        val id:           Long = 0,
        val appPackage:   String,
        val goalText:     String,
        val repeatCount:  Int,
        val suggestionText: String,
        val state:        String = STATE_PENDING,
        val createdAt:    Long   = System.currentTimeMillis(),
        val dismissedAt:  Long   = 0
    )

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS suggestions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                app_package     TEXT    NOT NULL,
                goal_text       TEXT    NOT NULL,
                repeat_count    INTEGER NOT NULL DEFAULT 0,
                suggestion_text TEXT    NOT NULL,
                state           TEXT    NOT NULL DEFAULT 'pending',
                created_at      INTEGER NOT NULL,
                dismissed_at    INTEGER NOT NULL DEFAULT 0
            )
        """.trimIndent())
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_sug_state ON suggestions(state, created_at DESC)")
    }

    override fun onUpgrade(db: SQLiteDatabase, old: Int, new: Int) {
        db.execSQL("DROP TABLE IF EXISTS suggestions")
        onCreate(db)
    }

    fun addSuggestion(suggestion: Suggestion): Long {
        pruneExpired()
        // Avoid duplicates — same app + same goal already pending
        val existing = readableDatabase.rawQuery(
            "SELECT id FROM suggestions WHERE app_package=? AND goal_text=? AND state=?",
            arrayOf(suggestion.appPackage, suggestion.goalText.take(120), STATE_PENDING)
        ).use { c -> if (c.moveToFirst()) c.getLong(0) else -1L }
        if (existing >= 0) return existing

        val id = writableDatabase.insert("suggestions", null, ContentValues().apply {
            put("app_package",     suggestion.appPackage)
            put("goal_text",       suggestion.goalText.take(120))
            put("repeat_count",    suggestion.repeatCount)
            put("suggestion_text", suggestion.suggestionText.take(300))
            put("state",           STATE_PENDING)
            put("created_at",      System.currentTimeMillis())
        })

        // Evict oldest if over limit
        val pendingCount = readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM suggestions WHERE state=?", arrayOf(STATE_PENDING)
        ).use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }

        if (pendingCount > MAX_PENDING) {
            writableDatabase.execSQL("""
                UPDATE suggestions SET state='$STATE_DISMISSED', dismissed_at=${System.currentTimeMillis()}
                WHERE state='$STATE_PENDING'
                ORDER BY created_at ASC LIMIT ${pendingCount - MAX_PENDING}
            """.trimIndent())
        }
        return id
    }

    fun getPending(): List<Suggestion> {
        pruneExpired()
        val now   = System.currentTimeMillis()
        val list  = mutableListOf<Suggestion>()
        readableDatabase.rawQuery(
            """SELECT * FROM suggestions
               WHERE state=? AND (dismissed_at=0 OR dismissed_at<?)
               ORDER BY created_at DESC LIMIT ?""",
            arrayOf(STATE_PENDING, (now - SNOOZE_MS).toString(), MAX_PENDING.toString())
        ).use { c ->
            while (c.moveToNext()) {
                list += Suggestion(
                    id             = c.getLong(c.getColumnIndexOrThrow("id")),
                    appPackage     = c.getString(c.getColumnIndexOrThrow("app_package")),
                    goalText       = c.getString(c.getColumnIndexOrThrow("goal_text")),
                    repeatCount    = c.getInt(c.getColumnIndexOrThrow("repeat_count")),
                    suggestionText = c.getString(c.getColumnIndexOrThrow("suggestion_text")),
                    state          = c.getString(c.getColumnIndexOrThrow("state")),
                    createdAt      = c.getLong(c.getColumnIndexOrThrow("created_at")),
                    dismissedAt    = c.getLong(c.getColumnIndexOrThrow("dismissed_at"))
                )
            }
        }
        return list
    }

    fun accept(id: Long) = setState(id, STATE_ACCEPTED)

    fun dismiss(id: Long) {
        writableDatabase.execSQL(
            "UPDATE suggestions SET state=?, dismissed_at=? WHERE id=?",
            arrayOf(STATE_DISMISSED, System.currentTimeMillis(), id)
        )
    }

    private fun setState(id: Long, state: String) {
        writableDatabase.execSQL(
            "UPDATE suggestions SET state=? WHERE id=?",
            arrayOf(state, id)
        )
    }

    private fun pruneExpired() {
        val cutoff = System.currentTimeMillis() - EXPIRE_MS
        writableDatabase.execSQL(
            "UPDATE suggestions SET state=? WHERE state=? AND created_at<?",
            arrayOf(STATE_EXPIRED, STATE_PENDING, cutoff)
        )
    }
}
