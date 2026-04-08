package com.ariaagent.mobile.core.memory

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/**
 * ObjectLabelStore — SQLite persistence for human-annotated UI element labels.
 *
 * The Object Labeler tool (future Phase UI) lets the user tap on a screenshot
 * and annotate each UI element with name, context, element type, and meaning.
 * The LLM then enriches each annotation with interaction hints and reasoning notes.
 *
 * These labels serve THREE purposes in the training pipeline:
 *
 *   1. PROMPT INJECTION (Phase 3 / AgentLoop)
 *      When the agent observes a screen, labels for that screen are retrieved
 *      and injected into the LLM prompt as [KNOWN ELEMENTS] context.
 *      This dramatically improves action accuracy on labeled apps.
 *
 *   2. LORA TRAINING DATA (Phase 5 / LoraTrainer)
 *      Labels are expert annotations of "on THIS screen, element X means Y
 *      and should be interacted with in this way." They become high-quality
 *      LoRA training pairs:
 *        Input:  screen summary + goal
 *        Output: correct action reasoning (from interaction_hint + reasoning_context)
 *      These outweigh plain experience tuples because they're human-verified.
 *
 *   3. RL REWARD SHAPING (Phase 5 / AgentLoop)
 *      When the agent interacts with a labeled element correctly (per interaction_hint),
 *      the reward is boosted by importance_score/10.
 *      This shapes the policy to prefer high-importance, well-understood elements.
 *
 * Schema: keyed by (app_package, screen_hash) so retrieval is O(1) on index.
 * screen_hash = hash of the OCR+a11y tree output for that screen state.
 */
class ObjectLabelStore private constructor(context: Context) :
    SQLiteOpenHelper(context, "aria_object_labels.db", null, 1) {

    companion object {
        @Volatile
        private var instance: ObjectLabelStore? = null

        fun getInstance(context: Context): ObjectLabelStore =
            instance ?: synchronized(this) {
                instance ?: ObjectLabelStore(context.applicationContext).also { instance = it }
            }
    }

    // ─── Data model ───────────────────────────────────────────────────────────

    enum class ElementType {
        BUTTON, TEXT, INPUT, ICON, IMAGE, CONTAINER, TOGGLE, LINK, UNKNOWN;

        companion object {
            fun fromString(s: String): ElementType =
                values().firstOrNull { it.name.equals(s, ignoreCase = true) } ?: UNKNOWN
        }
    }

    data class ObjectLabel(
        val id: String = UUID.randomUUID().toString(),
        val appPackage: String,
        val screenHash: String,
        val x: Float,                   // 0–1 normalized position on screenshot
        val y: Float,
        val name: String,               // user-assigned label name
        val context: String,            // user description of purpose
        val elementType: ElementType = ElementType.UNKNOWN,
        val ocrText: String = "",       // OCR text found at this location
        val meaning: String = "",       // LLM-generated: element meaning in app context
        val interactionHint: String = "",    // LLM-generated: how to interact with it
        val reasoningContext: String = "",   // LLM-generated: note for agent prompt
        val importanceScore: Int = 5,   // 0–10, LLM-generated
        val additionalFields: Map<String, String> = emptyMap(),
        val isEnriched: Boolean = false,
        val createdAt: Long = System.currentTimeMillis(),
        val updatedAt: Long = System.currentTimeMillis()
    ) {
        /**
         * Format this label for injection into the LLM prompt's [KNOWN ELEMENTS] section.
         * The agent uses this to understand what elements mean before it even taps them.
         */
        fun toPromptLine(): String {
            val enriched = if (isEnriched) "★" else "○"
            val hint = interactionHint.ifBlank { context }
            return "$enriched \"$name\" (${elementType.name.lowercase()}, importance ${importanceScore}/10): $hint" +
                if (reasoningContext.isNotBlank()) " | Agent note: $reasoningContext" else ""
        }

        /**
         * Format as a LoRA training target output for a given goal.
         * This is the "ideal" response the LLM should produce when it sees this element.
         */
        fun toLoraTargetAction(goal: String): String {
            val tool = when (elementType) {
                ElementType.BUTTON, ElementType.LINK -> "Click"
                ElementType.INPUT                    -> "Type"
                ElementType.TOGGLE                   -> "Click"
                else                                 -> "Click"
            }
            return """{"tool":"$tool","node_id":"#labeled","reason":"$reasoningContext. Goal: ${goal.take(40)}"}"""
        }
    }

    // ─── Schema ───────────────────────────────────────────────────────────────

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS object_labels (
                id TEXT PRIMARY KEY,
                app_package TEXT NOT NULL,
                screen_hash TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                name TEXT NOT NULL,
                context TEXT NOT NULL,
                element_type TEXT NOT NULL DEFAULT 'unknown',
                ocr_text TEXT NOT NULL DEFAULT '',
                meaning TEXT NOT NULL DEFAULT '',
                interaction_hint TEXT NOT NULL DEFAULT '',
                reasoning_context TEXT NOT NULL DEFAULT '',
                importance_score INTEGER NOT NULL DEFAULT 5,
                additional_fields TEXT NOT NULL DEFAULT '{}',
                is_enriched INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
        """.trimIndent())

        db.execSQL("CREATE INDEX IF NOT EXISTS idx_labels_screen ON object_labels(app_package, screen_hash)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_labels_app ON object_labels(app_package)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS object_labels")
        onCreate(db)
    }

    // ─── Write operations ─────────────────────────────────────────────────────

    fun save(label: ObjectLabel) {
        writableDatabase.insertWithOnConflict("object_labels", null, label.toContentValues(),
            SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun saveAll(labels: List<ObjectLabel>) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            labels.forEach { label ->
                db.insertWithOnConflict("object_labels", null, label.toContentValues(),
                    SQLiteDatabase.CONFLICT_REPLACE)
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun delete(id: String) {
        writableDatabase.delete("object_labels", "id=?", arrayOf(id))
    }

    fun clearForScreen(appPackage: String, screenHash: String) {
        writableDatabase.delete("object_labels",
            "app_package=? AND screen_hash=?", arrayOf(appPackage, screenHash))
    }

    fun clearAll() {
        writableDatabase.execSQL("DELETE FROM object_labels")
    }

    // ─── Read operations ──────────────────────────────────────────────────────

    /**
     * Get all labels for a specific screen state.
     * Called by AgentLoop before building the LLM prompt.
     * Sorted by importance_score DESC so the most critical elements appear first.
     */
    fun getByScreen(appPackage: String, screenHash: String): List<ObjectLabel> {
        val list = mutableListOf<ObjectLabel>()
        readableDatabase.rawQuery(
            "SELECT * FROM object_labels WHERE app_package=? AND screen_hash=? ORDER BY importance_score DESC",
            arrayOf(appPackage, screenHash)
        ).use { c ->
            while (c.moveToNext()) list.add(c.toObjectLabel())
        }
        return list
    }

    /**
     * Get all enriched labels for an entire app (across all screen states).
     * Used by LoraTrainer to build training data from labeled screens.
     */
    fun getEnrichedByApp(appPackage: String): List<ObjectLabel> {
        val list = mutableListOf<ObjectLabel>()
        readableDatabase.rawQuery(
            "SELECT * FROM object_labels WHERE app_package=? AND is_enriched=1 ORDER BY importance_score DESC",
            arrayOf(appPackage)
        ).use { c ->
            while (c.moveToNext()) list.add(c.toObjectLabel())
        }
        return list
    }

    /**
     * Get all labels across all apps — used for LoRA training dataset.
     * Only enriched labels are suitable for training (they have interaction_hint + reasoning_context).
     */
    fun getAllEnriched(limit: Int = 500): List<ObjectLabel> {
        val list = mutableListOf<ObjectLabel>()
        readableDatabase.rawQuery(
            "SELECT * FROM object_labels WHERE is_enriched=1 ORDER BY importance_score DESC LIMIT ?",
            arrayOf(limit.toString())
        ).use { c ->
            while (c.moveToNext()) list.add(c.toObjectLabel())
        }
        return list
    }

    fun count(): Int =
        readableDatabase.rawQuery("SELECT COUNT(*) FROM object_labels", null).use {
            if (it.moveToFirst()) it.getInt(0) else 0
        }

    fun countEnriched(): Int =
        readableDatabase.rawQuery("SELECT COUNT(*) FROM object_labels WHERE is_enriched=1", null).use {
            if (it.moveToFirst()) it.getInt(0) else 0
        }

    // ─── Serialization ────────────────────────────────────────────────────────

    private fun ObjectLabel.toContentValues() = ContentValues().apply {
        put("id", id)
        put("app_package", appPackage)
        put("screen_hash", screenHash)
        put("x", x)
        put("y", y)
        put("name", name)
        put("context", context)
        put("element_type", elementType.name.lowercase())
        put("ocr_text", ocrText)
        put("meaning", meaning)
        put("interaction_hint", interactionHint)
        put("reasoning_context", reasoningContext)
        put("importance_score", importanceScore)
        put("additional_fields", JSONObject(additionalFields).toString())
        put("is_enriched", if (isEnriched) 1 else 0)
        put("created_at", createdAt)
        put("updated_at", updatedAt)
    }

    private fun android.database.Cursor.toObjectLabel(): ObjectLabel {
        val additionalJson = getString(getColumnIndexOrThrow("additional_fields")) ?: "{}"
        val additionalMap = mutableMapOf<String, String>()
        runCatching {
            val obj = JSONObject(additionalJson)
            obj.keys().forEach { k -> additionalMap[k] = obj.getString(k) }
        }
        return ObjectLabel(
            id               = getString(getColumnIndexOrThrow("id")),
            appPackage       = getString(getColumnIndexOrThrow("app_package")),
            screenHash       = getString(getColumnIndexOrThrow("screen_hash")),
            x                = getFloat(getColumnIndexOrThrow("x")),
            y                = getFloat(getColumnIndexOrThrow("y")),
            name             = getString(getColumnIndexOrThrow("name")),
            context          = getString(getColumnIndexOrThrow("context")),
            elementType      = ElementType.fromString(getString(getColumnIndexOrThrow("element_type"))),
            ocrText          = getString(getColumnIndexOrThrow("ocr_text")) ?: "",
            meaning          = getString(getColumnIndexOrThrow("meaning")) ?: "",
            interactionHint  = getString(getColumnIndexOrThrow("interaction_hint")) ?: "",
            reasoningContext  = getString(getColumnIndexOrThrow("reasoning_context")) ?: "",
            importanceScore  = getInt(getColumnIndexOrThrow("importance_score")),
            additionalFields = additionalMap,
            isEnriched       = getInt(getColumnIndexOrThrow("is_enriched")) == 1,
            createdAt        = getLong(getColumnIndexOrThrow("created_at")),
            updatedAt        = getLong(getColumnIndexOrThrow("updated_at"))
        )
    }

    /**
     * Serialize a list of ObjectLabels to a JSON string.
     * Used for storage and inter-component transfer within the Kotlin layer.
     */
    fun toJson(labels: List<ObjectLabel>): String {
        val arr = JSONArray()
        labels.forEach { label ->
            arr.put(JSONObject().apply {
                put("id", label.id)
                put("appPackage", label.appPackage)
                put("screenHash", label.screenHash)
                put("x", label.x)
                put("y", label.y)
                put("name", label.name)
                put("context", label.context)
                put("elementType", label.elementType.name.lowercase())
                put("ocrText", label.ocrText)
                put("meaning", label.meaning)
                put("interactionHint", label.interactionHint)
                put("reasoningContext", label.reasoningContext)
                put("importanceScore", label.importanceScore)
                put("additionalFields", JSONObject(label.additionalFields))
                put("isEnriched", label.isEnriched)
                put("createdAt", label.createdAt)
                put("updatedAt", label.updatedAt)
            })
        }
        return arr.toString()
    }

    /**
     * Parse a JSON string back into ObjectLabels.
     */
    fun fromJson(json: String, appPackage: String, screenHash: String): List<ObjectLabel> {
        return runCatching {
            val arr = JSONArray(json)
            (0 until arr.length()).map { i ->
                val obj = arr.getJSONObject(i)
                val additionalMap = mutableMapOf<String, String>()
                runCatching {
                    val fields = obj.optJSONObject("additionalFields") ?: JSONObject()
                    fields.keys().forEach { k -> additionalMap[k] = fields.getString(k) }
                }
                ObjectLabel(
                    id              = obj.optString("id").ifBlank { UUID.randomUUID().toString() },
                    appPackage      = appPackage,
                    screenHash      = screenHash,
                    x               = obj.optDouble("x", 0.5).toFloat(),
                    y               = obj.optDouble("y", 0.5).toFloat(),
                    name            = obj.optString("name"),
                    context         = obj.optString("context"),
                    elementType     = ElementType.fromString(obj.optString("elementType", "unknown")),
                    ocrText         = obj.optString("ocrText"),
                    meaning         = obj.optString("meaning"),
                    interactionHint = obj.optString("interactionHint"),
                    reasoningContext = obj.optString("reasoningContext"),
                    importanceScore = obj.optInt("importanceScore", 5),
                    additionalFields = additionalMap,
                    isEnriched      = obj.optBoolean("isEnriched", false)
                )
            }
        }.getOrDefault(emptyList())
    }
}
