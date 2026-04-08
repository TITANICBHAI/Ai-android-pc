package com.ariaagent.mobile.core.memory

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * VisionEmbeddingStore — persists SmolVLM vision embeddings keyed by screenHash.
 *
 * Why: when an app has no accessibility tree (game, Unity, Flutter), text-based
 * memory retrieval fails — there is no text to embed. VisionEmbeddingStore
 * lets EmbeddingEngine retrieve past experiences by *visual similarity* instead:
 *   1. SmolVLM runs → produces a text description
 *   2. EmbeddingEngine.embed(description) → 384-dim float vector
 *   3. That vector is stored here, keyed by screenHash
 *   4. On future empty-a11y screens, retrieve by cosine similarity over stored vectors
 *      to find "I've seen this visual pattern before" even when OCR/a11y are silent
 *
 * Storage: each embedding = 384 × 4 bytes = 1.5 KB.
 * Cap: 2000 entries max (~3 MB) — oldest pruned on overflow.
 *
 * The embedding stored here is the MiniLM text embedding of SmolVLM's description,
 * NOT a raw pixel embedding. This keeps the model count at zero additional models.
 */
class VisionEmbeddingStore private constructor(context: Context) :
    SQLiteOpenHelper(context, "aria_vision_emb.db", null, 1) {

    companion object {
        @Volatile private var instance: VisionEmbeddingStore? = null
        fun getInstance(ctx: Context): VisionEmbeddingStore =
            instance ?: synchronized(this) {
                instance ?: VisionEmbeddingStore(ctx.applicationContext).also { instance = it }
            }

        private const val MAX_ROWS     = 2000
        private const val EMBEDDING_DIM = 384
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("""
            CREATE TABLE IF NOT EXISTS vision_embeddings (
                screen_hash      TEXT PRIMARY KEY,
                app_package      TEXT NOT NULL,
                description      TEXT NOT NULL,
                embedding_blob   BLOB NOT NULL,
                timestamp        INTEGER NOT NULL
            )
        """.trimIndent())
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_ve_ts ON vision_embeddings(timestamp DESC)")
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_ve_app ON vision_embeddings(app_package)")
    }

    override fun onUpgrade(db: SQLiteDatabase, old: Int, new: Int) {
        db.execSQL("DROP TABLE IF EXISTS vision_embeddings")
        onCreate(db)
    }

    /**
     * Store a vision embedding for a screen.
     * If the screenHash already exists, update the record.
     */
    fun store(screenHash: String, appPackage: String, description: String, embedding: FloatArray) {
        writableDatabase.insertWithOnConflict(
            "vision_embeddings", null,
            ContentValues().apply {
                put("screen_hash",    screenHash)
                put("app_package",    appPackage)
                put("description",    description.take(300))
                put("embedding_blob", floatsToBlob(embedding))
                put("timestamp",      System.currentTimeMillis())
            },
            SQLiteDatabase.CONFLICT_REPLACE
        )
        pruneIfNeeded()
    }

    /**
     * Retrieve top-K most visually similar screens for a given query embedding.
     * Returns list of (screenHash, appPackage, description, similarity) sorted descending.
     */
    fun retrieveSimilar(
        queryEmbedding: FloatArray,
        appPackage: String? = null,
        topK: Int = 5
    ): List<VisionMatch> {
        val where = if (appPackage != null) "WHERE app_package=?" else ""
        val args  = if (appPackage != null) arrayOf(appPackage)   else null

        val results = mutableListOf<Pair<VisionMatch, Float>>()

        readableDatabase.rawQuery(
            "SELECT screen_hash, app_package, description, embedding_blob FROM vision_embeddings $where ORDER BY timestamp DESC LIMIT 500",
            args
        ).use { c ->
            while (c.moveToNext()) {
                val blob      = c.getBlob(c.getColumnIndexOrThrow("embedding_blob"))
                val emb       = blobToFloats(blob)
                val sim       = cosineSimilarity(queryEmbedding, emb)
                results += Pair(
                    VisionMatch(
                        screenHash  = c.getString(0),
                        appPackage  = c.getString(1),
                        description = c.getString(2),
                        similarity  = sim
                    ),
                    sim
                )
            }
        }

        return results
            .sortedByDescending { it.second }
            .take(topK)
            .map { it.first }
    }

    data class VisionMatch(
        val screenHash:  String,
        val appPackage:  String,
        val description: String,
        val similarity:  Float
    )

    // ─── Serialisation helpers ────────────────────────────────────────────────

    private fun floatsToBlob(floats: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        floats.forEach { buf.putFloat(it) }
        return buf.array()
    }

    private fun blobToFloats(blob: ByteArray): FloatArray {
        val buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(blob.size / 4) { buf.getFloat() }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        val denom = kotlin.math.sqrt(na) * kotlin.math.sqrt(nb)
        return if (denom > 0f) dot / denom else 0f
    }

    private fun pruneIfNeeded() {
        val count = readableDatabase.rawQuery(
            "SELECT COUNT(*) FROM vision_embeddings", null
        ).use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }

        if (count > MAX_ROWS) {
            writableDatabase.execSQL(
                "DELETE FROM vision_embeddings WHERE screen_hash IN " +
                "(SELECT screen_hash FROM vision_embeddings ORDER BY timestamp ASC LIMIT ?)",
                arrayOf(count - MAX_ROWS)
            )
        }
    }
}
