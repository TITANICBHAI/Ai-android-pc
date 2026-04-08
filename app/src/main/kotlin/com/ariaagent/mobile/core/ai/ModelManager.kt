package com.ariaagent.mobile.core.ai

import android.content.Context
import java.io.File
import java.security.MessageDigest

/**
 * ModelManager — single source of truth for GGUF model state on device.
 *
 * Supports multiple models from ModelCatalog. The user can download any or
 * all catalog models and activate whichever one they want. The active model
 * ID is persisted in SharedPreferences (key: "active_model_id").
 *
 * Backward-compatible constants (MODEL_URL, MODEL_FILENAME, EXPECTED_SIZE_BYTES)
 * always delegate to the currently active catalog entry so callers that have
 * not yet been updated continue to work correctly.
 *
 * Local / custom GGUF:
 *   The user can also supply an absolute path to any GGUF they placed on the
 *   device themselves. This is stored separately (key: "custom_model_path").
 *   When a custom path is set and the file exists, it takes priority over the
 *   active catalog model.
 */
object ModelManager {

    private const val PREFS_NAME        = "aria_model_prefs"
    private const val KEY_ACTIVE_ID     = "active_model_id"
    private const val KEY_CUSTOM_PATH   = "custom_model_path"

    // ── Active model helpers ──────────────────────────────────────────────────

    fun activeModelId(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_ACTIVE_ID, "") ?: ""
    }

    fun setActiveModelId(context: Context, id: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_ACTIVE_ID, id).apply()
    }

    fun activeEntry(context: Context): CatalogModel? =
        ModelCatalog.findById(activeModelId(context))

    // ── Custom / local model path ─────────────────────────────────────────────

    /** Returns the user-set absolute path to a local GGUF, or null if not set. */
    fun customModelPath(context: Context): String? {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_CUSTOM_PATH, null)?.takeIf { it.isNotBlank() }
    }

    fun setCustomModelPath(context: Context, path: String?) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_CUSTOM_PATH, path?.takeIf { it.isNotBlank() }).apply()
    }

    // ── Custom model type + mmproj (for user-supplied GGUFs) ─────────────────

    /**
     * Classification the user assigns to their custom GGUF so ARIA knows how to load it.
     *
     *   TEXT_LLM       — pure text model (Llama, Mistral, Phi, Gemma …).
     *                    Loaded via LlamaEngine.load(). If SmolVLM helper is downloaded
     *                    it is used as a separate screen-reader.
     *
     *   MULTIMODAL_VLM — vision+text model (LLaVA, Moondream, InternVL, SmolVLM …).
     *                    Requires a matching mmproj GGUF. Loaded via LlamaEngine.loadUnified()
     *                    so a single model instance handles both vision and reasoning.
     */
    enum class CustomModelType { TEXT_LLM, MULTIMODAL_VLM }

    private const val KEY_CUSTOM_TYPE     = "custom_model_type"
    private const val KEY_CUSTOM_MMPROJ   = "custom_mmproj_path"

    fun customModelType(context: Context): CustomModelType {
        val raw = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_CUSTOM_TYPE, null)
        return if (raw == CustomModelType.MULTIMODAL_VLM.name) CustomModelType.MULTIMODAL_VLM
               else CustomModelType.TEXT_LLM
    }

    fun setCustomModelType(context: Context, type: CustomModelType) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_CUSTOM_TYPE, type.name).apply()
    }

    fun customMmProjPath(context: Context): String? {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_CUSTOM_MMPROJ, null)?.takeIf { it.isNotBlank() }
    }

    fun setCustomMmProjPath(context: Context, path: String?) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_CUSTOM_MMPROJ, path?.takeIf { it.isNotBlank() }).apply()
    }

    // ── Backward-compatible constants (always reflect the active entry) ────────

    val MODEL_URL: String get() = activeEntry_static?.url ?: ModelCatalog.SMOLVLM_256M.url
    val MODEL_FILENAME: String get() = activeEntry_static?.filename ?: ModelCatalog.SMOLVLM_256M.filename
    val EXPECTED_SIZE_BYTES: Long get() = activeEntry_static?.expectedSizeBytes ?: ModelCatalog.SMOLVLM_256M.expectedSizeBytes

    // Static cache updated each time the active model is resolved — allows
    // ModelDownloadService (which doesn't always have a Context handy) to read
    // the active values via the top-level constants.
    @Volatile private var activeEntry_static: CatalogModel? = null

    private fun resolveActive(context: Context): CatalogModel? {
        val entry = activeEntry(context)
        activeEntry_static = entry
        return entry
    }

    // ── Directories ───────────────────────────────────────────────────────────

    fun modelDir(context: Context): File {
        val internal = File(context.filesDir, "models").also { it.mkdirs() }
        if (internal.canWrite()) return internal
        return (context.getExternalFilesDir("models") ?: internal).also { it.mkdirs() }
    }

    // ── Paths for the ACTIVE model ────────────────────────────────────────────

    fun modelPath(context: Context): File {
        val custom = customModelPath(context)
        if (custom != null) return File(custom)
        val entry = resolveActive(context) ?: return File(modelDir(context), "no_model_selected.gguf")
        return File(modelDir(context), entry.filename)
    }

    fun partialPath(context: Context): File {
        val entry = resolveActive(context) ?: return File(modelDir(context), "no_model_selected.gguf.part")
        return File(modelDir(context), "${entry.filename}.part")
    }

    // ── Paths for ANY catalog model (used by SettingsScreen / download logic) ─

    fun modelPathFor(context: Context, modelId: String): File {
        val entry = ModelCatalog.findById(modelId) ?: return modelPath(context)
        return File(modelDir(context), entry.filename)
    }

    fun partialPathFor(context: Context, modelId: String): File {
        val entry = ModelCatalog.findById(modelId) ?: return partialPath(context)
        return File(modelDir(context), "${entry.filename}.part")
    }

    fun isModelDownloaded(context: Context, modelId: String): Boolean {
        val entry = ModelCatalog.findById(modelId) ?: return false
        val f = File(modelDir(context), entry.filename)
        return f.exists() && f.length() >= entry.expectedSizeBytes
    }

    // ── Ready check (active model or custom path) ─────────────────────────────

    /**
     * Returns true if the currently active GGUF (or custom path) is fully
     * present and meets the minimum expected size.
     */
    fun isModelReady(context: Context): Boolean {
        val custom = customModelPath(context)
        if (custom != null) {
            val f = File(custom)
            return f.exists() && f.length() > 0
        }
        val entry = resolveActive(context) ?: return false
        val f = File(modelDir(context), entry.filename)
        return f.exists() && f.length() >= entry.expectedSizeBytes
    }

    /** Bytes already on disk for the active model (partial file, for resume). */
    fun downloadedBytes(context: Context): Long {
        val partial = partialPath(context)
        return if (partial.exists()) partial.length() else 0L
    }

    /** Move the partial file to the final path after verified download. */
    fun finalizeDownload(context: Context): Boolean {
        val partial = partialPath(context)
        val final   = modelPath(context)
        return partial.renameTo(final)
    }

    /**
     * Finalize a download for a specific catalog model ID.
     * Used by ModelDownloadService when downloading a non-active model.
     */
    fun finalizeDownloadFor(context: Context, modelId: String): Boolean {
        val partial = partialPathFor(context, modelId)
        val final   = modelPathFor(context, modelId)
        return partial.renameTo(final)
    }

    // ── SHA-256 ───────────────────────────────────────────────────────────────

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(8192)
            var read: Int
            while (input.read(buffer).also { read = it } != -1) {
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
