package com.ariaagent.mobile.core.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

/**
 * VisionEngine — Multimodal screen understanding via SmolVLM-256M + llama.cpp mtmd.
 *
 * Architecture:
 *   • Vision base GGUF  (~150 MB)  — SmolVLM-256M-Instruct-Q4_K_M.gguf
 *   • mmproj GGUF       (~50 MB)   — mmproj-SmolVLM-256M-Instruct-f16.gguf
 *
 * Both are downloaded on demand (not bundled in the APK) to internal storage.
 * The vision model uses completely independent LlamaEngine handles so it does
 * not interfere with the primary Llama 3.2-1B text model.
 *
 * Frame compression (aspect-preserving letterbox):
 *   Phone screens are portrait (e.g. 1080×2400). Naive square scaling distorts the UI.
 *   compressFrame() letterboxes the bitmap into a 384×384 canvas with black bars,
 *   preserving the screen's true aspect ratio. The JPEG size is still ~35–55 KB.
 *
 * Frame caching:
 *   Each call to describe() stores (screenHash → description). If the next call
 *   has the same hash, the cached description is returned immediately — no inference.
 *   This saves ~400 ms on steps where the screen has not changed.
 *
 * Goal-aware prompting:
 *   describe() accepts the agent's current goal and injects it into the vision prompt,
 *   so SmolVLM answers a targeted question rather than a generic screen description.
 *   Example prompt: "Goal: Turn off Bluetooth. What UI elements relate to Bluetooth on this screen?"
 *
 * Thread safety:
 *   ensureLoaded() uses a @Synchronized guard so concurrent coroutines cannot double-load.
 *
 * Phase 17 — Multimodal.
 */
object VisionEngine {

    private const val TAG = "VisionEngine"

    // ── SmolVLM-256M from ggml-org on HuggingFace ────────────────────────────
    // Note: Q4_K_M is not available for this model; Q8_0 is the smallest quant.

    const val VISION_MODEL_URL =
        "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/" +
        "SmolVLM-256M-Instruct-Q8_0.gguf"

    const val MMPROJ_URL =
        "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/" +
        "mmproj-SmolVLM-256M-Instruct-f16.gguf"

    const val VISION_MODEL_FILENAME = "smolvlm-256m-q8_0.gguf"
    const val MMPROJ_FILENAME       = "smolvlm-256m-mmproj-f16.gguf"

    const val VISION_MODEL_MIN_BYTES = 155_000_000L   // ~163 MB expected (Q8_0)
    const val MMPROJ_MIN_BYTES       =  40_000_000L   // ~50 MB expected

    /** Native resolution expected by SmolVLM CLIP encoder. */
    private const val VISION_RES = 384

    /** JPEG quality — 82 keeps file ~35–55 KB while preserving UI text readability. */
    private const val JPEG_QUALITY = 82

    // ── Frame cache ───────────────────────────────────────────────────────────

    @Volatile private var lastCacheKey: String = ""
    @Volatile private var lastCacheDesc: String = ""

    // ── File paths ────────────────────────────────────────────────────────────

    fun modelDir(context: Context): File {
        val internal = File(context.filesDir, "models").also { it.mkdirs() }
        if (internal.canWrite()) return internal
        return (context.getExternalFilesDir("models") ?: internal).also { it.mkdirs() }
    }

    fun visionModelPath(context: Context): File =
        File(modelDir(context), VISION_MODEL_FILENAME)

    fun mmProjPath(context: Context): File =
        File(modelDir(context), MMPROJ_FILENAME)

    fun visionModelPartial(context: Context): File =
        File(modelDir(context), "$VISION_MODEL_FILENAME.part")

    fun mmProjPartial(context: Context): File =
        File(modelDir(context), "$MMPROJ_FILENAME.part")

    // ── Readiness checks ──────────────────────────────────────────────────────

    fun isVisionModelFileReady(context: Context): Boolean {
        val f = visionModelPath(context)
        return f.exists() && f.length() >= VISION_MODEL_MIN_BYTES
    }

    fun isMmProjReady(context: Context): Boolean {
        val f = mmProjPath(context)
        return f.exists() && f.length() >= MMPROJ_MIN_BYTES
    }

    /** Both files present → vision can be loaded into LlamaEngine. */
    fun isVisionModelReady(context: Context): Boolean =
        isVisionModelFileReady(context) && isMmProjReady(context)

    fun mmProjDownloadedBytes(context: Context): Long {
        val partial = mmProjPartial(context)
        return if (partial.exists()) partial.length()
        else if (mmProjPath(context).exists()) mmProjPath(context).length()
        else 0L
    }

    fun visionModelDownloadedBytes(context: Context): Long {
        val partial = visionModelPartial(context)
        return if (partial.exists()) partial.length()
        else if (visionModelPath(context).exists()) visionModelPath(context).length()
        else 0L
    }

    // ── Load / unload ─────────────────────────────────────────────────────────

    /**
     * Ensure the vision model is loaded into LlamaEngine.
     * Thread-safe — a @Synchronized lock prevents double-loading from parallel coroutines.
     * No-op if already loaded. Returns false if files are missing or load fails.
     */
    @Synchronized
    fun ensureLoaded(context: Context, nGpuLayers: Int = 32, gpuBackend: String = "opencl"): Boolean {
        if (LlamaEngine.isVisionLoaded()) return true
        if (!isVisionModelReady(context)) {
            Log.w(TAG, "Vision model files not ready — download first")
            return false
        }
        val ok = LlamaEngine.loadVision(
            visionModelPath = visionModelPath(context).absolutePath,
            mmProjPath      = mmProjPath(context).absolutePath,
            contextSize     = 2048,
            nGpuLayers      = nGpuLayers,
            gpuBackend      = gpuBackend
        )
        Log.i(TAG, if (ok) "Vision loaded OK (nGpuLayers=$nGpuLayers backend=$gpuBackend)" else "Vision load FAILED")
        return ok
    }

    fun unload() {
        lastCacheKey = ""
        lastCacheDesc = ""
        LlamaEngine.unloadVision()
        Log.i(TAG, "Vision unloaded — cache cleared")
    }

    /** True if the vision model is currently loaded into LlamaEngine. */
    fun isLoaded(): Boolean = LlamaEngine.isVisionLoaded()

    /**
     * Text-only inference using the vision model engine without a bitmap.
     *
     * VLMs like SmolVLM-256M and Moondream2 are instruction-tuned on image+text pairs
     * but can also process text-only prompts — their language head is fully capable of
     * text reasoning. This method uses LlamaEngine's standard text inference pipeline
     * (no bitmap encoding) but is called from the vision-model context.
     *
     * Use cases:
     *   - LlmRewardEnricher VISION_ONLY / BOTH mode: score experience tuples without
     *     a saved bitmap by using the VLM's UI-trained reasoning on text descriptions.
     *   - IRL VISION_ONLY action inference fallback when the LLM is not loaded.
     *
     * Returns "" if neither the vision model nor LlamaEngine is loaded.
     */
    suspend fun inferTextOnly(prompt: String, maxTokens: Int = 64): String {
        return try {
            if (!LlamaEngine.isLoaded() && !LlamaEngine.isVisionLoaded()) return ""
            LlamaEngine.infer(prompt, maxTokens = maxTokens)
        } catch (e: Exception) {
            Log.w(TAG, "inferTextOnly failed: ${e.message}")
            ""
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Describe the current screen frame, optionally grounded in [goal] and [annotation].
     *
     * Caching: if [screenHash] matches the previous call AND [annotation] is unchanged,
     * the cached description is returned without running inference. Annotations bypass the
     * cache (different annotation = different understanding needed, even for the same frame).
     *
     * Goal-aware prompting: when [goal] is non-empty, the vision prompt becomes a
     * targeted question rather than a generic description.
     *
     * Annotation enrichment: when [annotation] is non-empty (user has explained what is
     * happening in this frame), it is injected directly into the SmolVLM prompt so the
     * vision model reasons with the user's logic, not just pixel patterns.
     *
     * @param bitmap      Current screen frame from ScreenObserver
     * @param goal        Agent's current task goal (empty = generic description)
     * @param screenHash  Screen identity key for frame deduplication
     * @param annotation  User explanation for this frame (IRL labeling / training context)
     * @param maxTokens   Token cap — 96 tokens ≈ 2–3 sentences, enough for screen context
     * @return            Vision description, or "" on failure/unavailability
     */
    suspend fun describe(
        context: Context,
        bitmap: Bitmap,
        goal: String = "",
        screenHash: String = "",
        annotation: String = "",
        maxTokens: Int = 96
    ): String {
        // ── Cache hit — screen hasn't changed and no annotation override ───────
        // Annotations always bypass cache: the same frame described differently by the
        // user must produce a fresh, annotation-aware vision output.
        val cacheKey = if (annotation.isBlank()) screenHash else ""
        if (cacheKey.isNotEmpty() && cacheKey == lastCacheKey && lastCacheDesc.isNotEmpty()) {
            Log.d(TAG, "Vision cache hit: $screenHash")
            return lastCacheDesc
        }

        if (!ensureLoaded(context)) return ""

        return try {
            val prompt = buildVisionPrompt(goal, annotation)
            val imageBytes = compressFrame(bitmap)
            val desc = LlamaEngine.inferWithVision(imageBytes, prompt, maxTokens)

            // Store in cache only when no annotation (annotation-enriched results are frame-specific)
            if (cacheKey.isNotEmpty() && desc.isNotBlank()) {
                lastCacheKey  = cacheKey
                lastCacheDesc = desc
            }
            desc
        } catch (e: Exception) {
            Log.w(TAG, "Vision inference failed: ${e.message}")
            ""
        }
    }

    // ── Prompt construction ───────────────────────────────────────────────────

    /**
     * Build a goal-aware, annotation-enriched vision prompt.
     *
     * Annotation-enriched (IRL training):
     *   The user has explained what this frame shows and what action is happening.
     *   SmolVLM is asked to confirm and expand on that reasoning visually.
     *   Example: "User context: I tapped the Share button here. Which visual element
     *             on screen matches this, and what does the screen state tell you?"
     *
     * Goal-aware (live agent):
     *   "Goal: Open Bluetooth settings. Which UI elements on this Android screen
     *    relate to Bluetooth or Settings? Be concise."
     *
     * Generic fallback:
     *   "Describe the Android UI on screen. List visible interactive elements. 2–3 sentences."
     */
    private fun buildVisionPrompt(goal: String, annotation: String = ""): String {
        return when {
            annotation.isNotBlank() && goal.isNotBlank() -> {
                "Goal: $goal. " +
                "User context for this frame: ${annotation.take(200)}. " +
                "Looking at this Android screen, confirm what UI element or state matches the user's " +
                "context and describe what is visually happening. Be concise — 2 sentences max."
            }
            annotation.isNotBlank() -> {
                "User context for this frame: ${annotation.take(200)}. " +
                "Looking at this Android screen, identify the UI elements and state that match " +
                "the user's description. What do you see that confirms or adds to their reasoning? " +
                "Be concise — 2 sentences max."
            }
            goal.isNotBlank() -> {
                "Goal: $goal. " +
                "Which UI elements on this Android screen are relevant to achieving this goal? " +
                "Describe only what matters. Be concise — 2 sentences max."
            }
            else -> {
                "Describe the Android UI visible on this screen. " +
                "List key interactive elements (buttons, menus, text fields) and any visible text. " +
                "Be concise — 2–3 sentences max."
            }
        }
    }

    // ── Aspect-preserving frame compression ───────────────────────────────────

    /**
     * Public wrapper for callers outside VisionEngine that need to compress a bitmap
     * before passing it to [LlamaEngine.inferWithVision].
     *
     * Used by AgentLoop in unified VLM mode where the active catalog model receives the
     * screenshot directly (no separate VisionEngine.describe() pass needed).
     */
    fun compressFramePublic(bitmap: Bitmap): ByteArray = compressFrame(bitmap)

    /**
     * Letterbox [bitmap] into a VISION_RES×VISION_RES canvas (black background) and
     * JPEG-compress it to a byte array.
     *
     * Why letterbox instead of stretch:
     *   A phone screen at 1080×2400 has a 1:2.22 aspect ratio. Squishing it to 384×384
     *   compresses vertical space by 2.22×, making buttons appear 2× wider than tall and
     *   confusing SmolVLM's spatial reasoning. Letterboxing preserves true proportions.
     *
     * The black bars occupy ~22% of the 384×384 canvas (sides) for portrait content,
     * which SmolVLM ignores — it understands letterbox format from its training data.
     */
    private fun compressFrame(bitmap: Bitmap): ByteArray {
        val canvas = Bitmap.createBitmap(VISION_RES, VISION_RES, Bitmap.Config.RGB_565)
        val androidCanvas = Canvas(canvas)
        androidCanvas.drawColor(Color.BLACK)

        val srcW = bitmap.width.toFloat()
        val srcH = bitmap.height.toFloat()
        val scale = minOf(VISION_RES / srcW, VISION_RES / srcH)
        val dstW  = (srcW * scale).toInt()
        val dstH  = (srcH * scale).toInt()
        val dstX  = (VISION_RES - dstW) / 2
        val dstY  = (VISION_RES - dstH) / 2

        val dst = Rect(dstX, dstY, dstX + dstW, dstY + dstH)
        androidCanvas.drawBitmap(bitmap, null, dst, null as Paint?)

        return ByteArrayOutputStream(56_000).use { baos ->
            canvas.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, baos)
            canvas.recycle()
            baos.toByteArray()
        }
    }

    // ── Download helpers ──────────────────────────────────────────────────────

    /**
     * Download a single file from [url] to [dest], resuming from [partial].
     * Reports progress via [onProgress] (bytesDownloaded, totalBytes).
     * Finalizes by renaming [partial] → [dest].
     *
     * Called by AgentViewModel.downloadVisionModel() on an IO coroutine.
     */
    fun downloadFile(
        url: String,
        dest: File,
        partial: File,
        onProgress: (downloaded: Long, total: Long) -> Unit
    ): Boolean {
        val client = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .followRedirects(true)
            .followSslRedirects(true)
            .build()

        val resumeFrom = if (partial.exists()) partial.length() else 0L

        val request = Request.Builder()
            .url(url)
            .header("User-Agent", "AriaAgent/1.0")
            .apply { if (resumeFrom > 0) addHeader("Range", "bytes=$resumeFrom-") }
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e(TAG, "HTTP ${response.code} for $url")
                    throw Exception("HTTP ${response.code}: ${response.message}")
                }
                val body = response.body
                    ?: throw Exception("Empty response body for $url")
                val contentLength = body.contentLength()
                val totalBytes    = if (resumeFrom > 0 && contentLength > 0)
                    resumeFrom + contentLength else contentLength

                var downloaded = resumeFrom
                FileOutputStream(partial, resumeFrom > 0).use { out ->
                    body.byteStream().use { input ->
                        val buf = ByteArray(256 * 1024)
                        var n: Int
                        while (input.read(buf).also { n = it } != -1) {
                            out.write(buf, 0, n)
                            downloaded += n
                            onProgress(downloaded, totalBytes)
                        }
                    }
                }
                partial.renameTo(dest).also { renamed ->
                    if (!renamed) throw Exception("Failed to rename partial → final: $dest")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download failed for $url: ${e.message}")
            throw e
        }
    }
}
