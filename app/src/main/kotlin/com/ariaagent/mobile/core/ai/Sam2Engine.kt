package com.ariaagent.mobile.core.ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Rect
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import okhttp3.OkHttpClient
import okhttp3.Request
import java.util.concurrent.TimeUnit

/**
 * Sam2Engine — On-device pixel segmentation using MobileSAM (ViT-Tiny encoder).
 *
 * Architecture:
 *   Model: MobileSAM image encoder — ViT-Tiny variant (~38 MB ONNX)
 *   Source: Acly/MobileSAM on HuggingFace (ONNX-exported encoder)
 *
 * Pipeline:
 *   1. Resize + letterbox bitmap → 1024×1024 RGB float32
 *   2. Normalize: (pixel / 255 − mean) / std  (ImageNet stats)
 *   3. Run ONNX encoder → image_embeddings [1, 256, 64, 64]
 *   4. Collapse 256 channels → saliency map [64, 64] (sum of absolute activations)
 *   5. Peak detection → top-K regions → normalised (x, y) screen coordinates
 *
 * Output: List<SalientRegion> — each region is a likely interactive element.
 *   These coordinates feed directly into GestureEngine.tapXY() when the
 *   accessibility tree is empty (game / Flutter screens).
 *
 * ONNX Runtime:
 *   Uses the same ai.onnxruntime:onnxruntime-android:1.19.2 dependency as
 *   EmbeddingEngine. Session is loaded via reflection to avoid hard
 *   compile-time dependency on ONNX Runtime at class-load.
 *
 * Memory:
 *   Encoder weights: ~38 MB.
 *   Activation tensor: 1×256×64×64 × 4 bytes = ~4 MB peak.
 *   Total RAM peak: ~42 MB — safe on Exynos 9611 (6 GB LPDDR4X).
 *
 * Thread safety:
 *   ensureLoaded() is @Synchronized. segment() calls ensureLoaded() internally.
 *
 * Phase 18 — SAM2/MobileSAM pixel segmentation.
 */
object Sam2Engine {

    private const val TAG = "Sam2Engine"

    // ── MobileSAM ViT-Tiny encoder ONNX (Acly/MobileSAM on HuggingFace) ─────

    const val ENCODER_URL =
        "https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx"

    const val ENCODER_FILENAME = "mobilesam-encoder.onnx"
    const val ENCODER_MIN_BYTES = 30_000_000L          // ~38 MB expected

    /** SAM native resolution — encoder trained on 1024×1024 inputs. */
    private const val SAM_RES = 1024

    /** Feature map spatial size output by ViT-Tiny encoder. */
    private const val FEAT_SIZE = 64

    /** ImageNet mean / std for SAM's pixel normalisation. */
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD  = floatArrayOf(0.229f, 0.224f, 0.225f)

    // ── ONNX Runtime session ──────────────────────────────────────────────────

    @Volatile private var ortSession: OrtSession? = null
    @Volatile private var ortEnvironment: OrtEnvironment? = null

    // ── Data classes ──────────────────────────────────────────────────────────

    /**
     * A candidate interactive region detected by the SAM encoder.
     *
     * @param normX  Normalised X coordinate [0.0 – 1.0] relative to screen width
     * @param normY  Normalised Y coordinate [0.0 – 1.0] relative to screen height
     * @param score  Saliency activation score (higher = more likely interactive)
     */
    data class SalientRegion(
        val normX: Float,
        val normY: Float,
        val score: Float,
    )

    // ── File paths ────────────────────────────────────────────────────────────

    fun modelDir(context: Context): File {
        val internal = File(context.filesDir, "models").also { it.mkdirs() }
        if (internal.canWrite()) return internal
        return (context.getExternalFilesDir("models") ?: internal).also { it.mkdirs() }
    }

    fun encoderPath(context: Context): File =
        File(modelDir(context), ENCODER_FILENAME)

    fun encoderPartial(context: Context): File =
        File(modelDir(context), "$ENCODER_FILENAME.part")

    // ── Readiness ─────────────────────────────────────────────────────────────

    fun isModelReady(context: Context): Boolean {
        val f = encoderPath(context)
        return f.exists() && f.length() >= ENCODER_MIN_BYTES
    }

    fun isLoaded(): Boolean = ortSession != null

    fun downloadedBytes(context: Context): Long {
        val partial = encoderPartial(context)
        return if (partial.exists()) partial.length()
        else if (encoderPath(context).exists()) encoderPath(context).length()
        else 0L
    }

    // ── Load / Unload ─────────────────────────────────────────────────────────

    /**
     * Load the MobileSAM encoder into an ONNX Runtime session.
     * No-op if already loaded. Thread-safe via @Synchronized.
     * @return true if session is ready to run inference.
     */
    @Synchronized
    fun ensureLoaded(context: Context): Boolean {
        if (ortSession != null) return true
        if (!isModelReady(context)) {
            Log.w(TAG, "Encoder not downloaded — call downloadModel() first")
            return false
        }
        return try {
            val env = OrtEnvironment.getEnvironment()
            ortEnvironment = env
            ortSession = env.createSession(encoderPath(context).absolutePath)
            Log.i(TAG, "MobileSAM encoder loaded (${encoderPath(context).length() / 1_048_576} MB)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load ONNX session: ${e.message}")
            ortSession = null
            false
        }
    }

    /** Release ONNX session and free ~22 MB of RAM. */
    @Synchronized
    fun unload() {
        try {
            (ortSession as? AutoCloseable)?.close()
        } catch (_: Exception) {}
        ortSession = null
        Log.i(TAG, "MobileSAM encoder unloaded")
    }

    // ── Segmentation inference ────────────────────────────────────────────────

    /**
     * Identify the top [topK] most salient / interactive regions in [bitmap].
     *
     * Steps:
     *   1. Letterbox-resize to 1024×1024 (preserves aspect ratio)
     *   2. Convert to normalised float32 CHW tensor
     *   3. Run MobileSAM encoder → [1, 256, 64, 64] embeddings
     *   4. Collapse channels → 64×64 saliency map
     *   5. Non-maximum suppression with radius 5 → top-K peaks
     *   6. Map 64×64 peaks → original bitmap aspect → normalised screen coords
     *   7. If [annotationHint] is provided, re-rank regions toward the area described
     *      (e.g. "top button" → boost regions with normY < 0.4; "bottom" → normY > 0.6).
     *      This lets user IRL annotations guide which SAM regions ARIA focuses on.
     *
     * @param bitmap          Screen frame from ScreenObserver (any resolution)
     * @param topK            Maximum number of regions to return (default 8)
     * @param annotationHint  User annotation text for this frame — used to spatially re-rank
     *                        peaks so that the most annotation-relevant areas rank highest.
     * @return                List of SalientRegion sorted descending by score, or empty on failure
     */
    fun segment(
        context: Context,
        bitmap: Bitmap,
        topK: Int = 8,
        annotationHint: String = ""
    ): List<SalientRegion> {
        if (!ensureLoaded(context)) return emptyList()
        return try {
            val tensor = preprocess(bitmap)
            val embeddings = runEncoder(tensor) ?: return emptyList()
            val saliency = collapseChannels(embeddings)
            val peaks = extractPeaks(saliency, topK, bitmap.width.toFloat(), bitmap.height.toFloat())
            if (annotationHint.isBlank()) peaks else reRankByAnnotation(peaks, annotationHint)
        } catch (e: Exception) {
            Log.w(TAG, "Segmentation failed: ${e.message}")
            emptyList()
        }
    }

    /**
     * Re-rank SAM regions toward the spatial area implied by the user's annotation text.
     *
     * Annotation keywords → preferred screen zone:
     *   "top" / "header" / "title" / "status"  → prefer normY < 0.35
     *   "bottom" / "footer" / "nav" / "bar"     → prefer normY > 0.65
     *   "left"                                  → prefer normX < 0.35
     *   "right"                                 → prefer normX > 0.65
     *   "center" / "middle"                     → prefer normX 0.35–0.65, normY 0.35–0.65
     *
     * If no spatial keyword is found, the original saliency order is returned unchanged.
     * Regions that match the spatial zone receive a +50% score boost for sorting.
     */
    private fun reRankByAnnotation(
        regions: List<SalientRegion>,
        hint: String
    ): List<SalientRegion> {
        val h = hint.lowercase()

        val inZone: (SalientRegion) -> Boolean = when {
            h.containsAny("top", "header", "title", "status bar", "notification") ->
                { r -> r.normY < 0.35f }
            h.containsAny("bottom", "footer", "nav", "navigation bar", "toolbar") ->
                { r -> r.normY > 0.65f }
            h.containsAny("left", "sidebar") ->
                { r -> r.normX < 0.35f }
            h.containsAny("right") ->
                { r -> r.normX > 0.65f }
            h.containsAny("center", "middle", "center screen") ->
                { r -> r.normX in 0.35f..0.65f && r.normY in 0.35f..0.65f }
            else -> return regions   // no recognisable spatial keyword — keep original order
        }

        return regions
            .map { r -> if (inZone(r)) r.copy(score = r.score * 1.5f) else r }
            .sortedByDescending { it.score }
    }

    private fun String.containsAny(vararg keywords: String): Boolean =
        keywords.any { this.contains(it) }

    // ── Image preprocessing ───────────────────────────────────────────────────

    /**
     * Letterbox [bitmap] into SAM_RES×SAM_RES and convert to normalised
     * float32 CHW tensor [1, 3, SAM_RES, SAM_RES].
     *
     * Letterboxing: same strategy as VisionEngine — preserves aspect ratio so
     * spatial features in portrait phone screens are not distorted.
     */
    private fun preprocess(bitmap: Bitmap): FloatArray {
        // Letterbox onto SAM_RES × SAM_RES canvas
        val canvas = Bitmap.createBitmap(SAM_RES, SAM_RES, Bitmap.Config.ARGB_8888)
        val androidCanvas = Canvas(canvas)
        androidCanvas.drawColor(Color.BLACK)

        val srcW  = bitmap.width.toFloat()
        val srcH  = bitmap.height.toFloat()
        val scale = minOf(SAM_RES / srcW, SAM_RES / srcH)
        val dstW  = (srcW * scale).toInt()
        val dstH  = (srcH * scale).toInt()
        val dstX  = (SAM_RES - dstW) / 2
        val dstY  = (SAM_RES - dstH) / 2

        androidCanvas.drawBitmap(bitmap, null, Rect(dstX, dstY, dstX + dstW, dstY + dstH), null)

        // Convert to float32 CHW [1, 3, SAM_RES, SAM_RES]
        val pixels = IntArray(SAM_RES * SAM_RES)
        canvas.getPixels(pixels, 0, SAM_RES, 0, 0, SAM_RES, SAM_RES)
        canvas.recycle()

        val tensor = FloatArray(3 * SAM_RES * SAM_RES)
        val plane  = SAM_RES * SAM_RES
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr  8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            tensor[i]             = (r - MEAN[0]) / STD[0]   // R plane
            tensor[plane + i]     = (g - MEAN[1]) / STD[1]   // G plane
            tensor[plane * 2 + i] = (b - MEAN[2]) / STD[2]   // B plane
        }
        return tensor
    }

    // ── ONNX inference ────────────────────────────────────────────────────────

    /**
     * Run the encoder ONNX model.
     * @return Flat FloatArray of shape [1 * 256 * FEAT_SIZE * FEAT_SIZE], or null on error.
     *
     * Uses the direct ORT 1.19.2 Java API (OnnxTensor.createTensor + session.run).
     * Output is extracted by name via session.outputNames and getValue() as float[][][][].
     */
    private fun runEncoder(tensor: FloatArray): FloatArray? {
        val session = ortSession ?: return null
        val env     = ortEnvironment ?: return null
        return try {
            // Build input tensor: float32 [1, 3, SAM_RES, SAM_RES]
            val shape       = longArrayOf(1L, 3L, SAM_RES.toLong(), SAM_RES.toLong())
            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(tensor), shape)

            val inputs = LinkedHashMap<String, OnnxTensor>()
            inputs["input_image"] = inputTensor

            session.run(inputs).use { result ->
                val outputName = session.outputNames.first()
                val rawValue   = result.get(outputName).get().getValue()

                // rawValue is float[][][][]: [batch=1][channels=256][H=64][W=64]
                // Flatten into a single FloatArray for collapseChannels()
                @Suppress("UNCHECKED_CAST")
                val emb4d = rawValue as? Array<Array<Array<FloatArray>>> ?: return null
                val batch    = emb4d[0]              // [256][64][64]
                val channels = batch.size            // 256
                val hSize    = batch[0].size         // 64
                val wSize    = batch[0][0].size      // 64
                val flat     = FloatArray(channels * hSize * wSize)
                var idx = 0
                for (c in 0 until channels) {
                    for (h in 0 until hSize) {
                        for (w in 0 until wSize) {
                            flat[idx++] = batch[c][h][w]
                        }
                    }
                }
                flat
            }
        } catch (e: Exception) {
            Log.w(TAG, "Encoder inference error: ${e.message}")
            null
        }
    }

    // ── Saliency map computation ───────────────────────────────────────────────

    /**
     * Collapse 256 channels into a single [FEAT_SIZE × FEAT_SIZE] saliency map
     * by summing the absolute value of each spatial position across all channels.
     * High-activation positions correspond to rich texture / edge regions —
     * i.e. likely UI elements.
     */
    private fun collapseChannels(embeddings: FloatArray): FloatArray {
        val spatialSize = FEAT_SIZE * FEAT_SIZE
        val saliency    = FloatArray(spatialSize)
        val numChannels = embeddings.size / spatialSize   // should be 256

        for (c in 0 until numChannels) {
            val offset = c * spatialSize
            for (i in 0 until spatialSize) {
                saliency[i] += Math.abs(embeddings[offset + i])
            }
        }
        return saliency
    }

    // ── Peak extraction with NMS ──────────────────────────────────────────────

    /**
     * Find top-[topK] local maxima in [saliency] using non-maximum suppression
     * with a [nmsRadius] exclusion window. Convert 64×64 grid positions back to
     * normalised (0–1) screen coordinates accounting for letterbox padding.
     *
     * @param srcWidth   Original bitmap width (used to account for letterbox offset)
     * @param srcHeight  Original bitmap height
     */
    private fun extractPeaks(
        saliency: FloatArray,
        topK: Int,
        srcWidth: Float,
        srcHeight: Float,
        nmsRadius: Int = 5,
    ): List<SalientRegion> {
        // Compute letterbox scale + offset (mirrors preprocess logic)
        val scale = minOf(SAM_RES / srcWidth, SAM_RES / srcHeight)
        val dstW  = srcWidth  * scale
        val dstH  = srcHeight * scale
        val padX  = (SAM_RES - dstW) / 2f    // black bar width
        val padY  = (SAM_RES - dstH) / 2f    // black bar height

        val suppressed = BooleanArray(saliency.size)
        val results    = mutableListOf<SalientRegion>()

        repeat(topK) {
            // Find global max among non-suppressed cells
            var bestIdx   = -1
            var bestScore = Float.NEGATIVE_INFINITY
            for (i in saliency.indices) {
                if (!suppressed[i] && saliency[i] > bestScore) {
                    bestScore = saliency[i]
                    bestIdx   = i
                }
            }
            if (bestIdx < 0 || bestScore <= 0f) return results

            val row = bestIdx / FEAT_SIZE
            val col = bestIdx % FEAT_SIZE

            // Map 64×64 cell centre → SAM_RES pixel coords
            val pixX = (col + 0.5f) * (SAM_RES / FEAT_SIZE.toFloat())
            val pixY = (row + 0.5f) * (SAM_RES / FEAT_SIZE.toFloat())

            // Remove letterbox padding → content-relative coords
            val contentX = (pixX - padX).coerceIn(0f, dstW)
            val contentY = (pixY - padY).coerceIn(0f, dstH)

            // Normalise to [0, 1]
            val normX = (contentX / dstW).coerceIn(0f, 1f)
            val normY = (contentY / dstH).coerceIn(0f, 1f)

            results.add(SalientRegion(normX, normY, bestScore))

            // Suppress neighbours within nmsRadius
            for (dr in -nmsRadius..nmsRadius) {
                for (dc in -nmsRadius..nmsRadius) {
                    val nr = row + dr
                    val nc = col + dc
                    if (nr in 0 until FEAT_SIZE && nc in 0 until FEAT_SIZE) {
                        suppressed[nr * FEAT_SIZE + nc] = true
                    }
                }
            }
        }
        return results
    }

    // ── Download ──────────────────────────────────────────────────────────────

    /**
     * Download the MobileSAM encoder ONNX (~22 MB) to internal storage.
     * Supports HTTP range-resumption (partial downloads survive network drops).
     * Reports [onProgress](bytesDownloaded, totalBytes) for UI progress bars.
     * @return true on success.
     */
    fun downloadModel(
        context: Context,
        onProgress: (downloaded: Long, total: Long) -> Unit,
    ): Boolean {
        val dest    = encoderPath(context)
        val partial = encoderPartial(context)
        if (isModelReady(context)) return true

        val client = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .followRedirects(true)
            .followSslRedirects(true)
            .build()

        var resumeFrom = if (partial.exists()) partial.length() else 0L

        val request = Request.Builder()
            .url(ENCODER_URL)
            .header("User-Agent", "AriaAgent/1.0")
            .apply { if (resumeFrom > 0) addHeader("Range", "bytes=$resumeFrom-") }
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.e(TAG, "HTTP ${response.code} for $ENCODER_URL")
                    throw Exception("HTTP ${response.code}: ${response.message}")
                }
                // If we requested a range but server returned 200 (not 206),
                // it ignored the Range header — restart from scratch to avoid corruption.
                if (resumeFrom > 0 && response.code == 200) {
                    Log.w(TAG, "Server ignored Range header (returned 200), restarting from 0")
                    partial.delete()
                    resumeFrom = 0L
                }
                val body = response.body
                    ?: throw Exception("Empty response body for MobileSAM encoder")
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
                    if (renamed) Log.i(TAG, "MobileSAM encoder downloaded (${dest.length() / 1_048_576} MB)")
                    else throw Exception("Failed to rename partial → final: $dest")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download failed: ${e.message}")
            throw e
        }
    }
}
