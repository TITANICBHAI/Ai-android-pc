package com.ariaagent.mobile.core.perception

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream

/**
 * ObjectDetectorEngine — MediaPipe EfficientDet-Lite0 INT8 on-device object detection.
 *
 * Provides visual perception for UI elements that have no accessibility metadata:
 *   - Icons with no content description
 *   - Game characters, sprites, and HUD elements
 *   - Custom-drawn views (Flutter, Unity, WebView)
 *   - Images and graphics
 *
 * Architecture: Producer-Consumer pattern as specified in the feasibility analysis.
 *   - This engine is the Producer: runs detection every agent step when model is ready.
 *   - LlamaEngine is the Consumer: reads the resulting DetectedObject list via PromptBuilder.
 *   - They run on separate threads (Dispatchers.Default) and never block each other.
 *
 * Model: EfficientDet-Lite0 INT8
 *   - Source: MediaPipe Model Hub (Google CDN)
 *   - Size: ~4.4 MB on disk, ~15–30 MB runtime RAM
 *   - Latency: ~37ms per frame on Exynos 9611 Cortex-A73 — well within the agent step budget
 *   - Quantization: INT8 — allows Mali-G72 GPU acceleration with minimal thermal impact
 *   - Output: 80 COCO object categories + confidence + bounding box
 *
 * RAM safety (M31 budget):
 *   LLM (1,700 MB) + OCR (100 MB) + Detector (30 MB) + RL (5 MB) + misc (~200 MB) = ~2,035 MB
 *   Within the ~2,500–3,500 MB app budget. Safe.
 *
 * Adaptive sampling: Detection runs ONLY when ObjectDetectorEngine.isModelReady() is true
 *   and the agent explicitly requests a screen_read — not on every frame.
 *
 * Phase: 13 (Object Labeler — Auto-detect Extensions)
 */
object ObjectDetectorEngine {

    private const val TAG = "ObjectDetectorEngine"

    // MediaPipe Model Hub — EfficientDet-Lite0 INT8 (~4.4 MB, well under APK limit but kept external)
    private const val MODEL_URL =
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
    private const val MODEL_FILENAME = "efficientdet_lite0_int8.tflite"
    private const val MIN_MODEL_SIZE_BYTES = 3_000_000L   // 3 MB sanity check

    private const val SCORE_THRESHOLD = 0.40f   // ≥40% confidence to surface a detection
    private const val MAX_RESULTS     = 12       // cap to avoid bloating the LLM prompt

    private val http = OkHttpClient()

    // ─── Model management ────────────────────────────────────────────────────

    fun modelFile(context: Context): File {
        val internal = File(context.filesDir, "models").also { it.mkdirs() }
        val dir = if (internal.canWrite()) internal
                  else (context.getExternalFilesDir("models") ?: internal).also { it.mkdirs() }
        return File(dir, MODEL_FILENAME)
    }

    fun isModelReady(context: Context): Boolean {
        val f = modelFile(context)
        return f.exists() && f.length() >= MIN_MODEL_SIZE_BYTES
    }

    /**
     * Download the EfficientDet-Lite0 INT8 model if not already present.
     * Called from a background coroutine — never blocks the main thread.
     * Returns true when the model is ready to use.
     */
    suspend fun ensureModel(context: Context): Boolean = withContext(Dispatchers.IO) {
        if (isModelReady(context)) return@withContext true
        val file = modelFile(context)
        try {
            file.parentFile?.mkdirs()
            val req = Request.Builder().url(MODEL_URL).build()
            val resp = http.newCall(req).execute()
            if (!resp.isSuccessful) {
                Log.w(TAG, "Model download HTTP ${resp.code}")
                return@withContext false
            }
            FileOutputStream(file).use { out ->
                resp.body?.byteStream()?.copyTo(out)
            }
            Log.i(TAG, "EfficientDet-Lite0 INT8 downloaded: ${file.length()} bytes")
            isModelReady(context)
        } catch (e: Exception) {
            Log.e(TAG, "Model download failed: ${e.message}")
            file.delete()
            false
        }
    }

    fun downloadedBytes(context: Context): Long =
        modelFile(context).takeIf { it.exists() }?.length() ?: 0L

    // ─── Detection types ─────────────────────────────────────────────────────

    /**
     * A single detected object with normalized bounding box coordinates (0–1).
     * normX/normY are the CENTER of the bounding box — directly usable as pin coordinates.
     */
    data class DetectedObject(
        val label: String,          // COCO category name or custom label
        val confidence: Float,       // 0.0–1.0
        val normX: Float,            // bounding box center X, 0–1
        val normY: Float,            // bounding box center Y, 0–1
        val normW: Float,            // bounding box width,    0–1
        val normH: Float,            // bounding box height,   0–1
    ) {
        /** Short string injected into the LLM [VISUAL DETECTIONS] block. */
        fun toPromptLine(index: Int): String {
            val pct  = (confidence * 100).toInt()
            val cx   = (normX * 100).toInt()
            val cy   = (normY * 100).toInt()
            return "  det-${index + 1}: $label (${pct}%, center ${cx}%×${cy}%)"
        }
    }

    // ─── Inference ───────────────────────────────────────────────────────────

    /**
     * Run EfficientDet-Lite0 on a Bitmap (agent loop path — bitmap already in memory).
     * Returns an empty list immediately if the model is not ready.
     * ~37ms on Exynos 9611 — within the 800ms STEP_DELAY_MS agent loop budget.
     */
    suspend fun detect(context: Context, bitmap: Bitmap): List<DetectedObject> =
        withContext(Dispatchers.Default) {
            if (!isModelReady(context)) return@withContext emptyList()
            runDetection(context, bitmap)
        }

    /**
     * Run detection from a JPEG file path (Object Labeler UI path).
     * Decodes the JPEG to Bitmap then runs detection.
     */
    suspend fun detectFromPath(context: Context, imagePath: String): List<DetectedObject> =
        withContext(Dispatchers.Default) {
            if (!isModelReady(context)) return@withContext emptyList()
            val bitmap = BitmapFactory.decodeFile(imagePath)
                ?: return@withContext emptyList()
            runDetection(context, bitmap)
        }

    private fun runDetection(context: Context, bitmap: Bitmap): List<DetectedObject> {
        return try {
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(modelFile(context).absolutePath)
                        .build()
                )
                .setMaxResults(MAX_RESULTS)
                .setScoreThreshold(SCORE_THRESHOLD)
                .build()

            val detector = ObjectDetector.createFromOptions(context, options)
            val mpImage  = BitmapImageBuilder(bitmap).build()
            val result   = detector.detect(mpImage)
            detector.close()

            val imgW = bitmap.width.toFloat().coerceAtLeast(1f)
            val imgH = bitmap.height.toFloat().coerceAtLeast(1f)

            result.detections().mapNotNull { detection ->
                val category = detection.categories().maxByOrNull { it.score() } ?: return@mapNotNull null
                val box      = detection.boundingBox()
                DetectedObject(
                    label      = category.categoryName() ?: "object",
                    confidence = category.score(),
                    normX      = ((box.left + box.width()  / 2f) / imgW).coerceIn(0f, 1f),
                    normY      = ((box.top  + box.height() / 2f) / imgH).coerceIn(0f, 1f),
                    normW      = (box.width()  / imgW).coerceIn(0f, 1f),
                    normH      = (box.height() / imgH).coerceIn(0f, 1f),
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Detection error: ${e.message}")
            emptyList()
        }
    }
}
