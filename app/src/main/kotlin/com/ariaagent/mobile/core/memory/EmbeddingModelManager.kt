package com.ariaagent.mobile.core.memory

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

/**
 * EmbeddingModelManager — downloads and verifies the MiniLM-L6-v2 ONNX model.
 *
 * Model: all-MiniLM-L6-v2 (ONNX export, sentence-transformers community)
 *   - Source: HuggingFace — sentence-transformers/all-MiniLM-L6-v2
 *   - URL:    ...resolve/main/onnx/model.onnx  (verified file, ~22.7 MB)
 *   - Disk size: ~23 MB
 *   - Output dim: 384 floats per sentence
 *   - Runtime: ONNX Runtime Android (~50-80ms per inference on Exynos 9611)
 *
 * Why ONNX instead of TFLite:
 *   The sentence-transformers HuggingFace repos ship verified ONNX exports.
 *   A float16 TFLite conversion is not publicly hosted for this model.
 *   ONNX Runtime Android (com.microsoft.onnxruntime:onnxruntime-android) gives
 *   the same ~50ms latency on ARM64 with no extra conversion step.
 *
 * Download strategy:
 *   - HTTP Range header for resumable downloads
 *   - Partial file: models/minilm-l6-v2.onnx.part
 *   - Final file:   models/minilm-l6-v2.onnx
 *   - EmbeddingEngine checks for final file on embed() — uses hash fallback until present
 *
 * Phase: 4 (Data Collection — memory retrieval)
 */
object EmbeddingModelManager {

    private const val TAG = "EmbeddingModelManager"

    // MiniLM-L6-v2 ONNX — official sentence-transformers export on HuggingFace.
    // Output: 384-dim float32 sentence embeddings (mean-pooled, L2-normalised).
    private const val MODEL_URL =
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
    private const val MODEL_FILENAME    = "minilm-l6-v2.onnx"
    private const val MODEL_MIN_SIZE_BYTES = 18_000_000L   // ~18 MB minimum valid file

    // BERT vocab.txt — same repo, WordPiece vocabulary for real sub-word tokenisation.
    // ~230 KB. EmbeddingEngine loads this into a HashMap<String, Int> on first use.
    private const val VOCAB_URL      = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"
    private const val VOCAB_FILENAME = "bert-vocab.txt"
    private const val VOCAB_MIN_SIZE_BYTES = 200_000L      // ~230 KB minimum valid file

    private fun modelsDir(context: Context): File {
        val internal = File(context.filesDir, "models").also { it.mkdirs() }
        if (internal.canWrite()) return internal
        return (context.getExternalFilesDir("models") ?: internal).also { it.mkdirs() }
    }

    fun modelPath(context: Context): File =
        File(modelsDir(context), MODEL_FILENAME)

    fun partialPath(context: Context): File =
        File(modelsDir(context), "$MODEL_FILENAME.part")

    fun vocabPath(context: Context): File =
        File(modelsDir(context), VOCAB_FILENAME)

    fun isModelReady(context: Context): Boolean {
        val f = modelPath(context)
        return f.exists() && f.length() >= MODEL_MIN_SIZE_BYTES
    }

    fun isVocabReady(context: Context): Boolean {
        val f = vocabPath(context)
        return f.exists() && f.length() >= VOCAB_MIN_SIZE_BYTES
    }

    fun downloadedBytes(context: Context): Long {
        val partial = partialPath(context)
        return if (partial.exists()) partial.length() else 0L
    }

    /**
     * Download the MiniLM ONNX model and the BERT vocab.txt.
     * Both files are needed for full-quality WordPiece embeddings.
     * Resumes partial ONNX downloads automatically via HTTP Range header.
     * vocab.txt is small enough (~230 KB) that it is always re-fetched if absent.
     *
     * @param context Android context
     * @param onProgress (downloadedBytes, totalBytes) — called every ~1MB during ONNX download
     * @return true if both files are ready after this call
     */
    suspend fun download(
        context: Context,
        onProgress: ((downloaded: Long, total: Long) -> Unit)? = null
    ): Boolean = withContext(Dispatchers.IO) {
        modelsDir(context).mkdirs()

        // ── 1. Download ONNX model (resumable) ───────────────────────────────
        val onnxOk = downloadFile(
            context      = context,
            url          = MODEL_URL,
            finalFile    = modelPath(context),
            partialFile  = partialPath(context),
            minSizeBytes = MODEL_MIN_SIZE_BYTES,
            label        = "MiniLM ONNX",
            onProgress   = onProgress
        )
        if (!onnxOk) return@withContext false

        // ── 2. Download vocab.txt (small — no resume needed) ─────────────────
        val vocabOk = downloadFile(
            context      = context,
            url          = VOCAB_URL,
            finalFile    = vocabPath(context),
            partialFile  = null,               // skip resume for small file
            minSizeBytes = VOCAB_MIN_SIZE_BYTES,
            label        = "BERT vocab.txt",
            onProgress   = null
        )
        vocabOk   // both must succeed for WordPiece to work
    }

    /**
     * Generic resumable file downloader.
     * If partialFile is null the download is non-resumable (re-downloads from 0).
     */
    private suspend fun downloadFile(
        context: Context,
        url: String,
        finalFile: File,
        partialFile: File?,
        minSizeBytes: Long,
        label: String,
        onProgress: ((Long, Long) -> Unit)?
    ): Boolean {
        try {
            if (finalFile.exists() && finalFile.length() >= minSizeBytes) {
                Log.i(TAG, "$label already present — skipping download")
                return true
            }

            val resumeFrom = if (partialFile != null && partialFile.exists()) partialFile.length() else 0L

            val client = OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .build()

            val request = Request.Builder()
                .url(url)
                .apply { if (resumeFrom > 0) addHeader("Range", "bytes=$resumeFrom-") }
                .build()

            Log.i(TAG, "Downloading $label from offset=$resumeFrom")

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful && response.code != 206) {
                    Log.e(TAG, "$label HTTP ${response.code}: ${response.message}")
                    return false
                }

                val body = response.body ?: run {
                    Log.e(TAG, "$label empty response body")
                    return false
                }

                val serverLength = body.contentLength().takeIf { it > 0 } ?: (minSizeBytes - resumeFrom)
                val total = serverLength + resumeFrom
                var downloaded = resumeFrom
                var lastEmitAt = resumeFrom
                val emitEveryBytes = 1_000_000L

                val dest = partialFile ?: finalFile
                FileOutputStream(dest, resumeFrom > 0).use { out ->
                    body.byteStream().use { input ->
                        val buf = ByteArray(16_384)
                        var read: Int
                        while (input.read(buf).also { read = it } != -1) {
                            out.write(buf, 0, read)
                            downloaded += read
                            if (onProgress != null && downloaded - lastEmitAt >= emitEveryBytes) {
                                lastEmitAt = downloaded
                                onProgress.invoke(downloaded, total)
                            }
                        }
                    }
                }

                val savedFile = partialFile ?: finalFile
                if (savedFile.length() >= minSizeBytes) {
                    val ok = if (partialFile != null) partialFile.renameTo(finalFile) else true
                    Log.i(TAG, "$label download complete: ${finalFile.length()} bytes → ok=$ok")
                    return ok
                } else {
                    Log.e(TAG, "$label file too small after download: ${savedFile.length()} bytes")
                    return false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "$label download failed: ${e.message}")
            return false
        }
    }

    /**
     * Cancel a partial ONNX download by deleting the partial file.
     * Does not affect the final file if it already exists.
     */
    fun cancelDownload(context: Context) {
        partialPath(context).delete()
        Log.i(TAG, "MiniLM ONNX partial download cancelled")
    }
}
