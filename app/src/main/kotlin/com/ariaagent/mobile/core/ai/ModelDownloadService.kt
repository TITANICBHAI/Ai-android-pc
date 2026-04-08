package com.ariaagent.mobile.core.ai

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.ariaagent.mobile.core.events.AgentEventBus
import com.ariaagent.mobile.ui.ComposeMainActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

/**
 * ModelDownloadService — foreground service that downloads any catalog model.
 *
 * Multimodal catalog entries (mmprojFilename != null) download TWO files:
 * the base GGUF and the vision mmproj GGUF. Text-only entries (mmprojFilename == null)
 * download only the base GGUF and skip the mmproj step entirely.
 *
 * Reliability features
 * ────────────────────
 * • Resume support     — HTTP Range header continues partial downloads after kills.
 * • Retry with backoff — transient network errors retried up to MAX_RETRIES times
 *                        (5 s → 10 s → 20 s) before giving up.
 * • HTTP error handling — 4xx errors reported immediately without retrying.
 * • mmproj download    — automatically downloads the vision projector after the
 *                        base GGUF completes (skipped if already on disk).
 *
 * The model to download is specified by EXTRA_MODEL_ID in the start Intent.
 * Falls back to the currently active model in ModelManager if not provided.
 */
class ModelDownloadService : Service() {

    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(420, TimeUnit.SECONDS)   // 7 min — large GGUF files on slow connections
        .writeTimeout(60, TimeUnit.SECONDS)
        .callTimeout(0, TimeUnit.SECONDS)     // 0 = no per-call cap; controlled by read timeout
        .followRedirects(true)
        .followSslRedirects(true)
        .build()

    // Lightweight validation client — short timeout, HEAD only
    private val validationClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .callTimeout(15, TimeUnit.SECONDS)
        .followRedirects(true)
        .followSslRedirects(true)
        .build()

    companion object {
        private const val TAG              = "ModelDownload"
        private const val CHANNEL_ID       = "aria_model_download"
        private const val NOTIF_ID         = 1001
        private const val MAX_RETRIES      = 3
        private const val EMIT_EVERY_BYTES = 2_000_000L

        /** Pass a ModelCatalog ID string to download a specific model. */
        const val EXTRA_MODEL_ID = "aria_model_id"
    }

    private var targetModel: CatalogModel? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelId = intent?.getStringExtra(EXTRA_MODEL_ID)
        targetModel = if (modelId != null) {
            ModelCatalog.findById(modelId) ?: ModelManager.activeEntry(this)
        } else {
            ModelManager.activeEntry(this)  // may be null if user hasn't selected a model
        }
        startForeground(NOTIF_ID, buildNotification("Starting download…", 0, false))
        scope.launch { downloadAll() }
        return START_STICKY
    }

    // ── URL validation ────────────────────────────────────────────────────────

    /**
     * Validates a download URL with a HEAD request before starting the download.
     * Returns null on success, or a user-facing error string on failure.
     *
     * Accepts 200, 206 (partial content), 301/302/307/308 (redirects followed
     * automatically by the client), and 405 (HEAD not allowed — URL still usable).
     * Rejects 403/404 immediately to give the user a clear error instead of
     * waiting for a 7-minute timeout on a dead link.
     */
    private fun validateUrl(url: String): String? {
        return try {
            val req = Request.Builder().url(url).head().build()
            validationClient.newCall(req).execute().use { resp ->
                when (resp.code) {
                    200, 206, 405 -> null  // OK / partial / HEAD disallowed but URL valid
                    301, 302, 307, 308 -> null  // followed automatically
                    403 -> "URL returned 403 Forbidden — model may require HuggingFace login"
                    404 -> "URL returned 404 Not Found — model file may have moved or been renamed"
                    else -> "Unexpected HTTP ${resp.code} during URL check for $url"
                }
            }
        } catch (e: Exception) {
            // Network unavailable or DNS failure — don't block the download, let
            // the actual download attempt surface the real error with retries.
            Log.w(TAG, "URL validation network error (non-fatal): ${e.message}")
            null
        }
    }

    // ── Download both files ───────────────────────────────────────────────────

    private suspend fun downloadAll() {
        val model = targetModel ?: ModelManager.activeEntry(this) ?: run {
            Log.e(TAG, "downloadAll: no model selected — cannot download")
            stopSelf()
            return
        }

        // 0 ── Validate base URL before starting ─────────────────────────────
        updateNotification("Checking URL…", 0, false)
        val urlError = validateUrl(model.url)
        if (urlError != null) {
            Log.e(TAG, "URL validation failed for ${model.displayName}: $urlError")
            emitError(model.id, urlError)
            stopSelf()
            return
        }
        if (model.mmprojUrl != null) {
            val mmprojError = validateUrl(model.mmprojUrl)
            if (mmprojError != null) {
                Log.e(TAG, "mmproj URL validation failed for ${model.displayName}: $mmprojError")
                emitError(model.id, "Vision projector: $mmprojError")
                stopSelf()
                return
            }
        }

        // 1 ── Download base GGUF (with retry) ──────────────────────────────
        if (!ModelManager.isModelDownloaded(this, model.id)) {
            val baseResult = downloadFileWithRetry(
                modelId      = model.id,
                label        = model.displayName,
                url          = model.url,
                destFile     = ModelManager.modelPathFor(this, model.id),
                partialFile  = ModelManager.partialPathFor(this, model.id),
                minBytes     = model.expectedSizeBytes,
                phasePrefix  = "[1/2] Base model",
            )
            if (baseResult != null) {
                emitError(model.id, baseResult)
                stopSelf()
                return
            }
        } else {
            Log.i(TAG, "Base GGUF already on disk for ${model.displayName}")
        }

        // 2 ── Download mmproj (vision projector) if the model has one ───────
        val mmprojUrl      = model.mmprojUrl
        val mmprojFilename = model.mmprojFilename
        if (mmprojUrl != null && mmprojFilename != null) {
            val mmprojFile    = File(ModelManager.modelDir(this), mmprojFilename)
            val mmprojPartial = File(ModelManager.modelDir(this), "$mmprojFilename.part")

            if (mmprojFile.exists() && mmprojFile.length() > 0) {
                Log.i(TAG, "mmproj already on disk for ${model.displayName}")
            } else {
                updateNotification("[2/2] Vision projector…", 0, false)
                val mmprojResult = downloadFileWithRetry(
                    modelId     = model.id,
                    label       = "${model.displayName} (vision)",
                    url         = mmprojUrl,
                    destFile    = mmprojFile,
                    partialFile = mmprojPartial,
                    minBytes    = 50_000_000L,     // mmproj always > 50 MB
                    phasePrefix = "[2/2] Vision projector",
                )
                if (mmprojResult != null) {
                    emitError(model.id, mmprojResult)
                    stopSelf()
                    return
                }
            }
        }

        // Both files done
        emitComplete(model.id, ModelManager.modelPathFor(this, model.id).absolutePath)
        stopSelf()
    }

    // ── File download with retry ──────────────────────────────────────────────

    /**
     * Downloads a single file with retry on network errors.
     * @return null on success, or an error message string on failure.
     */
    private suspend fun downloadFileWithRetry(
        modelId:     String,
        label:       String,
        url:         String,
        destFile:    File,
        partialFile: File,
        minBytes:    Long,
        phasePrefix: String,
    ): String? {
        var lastError = "Download failed before any attempt"
        for (attempt in 0..MAX_RETRIES) {
            if (attempt > 0) {
                val waitSec = (5 * (1 shl (attempt - 1))).toLong()   // 5, 10, 20 s
                Log.i(TAG, "[$label] Retry $attempt/$MAX_RETRIES in ${waitSec}s — $lastError")
                updateNotification("$phasePrefix — retrying in ${waitSec}s…", 0, false)
                delay(waitSec * 1000L)
            }
            when (val result = trySingleDownload(modelId, label, url, destFile, partialFile, minBytes, phasePrefix)) {
                DownloadResult.SUCCESS          -> return null
                is DownloadResult.HTTP_ERROR    -> return result.message
                is DownloadResult.NETWORK_ERROR -> lastError = result.message
            }
        }
        return "Download failed after $MAX_RETRIES retries: $lastError"
    }

    private sealed class DownloadResult {
        object SUCCESS                                : DownloadResult()
        data class HTTP_ERROR(val message: String)    : DownloadResult()
        data class NETWORK_ERROR(val message: String) : DownloadResult()
    }

    private fun trySingleDownload(
        modelId:     String,
        label:       String,
        url:         String,
        destFile:    File,
        partialFile: File,
        minBytes:    Long,
        phasePrefix: String,
    ): DownloadResult {
        val resumeFrom = if (partialFile.exists()) partialFile.length() else 0L

        val request = Request.Builder()
            .url(url)
            .apply { if (resumeFrom > 0) addHeader("Range", "bytes=$resumeFrom-") }
            .build()

        return try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful && response.code !in 200..299) {
                    return DownloadResult.HTTP_ERROR(
                        "HTTP ${response.code} for $label: ${response.message}"
                    )
                }
                val body = response.body ?: return DownloadResult.HTTP_ERROR(
                    "Empty response body for $label"
                )

                val serverContentLength = body.contentLength().takeIf { it > 0L } ?: 0L
                val serverTotal = serverContentLength + resumeFrom

                var downloaded = resumeFrom
                var lastEmitAt = resumeFrom
                val startTime  = System.currentTimeMillis()

                FileOutputStream(partialFile, resumeFrom > 0L).use { out ->
                    body.byteStream().use { input ->
                        val buf = ByteArray(32_768)
                        var read: Int
                        while (input.read(buf).also { read = it } != -1) {
                            out.write(buf, 0, read)
                            downloaded += read

                            if (downloaded - lastEmitAt >= EMIT_EVERY_BYTES) {
                                lastEmitAt = downloaded
                                val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                                val speed   = if (elapsed > 0) (downloaded - resumeFrom) / elapsed / 1_000_000 else 0.0
                                val pct     = if (serverTotal > 0) {
                                    ((downloaded.toDouble() / serverTotal) * 100).roundToInt().coerceIn(0, 100)
                                } else 0
                                val dlMb    = downloaded / 1_000_000.0
                                val totalMb = if (serverTotal > 0) serverTotal / 1_000_000.0 else 0.0
                                val notifText = "$phasePrefix — $pct%"

                                updateNotification(notifText, pct, pct > 0)
                                emitProgress(modelId, pct, dlMb, totalMb, speed)
                            }
                        }
                    }
                }

                // Verify then rename to final path
                if (partialFile.length() < minBytes) {
                    Log.w(TAG, "[$label] File too small (${partialFile.length()} < $minBytes) — retrying")
                    return DownloadResult.NETWORK_ERROR("File smaller than expected — possible truncated response")
                }
                val renamed = partialFile.renameTo(destFile)
                if (renamed) DownloadResult.SUCCESS
                else DownloadResult.NETWORK_ERROR("Failed to rename partial file to $destFile")
            }
        } catch (e: IOException) {
            Log.w(TAG, "[$label] Network error: ${e.message}")
            DownloadResult.NETWORK_ERROR(e.message ?: "Network IO error")
        } catch (e: Exception) {
            Log.e(TAG, "[$label] Unexpected error: ${e.message}")
            DownloadResult.HTTP_ERROR(e.message ?: "Unexpected error")
        }
    }

    // ── Event helpers ─────────────────────────────────────────────────────────

    private fun emitProgress(
        modelId: String, pct: Int,
        dlMb: Double, totalMb: Double, speedMbps: Double
    ) {
        AgentEventBus.emit("model_download_progress", mapOf(
            "modelId"      to modelId,
            "percent"      to pct,
            "downloadedMb" to dlMb,
            "totalMb"      to totalMb,
            "speedMbps"    to speedMbps
        ))
    }

    private fun emitComplete(modelId: String, path: String) {
        AgentEventBus.emit("model_download_complete", mapOf(
            "modelId" to modelId,
            "path"    to path
        ))
    }

    private fun emitError(modelId: String, error: String) {
        AgentEventBus.emit("model_download_error", mapOf(
            "modelId" to modelId,
            "error"   to error
        ))
        Log.e(TAG, "Download error [$modelId]: $error")
    }

    // ── Notification ──────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "ARIA Model Download",
            NotificationManager.IMPORTANCE_LOW
        ).apply { description = "Downloading an on-device AI model" }
        (getSystemService(NotificationManager::class.java)).createNotificationChannel(channel)
    }

    private fun buildNotification(text: String, progress: Int, indeterminate: Boolean) =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("ARIA — Downloading model")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .setProgress(100, progress, indeterminate)
            .setContentIntent(
                PendingIntent.getActivity(
                    this, 0,
                    Intent(this, ComposeMainActivity::class.java),
                    PendingIntent.FLAG_IMMUTABLE
                )
            )
            .build()

    private fun updateNotification(text: String, progress: Int, indeterminate: Boolean) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIF_ID, buildNotification(text, progress, indeterminate))
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }
}
