package com.ariaagent.mobile.system.screen

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.IBinder
import androidx.core.app.NotificationCompat
import android.os.Handler
import android.os.HandlerThread
import java.io.File
import java.io.FileOutputStream

/**
 * ScreenCaptureService — MediaProjection screen capture.
 *
 * Captures the device display at an adaptive resolution that preserves the screen's
 * true aspect ratio. Resolution is computed at runtime from display metrics:
 *
 *   targetWidth  = min(540, physicalWidth / 2)  — capped at 720 for VRAM budget
 *   targetHeight = targetWidth * physicalHeight / physicalWidth
 *
 * Samsung M31 example (1080 × 2340):
 *   → capture: 540 × 1170 (9:20 ratio preserved)
 *   → OCR accuracy: excellent at 540 px wide
 *   → In-flight RAM: ~2.5 MB RGBA; on-disk JPEG: ~80–120 KB
 *
 * The old hardcoded 512 × 512 squashed a tall phone screen into a square, distorting
 * all UI coordinates and degrading vision model accuracy.
 *
 * Rolling frame buffer — dual-path (in-memory primary, disk secondary):
 *
 *   IN-MEMORY (primary, fast):
 *     [inMemoryFrames] — circular array of JPEG-compressed ByteArrays, one per slot.
 *     [captureLatest]  — decodes from the in-memory buffer (no disk round-trip).
 *     [captureLatestBytes] — returns the raw compressed ByteArray directly.
 *     At 540×1170 JPEG-85: ~80–120 KB per frame → 5 slots ≈ 400–600 KB total RAM.
 *
 *   DISK (secondary, async):
 *     The HandlerThread still writes JPEG files for persistence, session-replay export,
 *     and VLM models that require a file path rather than bytes. Disk writes happen on
 *     the background HandlerThread so they never block the agent loop or captureLatest().
 *
 *   Step-diff analysis uses [captureLatestBytes] → instant ByteArray comparison, no I/O.
 *   VLM/OCR callers use [captureLatest] → decoded Bitmap from memory, no disk read.
 *
 * Capture rate: event-driven (1–2 FPS net) — thermal safe on Exynos 9611.
 */
class ScreenCaptureService : Service() {

    companion object {
        private const val CHANNEL_ID = "aria_screen_capture"
        private const val NOTIF_ID   = 1002

        /** Number of frames kept in the rolling buffer (in-memory and on-disk). */
        const val FRAME_BUFFER_SIZE = 5

        var isActive = false
            private set

        /** Actual capture width chosen at setupCapture() time (aspect-ratio correct). */
        @Volatile var captureWidth  = 540
            private set

        /** Actual capture height chosen at setupCapture() time (aspect-ratio correct). */
        @Volatile var captureHeight = 1170
            private set

        /** Absolute path of the screenshots directory — set when service is created. */
        @Volatile private var screenshotsDir: String = ""

        /** Index of the most recently written frame slot (−1 = no frame yet). */
        @Volatile private var latestFrameIndex = -1

        /**
         * In-memory circular buffer of JPEG-compressed frame bytes.
         * Index matches latestFrameIndex / FRAME_BUFFER_SIZE slot convention.
         * Written on captureHandlerThread; read from any thread (volatile array refs).
         */
        private val inMemoryFrames = arrayOfNulls<ByteArray>(FRAME_BUFFER_SIZE)

        // ── Public read API ───────────────────────────────────────────────────

        /**
         * Decodes and returns the latest captured frame, or null if no frame is available.
         *
         * Fast path: decodes from the in-memory JPEG buffer — no disk I/O.
         * Fallback: reads from disk if the in-memory slot is empty (e.g. first boot).
         *
         * The bitmap has the correct screen aspect ratio (not a distorted square).
         */
        fun captureLatest(): Bitmap? {
            val idx = latestFrameIndex
            if (idx < 0) return null
            // Fast path: decode from in-memory JPEG bytes
            inMemoryFrames[idx]?.let { bytes ->
                return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }
            // Fallback: read from disk (should rarely happen after first frame)
            val dir = screenshotsDir.ifEmpty { return null }
            val file = File(dir, "frame_$idx.jpg")
            return if (file.exists()) android.graphics.BitmapFactory.decodeFile(file.absolutePath) else null
        }

        /**
         * Returns the raw JPEG bytes of the latest captured frame without decoding.
         * Use this for step-diff comparisons (hash the bytes) and network upload —
         * avoids a full decode/re-encode cycle.
         * Returns null if no frame has been captured yet.
         */
        fun captureLatestBytes(): ByteArray? {
            val idx = latestFrameIndex
            if (idx < 0) return null
            return inMemoryFrames[idx]
        }

        /**
         * Returns raw JPEG byte arrays for the last [count] frames, newest first.
         * Use for fast step-diff analysis at agent loop speed — pure memory access.
         */
        fun recentFrameBytes(count: Int = FRAME_BUFFER_SIZE): List<ByteArray> {
            val latest = latestFrameIndex
            if (latest < 0) return emptyList()
            return (0 until minOf(count, FRAME_BUFFER_SIZE)).mapNotNull { offset ->
                val idx = (latest - offset + FRAME_BUFFER_SIZE) % FRAME_BUFFER_SIZE
                inMemoryFrames[idx]
            }
        }

        /**
         * Returns file paths for the last [count] frames, newest first.
         * Use for session replay export or callers that need a file path (e.g. legacy VLM APIs).
         * Prefer [recentFrameBytes] for step-diff — it avoids disk I/O entirely.
         */
        fun recentFramePaths(count: Int = FRAME_BUFFER_SIZE): List<String> {
            val dir    = screenshotsDir.ifEmpty { return emptyList() }
            val latest = latestFrameIndex
            if (latest < 0) return emptyList()
            return (0 until minOf(count, FRAME_BUFFER_SIZE)).mapNotNull { offset ->
                val idx  = (latest - offset + FRAME_BUFFER_SIZE) % FRAME_BUFFER_SIZE
                val path = File(dir, "frame_$idx.jpg").absolutePath
                path.takeIf { File(it).exists() }
            }
        }
    }

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null

    private var captureHandlerThread: HandlerThread? = null
    private var captureHandler: Handler? = null

    /** Current write slot in the rolling buffer (advances mod FRAME_BUFFER_SIZE). */
    private var frameIndex = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIF_ID, buildNotification())

        val dir = File(filesDir, "screenshots").also { it.mkdirs() }
        screenshotsDir = dir.absolutePath

        captureHandlerThread = HandlerThread("ARIACaptureIO").also {
            it.start()
            captureHandler = Handler(it.looper)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val resultCode = intent?.getIntExtra("resultCode", 0) ?: return START_NOT_STICKY

        // API 33+ (Tiramisu): getParcelableExtra requires explicit class parameter;
        // the untyped overload is deprecated and returns null on some OEM ROMs.
        val projectionIntent: Intent? =
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra("projectionData", Intent::class.java)
            } else {
                @Suppress("DEPRECATION")
                intent.getParcelableExtra("projectionData")
            }
        projectionIntent ?: return START_NOT_STICKY

        val projectionManager = getSystemService(MediaProjectionManager::class.java)
        mediaProjection = projectionManager.getMediaProjection(resultCode, projectionIntent)

        // Android 14+ (API 34) requires registering a callback before createVirtualDisplay.
        // Without it the system silently kills the projection after a few seconds.
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            mediaProjection?.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() {
                    android.util.Log.i("ScreenCaptureService", "MediaProjection stopped by system")
                    isActive = false
                    virtualDisplay?.release()
                    virtualDisplay = null
                    imageReader?.close()
                    imageReader = null
                }
            }, null)
        }

        setupCapture()
        return START_STICKY
    }

    private fun setupCapture() {
        val mp      = mediaProjection ?: return
        val handler = captureHandler  ?: return
        val metrics = resources.displayMetrics

        // ── Adaptive resolution — preserves real screen aspect ratio ─────────
        //
        // Old behaviour:  512 × 512 (square) — wrong for all phones (9:20 ratio)
        // New behaviour:  targetWidth × (targetWidth * physH / physW)
        //
        // Width budget:
        //   ≥ 256 px  — minimum for ML Kit OCR and SmolVLM tokeniser
        //   ≤ 720 px  — memory/bandwidth ceiling (M31 Mali-G72 shared bus)
        //
        // Samsung M31 (1080 × 2340) example:
        //   physW / 2 = 540  → within [256, 720]  → captureWidth  = 540
        //   540 * 2340 / 1080 = 1170               → captureHeight = 1170
        val physW = metrics.widthPixels
        val physH = metrics.heightPixels
        val targetW = (physW / 2).coerceIn(256, 720)
        val targetH = (targetW.toFloat() * physH / physW).toInt().coerceAtLeast(256)

        captureWidth  = targetW
        captureHeight = targetH

        android.util.Log.i("ScreenCaptureService",
            "Adaptive capture: ${targetW}×${targetH} (device: ${physW}×${physH})")

        val reader = ImageReader.newInstance(
            targetW, targetH, PixelFormat.RGBA_8888, 2
        )
        imageReader = reader

        virtualDisplay = mp.createVirtualDisplay(
            "ARIACapture",
            targetW, targetH, metrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface, null, null
        )

        // Image callbacks run on captureHandlerThread — keeps disk I/O off main thread.
        reader.setOnImageAvailableListener({ r ->
            r.acquireLatestImage()?.use { image ->
                saveImage(image, targetW, targetH)
            }
        }, handler)

        isActive = true
    }

    private fun saveImage(image: android.media.Image, width: Int, height: Int) {
        try {
            val planes      = image.planes
            val buffer      = planes[0].buffer
            val rowStride   = planes[0].rowStride
            val pixelStride = planes[0].pixelStride
            val rowPadding  = rowStride - pixelStride * width

            val bitmap = Bitmap.createBitmap(
                width + rowPadding / pixelStride,
                height,
                Bitmap.Config.ARGB_8888
            )
            bitmap.copyPixelsFromBuffer(buffer)

            // Crop away any padding added by the ImageReader stride alignment.
            val cropped = if (rowPadding > 0)
                Bitmap.createBitmap(bitmap, 0, 0, width, height)
            else bitmap

            // ── Rolling buffer write — in-memory primary, disk secondary ─────
            //
            // 1. Compress to ByteArray (on this HandlerThread — no main-thread I/O).
            // 2. Store in the in-memory slot → captureLatest() can read instantly.
            // 3. Write to disk → session replay, VLM file-path APIs, persistence.
            //
            // captureLatest() checks inMemoryFrames[idx] first and skips disk read.
            // Step-diff callers use captureLatestBytes() for zero-copy byte access.
            val jpegBytes = java.io.ByteArrayOutputStream(128 * 1024).also { baos ->
                cropped.compress(Bitmap.CompressFormat.JPEG, 85, baos)
            }.toByteArray()

            // Store in-memory (atomic slot swap — readers see complete frame or previous)
            inMemoryFrames[frameIndex] = jpegBytes
            latestFrameIndex = frameIndex   // publish after bytes are stored

            // Write to disk for persistence / replay (same thread — non-blocking for callers)
            val dir = screenshotsDir
            if (dir.isNotEmpty()) {
                val frame = File(dir, "frame_$frameIndex.jpg")
                runCatching { frame.writeBytes(jpegBytes) }  // silently skip on storage error
            }

            frameIndex = (frameIndex + 1) % FRAME_BUFFER_SIZE

            if (cropped !== bitmap) bitmap.recycle()
            cropped.recycle()

        } catch (e: Exception) {
            android.util.Log.e("ScreenCaptureService", "saveImage failed: ${e.message}", e)
        }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "ARIA Screen Capture",
            NotificationManager.IMPORTANCE_LOW
        )
        (getSystemService(NotificationManager::class.java)).createNotificationChannel(channel)
    }

    private fun buildNotification() =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("ARIA Agent")
            .setContentText("Screen observation active")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .build()

    override fun onDestroy() {
        super.onDestroy()
        isActive         = false
        latestFrameIndex = -1
        virtualDisplay?.release()
        imageReader?.close()
        mediaProjection?.stop()
        captureHandlerThread?.quitSafely()
        captureHandlerThread = null
        captureHandler       = null
    }
}
