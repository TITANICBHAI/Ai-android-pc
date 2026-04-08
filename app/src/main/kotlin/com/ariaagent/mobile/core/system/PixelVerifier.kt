package com.ariaagent.mobile.core.system

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.ariaagent.mobile.core.ocr.OcrEngine
import com.ariaagent.mobile.system.screen.ScreenCaptureService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * PixelVerifier — Fast, low-overhead action-result verification.
 *
 * Problem: Full-screen MediaProjection screenshot is expensive (~40–80ms on Exynos 9611).
 * Taking one before AND after every gesture doubles the perception cost per step.
 *
 * Solution: Targeted pixel diff on a small region of interest (the acted-upon element),
 * plus optional targeted OCR on that same region to detect text label changes (e.g.
 * "Follow" → "Following").
 *
 * Verification pipeline:
 *   1. Capture PRE bitmap crop from ScreenCaptureService's latest 512×512 bitmap
 *   2. Dispatch gesture (done by GestureEngine)
 *   3. Wait SETTLE_MS for screen to update
 *   4. Capture POST bitmap crop from the same region
 *   5. Compute normalised pixel diff:
 *        diff = Σ(|R₁−R₂| + |G₁−G₂| + |B₁−B₂|) / (3 × W × H × 255)
 *   6. If diff ≥ PIXEL_DIFF_THRESHOLD → screen changed → action was registered
 *   7. If textVerify=true: run targeted ML Kit OCR on the POST crop
 *      Return the new text so the caller can confirm semantic change
 *      (e.g. the "Follow" button now reads "Following")
 *
 * Coordinate scaling:
 *   ScreenCaptureService captures at 512×512 regardless of actual display resolution.
 *   The Rect passed in must be in screen coordinates (from AccessibilityNodeInfo.getBoundsInScreen).
 *   PixelVerifier scales it down to the 512×512 space before cropping.
 *
 * Note on PixelCopy API:
 *   PixelCopy.request() (API 26+) copies a SurfaceView/Window region into a Bitmap with
 *   hardware-accelerated efficiency. It requires a live Window reference, which is not
 *   available in a Service context. This implementation achieves the same objective by
 *   operating on the Bitmap already held by ScreenCaptureService's ImageReader pipeline —
 *   giving us the region diff without needing an Activity Window reference.
 *   When ARIA moves to ComposeMainActivity as launcher (Phase 11 completion), this can be
 *   upgraded to true PixelCopy.request() targeting the display surface for sub-millisecond
 *   captures. The verification logic below is identical either way.
 *
 * Phase: 14.1 (Advanced Architecture — Fast Action Verification)
 */
object PixelVerifier {

    private const val TAG = "PixelVerifier"

    private const val SETTLE_MS = 250L
    private const val PIXEL_DIFF_THRESHOLD = 0.02  // 2% of pixels must change

    // Maximum region size to diff — larger rects are clamped to this for speed
    private const val MAX_CROP_PX = 128

    data class VerificationResult(
        val changed: Boolean,
        val pixelDiff: Double,       // 0.0–1.0 normalised
        val postCropOcr: String = "" // non-empty only if textVerify=true
    )

    /**
     * Capture the PRE-action pixel state for a screen-coordinate region.
     *
     * Call this BEFORE dispatching the gesture.
     *
     * @param screenRect  Bounding box in real screen coordinates (from AccessibilityNodeInfo)
     * @return            Cropped Bitmap in 512×512 space, or null if capture unavailable
     */
    fun capturePre(screenRect: Rect): Bitmap? {
        return cropFromLatest(screenRect)
    }

    /**
     * Verify an action by comparing PRE and POST pixel state.
     *
     * Call this AFTER dispatching the gesture.
     *
     * @param preBitmap   Result of capturePre() called before the gesture
     * @param screenRect  Same bounding box passed to capturePre()
     * @param textVerify  If true, run OCR on the POST crop to detect label changes
     * @return            VerificationResult with changed flag, diff score, and optional OCR text
     */
    suspend fun verify(
        preBitmap: Bitmap?,
        screenRect: Rect,
        textVerify: Boolean = false
    ): VerificationResult = withContext(Dispatchers.Default) {
        delay(SETTLE_MS)

        val postBitmap = cropFromLatest(screenRect)

        if (preBitmap == null || postBitmap == null) {
            Log.d(TAG, "verify: bitmap unavailable — returning changed=false")
            return@withContext VerificationResult(changed = false, pixelDiff = 0.0)
        }

        val diff = computeNormalisedDiff(preBitmap, postBitmap)
        val changed = diff >= PIXEL_DIFF_THRESHOLD

        Log.d(TAG, "verify: diff=%.4f changed=$changed rect=$screenRect".format(diff))

        var ocrText = ""
        if (textVerify && postBitmap.width > 0 && postBitmap.height > 0) {
            ocrText = runCatching { OcrEngine.run(postBitmap) }.getOrDefault("")
        }

        if (preBitmap !== postBitmap) {
            runCatching { postBitmap.recycle() }
        }

        VerificationResult(changed = changed, pixelDiff = diff, postCropOcr = ocrText)
    }

    // ─── Internal helpers ─────────────────────────────────────────────────────

    /**
     * Crop a region from ScreenCaptureService's latest 512×512 bitmap.
     *
     * Scales screen coordinates to the 512×512 capture space, then crops.
     * Returns null if no capture is available yet.
     */
    private fun cropFromLatest(screenRect: Rect): Bitmap? {
        val full = ScreenCaptureService.captureLatest() ?: return null

        val captureW = full.width.toFloat()
        val captureH = full.height.toFloat()

        // We don't have the real display dimensions here — assume standard portrait.
        // AgentAccessibilityService could pass actual display size; for now 1080×2340 covers M31.
        val displayW = 1080f
        val displayH = 2340f

        val scaleX = captureW / displayW
        val scaleY = captureH / displayH

        val left   = (screenRect.left   * scaleX).toInt().coerceIn(0, full.width  - 1)
        val top    = (screenRect.top    * scaleY).toInt().coerceIn(0, full.height - 1)
        val right  = (screenRect.right  * scaleX).toInt().coerceIn(left + 1, full.width)
        val bottom = (screenRect.bottom * scaleY).toInt().coerceIn(top  + 1, full.height)

        val rawW = right - left
        val rawH = bottom - top

        // Clamp to MAX_CROP_PX for speed — larger elements still produce valid diffs
        val cropW = rawW.coerceAtMost(MAX_CROP_PX)
        val cropH = rawH.coerceAtMost(MAX_CROP_PX)

        return runCatching {
            Bitmap.createBitmap(full, left, top, cropW, cropH)
        }.getOrNull()
    }

    /**
     * Compute a normalised pixel-level difference between two same-size bitmaps.
     *
     *   diff = Σ(|R₁−R₂| + |G₁−G₂| + |B₁−B₂|) / (3 × W × H × 255)
     *
     * Returns 0.0 (identical) to 1.0 (completely different).
     * Bitmaps are resized to the smaller of the two dimensions if they differ.
     */
    private fun computeNormalisedDiff(a: Bitmap, b: Bitmap): Double {
        val w = minOf(a.width,  b.width)
        val h = minOf(a.height, b.height)
        if (w <= 0 || h <= 0) return 0.0

        val pixelsA = IntArray(w * h)
        val pixelsB = IntArray(w * h)

        a.getPixels(pixelsA, 0, w, 0, 0, w, h)
        b.getPixels(pixelsB, 0, w, 0, 0, w, h)

        var total = 0L
        for (i in pixelsA.indices) {
            val pA = pixelsA[i]
            val pB = pixelsB[i]
            total += Math.abs(((pA shr 16) and 0xFF) - ((pB shr 16) and 0xFF)).toLong()  // R
            total += Math.abs(((pA shr  8) and 0xFF) - ((pB shr  8) and 0xFF)).toLong()  // G
            total += Math.abs(( pA         and 0xFF) - ( pB         and 0xFF)).toLong()  // B
        }

        return total.toDouble() / (3.0 * w * h * 255.0)
    }
}
