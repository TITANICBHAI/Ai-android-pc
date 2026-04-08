package com.ariaagent.mobile.core.ocr

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

/**
 * OcrEngine — ML Kit on-device text recognition.
 *
 * Wraps the ML Kit TextRecognition async API with a suspending coroutine bridge.
 * Used by AgentLoop and LabelerScreen (via AgentViewModel) for screen reading and labeled capture flows.
 *
 * RAM: ~100 MB peak (shared ML Kit process pool).
 * Latency: ~50–150ms per frame on mid-range hardware.
 */
object OcrEngine {

    private const val TAG = "OcrEngine"

    private val recognizer by lazy {
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }

    /**
     * Returns true if the ML Kit recognizer initialises without error.
     * ML Kit Text Recognition is bundled, so this is almost always true;
     * however the lazy initialisation can theoretically throw on severely
     * degraded devices or during unit-test runs without the ML Kit stub.
     */
    fun isAvailable(@Suppress("UNUSED_PARAMETER") context: android.content.Context): Boolean =
        runCatching { recognizer; true }.getOrDefault(false)

    /**
     * Run text recognition on the provided [bitmap].
     * Returns the full detected text as a single String, or an empty String on failure.
     * Safe to call from any coroutine dispatcher — bridges the ML Kit Task callback.
     */
    suspend fun run(bitmap: Bitmap): String = suspendCoroutine { continuation ->
        val image = InputImage.fromBitmap(bitmap, 0)
        recognizer.process(image)
            .addOnSuccessListener { result ->
                continuation.resume(result.text)
            }
            .addOnFailureListener { exception ->
                Log.e(TAG, "OCR failed: ${exception.message}")
                continuation.resume("")
            }
    }
}
