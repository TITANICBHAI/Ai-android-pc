package com.ariaagent.mobile.core.setup

import android.content.Context
import android.util.Log
import com.ariaagent.mobile.core.ai.ModelManager
import com.ariaagent.mobile.core.memory.EmbeddingModelManager
import com.ariaagent.mobile.core.perception.ObjectDetectorEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * ModelBootstrap — sequential on-device model download orchestrator.
 *
 * Download order (hard dependency chain):
 *
 *   Step 1 ── GGUF VLM  (size depends on selected model, HuggingFace)
 *             Active model GGUF + mmproj (vision projector) if needed.
 *             Downloaded by ModelDownloadService (foreground service, user-visible).
 *             bootstrap() WAITS here until ModelManager.isModelReady() is true.
 *             After: LlamaEngine can load the model and inference begins.
 *
 *   Step 2 ── MiniLM-L6-v2 ONNX  (~23 MB, HuggingFace)
 *             sentence-transformers/all-MiniLM-L6-v2 ONNX export.
 *             Only starts after Step 1 is complete — uses the same OkHttp
 *             connection pool and avoids competing for bandwidth.
 *             After: EmbeddingEngine switches from hash fallback → semantic search.
 *
 *   Step 3 ── EfficientDet-Lite0 INT8  (~4.4 MB, Google CDN)
 *             MediaPipe object detector for non-accessibility UI elements.
 *             Only starts after Step 2 is complete — tiny download, near instant.
 *             After: ObjectDetectorEngine enables bounding-box visual perception.
 *
 *   Step 4 ── READY
 *             All models present. Agent can start a full-capability loop.
 *             If LoRA adapter already exists on disk it is auto-loaded by
 *             LlamaEngine.loadLora() during normal agent startup — not here.
 *
 * Events emitted via the [emit] callback (forwarded to AgentEventBus):
 *   "bootstrap_stage" {
 *     stage:   "awaiting_gguf" | "downloading_minilm" | "downloading_detector" | "ready"
 *     step:    1 | 2 | 3 | 4          (which step we are on)
 *     totalSteps: 4
 *     percent: 0–100                  (progress within the current step)
 *     label:   "Downloading AI brain…" (human-readable status)
 *   }
 *
 * Usage in AgentViewModel:
 *   scope.launch(Dispatchers.IO) { ModelBootstrap.run(ctx, ::emitBootstrapEvent) }
 */
object ModelBootstrap {

    private const val TAG = "ModelBootstrap"
    private const val GGUF_POLL_INTERVAL_MS = 3_000L   // check every 3 s while waiting for GGUF

    data class BootstrapEvent(
        val stage: String,
        val step: Int,
        val totalSteps: Int = 4,
        val percent: Int,
        val label: String
    )

    /**
     * Run the full sequential bootstrap.
     * Suspend until all three models are confirmed ready.
     *
     * @param context Android context
     * @param emit     callback that forwards a [BootstrapEvent] to the JS layer
     */
    suspend fun run(
        context: Context,
        emit: (BootstrapEvent) -> Unit
    ) = withContext(Dispatchers.IO) {

        // ── Step 1: Wait for GGUF ─────────────────────────────────────────────
        val activeModel = ModelManager.activeEntry(context)
        if (activeModel == null) {
            Log.i(TAG, "No model selected yet — skipping GGUF wait (user must select from Modules)")
            emit(BootstrapEvent(
                stage   = "awaiting_gguf",
                step    = 1,
                percent = 0,
                label   = "No model selected — choose one in Modules"
            ))
        } else if (!ModelManager.isModelReady(context)) {
            Log.i(TAG, "Waiting for GGUF model (${activeModel.displayName}) — user must trigger download from Modules")
            emit(BootstrapEvent(
                stage   = "awaiting_gguf",
                step    = 1,
                percent = 0,
                label   = "Waiting for ${activeModel.displayName} download…"
            ))

            // Poll — ModelDownloadService runs in parallel and writes the file.
            while (!ModelManager.isModelReady(context)) {
                val downloaded = ModelManager.downloadedBytes(context)
                val total      = activeModel.expectedSizeBytes
                val pct        = if (total > 0) ((downloaded * 100L) / total).toInt() else 0
                emit(BootstrapEvent(
                    stage   = "awaiting_gguf",
                    step    = 1,
                    percent = pct,
                    label   = "${activeModel.displayName}: ${downloaded / 1_000_000} / ${total / 1_000_000} MB"
                ))
                delay(GGUF_POLL_INTERVAL_MS)
            }

            Log.i(TAG, "GGUF model ready — proceeding to MiniLM download")
        } else {
            Log.i(TAG, "GGUF already on disk — skipping step 1")
        }

        emit(BootstrapEvent(
            stage   = "awaiting_gguf",
            step    = 1,
            percent = 100,
            label   = "AI brain ready"
        ))

        // ── Step 2: MiniLM-L6-v2 ONNX ────────────────────────────────────────
        if (!EmbeddingModelManager.isModelReady(context)) {
            Log.i(TAG, "Downloading MiniLM-L6-v2 ONNX (~23 MB)…")
            emit(BootstrapEvent(
                stage   = "downloading_minilm",
                step    = 2,
                percent = 0,
                label   = "Downloading memory model (23 MB)…"
            ))

            val ok = EmbeddingModelManager.download(context) { downloaded, total ->
                val pct = if (total > 0) ((downloaded * 100L) / total).toInt() else 0
                emit(BootstrapEvent(
                    stage   = "downloading_minilm",
                    step    = 2,
                    percent = pct,
                    label   = "Memory model: ${downloaded / 1_000_000} / ${total / 1_000_000} MB"
                ))
            }

            if (ok) {
                Log.i(TAG, "MiniLM ONNX ready")
            } else {
                Log.w(TAG, "MiniLM download failed — EmbeddingEngine will use hash fallback")
            }
        } else {
            Log.i(TAG, "MiniLM already on disk — skipping step 2")
        }

        emit(BootstrapEvent(
            stage   = "downloading_minilm",
            step    = 2,
            percent = 100,
            label   = "Memory model ready"
        ))

        // ── Step 3: EfficientDet-Lite0 INT8 ──────────────────────────────────
        if (!ObjectDetectorEngine.isModelReady(context)) {
            Log.i(TAG, "Downloading EfficientDet-Lite0 INT8 (~4.4 MB)…")
            emit(BootstrapEvent(
                stage   = "downloading_detector",
                step    = 3,
                percent = 0,
                label   = "Downloading vision model (4.4 MB)…"
            ))

            val ok = ObjectDetectorEngine.ensureModel(context)

            if (ok) {
                Log.i(TAG, "EfficientDet-Lite0 ready")
            } else {
                Log.w(TAG, "EfficientDet download failed — object detection disabled until next launch")
            }
        } else {
            Log.i(TAG, "EfficientDet already on disk — skipping step 3")
        }

        emit(BootstrapEvent(
            stage   = "downloading_detector",
            step    = 3,
            percent = 100,
            label   = "Vision model ready"
        ))

        // ── Step 4: All done ──────────────────────────────────────────────────
        Log.i(TAG, "All models ready — ARIA agent fully operational")
        emit(BootstrapEvent(
            stage   = "ready",
            step    = 4,
            percent = 100,
            label   = "All models ready — ARIA is operational"
        ))
    }
}
