package com.ariaagent.mobile.core.ai

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * LlamaEngine — JNI wrapper around llama.cpp for on-device inference.
 *
 * Phase 1 status: Full JNI declarations wired. CMakeLists.txt written.
 * To activate: add llama.cpp as a submodule in android/app/src/main/cpp/
 *
 *   cd android/app/src/main/cpp
 *   git submodule add https://github.com/ggerganov/llama.cpp llama.cpp
 *   git submodule update --init --recursive
 *   Then run: ./gradlew assembleDebug (from android/)
 *
 * Hardware target: Samsung Galaxy M31 (Exynos 9611)
 *   - Mali-G72 MP3 → OpenCL 2.0 → n_gpu_layers = 32 (all layers offloaded)
 *   - Cortex-A73 big cores → n_threads = 4
 *   - mmap = true → CRITICAL: keeps RSS ~1700MB instead of ~2.5GB
 *   - mlock = false → do NOT lock pages (would cause OOM on 6GB device)
 *
 * Model: Llama-3.2-1B-Instruct-Q4_K_M.gguf
 *   - Disk: ~870MB
 *   - RAM (RSS): ~1500–1900MB with mmap
 *   - Speed: 8–15 tok/s on Exynos 9611 with OpenCL/Mali-G72 offload
 *
 * Token callback interface (streamed to ChatScreen / AgentViewModel):
 *   interface TokenCallback { fun onToken(token: String) }
 */
object LlamaEngine {

    private val engineMutex = Mutex()

    private var modelHandle: Long = 0L
    private var contextHandle: Long = 0L

    // Track last-used load params so GGUF hot-reload can reload with same settings
    private var lastModelPath: String = ""
    private var lastContextSize: Int = 4096
    private var lastNGpuLayers: Int = 32
    private var lastGpuBackend: String = "opencl"

    /**
     * Catalog model-ID of the currently loaded model.
     * Updated by [loadForRole] whenever a model is swapped in.
     * Empty string when nothing is loaded.
     */
    var currentModelId: String = ""
        private set

    var lastToksPerSec: Double = 0.0
        private set

    var memoryMb: Double = 0.0
        private set

    private var jniAvailable = false

    fun isLoaded(): Boolean = modelHandle != 0L

    /**
     * Load a specific catalog model by ID for a pipeline role (PLANNING, TOOL_USE, etc.)
     * and run [block] with it, then restore the previously loaded model if different.
     *
     * Used by the multi-model pipeline in AgentLoop to chain different specialist models:
     *   - PLANNING model runs at task start (once), then restores the REASONING model.
     *   - TOOL_USE model runs after each REASONING pass to convert thinking → JSON.
     *
     * Hot-path optimisation: if [targetModelId] matches [currentModelId], no swap is done —
     * [block] runs immediately on the already-loaded model.
     *
     * @param targetModelId  Catalog model ID to swap in.
     * @param context        Android context (used to resolve the model file path).
     * @param block          Suspend lambda to run with the target model loaded.
     * @return               Return value of [block], or null if the model file is missing.
     */
    suspend fun <T> withModel(
        targetModelId: String,
        context: android.content.Context,
        block: suspend () -> T
    ): T? {
        // Fast path: already this model
        if (currentModelId == targetModelId && isLoaded()) {
            return block()
        }

        val catalog = ModelCatalog.findById(targetModelId)
        if (catalog == null) {
            android.util.Log.w("LlamaEngine", "withModel: unknown model id '$targetModelId'")
            return null
        }

        val modelFile = java.io.File(context.filesDir, "models/${catalog.filename}")
        if (!modelFile.exists()) {
            android.util.Log.w("LlamaEngine", "withModel: model file not found: ${modelFile.path}")
            return null
        }

        // Snapshot the previously loaded model so we can restore it afterward
        val prevModelId   = currentModelId
        val prevModelPath = lastModelPath
        val prevCtxSize   = lastContextSize
        val prevGpuLayers = lastNGpuLayers
        val prevBackend   = lastGpuBackend

        android.util.Log.i("LlamaEngine",
            "withModel: swapping $prevModelId → $targetModelId (${modelFile.name})")

        return try {
            unload()
            load(modelFile.absolutePath)
            currentModelId = targetModelId
            block()
        } finally {
            // Restore previous model (if there was one and it's different)
            if (prevModelId.isNotEmpty() && prevModelId != targetModelId) {
                android.util.Log.i("LlamaEngine",
                    "withModel: restoring $prevModelId after $targetModelId pass")
                unload()
                load(prevModelPath, prevCtxSize, prevGpuLayers, prevBackend)
                currentModelId = prevModelId
            }
        }
    }

    /**
     * Load the GGUF model from disk.
     * Calls nativeLoadModel() → nativeCreateContext() via JNI.
     *
     * @param path        Absolute path to the .gguf file in internal storage
     * @param contextSize Token context window (4096 for M31 — 128K causes OOM)
     * @param nGpuLayers  Layers to offload to GPU (32 = all layers for Q4_K_M 1B)
     * @param gpuBackend  Backend to use: "opencl" or "cpu".
     *                    Default is "opencl" — Mali-G72 supports OpenCL 2.0.
     */
    fun load(path: String, contextSize: Int = 4096, nGpuLayers: Int = 32, gpuBackend: String = "opencl") {
        lastModelPath    = path
        lastContextSize  = contextSize
        lastNGpuLayers   = nGpuLayers
        lastGpuBackend   = gpuBackend
        if (jniAvailable) {
            modelHandle   = nativeLoadModel(path, contextSize, nGpuLayers, gpuBackend)
            contextHandle = if (modelHandle != 0L) nativeCreateContext(modelHandle) else 0L
            memoryMb      = if (isLoaded()) nativeGetMemoryMb() else 0.0
        } else {
            // JNI library not compiled yet — stay unloaded so isLoaded() returns false
            // and the user sees a proper "no model" message instead of a fake stub response.
            android.util.Log.w("LlamaEngine", "load(): JNI not available — model stays unloaded")
        }
    }

    /**
     * Run a single inference turn with streaming token callback.
     *
     * @param prompt    Full formatted prompt (built by PromptBuilder)
     * @param maxTokens Maximum tokens to generate (200–512 for action responses)
     * @param onToken   Streaming callback — called once per token as it generates
     * @return          Full generated text (complete before returning)
     */
    suspend fun infer(
        prompt: String,
        maxTokens: Int = 512,
        onToken: ((String) -> Unit)? = null
    ): String = engineMutex.withLock {
        if (!isLoaded()) throw IllegalStateException("Model not loaded. Call load() first.")

        if (jniAvailable) {
            val callback = onToken?.let { cb ->
                object : TokenCallback { override fun onToken(token: String) { cb(token) } }
            }
            val result = nativeRunInference(contextHandle, prompt, maxTokens, callback)
            lastToksPerSec = nativeGetToksPerSec()
            result
        } else {
            // Stub response for architecture testing without llama.cpp compiled
            val stub = """{"tool":"Click","node_id":"#1","reason":"stub inference — llama.cpp not compiled"}"""
            lastToksPerSec = 11.5
            onToken?.invoke(stub)
            stub
        }
    }

    fun unload() {
        // Wait for any in-progress inference to finish before freeing JNI memory.
        // Called from Dispatchers.IO so blocking is acceptable, BUT we cap the wait
        // at 10 seconds via withTimeoutOrNull so this can never cause an ANR.
        // If inference has not released the lock within 10 s, we log a warning and
        // skip the native free entirely — the model stays in RAM but the app stays
        // alive. The OS will reclaim memory when the process exits.
        kotlinx.coroutines.runBlocking {
            val completed = kotlinx.coroutines.withTimeoutOrNull(10_000L) {
                engineMutex.withLock {
                    if (unifiedMode) {
                        // In unified mode text handles are aliases to vision handles.
                        // Zero them out WITHOUT calling nativeFree — unloadVision() owns the memory.
                        modelHandle    = 0L
                        contextHandle  = 0L
                        unifiedMode    = false
                    } else {
                        val ctxSnap   = contextHandle
                        val mdlSnap   = modelHandle
                        modelHandle   = 0L
                        contextHandle = 0L
                        if (jniAvailable) {
                            if (ctxSnap != 0L) runCatching { nativeFreeContext(ctxSnap) }
                            if (mdlSnap != 0L) runCatching { nativeFreeModel(mdlSnap) }
                        }
                    }
                    lastToksPerSec = 0.0
                    memoryMb       = 0.0
                }
            }
            if (completed == null) {
                android.util.Log.w("LlamaEngine",
                    "unload(): inference lock not released within 10 s — skipping native free to prevent ANR. " +
                    "Memory will be reclaimed when the process exits.")
            }
        }
    }

    // ─── JNI declarations ────────────────────────────────────────────────────
    // Implemented in llama_jni.cpp — compiled by CMakeLists.txt via NDK.
    // System.loadLibrary() is called in companion object init.

    /**
     * Probe GPU backend availability without loading a model.
     *
     * Internally:
     *  1. Tries dlopen() on known Mali libOpenCL.so paths (bypasses Android linker namespace)
     *  2. Calls ggml_backend_load_all() to force-register all compiled-in backends
     *  3. Enumerates ggml_backend_dev_* and reports names
     *  4. Reports OpenCL backend status
     *
     * Returns a multi-line human-readable string suitable for display in Settings.
     * Call from a background thread (IO dispatcher) — may take ~200 ms on first call.
     */
    fun probeBackends(): String =
        if (jniAvailable) nativeProbeBackends()
        else "JNI library not loaded — cannot probe backends.\nInstall the APK built by CI."

    private external fun nativeProbeBackends(): String
    private external fun nativeLoadModel(path: String, ctxSize: Int, nGpuLayers: Int, gpuBackend: String): Long
    private external fun nativeCreateContext(modelHandle: Long): Long
    private external fun nativeRunInference(
        ctxHandle:     Long,
        prompt:        String,
        maxTokens:     Int,
        tokenCallback: TokenCallback?
    ): String
    private external fun nativeFreeModel(handle: Long)
    private external fun nativeFreeContext(handle: Long)
    private external fun nativeGetToksPerSec(): Double
    private external fun nativeGetMemoryMb(): Double

    // ─── Token streaming callback interface ───────────────────────────────────
    // Passed through JNI to C++ — each generated token calls onToken().

    interface TokenCallback {
        fun onToken(token: String)
    }

    // ─── Multimodal / Vision ──────────────────────────────────────────────────
    //
    // Vision requires a multimodal base model (e.g. moondream2, SmolVLM) plus a
    // matching CLIP projection file (mmproj). These are separate from the Llama 3.2-1B
    // text model — they are loaded into completely independent handles so they
    // can coexist in memory without interfering.
    //
    // Handle layout:
    //   visionModelHandle  — llama_model* for the vision base (e.g. moondream2-Q4)
    //   visionCtxHandle    — llama_context* for the vision base
    //   visionHandle       — mtmd_context* (CLIP encoder + projection layer)

    private var visionModelHandle: Long = 0L
    private var visionCtxHandle: Long = 0L
    private var visionHandle: Long = 0L

    /**
     * True when the active model was loaded in unified mode via [loadUnified].
     *
     * In unified mode the primary text handles (modelHandle/contextHandle) are
     * aliased to the vision handles — a single model instance handles BOTH
     * visual understanding and text reasoning.  The agent loop skips the
     * separate VisionEngine.describe() step and calls [inferWithVision]
     * directly with the raw screenshot bytes + full reasoning prompt.
     *
     * In non-unified mode the primary model is text-only and VisionEngine
     * auto-loads SmolVLM-256M as a separate screen-reading helper.
     */
    @Volatile private var unifiedMode: Boolean = false

    fun isUnifiedMode(): Boolean = unifiedMode
    fun isVisionLoaded(): Boolean = visionHandle != 0L

    /**
     * Load a multimodal vision model + its CLIP mmproj file.
     *
     * Must be called BEFORE [inferWithVision].
     * Safe to call even when the main text model (Llama 3.2-1B) is loaded —
     * they use independent handles and do not share state.
     *
     * @param visionModelPath  Absolute path to vision base GGUF (e.g. moondream2-Q4_K_M.gguf)
     * @param mmProjPath       Absolute path to the mmproj GGUF (e.g. moondream2-mmproj.gguf)
     * @param contextSize      KV-cache size for vision model (2048 is enough for moondream2)
     * @param nGpuLayers       GPU layers to offload; 32 = all layers for Mali-G72 OpenCL
     * @param gpuBackend       Backend to use: "opencl" or "cpu"
     * @return true if both model and mmproj loaded successfully
     */
    fun loadVision(
        visionModelPath: String,
        mmProjPath: String,
        contextSize: Int = 2048,
        nGpuLayers: Int = 32,
        gpuBackend: String = "opencl"
    ): Boolean {
        if (!jniAvailable) {
            android.util.Log.w("LlamaEngine", "loadVision: JNI not available — vision stays unloaded")
            return false
        }
        unloadVision()
        visionModelHandle = nativeLoadModel(visionModelPath, contextSize, nGpuLayers, gpuBackend)
        if (visionModelHandle == 0L) {
            android.util.Log.e("LlamaEngine", "loadVision: failed to load vision base model")
            return false
        }
        visionCtxHandle = nativeCreateContext(visionModelHandle)
        if (visionCtxHandle == 0L) {
            android.util.Log.e("LlamaEngine", "loadVision: failed to create vision context")
            nativeFreeModel(visionModelHandle); visionModelHandle = 0L
            return false
        }
        visionHandle = nativeInitVision(mmProjPath, visionModelHandle)
        if (visionHandle == 0L) {
            android.util.Log.e("LlamaEngine", "loadVision: mmproj init failed")
            nativeFreeContext(visionCtxHandle); visionCtxHandle = 0L
            nativeFreeModel(visionModelHandle); visionModelHandle = 0L
            return false
        }
        android.util.Log.i("LlamaEngine", "Vision loaded: model=$visionModelPath mmproj=$mmProjPath")
        return true
    }

    /**
     * Free vision model, context, and mmproj handles.
     * The main text model is NOT affected unless running in unified mode,
     * in which case [unload] must be called first to clear the aliases.
     */
    fun unloadVision() {
        if (jniAvailable) {
            if (visionHandle      != 0L) { nativeFreeVision(visionHandle);      visionHandle      = 0L }
            if (visionCtxHandle   != 0L) { nativeFreeContext(visionCtxHandle);  visionCtxHandle   = 0L }
            if (visionModelHandle != 0L) { nativeFreeModel(visionModelHandle);  visionModelHandle = 0L }
        } else {
            visionHandle = 0L; visionCtxHandle = 0L; visionModelHandle = 0L
        }
    }

    /**
     * Load a multimodal VLM as the SOLE model instance for both vision and reasoning.
     *
     * The model is loaded ONCE into the vision handles.  The primary text handles
     * (modelHandle / contextHandle) are then aliased to the same values so that
     * [isLoaded] returns true and callers that check the primary handles still work.
     * No second copy of the model is held in RAM — critical on M31 (6 GB).
     *
     * After this call:
     *   • [isUnifiedMode] == true
     *   • [isLoaded]      == true  (text handles aliased to vision handles)
     *   • [isVisionLoaded]== true
     *
     * The agent loop detects [isUnifiedMode] and calls [inferWithVision] directly
     * with the raw screenshot + full reasoning prompt, skipping the separate
     * VisionEngine.describe() step entirely.
     *
     * @param modelPath   Absolute path to the VLM base GGUF
     * @param mmProjPath  Absolute path to the matching mmproj GGUF (F16)
     * @param contextSize KV-cache size (4096 is sufficient for action-sized outputs)
     * @param nGpuLayers  GPU offload layers; 32 = all layers for Mali-G72 OpenCL
     * @param gpuBackend  Backend to use: "opencl" or "cpu"
     * @return true if both model + mmproj loaded successfully
     */
    fun loadUnified(
        modelPath:   String,
        mmProjPath:  String,
        contextSize: Int = 4096,
        nGpuLayers:  Int = 32,
        gpuBackend:  String = "opencl"
    ): Boolean {
        unload()
        unloadVision()
        val ok = loadVision(modelPath, mmProjPath, contextSize, nGpuLayers, gpuBackend)
        if (ok) {
            // Alias — no extra RAM cost, no extra JNI calls
            modelHandle   = visionModelHandle
            contextHandle = visionCtxHandle
            unifiedMode   = true
            memoryMb      = if (jniAvailable) nativeGetMemoryMb() else 1700.0
            android.util.Log.i("LlamaEngine",
                "Unified VLM mode active: $modelPath + $mmProjPath")
        }
        return ok
    }

    /**
     * Run vision inference: encode [imageBytes] (JPEG/PNG) through CLIP, then
     * generate a response to [prompt] grounded in the image.
     *
     * Requires [loadVision] to have been called successfully.
     * Falls back to a stub description when JNI is not compiled.
     *
     * @param imageBytes  Raw JPEG or PNG bytes — produced by Bitmap.compress(JPEG, 85)
     * @param prompt      Text question about the image (image marker is prepended automatically)
     * @param maxTokens   Generation cap — 128 is enough for screen descriptions
     * @param onToken     Optional streaming callback
     * @return Full generated text response
     */
    suspend fun inferWithVision(
        imageBytes: ByteArray,
        prompt: String,
        maxTokens: Int = 128,
        onToken: ((String) -> Unit)? = null
    ): String {
        if (!isVisionLoaded()) throw IllegalStateException("Vision not loaded. Call loadVision() first.")
        return if (jniAvailable) {
            val callback = onToken?.let { cb ->
                object : TokenCallback { override fun onToken(token: String) { cb(token) } }
            }
            val result = nativeRunVisionInference(
                visionCtxHandle, visionHandle, imageBytes, prompt, maxTokens, callback
            )
            lastToksPerSec = nativeGetToksPerSec()
            result
        } else {
            val stub = "[vision stub] Screen shows a mobile UI. Text visible via OCR. " +
                "No mmproj compiled yet — install libllama-jni.so to enable real vision."
            onToken?.invoke(stub)
            stub
        }
    }

    // ─── Vision JNI declarations (implemented in llama_jni.cpp) ──────────────

    private external fun nativeInitVision(mmProjPath: String, modelHandle: Long): Long
    private external fun nativeFreeVision(visionHandle: Long)
    private external fun nativeRunVisionInference(
        ctxHandle:     Long,
        visionHandle:  Long,
        imageBytes:    ByteArray,
        prompt:        String,
        maxTokens:     Int,
        tokenCallback: TokenCallback?
    ): String

    // ─── LoRA adapter loading ─────────────────────────────────────────────────

    private external fun nativeLoadLora(ctxHandle: Long, adapterPath: String, scale: Float): Boolean

    /**
     * Load a trained adapter or checkpoint on top of the current model.
     *
     * Detects the file type from its first 4 bytes:
     *   "GGUF" (0x47 0x47 0x55 0x46) → full GGUF model checkpoint produced by
     *     nativeTrainLora() via llama_model_save_to_file(). Unloads the current
     *     model and reloads using the checkpoint path, preserving contextSize and
     *     nGpuLayers from the original load() call.
     *   Otherwise → classic LoRA adapter binary; loaded via nativeLoadLora()
     *     (llama_adapter_lora_init + llama_set_adapters_lora).
     *
     * @param adapterPath Absolute path to the .gguf checkpoint or LoRA .bin file
     * @param scale       LoRA influence weight — ignored for GGUF checkpoints
     */
    fun loadLora(adapterPath: String, scale: Float = 0.8f): Boolean {
        if (!jniAvailable || !isLoaded()) return false
        return try {
            if (isGgufFile(adapterPath)) {
                // Full GGUF model checkpoint — unload inference model and reload
                // the fine-tuned weights as the new base model.
                android.util.Log.i("LlamaEngine",
                    "GGUF checkpoint detected — hot-reloading as base model: $adapterPath")
                unload()
                load(adapterPath, lastContextSize, lastNGpuLayers, lastGpuBackend)
                isLoaded()
            } else {
                // Classic LoRA adapter binary — apply on top of loaded base model
                val ok = nativeLoadLora(contextHandle, adapterPath, scale)
                android.util.Log.i("LlamaEngine",
                    "LoRA adapter loaded: $adapterPath (scale=$scale) → $ok")
                ok
            }
        } catch (e: Exception) {
            android.util.Log.w("LlamaEngine", "loadLora failed: ${e.message}")
            false
        }
    }

    /**
     * Returns true if the file starts with the GGUF magic bytes (0x47 0x47 0x55 0x46 = "GGUF").
     * Used by loadLora() to distinguish trained GGUF checkpoints from LoRA adapter binaries.
     */
    private fun isGgufFile(path: String): Boolean {
        return try {
            java.io.RandomAccessFile(path, "r").use { f ->
                val magic = ByteArray(4)
                f.readFully(magic)
                magic[0] == 0x47.toByte() &&   // 'G'
                magic[1] == 0x47.toByte() &&   // 'G'
                magic[2] == 0x55.toByte() &&   // 'U'
                magic[3] == 0x46.toByte()      // 'F'
            }
        } catch (e: Exception) { false }
    }

    // companion object is invalid inside an `object` — use a bare init block instead.
    init {
        try {
            System.loadLibrary("llama-jni")
            // Library loaded successfully — activate real JNI inference path
            markJniAvailable()
            android.util.Log.i("LlamaEngine", "llama-jni loaded — real inference active")
        } catch (e: UnsatisfiedLinkError) {
            // llama.cpp not compiled yet (submodule missing) — stub mode active
            android.util.Log.w("LlamaEngine", "llama-jni not found — running in stub mode")
        }
    }

    /**
     * Called by init block after System.loadLibrary succeeds.
     * Switches from stub → real JNI inference.
     */
    internal fun markJniAvailable() {
        jniAvailable = true
    }
}
