package com.ariaagent.mobile.core.pc

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * PcBridgeClient — Offloads LLM inference and LoRA training to a PC over USB.
 *
 * Architecture
 * ────────────
 *   Android ─── USB ─── PC
 *                │
 *                ├── adb forward tcp:11435 tcp:11435  (run on PC before starting ARIA)
 *                │
 *                └── aria_pc_server.py  (Python server running on PC, port 11435)
 *
 * The Android app connects to http://localhost:11435 — requests are forwarded
 * through the ADB TCP tunnel to the PC's server on the same port.
 * No Wi-Fi or network setup required; USB only.
 *
 * Benefits over on-device inference:
 *   • PC can run 7B–70B parameter models (vs 1B limit on M31)
 *   • GPU inference at 60–200 tok/s (vs 8–15 tok/s on Exynos 9611)
 *   • LoRA training in seconds (vs minutes on-device or not feasible)
 *   • No thermal throttling on phone during inference
 *
 * Fallback behaviour:
 *   [isConnected] is checked by AgentLoop before routing inference here.
 *   If false, AgentLoop falls back to on-device LlamaEngine transparently.
 *   Disconnections update [isConnected] = false immediately, so the next
 *   AgentLoop step automatically routes back to on-device.
 *
 * Setup steps (run on PC before connecting):
 *   1. Install Python deps:  pip install llama-cpp-python flask
 *   2. Start server:         python aria_pc_server.py --model /path/to/model.gguf
 *   3. Enable ADB forward:   adb forward tcp:11435 tcp:11435
 *   4. In ARIA app:          tap "Connect to PC" in Settings → PC Bridge
 *
 * HTTP API:
 *   GET  /health          → {"status":"ok","model":"<model name>"}
 *   POST /infer           → body: {"prompt":"…","goal":"…"} → {"action_json":"…"}
 *   POST /sync            → body: {"experiences":[…]} → {"stored":N}
 *   GET  /adapter         → binary LoRA adapter .bin download
 *   POST /train           → body: {"experiences":[…]} → {"done":true,"loss":0.12}
 */
object PcBridgeClient {

    private const val TAG              = "PcBridgeClient"
    const val DEFAULT_PORT             = 11435   // must match aria_pc_server.py
    private const val CONNECT_TIMEOUT  = 2_000   // ms — fast health check
    private const val READ_TIMEOUT     = 90_000  // ms — allow slow GPU inference
    private const val WRITE_TIMEOUT    = 15_000  // ms

    @Volatile var port: Int = DEFAULT_PORT

    private val _isConnected = AtomicBoolean(false)
    val isConnected: Boolean get() = _isConnected.get()

    /** Human-readable name of the model loaded on the PC, or "" if unknown. */
    @Volatile var remoteModelName: String = ""
        private set

    // ── Auto-reconnect state ──────────────────────────────────────────────────
    // Staggered backoff: 5s → 15s → 45s (×3 each step), capped at 60s
    private val reconnectAttempts   = AtomicInteger(0)
    private val lastDisconnectEpoch = AtomicLong(0L)
    private var reconnectJob: Job?  = null

    private fun baseUrl() = "http://127.0.0.1:$port"

    // ── Connection management ─────────────────────────────────────────────────

    /**
     * Ping the PC server's /health endpoint.
     * Returns true and updates [isConnected] if the server responds successfully.
     * Must be called from a coroutine — performs network I/O on [Dispatchers.IO].
     */
    suspend fun checkConnection(): Boolean = withContext(Dispatchers.IO) {
        try {
            val conn = openConn("${baseUrl()}/health", "GET", connectTimeout = CONNECT_TIMEOUT)
            conn.connect()
            val ok = conn.responseCode == 200
            if (ok) {
                val body = conn.inputStream.bufferedReader().readText()
                runCatching {
                    val json = JSONObject(body)
                    remoteModelName = json.optString("model", "")
                }
                Log.i(TAG, "PC bridge connected — model: ${remoteModelName.ifBlank { "unknown" }}")
                reconnectAttempts.set(0)  // reset backoff on successful connect
            }
            conn.disconnect()
            _isConnected.set(ok)
            ok
        } catch (e: Exception) {
            Log.d(TAG, "PC bridge not reachable: ${e.message}")
            _isConnected.set(false)
            false
        }
    }

    fun disconnect() {
        _isConnected.set(false)
        remoteModelName = ""
        lastDisconnectEpoch.set(System.currentTimeMillis())
        Log.i(TAG, "PC bridge disconnected")
    }

    /**
     * Start a background auto-reconnect loop.
     *
     * When the ADB tunnel drops (USB disconnect, PC sleep, server crash), the next
     * infer() call immediately falls back to on-device inference. Meanwhile this
     * coroutine keeps probing /health with exponential backoff so the bridge
     * reconnects automatically when the PC becomes available again.
     *
     * Backoff schedule: 5s → 15s → 45s, then capped at 60s per attempt.
     * Attempts reset to 0 on successful reconnect.
     *
     * Call once from AgentViewModel when PC bridge is enabled in settings.
     * Call [stopAutoReconnect] when the user disables PC bridge or agent stops.
     */
    fun startAutoReconnect(scope: CoroutineScope) {
        reconnectJob?.cancel()
        reconnectJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                if (!_isConnected.get()) {
                    val attempt = reconnectAttempts.incrementAndGet()
                    // Backoff schedule: 5s → 15s → 45s → 60s cap
                    val backoffMs = when (attempt) {
                        1    -> 5_000L
                        2    -> 15_000L
                        3    -> 45_000L
                        else -> 60_000L
                    }
                    Log.d(TAG, "Auto-reconnect attempt $attempt — waiting ${backoffMs / 1000}s")
                    delay(backoffMs)
                    if (!isActive) break
                    val ok = checkConnection()
                    if (ok) {
                        Log.i(TAG, "Auto-reconnect succeeded after $attempt attempt(s)")
                        reconnectAttempts.set(0)
                    }
                } else {
                    reconnectAttempts.set(0)
                    delay(15_000L)  // healthy — probe every 15s to detect silent drops
                    checkConnection()  // re-validate; sets _isConnected=false if dropped
                }
            }
        }
        Log.i(TAG, "PC bridge auto-reconnect loop started")
    }

    /** Cancel the auto-reconnect loop. */
    fun stopAutoReconnect() {
        reconnectJob?.cancel()
        reconnectJob = null
        Log.i(TAG, "PC bridge auto-reconnect loop stopped")
    }

    /**
     * Mark as disconnected and trigger reconnect attempt immediately
     * (reduces the next backoff to the minimum 5s instead of the current position).
     */
    private fun onTransientFailure(reason: String) {
        if (_isConnected.get()) {
            Log.w(TAG, "PC bridge transient failure ($reason) — marking disconnected, reconnect scheduled")
            _isConnected.set(false)
            lastDisconnectEpoch.set(System.currentTimeMillis())
            reconnectAttempts.set(0)  // start backoff from scratch so next attempt is quick (5s)
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Send the full prompt to the PC server for LLM inference.
     *
     * The PC server runs the same Llama-format prompt as the on-device engine,
     * so no prompt transformation is needed — just pass the PromptBuilder output.
     *
     * @param prompt  Full prompt string from [PromptBuilder.build]
     * @param goal    User's goal string (for PC-side logging)
     * @return The action JSON string, or null on failure (caller falls back to on-device).
     */
    suspend fun infer(prompt: String, goal: String = ""): String? = withContext(Dispatchers.IO) {
        if (!_isConnected.get()) return@withContext null
        try {
            val body = JSONObject().apply {
                put("prompt", prompt)
                put("goal",   goal)
            }.toString()

            val conn = openConn("${baseUrl()}/infer", "POST")
            conn.setRequestProperty("Content-Type", "application/json")
            conn.outputStream.bufferedWriter().use { it.write(body) }

            if (conn.responseCode != 200) {
                Log.w(TAG, "PC /infer returned ${conn.responseCode}")
                onTransientFailure("http_${conn.responseCode}")
                conn.disconnect()
                return@withContext null
            }

            val response = conn.inputStream.bufferedReader().readText()
            conn.disconnect()
            val parsed = JSONObject(response)
            parsed.optString("action_json").ifBlank { null }
        } catch (e: Exception) {
            Log.w(TAG, "PC infer failed: ${e.message}")
            onTransientFailure(e.javaClass.simpleName)
            null
        }
    }

    // ── Experience sync ───────────────────────────────────────────────────────

    /**
     * Upload serialised experience tuples to the PC for training.
     * The PC server appends them to its local dataset and optionally triggers
     * a LoRA fine-tuning pass.
     *
     * @param experiencesJson JSON-encoded list of [ExperienceTuple]-like objects
     * @return Number of tuples accepted by the server, or -1 on failure.
     */
    suspend fun syncExperiences(experiencesJson: String): Int = withContext(Dispatchers.IO) {
        if (!_isConnected.get()) return@withContext -1
        try {
            val body = JSONObject().apply {
                put("experiences", JSONArray(experiencesJson))
            }.toString()

            val conn = openConn("${baseUrl()}/sync", "POST")
            conn.setRequestProperty("Content-Type", "application/json")
            conn.outputStream.bufferedWriter().use { it.write(body) }

            if (conn.responseCode != 200) {
                conn.disconnect()
                return@withContext -1
            }

            val response = conn.inputStream.bufferedReader().readText()
            conn.disconnect()
            JSONObject(response).optInt("stored", 0)
        } catch (e: Exception) {
            Log.w(TAG, "PC sync failed: ${e.message}")
            -1
        }
    }

    // ── LoRA training ─────────────────────────────────────────────────────────

    /**
     * Ask the PC server to run a LoRA training pass on its accumulated experiences.
     *
     * @param experiencesJson Optional fresh experiences to add before training.
     *                        Pass null to train on whatever the server already has.
     * @return The training loss reported by the server, or null on failure.
     */
    suspend fun requestTraining(experiencesJson: String? = null): Float? = withContext(Dispatchers.IO) {
        if (!_isConnected.get()) return@withContext null
        try {
            val body = JSONObject().apply {
                experiencesJson?.let { put("experiences", JSONArray(it)) }
            }.toString()

            val conn = openConn("${baseUrl()}/train", "POST", readTimeout = 300_000)
            conn.setRequestProperty("Content-Type", "application/json")
            conn.outputStream.bufferedWriter().use { it.write(body) }

            if (conn.responseCode != 200) {
                conn.disconnect()
                return@withContext null
            }

            val response = conn.inputStream.bufferedReader().readText()
            conn.disconnect()
            JSONObject(response).optDouble("loss", Double.NaN).toFloat()
                .takeIf { !it.isNaN() }
        } catch (e: Exception) {
            Log.w(TAG, "PC training failed: ${e.message}")
            null
        }
    }

    // ── LoRA adapter download ─────────────────────────────────────────────────

    /**
     * Download the LoRA adapter weights trained on the PC to the device.
     * The adapter can then be loaded by [LoraTrainer] for on-device inference.
     *
     * @param destPath  Absolute path on the device to write the adapter binary.
     * @return True if the download succeeded and the file was written.
     */
    suspend fun downloadAdapter(destPath: String): Boolean = withContext(Dispatchers.IO) {
        if (!_isConnected.get()) return@withContext false
        try {
            val conn = openConn("${baseUrl()}/adapter", "GET", readTimeout = 120_000)
            conn.connect()
            if (conn.responseCode != 200) {
                conn.disconnect()
                return@withContext false
            }
            val bytes = conn.inputStream.readBytes()
            conn.disconnect()
            if (bytes.isEmpty()) return@withContext false
            File(destPath).writeBytes(bytes)
            Log.i(TAG, "Downloaded PC adapter to $destPath (${bytes.size / 1024} KB)")
            true
        } catch (e: Exception) {
            Log.w(TAG, "PC adapter download failed: ${e.message}")
            false
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private fun openConn(
        urlString: String,
        method: String,
        connectTimeout: Int = CONNECT_TIMEOUT,
        readTimeout: Int = READ_TIMEOUT
    ): HttpURLConnection {
        val conn = URL(urlString).openConnection() as HttpURLConnection
        conn.requestMethod       = method
        conn.connectTimeout      = connectTimeout
        conn.readTimeout         = readTimeout
        conn.doInput             = true
        if (method == "POST") conn.doOutput = true
        return conn
    }
}
