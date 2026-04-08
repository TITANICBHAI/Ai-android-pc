package com.ariaagent.mobile.core.monitoring

import android.content.Context
import android.util.Log
import com.ariaagent.mobile.core.agent.AgentLoop
import com.ariaagent.mobile.core.agent.AppSkillRegistry
import com.ariaagent.mobile.core.agent.TaskQueueManager
import com.ariaagent.mobile.core.events.AgentEventBus
import com.ariaagent.mobile.core.memory.ExperienceStore
import com.ariaagent.mobile.core.rl.LoraTrainer
import com.ariaagent.mobile.core.rl.PolicyNetwork
import com.ariaagent.mobile.core.system.ThermalGuard
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * MonitoringPusher — Phase 16 real-time agent state updater.
 *
 * Subscribes to [AgentEventBus] and updates [LocalSnapshotStore] whenever
 * significant agent events fire. [LocalDeviceServer] then serves these
 * snapshots over HTTP so the web dashboard can connect on the same Wi-Fi network.
 *
 * Also writes a flat file snapshot to {filesDir}/monitoring/snapshot.json
 * for ADB pull debugging.
 *
 * Architecture:
 *   AgentEventBus → MonitoringPusher → LocalSnapshotStore ← LocalDeviceServer → browser
 *
 * Fully local — no cloud, no OkHttp push, no external server.
 * Throttled to ≤1 update per [MIN_UPDATE_INTERVAL_MS] milliseconds.
 *
 * Phase: 16 (Local Monitoring)
 */
object MonitoringPusher {

    private const val TAG                    = "MonitoringPusher"
    private const val MIN_UPDATE_INTERVAL_MS = 3_000L
    private const val SNAPSHOT_DIR           = "monitoring"
    private const val SNAPSHOT_FILENAME      = "snapshot.json"

    private val scope    = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var watchJob: Job? = null

    @Volatile private var lastUpdateMs = 0L

    private val TRIGGER_EVENTS = setOf(
        "action_performed",
        "agent_status_changed",
        "thermal_status_changed",
        "learning_cycle_complete",
        "game_loop_status",
        "skill_updated",
        "task_chain_advanced"
    )

    // ─── Lifecycle ────────────────────────────────────────────────────────────

    fun start(context: Context) {
        watchJob?.cancel()
        watchJob = scope.launch {
            AgentEventBus.flow.collect { (name, _) ->
                if (!isActive) return@collect
                if (name in TRIGGER_EVENTS) {
                    val now = System.currentTimeMillis()
                    if (now - lastUpdateMs >= MIN_UPDATE_INTERVAL_MS) {
                        lastUpdateMs = now
                        pushAll(context)
                    }
                }
            }
        }
        // Push initial snapshot so the dashboard shows live data immediately
        scope.launch(Dispatchers.IO) { pushAll(context) }
        Log.d(TAG, "started")
    }

    fun stop() {
        watchJob?.cancel()
        watchJob = null
        Log.d(TAG, "stopped")
    }

    /** Returns path to the latest file-based snapshot, or null if none written yet. */
    fun snapshotPath(context: Context): String? {
        val f = File(context.filesDir, "$SNAPSHOT_DIR/$SNAPSHOT_FILENAME")
        return if (f.exists()) f.absolutePath else null
    }

    // ─── Per-endpoint push ────────────────────────────────────────────────────

    private fun pushAll(context: Context) {
        try { LocalSnapshotStore.status   = buildStatus(context)   } catch (_: Exception) {}
        try { LocalSnapshotStore.thermal  = buildThermal()         } catch (_: Exception) {}
        try { LocalSnapshotStore.rl       = buildRl(context)       } catch (_: Exception) {}
        try { LocalSnapshotStore.lora     = buildLora(context)     } catch (_: Exception) {}
        try { LocalSnapshotStore.memory   = buildMemory(context)   } catch (_: Exception) {}
        try { LocalSnapshotStore.activity = buildActivity(context) } catch (_: Exception) {}
        try { LocalSnapshotStore.modules  = buildModules(context)  } catch (_: Exception) {}

        // Persist flat file snapshot for ADB pull debugging
        scope.launch(Dispatchers.IO) {
            try {
                val dir = File(context.filesDir, SNAPSHOT_DIR).also { it.mkdirs() }
                val tmp = File(dir, "$SNAPSHOT_FILENAME.tmp")
                val tgt = File(dir, SNAPSHOT_FILENAME)
                tmp.writeText(JSONObject().apply {
                    put("status",    LocalSnapshotStore.status)
                    put("thermal",   LocalSnapshotStore.thermal)
                    put("rl",        LocalSnapshotStore.rl)
                    put("lora",      LocalSnapshotStore.lora)
                    put("memory",    LocalSnapshotStore.memory)
                    put("activity",  LocalSnapshotStore.activity)
                    put("modules",   LocalSnapshotStore.modules)
                    put("writtenAt", System.currentTimeMillis())
                }.toString(2), Charsets.UTF_8)
                tmp.renameTo(tgt)
            } catch (_: Exception) {}
        }
    }

    // ─── Endpoint builders ────────────────────────────────────────────────────

    private fun buildStatus(context: Context): JSONObject {
        val s = AgentLoop.state
        return JSONObject().apply {
            put("status",              s.status.name.lowercase())
            put("currentTask",         s.goal.ifBlank { null } ?: JSONObject.NULL)
            put("currentApp",          s.appPackage.ifBlank { null } ?: JSONObject.NULL)
            put("stepCount",           s.stepCount)
            put("lastAction",          s.lastAction)
            put("lastError",           s.lastError)
            put("gameMode",            s.gameMode)
            put("modelReady",          true)
            put("llmLoaded",           true)
            put("accessibilityActive", false)
            put("screenCaptureActive", false)
            put("memoryUsedMb",        Runtime.getRuntime().let {
                ((it.totalMemory() - it.freeMemory()) / 1_048_576L).toInt()
            })
        }
    }

    private fun buildThermal(): JSONObject = JSONObject().apply {
        put("level",           ThermalGuard.currentLevel.name.lowercase())
        put("inferenceSafe",   ThermalGuard.isInferenceSafe())
        put("trainingSafe",    ThermalGuard.isTrainingSafe())
        put("throttleCapture", ThermalGuard.shouldThrottleCapture())
        put("emergency",       ThermalGuard.isEmergency())
    }

    private fun buildRl(context: Context): JSONObject {
        val store = ExperienceStore.getInstance(context)
        val episodesRun = try {
            store.countByResult("success") + store.countByResult("failure")
        } catch (_: Exception) { 0 }
        val loraVer = try { LoraTrainer.currentVersion(context) } catch (_: Exception) { 0 }
        return JSONObject().apply {
            put("episodesRun",       episodesRun)
            put("loraVersion",       loraVer)
            put("adapterLoaded",     loraVer > 0)
            put("untrainedSamples",  try { store.countByResult("success") } catch (_: Exception) { 0 })
            put("adamStep",          PolicyNetwork.adamStepCount)
            put("lastPolicyLoss",    PolicyNetwork.lastPolicyLoss)
            put("rewardHistory",     JSONArray())
            put("policyLossHistory", JSONArray())
        }
    }

    private fun buildLora(context: Context): JSONArray {
        val arr = JSONArray()
        val version = try { LoraTrainer.currentVersion(context) } catch (_: Exception) { 0 }
        if (version > 0) {
            val isoFmt = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", java.util.Locale.US)
                .also { it.timeZone = java.util.TimeZone.getTimeZone("UTC") }
            arr.put(JSONObject().apply {
                put("version",          version)
                put("path",             try { LoraTrainer.latestAdapterPath(context) ?: "" } catch (_: Exception) { "" })
                put("trainedAt",        isoFmt.format(java.util.Date()))
                put("samplesUsed",      0)
                put("successRateDelta", 0.0)
            })
        }
        return arr
    }

    private fun buildMemory(context: Context): JSONObject {
        val store = ExperienceStore.getInstance(context)
        return JSONObject().apply {
            put("embeddingCount", try { store.count() } catch (_: Exception) { 0 })
            put("dbSizeKb",       0)
            put("edgeCaseCount",  try { store.edgeCaseCount() } catch (_: Exception) { 0 })
            put("miniLmReady",    false)
        }
    }

    private fun buildActivity(context: Context): JSONArray {
        val arr = JSONArray()
        try {
            ExperienceStore.getInstance(context).getRecent(50).forEach { exp ->
                arr.put(JSONObject().apply {
                    put("id",          exp.id)
                    put("timestamp",   exp.timestamp)
                    put("type",        exp.taskType)
                    put("description", exp.screenSummary.take(100))
                    put("app",         exp.appPackage)
                    put("success",     exp.result == "success")
                    put("rewardSignal",exp.reward)
                })
            }
        } catch (_: Exception) {}
        return arr
    }

    private fun buildModules(context: Context): JSONObject {
        val store = ExperienceStore.getInstance(context)
        val loraVer = try { LoraTrainer.currentVersion(context) } catch (_: Exception) { 0 }
        val episodes = try {
            store.countByResult("success") + store.countByResult("failure")
        } catch (_: Exception) { 0 }
        return JSONObject().apply {
            put("llm", JSONObject().apply {
                put("loaded",          true)
                put("modelName",       "Llama-3.2-1B-Instruct")
                put("quantization",    "Q4_K_M")
                put("contextLength",   4096)
                put("tokensPerSecond", 0)
                put("memoryMb",        0)
            })
            put("ocr", JSONObject().apply {
                put("ready",  true)
                put("engine", "ML Kit Text Recognition v2")
            })
            put("rl", JSONObject().apply {
                put("ready",            PolicyNetwork.isReady())
                put("episodesRun",      episodes)
                put("loraVersion",      loraVer)
                put("adapterLoaded",    loraVer > 0)
                put("untrainedSamples", 0)
                put("adamStep",         PolicyNetwork.adamStepCount)
                put("lastPolicyLoss",   PolicyNetwork.lastPolicyLoss)
            })
            put("memory", JSONObject().apply {
                put("ready",          true)
                put("embeddingCount", try { store.count() } catch (_: Exception) { 0 })
                put("dbSizeKb",       0)
            })
            put("accessibility", JSONObject().apply {
                put("granted", false)
                put("active",  false)
            })
            put("screenCapture", JSONObject().apply {
                put("granted", false)
                put("active",  false)
            })
        }
    }
}
