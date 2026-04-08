package com.ariaagent.mobile.core.monitoring

import org.json.JSONArray
import org.json.JSONObject

/**
 * LocalSnapshotStore — Thread-safe in-memory snapshot of the latest agent state.
 *
 * Updated by [MonitoringPusher] whenever significant agent events fire.
 * Served by [LocalDeviceServer] via HTTP to the web dashboard on the same LAN.
 *
 * All fields are @Volatile so any thread sees the latest write immediately.
 * JSONObject/JSONArray are treated as immutable once written — no lock needed.
 *
 * Phase: 16 (Local Monitoring)
 */
object LocalSnapshotStore {

    @Volatile var status:   JSONObject = defaultStatus()
    @Volatile var thermal:  JSONObject = defaultThermal()
    @Volatile var rl:       JSONObject = defaultRL()
    @Volatile var lora:     JSONArray  = JSONArray()
    @Volatile var memory:   JSONObject = defaultMemory()
    @Volatile var activity: JSONArray  = JSONArray()
    @Volatile var modules:  JSONObject = defaultModules()

    // ─── Defaults ─────────────────────────────────────────────────────────────

    private fun defaultStatus() = JSONObject().apply {
        put("status",              "idle")
        put("currentTask",         JSONObject.NULL)
        put("currentApp",          JSONObject.NULL)
        put("tokenRate",           0)
        put("memoryUsedMb",        0)
        put("sessionStartedAt",    JSONObject.NULL)
        put("actionsPerformed",    0)
        put("successRate",         0)
        put("modelReady",          false)
        put("llmLoaded",           false)
        put("accessibilityActive", false)
        put("screenCaptureActive", false)
        put("gameMode",            "none")
    }

    private fun defaultThermal() = JSONObject().apply {
        put("level",           "safe")
        put("inferenceSafe",   true)
        put("trainingSafe",    true)
        put("throttleCapture", false)
        put("emergency",       false)
    }

    private fun defaultRL() = JSONObject().apply {
        put("episodesRun",       0)
        put("loraVersion",       0)
        put("adapterLoaded",     false)
        put("untrainedSamples",  0)
        put("adamStep",          0)
        put("lastPolicyLoss",    0.0)
        put("rewardHistory",     JSONArray())
        put("policyLossHistory", JSONArray())
    }

    private fun defaultMemory() = JSONObject().apply {
        put("embeddingCount", 0)
        put("dbSizeKb",       0)
        put("edgeCaseCount",  0)
        put("miniLmReady",    false)
    }

    private fun defaultModules() = JSONObject().apply {
        put("llm", JSONObject().apply {
            put("loaded",          false)
            put("modelName",       "Llama-3.2-1B-Instruct")
            put("quantization",    "Q4_K_M")
            put("contextLength",   4096)
            put("tokensPerSecond", 0)
            put("memoryMb",        0)
        })
        put("ocr", JSONObject().apply {
            put("ready",  false)
            put("engine", "ML Kit Text Recognition v2")
        })
        put("rl", JSONObject().apply {
            put("ready",            false)
            put("episodesRun",      0)
            put("loraVersion",      0)
            put("adapterLoaded",    false)
            put("untrainedSamples", 0)
            put("adamStep",         0)
            put("lastPolicyLoss",   0.0)
        })
        put("memory", JSONObject().apply {
            put("ready",          false)
            put("embeddingCount", 0)
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
