package com.ariaagent.mobile.core.config

import android.content.Context
import android.util.Log
import com.ariaagent.mobile.ui.viewmodel.SafetyConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * SafetyConfigStore — JSON-file-backed persistence for SafetyConfig.
 *
 * File: filesDir/aria_safety_config.json
 * Thread-safe via @Synchronized (reads and writes are fast file I/O, not DB).
 *
 * Schema:
 * {
 *   "globalKillActive": false,
 *   "confirmMode": false,
 *   "blockedPackages": ["com.example.bank"],
 *   "allowlistMode": false,
 *   "allowedPackages": []
 * }
 */
object SafetyConfigStore {

    private const val TAG  = "SafetyConfigStore"
    private const val FILE = "aria_safety_config.json"

    @Synchronized
    fun load(context: Context): SafetyConfig {
        return runCatching {
            val file = File(context.filesDir, FILE)
            if (!file.exists()) return SafetyConfig()
            val obj = JSONObject(file.readText())
            SafetyConfig(
                globalKillActive = obj.optBoolean("globalKillActive", false),
                confirmMode      = obj.optBoolean("confirmMode",      false),
                blockedPackages  = obj.optJSONArray("blockedPackages").toStringSet(),
                allowlistMode    = obj.optBoolean("allowlistMode",    false),
                allowedPackages  = obj.optJSONArray("allowedPackages").toStringSet(),
            )
        }.getOrElse {
            Log.w(TAG, "Failed to load safety config: ${it.message}")
            SafetyConfig()
        }
    }

    @Synchronized
    fun save(context: Context, config: SafetyConfig) {
        runCatching {
            val obj = JSONObject().apply {
                put("globalKillActive", config.globalKillActive)
                put("confirmMode",      config.confirmMode)
                put("blockedPackages",  config.blockedPackages.toJSONArray())
                put("allowlistMode",    config.allowlistMode)
                put("allowedPackages",  config.allowedPackages.toJSONArray())
            }
            File(context.filesDir, FILE).writeText(obj.toString(2))
        }.onFailure {
            Log.e(TAG, "Failed to save safety config: ${it.message}")
        }
    }

    private fun JSONArray?.toStringSet(): Set<String> {
        if (this == null) return emptySet()
        return (0 until length()).mapNotNull { optString(it).takeIf { s -> s.isNotBlank() } }.toSet()
    }

    private fun Set<String>.toJSONArray(): JSONArray {
        val arr = JSONArray()
        forEach { arr.put(it) }
        return arr
    }
}
