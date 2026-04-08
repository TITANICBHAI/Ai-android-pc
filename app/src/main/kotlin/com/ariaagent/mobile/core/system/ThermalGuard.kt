package com.ariaagent.mobile.core.system

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.util.Log
import androidx.annotation.RequiresApi
import com.ariaagent.mobile.core.events.AgentEventBus

/**
 * ThermalGuard — protects the Exynos 9611 from thermal throttling and damage.
 *
 * The M31 has NO dedicated NPU. All AI runs on the 4× Cortex-A73 big cores.
 * Sustained inference generates significant heat. Without throttling:
 *   - The OS forces clock reduction → tok/s drops from 12 to ~4 (system throttle)
 *   - Battery temperature climbs → accelerated cell degradation
 *   - Extreme cases: ANR or force-close due to system resource pressure
 *
 * Thermal status levels (ThermalManager API 29+):
 *   THERMAL_STATUS_NONE (0)       → safe, full speed
 *   THERMAL_STATUS_LIGHT (1)      → slight warmth — throttle screen capture
 *   THERMAL_STATUS_MODERATE (2)   → warm — pause RL training, reduce capture FPS
 *   THERMAL_STATUS_SEVERE (3)     → hot — pause all inference, notify JS
 *   THERMAL_STATUS_CRITICAL (4)   → very hot — emergency stop everything
 *   THERMAL_STATUS_EMERGENCY (5)  → Android shuts down anyway
 *
 * Fallback for API < 29: battery temperature from ACTION_BATTERY_CHANGED.
 * Rule: < 37°C = safe, 37–40°C = warm, > 40°C = hot
 *
 * Phase: 8 (Optimization & Thermal Management)
 */
object ThermalGuard {

    private const val TAG = "ThermalGuard"

    enum class ThermalLevel {
        SAFE,       // full speed — all operations permitted
        LIGHT,      // screen capture throttled to 0.5 FPS
        MODERATE,   // RL training paused, capture at 0.5 FPS
        SEVERE,     // inference paused, user notified
        CRITICAL    // everything stopped, emergency
    }

    @Volatile
    var currentLevel: ThermalLevel = ThermalLevel.SAFE
        private set

    private var listener: ThermalListener? = null

    /**
     * Stored reference to the OnThermalStatusChangedListener passed to addThermalStatusListener.
     * Must be kept so removeThermalStatusListener() can remove the exact same instance.
     * PowerManager.clearThermalStatusListeners() does NOT exist — removal is by reference only.
     */
    @Volatile
    private var thermalConsumer: PowerManager.OnThermalStatusChangedListener? = null

    interface ThermalListener {
        fun onThermalLevelChanged(level: ThermalLevel)
    }

    // ─── Register ─────────────────────────────────────────────────────────────

    fun register(context: Context, listener: ThermalListener) {
        this.listener = listener
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            registerThermalManager(context, listener)
        } else {
            Log.i(TAG, "API < 29 — using battery temp polling fallback")
        }
    }

    fun unregister(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            unregisterThermalManager(context)
        }
        listener = null
    }

    // ─── ThermalManager (API 29+) ──────────────────────────────────────────────

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun registerThermalManager(context: Context, @Suppress("UNUSED_PARAMETER") listener: ThermalListener) {
        try {
            val thermalManager = context.getSystemService(PowerManager::class.java)
            val thermalListener = PowerManager.OnThermalStatusChangedListener { status ->
                val level = thermalStatusToLevel(status)
                updateLevel(level)
            }
            thermalConsumer = thermalListener
            thermalManager?.addThermalStatusListener(context.mainExecutor, thermalListener)
            Log.i(TAG, "ThermalManager listener registered (API 29+)")
        } catch (e: Exception) {
            Log.w(TAG, "ThermalManager not available: ${e.message}")
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun unregisterThermalManager(context: Context) {
        try {
            val thermalManager = context.getSystemService(PowerManager::class.java)
            thermalConsumer?.let { consumer ->
                thermalManager?.removeThermalStatusListener(consumer)
            }
            thermalConsumer = null
        } catch (e: Exception) {
            Log.w(TAG, "Failed to unregister ThermalManager: ${e.message}")
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun thermalStatusToLevel(status: Int): ThermalLevel = when (status) {
        PowerManager.THERMAL_STATUS_NONE      -> ThermalLevel.SAFE
        PowerManager.THERMAL_STATUS_LIGHT     -> ThermalLevel.LIGHT
        PowerManager.THERMAL_STATUS_MODERATE  -> ThermalLevel.MODERATE
        PowerManager.THERMAL_STATUS_SEVERE    -> ThermalLevel.SEVERE
        PowerManager.THERMAL_STATUS_CRITICAL  -> ThermalLevel.CRITICAL
        PowerManager.THERMAL_STATUS_EMERGENCY -> ThermalLevel.CRITICAL
        else -> ThermalLevel.SAFE
    }

    // ─── Fallback: battery temperature poll (API < 29) ────────────────────────

    /**
     * Call this from LearningScheduler.isThermalSafe() on API < 29.
     * temp is in tenths of degrees Celsius (from BatteryManager.EXTRA_TEMPERATURE).
     */
    fun updateFromBatteryTemp(tempTenths: Int) {
        val level = when {
            tempTenths >= 420 -> ThermalLevel.CRITICAL   // ≥ 42°C
            tempTenths >= 400 -> ThermalLevel.SEVERE     // ≥ 40°C
            tempTenths >= 370 -> ThermalLevel.MODERATE   // ≥ 37°C
            tempTenths >= 350 -> ThermalLevel.LIGHT      // ≥ 35°C
            else              -> ThermalLevel.SAFE
        }
        updateLevel(level)
    }

    // ─── State queries ────────────────────────────────────────────────────────

    /** True if inference can safely proceed. */
    fun isInferenceSafe(): Boolean = currentLevel <= ThermalLevel.MODERATE

    /** True if RL / LoRA training can safely run. */
    fun isTrainingSafe(): Boolean = currentLevel <= ThermalLevel.LIGHT

    /** True if screen capture should throttle to 0.5 FPS instead of 1-2 FPS. */
    fun shouldThrottleCapture(): Boolean = currentLevel >= ThermalLevel.LIGHT

    /**
     * True when the device is warm but not yet at emergency level.
     * At MODERATE: inference is still permitted but the agent loop should
     * extend its step delay (1600ms instead of 800ms) and skip optional
     * heavy-weight VLM calls (VisionEngine.describe) to reduce heat generation.
     * RL training should already be paused via [isTrainingSafe].
     */
    fun isModerate(): Boolean = currentLevel == ThermalLevel.MODERATE

    /** True if all AI operations must stop immediately. */
    fun isEmergency(): Boolean = currentLevel >= ThermalLevel.SEVERE

    // ─── Internal ─────────────────────────────────────────────────────────────

    private fun updateLevel(newLevel: ThermalLevel) {
        if (newLevel == currentLevel) return
        val prev = currentLevel
        currentLevel = newLevel
        Log.i(TAG, "Thermal level: $prev → $newLevel")
        listener?.onThermalLevelChanged(newLevel)
        AgentEventBus.emit("thermal_status_changed", mapOf(
            "level"         to newLevel.name.lowercase(),
            "inferenceSafe" to isInferenceSafe(),
            "trainingSafe"  to isTrainingSafe(),
            "emergency"     to isEmergency(),
        ))
    }
}
