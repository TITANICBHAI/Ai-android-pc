package com.ariaagent.mobile.core.system

import android.app.Activity
import android.util.Log
import java.lang.ref.WeakReference

/**
 * SustainedPerformanceManager — Android Sustained Performance Mode integration.
 *
 * Problem: The Exynos 9611 (Samsung M31) thermal design is aggressive. During sustained
 * LLM inference (~10–15 tok/s, CPU-heavy), the SoC's DVFS governor ramps clocks to
 * maximum, overheats, then throttles to minimum — creating a "cycle of death":
 *
 *   Max speed → overheat → thermal throttle → crawl → cool down → ramp again
 *
 * This produces highly variable tokens/sec and can drop mid-inference performance below
 * the 4–5 tok/s threshold where the agent becomes too slow to be useful.
 *
 * Solution: Window.setSustainedPerformanceMode(true) (API 24+)
 *
 * This signals the Android OS to keep CPU/GPU at a lower, STABLE clock frequency.
 * The SoC runs slightly cooler, avoiding the ramp-throttle cycle. The result:
 *   - Slightly lower peak throughput (~8 tok/s vs ~15 tok/s burst)
 *   - Consistent throughput — no mid-inference throttle dips
 *   - Longer sessions before ThermalGuard triggers emergency pause
 *
 * This is especially important for:
 *   - Long reasoning turns (>30 tokens output)
 *   - IRL video processing (sustained OCR + LLM over many frames)
 *   - LoRA training cycles during charging
 *
 * Usage:
 *   1. Call SustainedPerformanceManager.register(activity) from MainActivity.onCreate()
 *   2. Call enable() before starting inference in AgentLoop / LearningScheduler
 *   3. Call disable() after inference completes or when agent stops
 *
 * Design note: Uses WeakReference<Activity> so the manager never prevents GC of the
 * Activity. If the Activity is recreated (rotation etc.), re-register.
 *
 * Phase: 14.3 (Advanced Architecture — Thermal Control)
 */
object SustainedPerformanceManager {

    private const val TAG = "SustainedPerfMgr"

    @Volatile
    private var activityRef: WeakReference<Activity>? = null

    @Volatile
    var isEnabled: Boolean = false
        private set

    /**
     * Register the current Activity so sustained performance can be toggled.
     * Call from MainActivity.onCreate() or ComposeMainActivity.onCreate().
     *
     * Safe to call multiple times — old reference is replaced.
     */
    fun register(activity: Activity) {
        activityRef = WeakReference(activity)
        Log.d(TAG, "Activity registered: ${activity.javaClass.simpleName}")
    }

    /**
     * Enable Sustained Performance Mode.
     *
     * Signals the OS to maintain stable, lower CPU/GPU clocks.
     * Must be called on the main thread (Window operation).
     *
     * Does nothing if:
     *   - No Activity is registered
     *   - The Activity has been GC'd (WeakReference expired)
     *   - Already enabled
     */
    fun enable() {
        if (isEnabled) return
        val activity = activityRef?.get()
        if (activity == null) {
            Log.w(TAG, "enable() — no registered Activity, cannot set sustained performance mode")
            return
        }
        activity.runOnUiThread {
            try {
                activity.window.setSustainedPerformanceMode(true)
                isEnabled = true
                Log.i(TAG, "Sustained performance mode ENABLED — clocks stabilised for inference")
            } catch (e: Exception) {
                Log.w(TAG, "setSustainedPerformanceMode(true) failed: ${e.message}")
            }
        }
    }

    /**
     * Disable Sustained Performance Mode.
     *
     * Restores normal DVFS governor behaviour — OS can ramp clocks freely again.
     * Call when the agent stops, pauses, or goes idle.
     */
    fun disable() {
        if (!isEnabled) return
        val activity = activityRef?.get()
        if (activity == null) {
            isEnabled = false  // Activity gone — mode already gone with it
            return
        }
        activity.runOnUiThread {
            try {
                activity.window.setSustainedPerformanceMode(false)
                isEnabled = false
                Log.i(TAG, "Sustained performance mode DISABLED — normal DVFS restored")
            } catch (e: Exception) {
                Log.w(TAG, "setSustainedPerformanceMode(false) failed: ${e.message}")
            }
        }
    }

    /**
     * Unregister the Activity reference.
     * Call from MainActivity.onDestroy() to release the WeakReference early.
     */
    fun unregister() {
        disable()
        activityRef = null
        Log.d(TAG, "Activity unregistered")
    }
}
