package com.ariaagent.mobile.system.actions

import android.accessibilityservice.AccessibilityService.GestureResultCallback
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.graphics.Rect
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo
import com.ariaagent.mobile.system.accessibility.AgentAccessibilityService
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import org.json.JSONObject
import kotlin.coroutines.resume

/**
 * GestureEngine — executes LLM action decisions as physical gestures.
 *
 * The LLM outputs JSON: {"tool":"Tap","node_id":"#3","reason":"..."}
 * GestureEngine resolves node_id → screen coordinates → gesture dispatch.
 *
 * ── Supported tools ──────────────────────────────────────────────────────────
 *
 *  Node-ID actions (accessibility tree available):
 *    Tap / Click          {"tool":"Tap","node_id":"#3"}
 *    DoubleTap            {"tool":"DoubleTap","node_id":"#3"}
 *    LongPress            {"tool":"LongPress","node_id":"#3"}
 *    Swipe                {"tool":"Swipe","node_id":"#7","direction":"up"}
 *    Scroll               {"tool":"Scroll","node_id":"#5","direction":"down"}
 *    Type                 {"tool":"Type","node_id":"#2","text":"hello"}
 *    Copy                 {"tool":"Copy","node_id":"#4"}
 *    Paste                {"tool":"Paste","node_id":"#4"}
 *
 *  XY coordinate actions (game / Flutter / Unity — no accessibility tree):
 *    TapXY                {"tool":"TapXY","x":0.5,"y":0.6}
 *    DoubleTapXY          {"tool":"DoubleTapXY","x":0.5,"y":0.6}
 *    SwipeXY              {"tool":"SwipeXY","x1":0.5,"y1":0.8,"x2":0.5,"y2":0.2}
 *    PinchZoom            {"tool":"PinchZoom","x":0.5,"y":0.5,"scale":2.0}
 *                           scale > 1.0 = zoom in, scale < 1.0 = zoom out
 *
 *  System actions:
 *    Back                 {"tool":"Back"}
 *    Home                 {"tool":"Home"}
 *    Notifications        {"tool":"Notifications"}
 *    HideKeyboard         {"tool":"HideKeyboard"}
 *    OpenApp              {"tool":"OpenApp","package":"com.google.android.youtube"}
 *
 *  Agent control (handled by AgentLoop, not here):
 *    Wait                 {"tool":"Wait","duration_ms":1000}
 *    Done                 {"tool":"Done","reason":"task complete"}
 *
 * ── Coordinate modes ─────────────────────────────────────────────────────────
 *
 *  1. Node-ID mode: GestureEngine resolves the node's bounding box from the
 *     accessibility tree → computes center → dispatches gesture.
 *  2. XY mode: Coordinates are normalised [0.0–1.0] relative to screen size.
 *     AgentAccessibilityService.getScreenSize() converts to real pixels.
 *
 * Phase: 3 (Action Layer) — expanded in multi-model pipeline update.
 */
object GestureEngine {

    private const val TAG = "GestureEngine"

    /**
     * Parse and execute an action from the LLM's JSON output.
     * @return true if the action was dispatched successfully.
     */
    suspend fun executeFromJson(actionJson: String): Boolean {
        return try {
            val json      = JSONObject(actionJson)
            val tool      = json.optString("tool", "")
            val nodeId    = json.optString("node_id", "")
            val direction = json.optString("direction", "")
            val text      = json.optString("text", "")

            when (tool.lowercase()) {

                // ── Node-ID actions ───────────────────────────────────────────
                "click", "tap"            -> tap(nodeId)
                "doubletap", "double_tap" -> doubleTap(nodeId)
                "longpress", "long_press" -> longPress(nodeId)
                "swipe"                   -> swipe(nodeId, direction)
                "scroll"                  -> scroll(nodeId, direction)
                "type", "typetext"        -> type(nodeId, text)
                "copy"                    -> copy(nodeId)
                "paste"                   -> paste(nodeId)

                // ── XY coordinate actions ─────────────────────────────────────
                "tapxy" -> {
                    val x = json.optDouble("x", 0.5).toFloat().coerceIn(0f, 1f)
                    val y = json.optDouble("y", 0.5).toFloat().coerceIn(0f, 1f)
                    tapXY(x, y)
                }
                "doubletapxy", "double_tap_xy" -> {
                    val x = json.optDouble("x", 0.5).toFloat().coerceIn(0f, 1f)
                    val y = json.optDouble("y", 0.5).toFloat().coerceIn(0f, 1f)
                    doubleTapXY(x, y)
                }
                "swipexy" -> {
                    val x1 = json.optDouble("x1", 0.3).toFloat().coerceIn(0f, 1f)
                    val y1 = json.optDouble("y1", 0.8).toFloat().coerceIn(0f, 1f)
                    val x2 = json.optDouble("x2", 0.3).toFloat().coerceIn(0f, 1f)
                    val y2 = json.optDouble("y2", 0.2).toFloat().coerceIn(0f, 1f)
                    swipeXY(x1, y1, x2, y2)
                }
                "pinchzoom", "pinch_zoom", "pinch" -> {
                    val x     = json.optDouble("x", 0.5).toFloat().coerceIn(0f, 1f)
                    val y     = json.optDouble("y", 0.5).toFloat().coerceIn(0f, 1f)
                    val scale = json.optDouble("scale", 2.0).toFloat().coerceIn(0.1f, 10f)
                    pinchZoom(x, y, scale)
                }

                // ── System actions ────────────────────────────────────────────
                "back" -> {
                    AgentAccessibilityService.performBack()
                    true
                }
                "home" -> {
                    AgentAccessibilityService.performHome()
                    true
                }
                "notifications" -> {
                    AgentAccessibilityService.performNotifications()
                    true
                }
                "hidekeyboard", "hide_keyboard", "dismisskeyboard", "dismiss_keyboard" -> {
                    hideKeyboard()
                }
                "openapp", "open_app", "launchapp", "launch_app" -> {
                    val pkg = json.optString("package", "").ifBlank {
                        json.optString("packageName", "")
                    }
                    openApp(pkg)
                }

                else -> {
                    Log.w(TAG, "executeFromJson: unrecognised tool='$tool' — json=$actionJson")
                    false
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "executeFromJson: JSON parse/dispatch failed — ${e.message} | json=$actionJson")
            false
        }
    }

    // ── Node-ID based actions ─────────────────────────────────────────────────

    /** Tap the center of [nodeId]. */
    suspend fun tap(nodeId: String): Boolean {
        val (cx, cy) = centerOf(nodeId) ?: return false
        return dispatchTapAt(cx, cy)
    }

    /**
     * Double-tap [nodeId] — two taps 80 ms apart.
     * Required by some apps (e.g. like buttons, map zoom in, photo open).
     */
    suspend fun doubleTap(nodeId: String): Boolean {
        val (cx, cy) = centerOf(nodeId) ?: return false
        val first = dispatchTapAt(cx, cy)
        delay(80L)
        val second = dispatchTapAt(cx, cy)
        return first || second
    }

    /** Long-press [nodeId] to trigger context menus / drag handles. */
    suspend fun longPress(nodeId: String): Boolean {
        val (cx, cy) = centerOf(nodeId) ?: return false
        return suspendCancellableCoroutine { cont ->
            AgentAccessibilityService.dispatchLongPress(cx, cy, object : GestureResultCallback() {
                override fun onCompleted(g: android.accessibilityservice.GestureDescription) { cont.resume(true) }
                override fun onCancelled(g: android.accessibilityservice.GestureDescription) {
                    Log.w(TAG, "longPress cancelled by OS for node $nodeId")
                    cont.resume(false)
                }
            })
        }
    }

    /** Swipe within [nodeId] in the given [direction] (up/down/left/right). */
    suspend fun swipe(nodeId: String, direction: String): Boolean {
        val node = AgentAccessibilityService.getNodeById(nodeId) ?: return false
        val rect = Rect().also { node.getBoundsInScreen(it) }
        val (x1, y1, x2, y2) = when (direction.lowercase()) {
            "up"    -> floatArrayOf(rect.centerX().f, rect.bottom.f * 0.8f, rect.centerX().f, rect.top.f * 1.2f)
            "down"  -> floatArrayOf(rect.centerX().f, rect.top.f    * 1.2f, rect.centerX().f, rect.bottom.f * 0.8f)
            "left"  -> floatArrayOf(rect.right.f * 0.8f, rect.centerY().f, rect.left.f  * 1.2f, rect.centerY().f)
            "right" -> floatArrayOf(rect.left.f  * 1.2f, rect.centerY().f, rect.right.f * 0.8f, rect.centerY().f)
            else    -> return false
        }
        return suspendCancellableCoroutine { cont ->
            AgentAccessibilityService.dispatchSwipe(x1, y1, x2, y2, object : GestureResultCallback() {
                override fun onCompleted(g: android.accessibilityservice.GestureDescription) { cont.resume(true) }
                override fun onCancelled(g: android.accessibilityservice.GestureDescription) {
                    Log.w(TAG, "swipe cancelled: node=$nodeId dir=$direction")
                    cont.resume(false)
                }
            })
        }
    }

    /** Scroll [nodeId] using accessibility actions (no gesture path needed). */
    fun scroll(nodeId: String, direction: String): Boolean {
        val node = AgentAccessibilityService.getNodeById(nodeId) ?: return false
        return when (direction.lowercase()) {
            "up"   -> node.performAction(AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD)
            "down" -> node.performAction(AccessibilityNodeInfo.ACTION_SCROLL_FORWARD)
            else   -> false
        }
    }

    /** Type [text] into an editable [nodeId] using ACTION_SET_TEXT. */
    fun type(nodeId: String, text: String): Boolean {
        val node = AgentAccessibilityService.getNodeById(nodeId) ?: return false
        if (!node.isEditable) return false
        val args = android.os.Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        }
        return node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)
    }

    /**
     * Copy the selected or full text of [nodeId] to the Android clipboard.
     * The agent can then paste it elsewhere with [paste].
     */
    fun copy(nodeId: String): Boolean {
        val node = AgentAccessibilityService.getNodeById(nodeId) ?: return false
        // Select all first so there is something to copy
        node.performAction(AccessibilityNodeInfo.ACTION_SELECT)
        return node.performAction(AccessibilityNodeInfo.ACTION_COPY)
    }

    /**
     * Paste clipboard content into the editable field [nodeId].
     * Works with text copied by [copy] or any system clipboard content.
     */
    fun paste(nodeId: String): Boolean {
        val node = AgentAccessibilityService.getNodeById(nodeId) ?: return false
        if (!node.isEditable) return false
        return node.performAction(AccessibilityNodeInfo.ACTION_PASTE)
    }

    // ── XY coordinate actions (vision-only / SAM fallback) ────────────────────

    /** Tap at normalised screen coordinates [0.0–1.0]. */
    suspend fun tapXY(normX: Float, normY: Float): Boolean {
        val (w, h) = AgentAccessibilityService.getScreenSize()
        return dispatchTapAt(normX * w, normY * h)
    }

    /**
     * Double-tap at normalised screen coordinates — two taps 80 ms apart.
     * Useful for zooming into maps or photos in vision-only screens.
     */
    suspend fun doubleTapXY(normX: Float, normY: Float): Boolean {
        val (w, h) = AgentAccessibilityService.getScreenSize()
        val px = normX * w
        val py = normY * h
        val first = dispatchTapAt(px, py)
        delay(80L)
        val second = dispatchTapAt(px, py)
        return first || second
    }

    /** Swipe between two normalised screen coordinates [0.0–1.0]. */
    suspend fun swipeXY(normX1: Float, normY1: Float, normX2: Float, normY2: Float): Boolean {
        val (w, h) = AgentAccessibilityService.getScreenSize()
        return suspendCancellableCoroutine { cont ->
            AgentAccessibilityService.dispatchSwipe(
                normX1 * w, normY1 * h, normX2 * w, normY2 * h,
                object : GestureResultCallback() {
                    override fun onCompleted(g: android.accessibilityservice.GestureDescription) { cont.resume(true) }
                    override fun onCancelled(g: android.accessibilityservice.GestureDescription) {
                        Log.w(TAG, "swipeXY cancelled (%.2f,%.2f)→(%.2f,%.2f)".format(normX1, normY1, normX2, normY2))
                        cont.resume(false)
                    }
                })
        }
    }

    /**
     * Two-finger pinch-zoom centred at ([normX], [normY]).
     *
     * [scale] controls the direction and magnitude:
     *   scale > 1.0  — zoom IN  (fingers start apart, move toward centre)
     *   scale < 1.0  — zoom OUT (fingers start close, move away from centre)
     *   scale == 2.0 — good default for maps / photos zoom in
     *   scale == 0.5 — good default for zoom out
     *
     * The gesture uses two simultaneous GestureDescription strokes dispatched
     * through the accessibility service, which the OS interprets as a real
     * two-finger pinch exactly like a human would perform.
     */
    suspend fun pinchZoom(normX: Float, normY: Float, scale: Float): Boolean {
        val (w, h) = AgentAccessibilityService.getScreenSize()
        val cx = normX * w
        val cy = normY * h

        // Spread: distance from centre to each finger tip (15% of screen width)
        val spread = w * 0.15f

        // For zoom in:  fingers move FROM far → TO close
        // For zoom out: fingers move FROM close → TO far
        val zoomIn      = scale >= 1.0f
        val startOffset = if (zoomIn) spread else spread * 0.25f
        val endOffset   = if (zoomIn) spread * 0.25f else spread

        val path1 = Path().apply { moveTo(cx - startOffset, cy); lineTo(cx - endOffset, cy) }
        val path2 = Path().apply { moveTo(cx + startOffset, cy); lineTo(cx + endOffset, cy) }

        val duration = 400L
        val gesture = GestureDescription.Builder()
            .addStroke(GestureDescription.StrokeDescription(path1, 0L, duration))
            .addStroke(GestureDescription.StrokeDescription(path2, 0L, duration))
            .build()

        return suspendCancellableCoroutine { cont ->
            AgentAccessibilityService.dispatchGesture(gesture, object : GestureResultCallback() {
                override fun onCompleted(g: android.accessibilityservice.GestureDescription) { cont.resume(true) }
                override fun onCancelled(g: android.accessibilityservice.GestureDescription) {
                    Log.w(TAG, "pinchZoom cancelled at norm(%.2f,%.2f) scale=%.2f".format(normX, normY, scale))
                    cont.resume(false)
                }
            })
        }
    }

    // ── System actions ────────────────────────────────────────────────────────

    /**
     * Dismiss the soft keyboard if it is currently visible.
     *
     * Sends a Back event which Android intercepts to close the IME before any
     * navigation occurs — so the current screen and focus are preserved.
     * Returns false immediately if the keyboard is already hidden.
     *
     * Example: {"tool":"HideKeyboard"}
     */
    fun hideKeyboard(): Boolean = AgentAccessibilityService.performHideKeyboard()

    /**
     * Launch an app directly by its [packageName] without navigating via the launcher.
     * Uses the package manager's launch intent — identical to tapping the app icon,
     * but instant regardless of where the launcher is.
     *
     * Example: {"tool":"OpenApp","package":"com.google.android.youtube"}
     */
    fun openApp(packageName: String): Boolean {
        if (packageName.isBlank()) {
            Log.w(TAG, "openApp: empty package name")
            return false
        }
        val ctx = AgentAccessibilityService.getContext() ?: run {
            Log.w(TAG, "openApp: accessibility service not active")
            return false
        }
        val intent = ctx.packageManager.getLaunchIntentForPackage(packageName) ?: run {
            Log.w(TAG, "openApp: no launch intent for package '$packageName' — app not installed?")
            return false
        }
        intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
        ctx.startActivity(intent)
        Log.i(TAG, "openApp: launched '$packageName'")
        return true
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * Resolve the screen-centre coordinates of [nodeId].
     * Returns null if the node cannot be found in the accessibility registry.
     */
    private fun centerOf(nodeId: String): Pair<Float, Float>? {
        val node = AgentAccessibilityService.getNodeById(nodeId) ?: return null
        val rect = Rect().also { node.getBoundsInScreen(it) }
        return Pair(rect.centerX().f, rect.centerY().f)
    }

    /**
     * Dispatch a single tap at absolute pixel coordinates with automatic retry.
     *
     * The accessibility gesture dispatcher can cancel taps during heavy load,
     * fast animations, or when the target surface briefly loses focus.
     * Up to [MAX_TAP_RETRIES] retries are attempted, each with a small random
     * coordinate jitter (±4 px) so repeat taps don't land in exactly the same
     * rejected location (helps with touch-slop edge cases on some OEM ROMs).
     *
     * On M31 (Exynos 9611) cancellations are rare (~1 in 40 taps under load);
     * retries add at most 2 × 60 ms ≈ 120 ms extra latency per affected step.
     */
    private suspend fun dispatchTapAt(px: Float, py: Float): Boolean {
        repeat(MAX_TAP_RETRIES + 1) { attempt ->
            val jitter = if (attempt == 0) 0f else ((-4..4).random().toFloat())
            val result = suspendCancellableCoroutine<Boolean> { cont ->
                AgentAccessibilityService.dispatchTap(
                    px + jitter, py + jitter,
                    object : GestureResultCallback() {
                        override fun onCompleted(g: android.accessibilityservice.GestureDescription) {
                            cont.resume(true)
                        }
                        override fun onCancelled(g: android.accessibilityservice.GestureDescription) {
                            Log.w(TAG, "tapAt attempt ${attempt + 1} cancelled by OS at " +
                                "(%.1f, %.1f)".format(px + jitter, py + jitter))
                            cont.resume(false)
                        }
                    }
                )
            }
            if (result) return true
            if (attempt < MAX_TAP_RETRIES) delay(TAP_RETRY_DELAY_MS)
        }
        Log.e(TAG, "tapAt failed after ${MAX_TAP_RETRIES + 1} attempts at (%.1f, %.1f)".format(px, py))
        return false
    }

    private const val MAX_TAP_RETRIES   = 2
    private const val TAP_RETRY_DELAY_MS = 60L

    // Extension to reduce Int.toFloat() noise throughout
    private val Int.f: Float get() = toFloat()

    private operator fun FloatArray.component1() = this[0]
    private operator fun FloatArray.component2() = this[1]
    private operator fun FloatArray.component3() = this[2]
    private operator fun FloatArray.component4() = this[3]
}
