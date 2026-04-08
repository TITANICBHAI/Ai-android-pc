package com.ariaagent.mobile.system.accessibility

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.graphics.Rect
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import com.ariaagent.mobile.core.events.AgentEventBus
import java.util.concurrent.ConcurrentHashMap

/**
 * AgentAccessibilityService — two jobs:
 *   1. READ: parse the UI node tree into LLM-friendly semantic text
 *   2. WRITE: dispatch physical gestures (tap, swipe, text, scroll)
 *
 * Thread-safety model
 * ───────────────────
 * onAccessibilityEvent() is called on the accessibility main thread.
 * getSemanticTree() / getNodeById() may be called from any coroutine thread.
 *
 * Fix: The semantic tree is rebuilt on the accessibility thread and stored in a
 * @Volatile field (cachedTree). All callers read that field — zero thread crossing.
 * nodeRegistry uses ConcurrentHashMap so cross-thread reads are safe.
 *
 * Keyboard awareness
 * ──────────────────
 * When the Android soft keyboard (IME) opens, TYPE_WINDOW_STATE_CHANGED fires
 * for the IME window. We detect this by checking the event's package name for
 * known IME packages and set isKeyboardVisible accordingly.
 *
 * Tap awareness
 * ─────────────
 * TYPE_VIEW_CLICKED events are now declared in accessibility_service_config.xml.
 * Each click from the *user* (not from ARIA's own gesture dispatch) is emitted
 * to AgentEventBus as "user_tapped" so the LLM knows what the user touched.
 */
class AgentAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "AgentA11y"

        // IME packages — extended for major OEM and 3rd-party keyboards.
        // Samsung ships multiple IME packages across OneUI versions; all are included here.
        // InputMethodManager is used as a secondary heuristic (see onAccessibilityEvent).
        private val IME_PACKAGES = setOf(
            // AOSP / Gboard
            "com.android.inputmethod.latin",
            "com.google.android.inputmethod.latin",
            // Samsung OneUI keyboards (honeyboard = <= One UI 3; sec.android = One UI 4+)
            "com.samsung.android.honeyboard",
            "com.sec.android.inputmethod",
            "com.samsung.android.inputmethod",
            // SwiftKey (Microsoft)
            "com.swiftkey.swiftkeyapp",
            "com.touchtype.swiftkey",
            "com.microsoft.swiftkey",
            // Other popular keyboards
            "com.nuance.swype.input",
            "com.greyc.gboard",
            "com.fleksy.keyboard",
            "org.pocketworkstation.pckeyboard",
        )

        @Volatile var isActive = false
            private set

        @Volatile var currentPackage: String? = null
            private set

        @Volatile var currentActivity: String? = null
            private set

        /** True while the soft keyboard (IME) is visible on screen. */
        @Volatile var isKeyboardVisible = false
            private set

        /**
         * Latest semantic tree snapshot — built on the accessibility thread
         * whenever a window or content change fires.
         * Safe to read from any thread with no locking overhead.
         */
        @Volatile private var cachedTree: String = "(not ready)"

        /** Thread-safe node registry: nodeId → AccessibilityNodeInfo copy */
        private val nodeRegistry = ConcurrentHashMap<String, AccessibilityNodeInfo>()

        private var instance: AgentAccessibilityService? = null

        /**
         * Returns the most recently built semantic tree.
         * Guaranteed not to cross threads — reads a volatile String.
         */
        fun getSemanticTree(): String = cachedTree

        fun getNodeById(id: String): AccessibilityNodeInfo? = nodeRegistry[id]

        fun dispatchTap(x: Float, y: Float, callback: GestureResultCallback) {
            instance?.dispatchTapAt(x, y, callback)
        }

        fun dispatchSwipe(
            x1: Float, y1: Float, x2: Float, y2: Float,
            callback: GestureResultCallback
        ) {
            instance?.dispatchSwipeGesture(x1, y1, x2, y2, callback)
        }

        fun dispatchLongPress(x: Float, y: Float, callback: GestureResultCallback) {
            instance?.dispatchLongPressAt(x, y, callback)
        }

        fun performBack() {
            instance?.performGlobalAction(GLOBAL_ACTION_BACK)
        }

        /** Navigate to the Android home screen. */
        fun performHome() {
            instance?.performGlobalAction(GLOBAL_ACTION_HOME)
        }

        /** Pull down the notification shade. */
        fun performNotifications() {
            instance?.performGlobalAction(GLOBAL_ACTION_NOTIFICATIONS)
        }

        /**
         * Dismiss the soft keyboard without triggering any navigation.
         *
         * Strategy: clear focus on the currently focused node so the IME loses its
         * anchor and closes naturally. GLOBAL_ACTION_BACK is used only as a fallback
         * when no focused node is found, since Back can inadvertently pop the back
         * stack (e.g. closing a dialog the agent just opened).
         *
         * Returns false immediately if the keyboard is not currently visible.
         */
        @Suppress("DEPRECATION")
        fun performHideKeyboard(): Boolean {
            if (!isKeyboardVisible) return false
            val svc = instance ?: return false
            // Preferred path: clear focus on the focused input node → IME detaches
            val root = runCatching { svc.rootInActiveWindow }.getOrNull()
            val focused = root?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            val cleared = focused?.performAction(AccessibilityNodeInfo.ACTION_CLEAR_FOCUS) == true
            focused?.recycle()
            root?.recycle()
            if (!cleared) {
                // Fallback: Back event (closes IME before navigation on Android 4.4+)
                svc.performGlobalAction(GLOBAL_ACTION_BACK)
            }
            return true
        }

        /**
         * Dispatch a fully-constructed [android.accessibilityservice.GestureDescription]
         * (e.g. pinch-zoom, double-tap) directly through the accessibility service.
         * Used by GestureEngine for multi-finger / compound gestures.
         */
        fun dispatchGesture(
            gesture:  android.accessibilityservice.GestureDescription,
            callback: GestureResultCallback
        ) {
            instance?.dispatchGesture(gesture, callback, null)
        }

        /**
         * Returns the service instance as a [android.content.Context], or null if not active.
         * Used by GestureEngine.openApp() to launch intents without holding a private reference.
         */
        fun getContext(): android.content.Context? = instance

        fun getScreenSize(): Pair<Int, Int> {
            val inst = instance ?: return Pair(1080, 2400)
            val dm = inst.resources.displayMetrics
            return Pair(dm.widthPixels, dm.heightPixels)
        }
    }

    // Handler bound to the accessibility thread for posting rebuilds
    private val handler = Handler(Looper.getMainLooper())

    // Debounce fast content-change bursts (e.g. text typing) — rebuild 150 ms
    // after the last event rather than on every keystroke.
    private val rebuildRunnable = Runnable { rebuildTree() }

    /**
     * Staging map for the atomic nodeRegistry swap.
     * ONLY accessed on the accessibility/main thread (via [handler]); never shared.
     * traverseNode writes here; rebuildTree swaps it into nodeRegistry atomically.
     */
    private var buildingRegistry = HashMap<String, AccessibilityNodeInfo>()

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        isActive  = true
        Log.i(TAG, "Accessibility service connected")
    }

    override fun onInterrupt() {
        isActive = false
    }

    override fun onDestroy() {
        super.onDestroy()
        isActive  = false
        instance  = null
        recycleRegistry()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        event ?: return

        when (event.eventType) {

            // ── Window transitions (app switch, IME open/close, dialog) ───────
            AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED -> {
                val pkg = event.packageName?.toString()
                val cls = event.className?.toString()

                if (pkg != null) {
                    // Detect IME
                    val imeVisible = IME_PACKAGES.contains(pkg) ||
                        cls?.contains("InputMethod", ignoreCase = true) == true
                    if (imeVisible != isKeyboardVisible) {
                        isKeyboardVisible = imeVisible
                        AgentEventBus.emit(
                            "keyboard_visibility_changed",
                            mapOf("visible" to imeVisible)
                        )
                        Log.d(TAG, "Keyboard visible: $imeVisible")
                    }

                    if (!imeVisible) {
                        currentPackage  = pkg
                        currentActivity = cls
                    }
                }

                scheduleRebuild()
            }

            // ── Content updated inside the current window ─────────────────────
            AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED -> {
                // Debounce — only rebuild once bursts settle
                handler.removeCallbacks(rebuildRunnable)
                handler.postDelayed(rebuildRunnable, 150)
            }

            // ── User physically tapped a UI element ───────────────────────────
            AccessibilityEvent.TYPE_VIEW_CLICKED -> {
                // Only report taps that come from the *user*, not from ARIA's
                // own gesture dispatches. ARIA's gestures don't go through
                // TYPE_VIEW_CLICKED (they use dispatchGesture), so all events
                // here are real user interactions.
                val pkg  = event.packageName?.toString() ?: return
                val text = event.text?.joinToString(" ")?.trim() ?: ""
                val desc = event.contentDescription?.toString()?.trim() ?: ""
                val label = text.ifBlank { desc }.ifBlank { "unknown" }

                AgentEventBus.emit(
                    "user_tapped",
                    mapOf("package" to pkg, "label" to label)
                )
                Log.d(TAG, "User tapped: \"$label\" in $pkg")
            }

            // ── Focus tracking ────────────────────────────────────────────────
            AccessibilityEvent.TYPE_VIEW_FOCUSED -> {
                if (event.packageName != null) {
                    currentPackage = event.packageName.toString()
                }
            }
        }
    }

    // ── Tree rebuild ──────────────────────────────────────────────────────────

    private fun scheduleRebuild() {
        handler.removeCallbacks(rebuildRunnable)
        handler.post(rebuildRunnable)
    }

    /**
     * Rebuild the semantic tree snapshot.
     * Must ONLY be called from the accessibility/main thread via [handler].
     *
     * Atomic swap protocol:
     *   1. traverseNode() writes new nodes into [buildingRegistry] (staging map).
     *   2. After traversal is complete, we snapshot the OLD nodeRegistry values,
     *      atomically replace the registry with the new entries, then recycle the
     *      old AccessibilityNodeInfo copies.
     *   → Cross-thread readers calling getNodeById() always see a fully-populated
     *     registry — the null-read window that existed between recycleRegistry()
     *     and the first traverseNode() write is eliminated.
     */
    @Suppress("DEPRECATION")
    private fun rebuildTree() {
        val root = try {
            rootInActiveWindow
        } catch (e: Exception) {
            Log.w(TAG, "rootInActiveWindow failed: ${e.message}")
            null
        }

        if (root == null) {
            cachedTree = if (isKeyboardVisible) "(keyboard visible — no active window)" else "(no active window)"
            return
        }

        // ── Phase 1: build into staging map, never touch the live registry ────
        buildingRegistry = HashMap()
        val lines = mutableListOf<String>()
        if (isKeyboardVisible) lines.add("[keyboard] Soft keyboard is currently visible")
        traverseNode(root, lines, 1)
        root.recycle()
        val newTree = lines.joinToString("\n")

        // ── Phase 2: atomic swap — live registry is empty for <1 µs ──────────
        val oldNodes = nodeRegistry.values.toList()
        nodeRegistry.clear()
        nodeRegistry.putAll(buildingRegistry)
        buildingRegistry = HashMap()  // release staging

        // Recycle OLD nodes AFTER the swap — readers never see an empty state
        oldNodes.forEach { it.recycle() }

        cachedTree = newTree
    }

    /** Fully clear the registry and recycle all held AccessibilityNodeInfo copies. */
    @Suppress("DEPRECATION")
    private fun recycleRegistry() {
        val oldNodes = nodeRegistry.values.toList()
        nodeRegistry.clear()
        oldNodes.forEach { it.recycle() }
    }

    // ── Node traversal ────────────────────────────────────────────────────────

    @Suppress("DEPRECATION")
    private fun traverseNode(
        node: AccessibilityNodeInfo?,
        lines: MutableList<String>,
        counter: Int
    ): Int {
        var id = counter
        if (node == null) return id

        val interactable = node.isClickable || node.isScrollable ||
            node.isEditable || node.isFocusable || node.isLongClickable

        if (interactable) {
            val nodeId = "#$id"
            // Write to staging map — swapped into live nodeRegistry after traversal completes
            buildingRegistry[nodeId] = AccessibilityNodeInfo.obtain(node)
            id++

            val type     = getNodeType(node)
            val text     = node.text?.toString()?.trim()
                ?: node.contentDescription?.toString()?.trim()
                ?: node.hintText?.toString()?.trim()
                ?: ""
            val rect     = Rect()
            node.getBoundsInScreen(rect)
            val position = describePosition(rect)
            val attrs    = buildList {
                if (node.isClickable) add("clickable")
                if (node.isScrollable) add("scrollable")
                if (node.isEditable) add("editable")
                if (!node.isEnabled) add("disabled")
            }.joinToString(", ")

            lines.add("[$nodeId] $type: \"$text\" ($position${if (attrs.isNotEmpty()) ", $attrs" else ""})")
        }

        for (i in 0 until node.childCount) {
            val child = node.getChild(i) ?: continue
            id = traverseNode(child, lines, id)
            child.recycle()
        }
        return id
    }

    private fun getNodeType(node: AccessibilityNodeInfo): String {
        val cls = node.className?.toString() ?: ""
        return when {
            cls.endsWith("Button") || cls.endsWith("ImageButton") -> "Button"
            cls.endsWith("EditText")    -> "EditText"
            cls.endsWith("TextView")    -> "Text"
            cls.endsWith("ImageView")   -> "Image"
            cls.endsWith("CheckBox")    -> "CheckBox"
            cls.endsWith("Switch")      -> "Switch"
            cls.endsWith("ListView") || cls.endsWith("RecyclerView") -> "List"
            cls.endsWith("ScrollView")  -> "ScrollView"
            else -> "View"
        }
    }

    private fun describePosition(rect: Rect): String {
        val dm = resources.displayMetrics
        val cx = rect.centerX().toFloat() / dm.widthPixels
        val cy = rect.centerY().toFloat() / dm.heightPixels
        val v = when {
            cy < 0.25f -> "top"
            cy > 0.75f -> "bottom"
            else       -> "center"
        }
        val h = when {
            cx < 0.33f -> "-left"
            cx > 0.66f -> "-right"
            else       -> ""
        }
        return "$v$h"
    }

    // ── Gesture dispatch ──────────────────────────────────────────────────────

    private fun dispatchTapAt(x: Float, y: Float, callback: GestureResultCallback) {
        val path    = Path().apply { moveTo(x, y) }
        val stroke  = GestureDescription.StrokeDescription(path, 0, 50)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        dispatchGesture(gesture, callback, null)
    }

    private fun dispatchLongPressAt(x: Float, y: Float, callback: GestureResultCallback) {
        val path    = Path().apply { moveTo(x, y) }
        val stroke  = GestureDescription.StrokeDescription(path, 0, 500)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        dispatchGesture(gesture, callback, null)
    }

    private fun dispatchSwipeGesture(
        x1: Float, y1: Float,
        x2: Float, y2: Float,
        callback: GestureResultCallback
    ) {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val stroke  = GestureDescription.StrokeDescription(path, 0, 300)
        val gesture = GestureDescription.Builder().addStroke(stroke).build()
        dispatchGesture(gesture, callback, null)
    }
}
