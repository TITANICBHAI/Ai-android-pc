package com.ariaagent.mobile.system.overlay

import android.app.Service
import android.content.Intent
import android.graphics.PixelFormat
import android.os.IBinder
import android.view.Gravity
import android.view.WindowManager
import androidx.compose.ui.platform.ComposeView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.savedstate.SavedStateRegistry
import androidx.savedstate.SavedStateRegistryController
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import com.ariaagent.mobile.core.events.AgentEventBus

/**
 * FloatingChatService — draws a live ARIA chat overlay over any app.
 *
 * Uses TYPE_APPLICATION_OVERLAY (requires SYSTEM_ALERT_WINDOW permission).
 * The overlay is a ComposeView attached to the WindowManager — it floats over
 * the current foreground app while ARIA is acting, letting the user:
 *   - Watch live action + reason in real time
 *   - Type instructions that are injected into the AgentLoop
 *   - Draw gestures on screen — captured as frame annotations
 *
 * Keyboard fix
 * ────────────
 * Overlay windows with FLAG_NOT_FOCUSABLE cannot receive keyboard input — the
 * soft keyboard will never appear for the text field. We start with that flag
 * set (so taps on the underlying app still pass through), then *temporarily*
 * clear it when the user focuses the text field, and restore it on blur.
 * This is the standard Android pattern for overlay text input.
 *
 * Lifecycle: started by AgentForegroundService when AgentLoop starts;
 *            stopped when AgentLoop finishes or is paused.
 */
class FloatingChatService : Service(), LifecycleOwner, SavedStateRegistryOwner {

    private val lifecycleRegistry    = LifecycleRegistry(this)
    private val savedStateController = SavedStateRegistryController.create(this)
    override val lifecycle: Lifecycle
        get() = lifecycleRegistry
    override val savedStateRegistry: SavedStateRegistry
        get() = savedStateController.savedStateRegistry

    private var windowManager: WindowManager? = null
    private var overlayView: ComposeView?     = null

    /** Kept so updateViewLayout() can toggle FLAG_NOT_FOCUSABLE without rebuilding. */
    private var overlayParams: WindowManager.LayoutParams? = null

    override fun onCreate() {
        super.onCreate()
        savedStateController.performAttach()
        savedStateController.performRestore(null)
        lifecycleRegistry.currentState = Lifecycle.State.CREATED
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            teardown()
            stopSelf()
            return START_NOT_STICKY
        }
        if (overlayView == null) setup()
        lifecycleRegistry.currentState = Lifecycle.State.STARTED
        return START_STICKY
    }

    override fun onDestroy() {
        teardown()
        lifecycleRegistry.currentState = Lifecycle.State.DESTROYED
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ── Window setup ──────────────────────────────────────────────────────────

    private fun setup() {
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            // FLAG_NOT_FOCUSABLE  — taps on underlying app pass through
            // FLAG_NOT_TOUCH_MODAL — touches outside the window bounds don't cancel it
            // (FLAG_NOT_FOCUSABLE is cleared temporarily when text field is focused)
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.BOTTOM or Gravity.START
            x = 24
            y = 120
        }
        overlayParams = params

        val composeView = ComposeView(this).apply {
            setViewTreeLifecycleOwner(this@FloatingChatService)
            setViewTreeSavedStateRegistryOwner(this@FloatingChatService)
            setContent {
                FloatingChatOverlay(
                    onInstruction = { text ->
                        AgentEventBus.emit(
                            "user_instruction",
                            mapOf("text" to text, "source" to "floating_chat")
                        )
                    },
                    onGestureAnnotation = { annotation ->
                        AgentEventBus.emit(
                            "user_gesture_annotation",
                            mapOf("annotation" to annotation, "source" to "floating_chat")
                        )
                    },
                    onDismiss = { teardown(); stopSelf() },
                    onInputFocused = { focused -> setFocusable(focused) }
                )
            }
        }

        windowManager?.addView(composeView, params)
        overlayView = composeView
    }

    /**
     * Toggle focusability of the overlay window.
     *
     * focused=true  → clear FLAG_NOT_FOCUSABLE → keyboard can attach to this window
     * focused=false → restore FLAG_NOT_FOCUSABLE → taps pass through to underlying app
     */
    private fun setFocusable(focused: Boolean) {
        val params = overlayParams ?: return
        val view   = overlayView   ?: return
        if (focused) {
            params.flags = params.flags and WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE.inv()
        } else {
            params.flags = params.flags or WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
        }
        try {
            windowManager?.updateViewLayout(view, params)
        } catch (e: Exception) {
            android.util.Log.w("FloatingChatService", "updateViewLayout failed: ${e.message}")
        }
    }

    private fun teardown() {
        overlayView?.let {
            runCatching { windowManager?.removeView(it) }
            overlayView   = null
            overlayParams = null
        }
    }

    companion object {
        const val ACTION_STOP = "com.ariaagent.mobile.STOP_FLOATING_CHAT"
    }
}
