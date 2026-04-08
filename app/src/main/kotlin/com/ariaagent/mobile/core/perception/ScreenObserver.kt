package com.ariaagent.mobile.core.perception

import android.graphics.Bitmap
import com.ariaagent.mobile.core.ocr.OcrEngine
import com.ariaagent.mobile.system.accessibility.AgentAccessibilityService
import com.ariaagent.mobile.system.screen.ScreenCaptureService

/**
 * ScreenObserver — Semantic perception fusion layer.
 *
 * Combines three signal sources into a single LLM-ready observation string:
 *   1. Accessibility tree  → structural UI nodes with stable IDs
 *   2. ML Kit OCR          → raw text visible on screen
 *   3. ScreenCapture       → bitmap for when the a11y tree is empty (games, custom views)
 *
 * Output format understood by the LLM system prompt:
 *   APP: com.example.app  SCREEN: MainActivity
 *   [NODES]
 *   #1 Button "Submit" [enabled] bounds(100,400,300,460)
 *   #2 EditText "Email" [focusable] text="user@example.com"
 *   ...
 *   [OCR]
 *   Welcome back! Enter your credentials below.
 *   Submit
 *
 * The fused output is passed directly to LlamaEngine.infer() as part of the prompt.
 *
 * Phase: 2 (Perception) — depends on Phase 1 (LLM) for prompt consumption.
 */
object ScreenObserver {

    data class ScreenSnapshot(
        val appPackage: String,
        val activityName: String,
        val a11yTree: String,
        val ocrText: String,
        val bitmap: Bitmap?,
        val timestamp: Long = System.currentTimeMillis()
    ) {
        /**
         * Produces the exact string format injected into the LLM prompt.
         * The LLM system prompt defines this schema — never change format
         * without updating the system prompt in PromptBuilder.
         */
        fun toLlmString(): String {
            val sb = StringBuilder()
            sb.appendLine("APP: $appPackage  SCREEN: ${activityName.substringAfterLast('.')}")
            sb.appendLine()

            if (a11yTree.isNotBlank()) {
                // NODES = interactive elements from the accessibility tree.
                // Prefer these for ALL tap/type/scroll actions — they give exact node IDs
                // that GestureEngine can route directly without coordinate math.
                sb.appendLine("[NODES] (interactive UI elements — USE these for Click/Type/Scroll)")
                sb.appendLine(a11yTree.trim())
                sb.appendLine()
            }

            if (ocrText.isNotBlank()) {
                // OCR = raw pixel text read by ML Kit from the screen bitmap.
                // Useful for reading labels, values, and non-interactive text that
                // the accessibility tree may omit (e.g. canvas text in WebViews).
                // Do NOT use OCR coordinates for clicking — always prefer [NODES].
                sb.appendLine("[OCR TEXT] (non-interactive text only — read values here, click via [NODES])")
                sb.appendLine(ocrText.trim())
            }

            return sb.toString().trim()
        }

        /**
         * A stable hash of the observation used for SQLite lookup, stuck detection, and
         * edge-case matching.
         *
         * Quality improvements over the old 200-char / 100-char version:
         *   • activityName included — catches in-app navigations that don't change the
         *     package but do change the screen (e.g. Settings subscreen navigation).
         *   • a11y window extended to 500 chars — reduces false-positive "stuck" signals
         *     when a dialog or list item appears beyond the first 200 chars of the tree.
         *   • OCR window extended to 200 chars — catches text-only screen changes that
         *     produce no new a11y nodes (e.g. dynamic content inside a WebView).
         *   • Uses SHA-256 truncated to 16 hex chars instead of Java's signed 32-bit
         *     hashCode() — eliminates sign-bit collisions and reduces birthday-paradox
         *     false matches on long agent sessions.
         */
        fun screenHash(): String {
            val combined = "$appPackage|${activityName.substringAfterLast('.')}|${a11yTree.take(500)}|${ocrText.take(200)}"
            return try {
                val digest = java.security.MessageDigest.getInstance("SHA-256")
                val bytes  = digest.digest(combined.toByteArray(Charsets.UTF_8))
                bytes.take(8).joinToString("") { "%02x".format(it) }
            } catch (_: Exception) {
                // Fallback to hashCode if MessageDigest is somehow unavailable
                combined.hashCode().toString(16)
            }
        }

        fun isEmpty(): Boolean = a11yTree.isBlank() && ocrText.isBlank()
    }

    /**
     * Capture a full semantic snapshot of the current screen.
     * Must be called from a coroutine (IO or Default dispatcher).
     *
     * If the accessibility service is not active, falls back to OCR-only.
     * If both are empty, returns an empty snapshot (agent should wait and retry).
     */
    suspend fun capture(): ScreenSnapshot {
        val a11yTree   = AgentAccessibilityService.getSemanticTree()
        val appPackage = AgentAccessibilityService.currentPackage ?: "unknown"
        val activity   = AgentAccessibilityService.currentActivity ?: "unknown"

        val bitmap = ScreenCaptureService.captureLatest()
        val ocrText = if (bitmap != null) {
            runCatching { OcrEngine.run(bitmap) }.getOrDefault("")
        } else {
            ""
        }

        return ScreenSnapshot(
            appPackage   = appPackage,
            activityName = activity,
            a11yTree     = a11yTree,
            ocrText      = ocrText,
            bitmap       = bitmap
        )
    }

    /**
     * Fast check: has the screen changed since the last observation?
     * Used by the agent loop to skip inference when nothing has changed.
     */
    fun hasChanged(previous: ScreenSnapshot?, current: ScreenSnapshot): Boolean {
        if (previous == null) return true
        return previous.screenHash() != current.screenHash()
    }

    /**
     * Lightweight check that does NOT capture a bitmap.
     * Only reads the accessibility tree — faster than full capture.
     * Use between agent steps to detect state transitions.
     */
    fun quickCheck(): String {
        return AgentAccessibilityService.getSemanticTree()
    }
}
