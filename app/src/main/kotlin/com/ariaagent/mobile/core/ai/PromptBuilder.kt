package com.ariaagent.mobile.core.ai

import com.ariaagent.mobile.core.memory.ObjectLabelStore
import com.ariaagent.mobile.core.perception.ObjectDetectorEngine
import com.ariaagent.mobile.core.perception.ScreenObserver

/**
 * PromptBuilder — Assembles the full LLM prompt for each reasoning step.
 *
 * The LLM (Llama 3.2-1B Instruct) expects a specific chat template:
 *   <|begin_of_text|>
 *   <|start_header_id|>system<|end_header_id|>
 *   {system prompt}
 *   <|eot_id|>
 *   <|start_header_id|>user<|end_header_id|>
 *   {current observation}
 *   <|eot_id|>
 *   <|start_header_id|>assistant<|end_header_id|>
 *
 * AVAILABLE ACTIONS (full set):
 *   Node-ID actions (accessibility tree present):
 *     {"tool":"Click","node_id":"#N","reason":"..."}
 *     {"tool":"Type","node_id":"#N","text":"...","reason":"..."}
 *     {"tool":"Swipe","direction":"up|down|left|right","reason":"..."}
 *     {"tool":"Scroll","node_id":"#N","direction":"up|down","reason":"..."}
 *     {"tool":"LongPress","node_id":"#N","reason":"..."}
 *
 *   XY coordinate actions (game / Flutter / Unity — no a11y tree):
 *     {"tool":"TapXY","x":0.45,"y":0.60,"reason":"..."}    ← normalised 0.0–1.0
 *     {"tool":"SwipeXY","x1":0.5,"y1":0.8,"x2":0.5,"y2":0.2,"reason":"..."}
 *
 *   Control actions:
 *     {"tool":"Back","reason":"..."}
 *     {"tool":"Wait","duration_ms":500,"reason":"..."}
 *     {"tool":"Done","reason":"Goal achieved"}
 *
 * Context sections injected (in priority order):
 *   [KNOWN ELEMENTS]      — human-annotated labels (highest quality)
 *   [VISION DESCRIPTION]  — SmolVLM-256M pixel-level analysis (Phase 17)
 *   [SAM REGIONS]         — MobileSAM salient tap targets (Phase 18) — game / Flutter fallback
 *   [VISUAL DETECTIONS]   — MediaPipe EfficientDet-Lite0 detections
 *   [NODES]               — raw accessibility tree
 *   [APP KNOWLEDGE]       — per-app accumulated skill hint
 *   RELEVANT MEMORY       — past experience snippets (EmbeddingEngine)
 *
 * Token budget:
 *   Llama 3.2-1B context = 4096 tokens ≈ 16 384 chars at ~4 chars/token.
 *   System prompt + template overhead ≈ 2000 chars.
 *   Safety headroom (response = 200 tokens ≈ 800 chars) leaves ~13 584 chars for user content.
 *   [NODES] section is the largest; it is trimmed first if the prompt exceeds budget.
 *
 * Phase: 1 (LLM) — used by AgentLoop from Phase 3 onward.
 */
object PromptBuilder {

    /**
     * Approximate char budget for the user section of the prompt.
     * 4096 tokens − 500 system tokens − 200 response tokens = 3396 user tokens ≈ 13 584 chars.
     * We cap at 12 000 to leave headroom for template markers.
     */
    private const val USER_CHAR_BUDGET = 12_000

    private const val SYSTEM_PROMPT = """You are ARIA — an autonomous Android UI agent running on-device.

Your job: given a screen description and a goal, output exactly ONE JSON action.

AVAILABLE ACTIONS:

  Node-ID actions (when [NODES] section is present — always prefer these):
    {"tool":"Tap","node_id":"#N","reason":"..."}
    {"tool":"DoubleTap","node_id":"#N","reason":"..."}
    {"tool":"LongPress","node_id":"#N","reason":"..."}
    {"tool":"Swipe","node_id":"#N","direction":"up|down|left|right","reason":"..."}
    {"tool":"Scroll","node_id":"#N","direction":"up|down","reason":"..."}
    {"tool":"Type","node_id":"#N","text":"<text>","reason":"..."}
    {"tool":"Copy","node_id":"#N","reason":"..."}
    {"tool":"Paste","node_id":"#N","reason":"..."}

  XY coordinate actions (when [SAM REGIONS] or [VISION DESCRIPTION] is the only signal):
    {"tool":"TapXY","x":0.5,"y":0.6,"reason":"..."}
    {"tool":"DoubleTapXY","x":0.5,"y":0.6,"reason":"..."}
    {"tool":"SwipeXY","x1":0.5,"y1":0.8,"x2":0.5,"y2":0.2,"reason":"..."}
    {"tool":"PinchZoom","x":0.5,"y":0.5,"scale":2.0,"reason":"..."}
      (x,y = centre of pinch; scale > 1.0 = zoom in, < 1.0 = zoom out)
    (all coordinates normalised 0.0–1.0; 0,0 = top-left; 1,1 = bottom-right)

  System actions:
    {"tool":"Back","reason":"..."}
    {"tool":"Home","reason":"..."}
    {"tool":"Notifications","reason":"..."}
    {"tool":"HideKeyboard","reason":"..."}
    {"tool":"OpenApp","package":"com.example.app","reason":"..."}
    {"tool":"Wait","duration_ms":500,"reason":"..."}
    {"tool":"Done","reason":"Goal achieved"}

RULES:
- Output ONLY valid JSON. No explanation outside the JSON object.
- Prefer node_id from [NODES] when available. Use XY tools only when no [NODES] exist.
- If [KNOWN ELEMENTS] section exists, prefer those elements — they are human-verified.
- If [VISION DESCRIPTION] section exists, use it for pixel-level context not visible in the node tree.
- If [SAM REGIONS] section exists and [NODES] is absent, use TapXY on the best region.
- Use OpenApp to launch an app directly rather than navigating to the home screen launcher.
- Use DoubleTap for actions like liking photos, zooming into maps, or opening items that require double-tap.
- Use PinchZoom to zoom in/out on maps, images, or web pages.
- Use Copy then Paste to move text between fields.
- Use HideKeyboard after finishing text input to dismiss the keyboard before tapping other elements.
- If the screen is loading, use Wait.
- If the goal is complete, use Done.
- Think step by step inside the "reason" field (max 30 words).
- Never use a node marked "disabled"."""

    /**
     * Build a full inference prompt for one Observe→Reason step.
     *
     * @param snapshot           The current screen observation (a11y tree + OCR)
     * @param goal               The user's task description
     * @param history            Last N actions taken (prevents repetition loops)
     * @param memory             Relevant past experience snippets (from EmbeddingEngine retrieval)
     * @param objectLabels       Human-annotated UI elements for this screen (highest-quality context)
     * @param detectedObjects    MediaPipe detections for visual elements not in the a11y tree (Phase 13)
     * @param appKnowledge       Compact one-liner from AppSkillRegistry for the current app (Phase 15)
     * @param visionDescription  SmolVLM-256M pixel-level description of the current frame (Phase 17)
     * @param samRegions         MobileSAM tap candidates as formatted strings (Phase 18)
     * @param stuckHint          Injected when stuck detector fires — prompts model to try different approach
     * @param goalPlan           Full task plan with checkmarks from TaskDecomposer (Phase 19).
     *                           Shows the agent the full multi-step plan and which step is current.
     *                           Empty string when goal has only one step (no overhead).
     */
    fun build(
        snapshot: ScreenObserver.ScreenSnapshot,
        goal: String,
        history: List<String> = emptyList(),
        memory: List<String> = emptyList(),
        objectLabels: List<ObjectLabelStore.ObjectLabel> = emptyList(),
        detectedObjects: List<ObjectDetectorEngine.DetectedObject> = emptyList(),
        appKnowledge: String = "",
        crossAppKnowledge: String = "",
        visionDescription: String = "",
        samRegions: List<String> = emptyList(),
        stuckHint: String = "",
        goalPlan: String = "",
    ): String {
        val sb = StringBuilder()

        sb.append("<|begin_of_text|>")
        sb.append("<|start_header_id|>system<|end_header_id|>\n")
        sb.append(SYSTEM_PROMPT)

        if (memory.isNotEmpty()) {
            sb.append("\n\nRELEVANT MEMORY (past experiences on similar screens):\n")
            memory.take(3).forEach { sb.appendLine("- $it") }
        }

        // ── App Skill Registry hint (Phase 15) ────────────────────────────────
        if (appKnowledge.isNotEmpty()) {
            sb.append("\n\n[APP KNOWLEDGE] (ARIA's prior experience with this app)\n")
            sb.appendLine(appKnowledge)
        }

        // ── Cross-app knowledge (compact hints for other known apps) ──────────
        // Allows the agent to plan app-switching steps informed by prior experience
        // on apps it may navigate to during a multi-step goal.
        if (crossAppKnowledge.isNotEmpty()) {
            sb.append("\n\n[OTHER KNOWN APPS] (compact hints — available if you need to switch apps)\n")
            sb.appendLine(crossAppKnowledge)
        }

        // ── Stuck hint (injected by AgentLoop stuck detector) ─────────────────
        if (stuckHint.isNotEmpty()) {
            sb.append("\n\n⚠ STUCK ALERT: $stuckHint\n")
        }

        sb.append("\n<|eot_id|>\n")

        sb.append("<|start_header_id|>user<|end_header_id|>\n")
        sb.append("GOAL: $goal\n\n")

        // ── Phase 19: Task plan — shows full multi-step plan with [x]/[ ] checkmarks ──
        // Only present when the goal was decomposed into ≥ 2 steps. Single-step goals
        // have an empty goalPlan so there is zero overhead for simple tasks.
        if (goalPlan.isNotEmpty()) {
            sb.appendLine("[TASK PLAN]")
            sb.appendLine(goalPlan)
            sb.appendLine()
        }

        // ── Object labels injected BEFORE the raw node tree ───────────────────
        if (objectLabels.isNotEmpty()) {
            sb.appendLine("[KNOWN ELEMENTS] (human-annotated — use these when relevant to goal)")
            objectLabels
                .sortedByDescending { it.importanceScore }
                .take(8)
                .forEach { label -> sb.appendLine(label.toPromptLine()) }
            sb.appendLine()
        }

        // ── Vision description (Phase 17) ─────────────────────────────────────
        if (visionDescription.isNotBlank()) {
            sb.appendLine("[VISION DESCRIPTION] (SmolVLM-256M — pixel-level screen analysis)")
            sb.appendLine(visionDescription.trim())
            sb.appendLine()
        }

        // ── SAM regions (Phase 18) — game / Flutter / Unity fallback ──────────
        // Injected when MobileSAM has identified salient tap candidates and no
        // accessibility nodes are available. The LLM must use TapXY to act on these.
        if (samRegions.isNotEmpty()) {
            sb.appendLine("[SAM REGIONS] (MobileSAM — tap targets on game/custom screen; use TapXY)")
            samRegions.forEach { sb.appendLine(it) }
            sb.appendLine()
        }

        // ── Visual detections (Phase 13) ──────────────────────────────────────
        if (detectedObjects.isNotEmpty()) {
            sb.appendLine("[VISUAL DETECTIONS] (MediaPipe — elements not in accessibility tree)")
            detectedObjects
                .sortedByDescending { it.confidence }
                .take(8)
                .forEachIndexed { i, obj -> sb.appendLine(obj.toPromptLine(i)) }
            sb.appendLine()
        }

        // ── Accessibility node tree — trimmed to token budget ─────────────────
        // The node tree is by far the largest section. If we are already near the
        // character budget, trim it so we don't silently overflow the 4096-token context.
        val usedSoFar = sb.length
        val remaining = USER_CHAR_BUDGET - usedSoFar
        val nodeTree  = snapshot.toLlmString()
        sb.append(
            if (nodeTree.length > remaining && remaining > 200) {
                nodeTree.take(remaining) + "\n…[node tree trimmed — token budget]"
            } else {
                nodeTree
            }
        )

        // ── Recent action history ──────────────────────────────────────────────
        if (history.isNotEmpty()) {
            sb.append("\n\nRECENT ACTIONS (avoid repeating these):\n")
            history.takeLast(5).forEachIndexed { i, action ->
                sb.appendLine("${i + 1}. $action")
            }
        }

        sb.append("\n<|eot_id|>\n")
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n")

        return sb.toString()
    }

    /**
     * Extract a valid JSON action from the raw LLM output.
     * Handles common formatting noise (markdown code fences, leading text).
     */
    fun parseAction(rawOutput: String): String {
        val cleaned = rawOutput.trim()

        val jsonStart = cleaned.indexOfFirst { it == '{' }
        if (jsonStart == -1) return """{"tool":"Wait","duration_ms":500,"reason":"no action parsed"}"""

        val jsonEnd = cleaned.lastIndexOf('}')
        if (jsonEnd <= jsonStart) return """{"tool":"Wait","duration_ms":500,"reason":"malformed json"}"""

        return cleaned.substring(jsonStart, jsonEnd + 1).trim()
    }
}
