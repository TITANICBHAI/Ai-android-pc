package com.ariaagent.mobile.core.ai

import android.util.Log
import com.ariaagent.mobile.ui.viewmodel.LlmRole
import com.ariaagent.mobile.ui.viewmodel.LoadedLlmEntry

/**
 * ModelDispatcher — routes inference requests to the correct loaded model(s).
 *
 * ─── Design ───────────────────────────────────────────────────────────────────
 *
 * ARIA can have up to 3 models registered in the slot system at once.
 * Each model is assigned a [LlmRole] by the user (or defaults to REASONING).
 * The dispatcher selects which model runs for each request type and, when
 * needed, chains model outputs together so they fill each other's gaps.
 *
 * ─── Role Priority (highest → lowest) ────────────────────────────────────────
 *
 *   1. EVERYTHING      — single model handles everything; no chaining.
 *   2. Specialist role — VISION, PLANNING, REASONING, TOOL_USE, CHAT.
 *   3. EVERYTHING_ELSE — explicit fallback for unmatched request types.
 *   4. First loaded    — last-resort fallback when nothing else matches.
 *
 * ─── Chaining rules ───────────────────────────────────────────────────────────
 *
 *   Vision  → Reasoning:   Vision model produces a text description of the
 *             screen/image. That description is injected into the Reasoning
 *             model's context as a [VISION] block. The Reasoning model then
 *             produces the action JSON. This is the same flow used when
 *             SmolVLM auto-helper is active — we just generalise it.
 *
 *   Planning → Reasoning:  Planning model produces a numbered sub-task list.
 *             ARIA's AgentLoop runs each sub-task through the Reasoning model.
 *             Planning runs once at goal-start, not every step.
 *
 *   EVERYTHING_ELSE:        Catches any request type that has no specialist.
 *             No chaining — treated exactly like REASONING.
 *
 * ─── Single-model path ────────────────────────────────────────────────────────
 *
 *   If only one model is loaded (or it has role EVERYTHING):
 *     • text-only model: does reasoning directly; VisionEngine helper provides
 *       screen description as text prefix (existing auto-helper logic).
 *     • VLM: receives image bytes directly via inferWithVision().
 *
 * ─── Thread safety ────────────────────────────────────────────────────────────
 *
 *   All public functions are pure reads on immutable snapshots — no state is
 *   mutated here. Call from any thread.
 */
object ModelDispatcher {

    private const val TAG = "ModelDispatcher"

    // ─── DispatchPlan ─────────────────────────────────────────────────────────

    /**
     * Describes how a single inference request should be executed.
     *
     * Full 3-stage pipeline: VISION → REASONING → TOOL_USE
     *
     * @param primaryRole         The role of the model that should run first.
     * @param primaryModelId      The model ID to use in the primary step.
     * @param chainRole           If non-null, the role whose model runs after [primaryRole]
     *                            receives the primary output as additional context.
     *                            Used for VISION → REASONING chaining.
     * @param chainModelId        Model ID for the chained step (null iff chainRole is null).
     * @param toolUseModelId      If non-null, a TOOL_USE-role model runs as the final pass
     *                            after REASONING completes. It receives the reasoning text
     *                            and emits the final JSON action.
     *                            When equal to [primaryModelId] no model swap occurs —
     *                            the same engine runs two inference passes.
     * @param useVisionOnPrimary  True when the primary model should receive raw image bytes.
     * @param systemPromptOverride Non-null when the slot has a custom system prompt.
     */
    data class DispatchPlan(
        val primaryRole:          LlmRole,
        val primaryModelId:       String,
        val chainRole:            LlmRole?  = null,
        val chainModelId:         String?   = null,
        val toolUseModelId:       String?   = null,
        val useVisionOnPrimary:   Boolean   = false,
        val systemPromptOverride: String?   = null,
    ) {
        val isChained:       Boolean get() = chainRole != null && chainModelId != null
        val hasToolUsePass:  Boolean get() = toolUseModelId != null

        /**
         * True when the TOOL_USE model is a different physical model than the primary.
         * When false AgentLoop re-uses the already-loaded engine (second pass, no swap).
         */
        val toolUseNeedsSwap: Boolean
            get() = toolUseModelId != null && toolUseModelId != primaryModelId &&
                    toolUseModelId != chainModelId
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    /**
     * Compute the dispatch plan for an inference request.
     *
     * @param slots     Current map of loaded model entries (from AgentViewModel).
     * @param requestType The type of inference needed (REASONING by default).
     * @param hasImage  Whether the request includes an image/screen bitmap.
     * @return A [DispatchPlan], or null if no loaded model can handle the request.
     */
    fun plan(
        slots:       Map<String, LoadedLlmEntry>,
        requestType: LlmRole  = LlmRole.REASONING,
        hasImage:    Boolean  = false,
    ): DispatchPlan? {
        val loaded = slots.values.filter { it.isLoaded }
        if (loaded.isEmpty()) {
            Log.w(TAG, "plan(): no loaded models — cannot dispatch")
            return null
        }

        // ── Shared helper: find the TOOL_USE specialist (if any) ──────────────
        // The TOOL_USE model is the last stage in every pipeline — it receives the
        // reasoning text and converts it to a precise JSON action.
        // If no dedicated TOOL_USE slot exists the primary model does both passes.
        val toolUseSlot = loaded.firstOrNull { it.role == LlmRole.TOOL_USE }

        // ── Single model path ─────────────────────────────────────────────────
        if (loaded.size == 1) {
            val entry  = loaded.first()
            val isVlm  = ModelCatalog.findById(entry.modelId)?.isTextOnly == false
            return DispatchPlan(
                primaryRole          = entry.role,
                primaryModelId       = entry.modelId,
                useVisionOnPrimary   = hasImage && isVlm,
                systemPromptOverride = entry.systemPrompt.ifBlank { null },
            ).also { Log.d(TAG, "Single-model path: ${entry.modelId} role=${entry.role} vision=${it.useVisionOnPrimary}") }
        }

        // ── Multi-model path ──────────────────────────────────────────────────

        // 1. EVERYTHING role takes absolute precedence — use it for everything.
        val everything = loaded.firstOrNull { it.role == LlmRole.EVERYTHING }
        if (everything != null) {
            val isVlm = ModelCatalog.findById(everything.modelId)?.isTextOnly == false
            return DispatchPlan(
                primaryRole          = LlmRole.EVERYTHING,
                primaryModelId       = everything.modelId,
                useVisionOnPrimary   = hasImage && isVlm,
                systemPromptOverride = everything.systemPrompt.ifBlank { null },
            ).also { Log.d(TAG, "EVERYTHING model: ${everything.modelId}") }
        }

        // 2. Vision chaining — if request involves an image and there is a dedicated
        //    VISION specialist, run it first then chain into the reasoning model,
        //    then optionally into the TOOL_USE model for JSON extraction.
        if (hasImage) {
            val visionSlot = loaded.firstOrNull { it.role == LlmRole.VISION }
            if (visionSlot != null) {
                val reasoningSlot = loaded.firstOrNull { it.role == LlmRole.REASONING }
                    ?: loaded.firstOrNull { it.role == LlmRole.EVERYTHING_ELSE }
                    ?: loaded.first { it.modelId != visionSlot.modelId }
                val visionIsVlm = ModelCatalog.findById(visionSlot.modelId)?.isTextOnly == false
                // TOOL_USE pass: dedicated slot if different from the reasoning model
                val toolId = toolUseSlot
                    ?.takeIf { it.modelId != reasoningSlot.modelId }
                    ?.modelId
                return DispatchPlan(
                    primaryRole          = LlmRole.VISION,
                    primaryModelId       = visionSlot.modelId,
                    useVisionOnPrimary   = visionIsVlm,
                    chainRole            = reasoningSlot.role,
                    chainModelId         = reasoningSlot.modelId,
                    toolUseModelId       = toolId ?: reasoningSlot.modelId,
                    systemPromptOverride = visionSlot.systemPrompt.ifBlank { null },
                ).also {
                    val chain = "${visionSlot.modelId} → ${reasoningSlot.modelId}" +
                        if (toolId != null) " → $toolId" else " → (same, 2-pass)"
                    Log.d(TAG, "Vision→Reasoning→ToolUse chain: $chain")
                }
            }
        }

        // 3. Exact specialist match for the requested role.
        //    Attach the TOOL_USE model as the final pass when available.
        val specialist = loaded.firstOrNull { it.role == requestType }
        if (specialist != null) {
            val isVlm = ModelCatalog.findById(specialist.modelId)?.isTextOnly == false
            val toolId = toolUseSlot
                ?.takeIf { it.modelId != specialist.modelId }
                ?.modelId
                ?: specialist.modelId   // same model = two-pass, no swap
            return DispatchPlan(
                primaryRole          = specialist.role,
                primaryModelId       = specialist.modelId,
                useVisionOnPrimary   = hasImage && isVlm,
                toolUseModelId       = toolId,
                systemPromptOverride = specialist.systemPrompt.ifBlank { null },
            ).also {
                Log.d(TAG, "Specialist match: ${specialist.modelId} role=$requestType toolUse=$toolId")
            }
        }

        // 4. EVERYTHING_ELSE fallback.
        val fallback = loaded.firstOrNull { it.role == LlmRole.EVERYTHING_ELSE }
        if (fallback != null) {
            val isVlm = ModelCatalog.findById(fallback.modelId)?.isTextOnly == false
            val toolId = toolUseSlot
                ?.takeIf { it.modelId != fallback.modelId }
                ?.modelId
                ?: fallback.modelId
            return DispatchPlan(
                primaryRole          = LlmRole.EVERYTHING_ELSE,
                primaryModelId       = fallback.modelId,
                useVisionOnPrimary   = hasImage && isVlm,
                toolUseModelId       = toolId,
                systemPromptOverride = fallback.systemPrompt.ifBlank { null },
            ).also { Log.d(TAG, "EVERYTHING_ELSE fallback: ${fallback.modelId} toolUse=$toolId") }
        }

        // 5. Last resort — first loaded model, no chaining.
        val last = loaded.first()
        val lastIsVlm = ModelCatalog.findById(last.modelId)?.isTextOnly == false
        val lastToolId = toolUseSlot
            ?.takeIf { it.modelId != last.modelId }
            ?.modelId
            ?: last.modelId
        return DispatchPlan(
            primaryRole          = last.role,
            primaryModelId       = last.modelId,
            useVisionOnPrimary   = hasImage && lastIsVlm,
            toolUseModelId       = lastToolId,
            systemPromptOverride = last.systemPrompt.ifBlank { null },
        ).also { Log.d(TAG, "Last-resort fallback: ${last.modelId} toolUse=$lastToolId") }
    }

    /**
     * Select the model that should handle Planning (goal decomposition).
     * Prefers PLANNING role, falls back to REASONING or EVERYTHING_ELSE.
     */
    fun planningModel(slots: Map<String, LoadedLlmEntry>): LoadedLlmEntry? {
        val loaded = slots.values.filter { it.isLoaded }
        return loaded.firstOrNull { it.role == LlmRole.PLANNING }
            ?: loaded.firstOrNull { it.role == LlmRole.EVERYTHING }
            ?: loaded.firstOrNull { it.role == LlmRole.REASONING }
            ?: loaded.firstOrNull { it.role == LlmRole.EVERYTHING_ELSE }
            ?: loaded.firstOrNull()
    }

    /**
     * Select the model that should handle Chat turns.
     * Prefers CHAT role, falls back to REASONING or EVERYTHING_ELSE.
     */
    fun chatModel(slots: Map<String, LoadedLlmEntry>): LoadedLlmEntry? {
        val loaded = slots.values.filter { it.isLoaded }
        return loaded.firstOrNull { it.role == LlmRole.CHAT }
            ?: loaded.firstOrNull { it.role == LlmRole.EVERYTHING }
            ?: loaded.firstOrNull { it.role == LlmRole.REASONING }
            ?: loaded.firstOrNull { it.role == LlmRole.EVERYTHING_ELSE }
            ?: loaded.firstOrNull()
    }

    /**
     * Select the model for tool-use / function calling.
     * Prefers TOOL_USE, then REASONING, then EVERYTHING_ELSE.
     */
    fun toolUseModel(slots: Map<String, LoadedLlmEntry>): LoadedLlmEntry? {
        val loaded = slots.values.filter { it.isLoaded }
        return loaded.firstOrNull { it.role == LlmRole.TOOL_USE }
            ?: loaded.firstOrNull { it.role == LlmRole.EVERYTHING }
            ?: loaded.firstOrNull { it.role == LlmRole.REASONING }
            ?: loaded.firstOrNull { it.role == LlmRole.EVERYTHING_ELSE }
            ?: loaded.firstOrNull()
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    /**
     * Returns a human-readable summary of the current dispatch topology.
     * Shown in the Modules screen as a connection diagram.
     */
    fun topologySummary(slots: Map<String, LoadedLlmEntry>): String {
        val loaded = slots.values.filter { it.isLoaded }
        if (loaded.isEmpty()) return "No models loaded"
        if (loaded.size == 1) {
            val e = loaded.first()
            val name = ModelCatalog.findById(e.modelId)?.displayName ?: e.modelId
            return if (e.role == LlmRole.EVERYTHING) "$name handles everything"
            else "$name (${e.role.label}) — solo mode"
        }

        val lines = mutableListOf<String>()
        val everything = loaded.firstOrNull { it.role == LlmRole.EVERYTHING }
        if (everything != null) {
            val name = ModelCatalog.findById(everything.modelId)?.displayName ?: everything.modelId
            lines += "$name (EVERYTHING) — handles all requests"
            val rest = loaded.filter { it.modelId != everything.modelId }
            rest.forEach { e ->
                val n = ModelCatalog.findById(e.modelId)?.displayName ?: e.modelId
                lines += "  $n (${e.role.label}) — idle (EVERYTHING model active)"
            }
            return lines.joinToString("\n")
        }

        val visionSlot    = loaded.firstOrNull { it.role == LlmRole.VISION }
        val reasonSlot    = loaded.firstOrNull { it.role == LlmRole.REASONING }
        val planningSlot  = loaded.firstOrNull { it.role == LlmRole.PLANNING }
        val toolSlot      = loaded.firstOrNull { it.role == LlmRole.TOOL_USE }
        val chatSlot      = loaded.firstOrNull { it.role == LlmRole.CHAT }
        val fallbackSlot  = loaded.firstOrNull { it.role == LlmRole.EVERYTHING_ELSE }

        // Planning chain: PLANNING decomposes goal → sub-tasks fed to REASONING
        if (planningSlot != null && reasonSlot != null) {
            val pn = ModelCatalog.findById(planningSlot.modelId)?.displayName ?: planningSlot.modelId
            val rn = ModelCatalog.findById(reasonSlot.modelId)?.displayName ?: reasonSlot.modelId
            lines += "Goal → $pn (PLANNING) → sub-tasks → $rn (REASONING)"
        }

        // Full 3-stage pipeline: VISION → REASONING → TOOL_USE
        val reasonName = reasonSlot?.let { ModelCatalog.findById(it.modelId)?.displayName ?: it.modelId }
        if (visionSlot != null && reasonSlot != null) {
            val vn = ModelCatalog.findById(visionSlot.modelId)?.displayName ?: visionSlot.modelId
            if (toolSlot != null) {
                val tn = ModelCatalog.findById(toolSlot.modelId)?.displayName ?: toolSlot.modelId
                lines += "Image → $vn (VISION) → description → $reasonName (REASONING) → thinking → $tn (TOOL USE) → JSON action"
            } else {
                lines += "Image → $vn (VISION) → description → $reasonName (REASONING) → 2-pass JSON action"
            }
        } else if (reasonSlot != null && toolSlot != null) {
            // No vision, but REASONING → TOOL_USE chain exists
            val tn = ModelCatalog.findById(toolSlot.modelId)?.displayName ?: toolSlot.modelId
            lines += "$reasonName (REASONING) → thinking → $tn (TOOL USE) → JSON action"
        } else if (reasonSlot != null) {
            lines += "$reasonName (REASONING) → single-pass JSON action"
        }

        // TOOL_USE shown separately only when it's not already part of a chain above
        if (toolSlot != null && reasonSlot == null && visionSlot == null) {
            val n = ModelCatalog.findById(toolSlot.modelId)?.displayName ?: toolSlot.modelId
            lines += "$n (TOOL USE) → JSON action"
        }
        chatSlot?.let {
            val n = ModelCatalog.findById(it.modelId)?.displayName ?: it.modelId
            lines += "$n (CHAT) → user conversation"
        }
        fallbackSlot?.let {
            val n = ModelCatalog.findById(it.modelId)?.displayName ?: it.modelId
            lines += "$n (EVERYTHING ELSE) → unmatched requests"
        }
        if (lines.isEmpty()) {
            loaded.forEach { e ->
                val n = ModelCatalog.findById(e.modelId)?.displayName ?: e.modelId
                lines += "$n (${e.role.label})"
            }
        }
        return lines.joinToString("\n")
    }

    /**
     * Returns true if the given role assignment would create a conflict
     * (e.g. assigning EVERYTHING when another model already has EVERYTHING).
     */
    fun isRoleConflict(
        slots:      Map<String, LoadedLlmEntry>,
        modelId:    String,
        newRole:    LlmRole,
    ): Boolean {
        if (newRole == LlmRole.CUSTOM) return false
        return slots.values.any { it.modelId != modelId && it.role == newRole && newRole == LlmRole.EVERYTHING }
    }
}
