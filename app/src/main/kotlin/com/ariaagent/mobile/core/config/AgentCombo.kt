package com.ariaagent.mobile.core.config

/**
 * AgentCombo — which major ARIA modules are active during an agent session.
 *
 * The agent is normally operated with ALL modules running (default). Users who want
 * faster cycles, less RAM, or purpose-specific training runs can restrict the combo.
 *
 * Semantics:
 *   usesLlm  — main LLM (LlamaEngine / ModelDispatcher) is consulted at each step.
 *              When false the agent falls back to PolicyNetwork (the fast RL policy)
 *              for action selection. This saves significant RAM and time but requires
 *              a trained policy — a cold-start device with no RL data will stall.
 *   usesRl   — agent's STORE step (ExperienceStore.save) and LearningScheduler
 *              LoRA training cycle are active. When false no experiences are written
 *              and no background LoRA jobs fire.
 *   usesIrl  — IrlModule.processVideo() is permitted from the UI and respects this
 *              flag. IRL does not run automatically in the main loop regardless.
 */
enum class AgentCombo(
    val label:    String,
    val usesLlm:  Boolean,
    val usesRl:   Boolean,
    val usesIrl:  Boolean,
) {
    ALL           ("All modules",           usesLlm = true,  usesRl = true,  usesIrl = true),
    LLM_ONLY      ("LLM only",              usesLlm = true,  usesRl = false, usesIrl = false),
    RL_ONLY       ("RL only (policy net)",  usesLlm = false, usesRl = true,  usesIrl = false),
    IRL_ONLY      ("IRL only",              usesLlm = false, usesRl = false, usesIrl = true),
    LLM_RL        ("LLM + RL",              usesLlm = true,  usesRl = true,  usesIrl = false),
    LLM_IRL       ("LLM + IRL",             usesLlm = true,  usesRl = false, usesIrl = true),
    RL_IRL        ("RL + IRL",              usesLlm = false, usesRl = true,  usesIrl = true);

    companion object {
        fun fromName(name: String?): AgentCombo =
            entries.firstOrNull { it.name == name } ?: ALL
    }
}

/**
 * ModuleVisionMode — which model(s) a learning module (RL or IRL) uses for
 * perception and reasoning during its own internal pipeline.
 *
 * This is separate from the main agent's ModelDispatcher routing — it controls
 * how ARIA's training subsystems (IrlModule, LlmRewardEnricher) pick their models.
 *
 * AUTO   — use whatever is available: vision model first, LLM fallback.
 *          This is the recommended default for most devices.
 * LLM_ONLY
 *        — use only the text LLM (LlamaEngine / REASONING slot). Fastest; skips
 *          VisionEngine and Sam2 entirely. Good when RAM is tight or vision model
 *          is not loaded.
 * VISION_ONLY
 *        — use only the vision model (VisionEngine + Sam2). The text LLM is not
 *          consulted. For IRL this means scene descriptions come from the VLM;
 *          action inference falls back to heuristics. For reward enrichment this
 *          means the VLM scores experiences without a separate reasoning pass.
 * BOTH   — chain vision + text LLM (most accurate; highest RAM/time cost).
 *          VisionEngine runs first to produce a rich screen description, then the
 *          text LLM uses that description for action inference / reward scoring.
 *          Falls back gracefully: if only one model type is loaded, acts like AUTO.
 */
enum class ModuleVisionMode(val label: String) {
    AUTO         ("Auto — best available"),
    LLM_ONLY     ("Text LLM only"),
    VISION_ONLY  ("Vision model only"),
    BOTH         ("LLM + Vision (chained)");

    companion object {
        fun fromName(name: String?): ModuleVisionMode =
            entries.firstOrNull { it.name == name } ?: AUTO
    }
}
