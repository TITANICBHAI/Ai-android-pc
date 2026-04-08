package com.ariaagent.mobile.core.ai

/**
 * ModelCatalog — single source of truth for all downloadable LLM/VLM models.
 *
 * Two categories — both work with ARIA's training pipeline:
 *
 *   MULTIMODAL (vision + text): single model handles screen reading and reasoning.
 *     mmproj field is set; loaded via LlamaEngine.loadUnified().
 *
 *   TEXT-ONLY: pure reasoning model; screen reading delegated to SmolVLM helper
 *     if downloaded, otherwise falls back to accessibility tree + OCR.
 *     mmprojFilename / mmprojUrl are null; loaded via LlamaEngine.load().
 *
 * Training data is fully shared across model types: ExperienceStore records
 * screen summaries as plain text regardless of how they were produced, so
 * a text LLM can train on vision-enriched experiences and vice versa.
 *
 * Model            │ Type        │ RAM    │ Source repo
 * ─────────────────┼─────────────┼────────┼──────────────────────────────────
 * SmolVLM 256M     │ VLM         │ <1 GB  │ ggml-org/SmolVLM-256M-Instruct-GGUF
 * SmolVLM 500M     │ VLM         │ ~1.2 GB│ Mungert/SmolVLM-500M-Instruct-GGUF
 * Moondream2       │ VLM         │ ~2 GB  │ cjpais/moondream2-llamafile
 * Qwen2.5-VL 3B   │ VLM         │ ~3 GB  │ ggml-org/Qwen2.5-VL-3B-Instruct-GGUF
 * MiniCPM-V 2.6   │ VLM Q4_0   │ ~5.5 GB│ openbmb/MiniCPM-V-2_6-gguf
 * Llama 3.2 V 11B │ VLM ⚠heavy  │ 8 GB+  │ ggml-org/Llama-3.2-11B-Vision-...
 * Llama 3.2 1B    │ Text        │ ~1.2 GB│ bartowski/Llama-3.2-1B-Instruct-GGUF
 * Gemma 3 1B      │ Text        │ ~1.1 GB│ ggml-org/gemma-3-1b-it-GGUF
 * Qwen2.5 1.5B    │ Text        │ ~1.5 GB│ ggml-org/Qwen2.5-1.5B-Instruct-GGUF
 * Llama 3.2 3B    │ Text        │ ~2.5 GB│ bartowski/Llama-3.2-3B-Instruct-GGUF
 * Gemma 3 4B      │ Text        │ ~3 GB  │ ggml-org/gemma-3-4b-it-GGUF
 *
 * DEFAULT_ID is smolvlm-256m — fastest, lowest-RAM, works on any Android phone.
 */
data class CatalogModel(
    /** Stable identifier stored in SharedPreferences. Never change once shipped. */
    val id: String,
    val displayName: String,
    val description: String,
    /** Filename written to the models directory on device. */
    val filename: String,
    /** Direct HTTPS download URL for the GGUF (follows HF redirects automatically). */
    val url: String,
    /** Minimum expected byte count — used to confirm download is complete. */
    val expectedSizeBytes: Long,
    /** CLIP projection filename required for vision inference (null = text-only model). */
    val mmprojFilename: String? = null,
    /** Download URL for the mmproj GGUF. Null only if model has no vision head. */
    val mmprojUrl: String? = null,
    /** Approximate combined on-disk size in MB shown in the UI. */
    val displaySizeMb: Int = (expectedSizeBytes / 1_048_576L).toInt(),
    /**
     * When true the model is shown in the catalog but flagged with a warning.
     * Reasons: too large for 6 GB RAM, unstable at mobile quant, etc.
     * The user can still download and activate it — this is advisory only.
     */
    val notRecommended: Boolean = false,
) {
    /** True when this model has no vision projector and runs in text-only mode. */
    val isTextOnly: Boolean get() = mmprojFilename == null && mmprojUrl == null
}

object ModelCatalog {

    // ── SmolVLM 256M (smallest VLM — default model) ───────────────────────────
    // Repo:    ggml-org/SmolVLM-256M-Instruct-GGUF
    //          ggml-org/SmolVLM-256M-Instruct-GGUF (mmproj)
    // Params:  ~256M  │  Q8_0 ≈ 163 MB + ~50 MB mmproj  (Q4_K_M not available)
    // RAM:     under 1 GB — works on any Android phone

    val SMOLVLM_256M = CatalogModel(
        id                = "smolvlm-256m",
        displayName       = "SmolVLM 256M",
        description       = "World's smallest multimodal model. Understands images and text with under 1 GB RAM. Ideal default for any Android phone.",
        filename          = "smolvlm-256m-q8_0.gguf",
        url               = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/" +
                            "SmolVLM-256M-Instruct-Q8_0.gguf",
        expectedSizeBytes = 155_000_000L,
        displaySizeMb     = 220,
        mmprojFilename    = "smolvlm-256m-mmproj-f16.gguf",
        mmprojUrl         = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/" +
                            "mmproj-SmolVLM-256M-Instruct-f16.gguf",
    )

    // ── SmolVLM 500M (mid-range VLM) ──────────────────────────────────────────
    // Repo:    Mungert/SmolVLM-500M-Instruct-GGUF   (Q4_K_M base)
    //          ggml-org/SmolVLM-500M-Instruct-GGUF  (mmproj F16)
    // Params:  ~500M  │  Q4_K_M ≈ 289 MB + 199 MB mmproj
    // RAM:     ~1.2 GB

    val SMOLVLM_500M = CatalogModel(
        id                = "smolvlm-500m",
        displayName       = "SmolVLM 500M",
        description       = "500M multimodal model. Better vision understanding than 256M with still very low RAM — great for mid-range devices.",
        filename          = "smolvlm-500m-q4_k_m.gguf",
        url               = "https://huggingface.co/Mungert/SmolVLM-500M-Instruct-GGUF/resolve/main/" +
                            "SmolVLM-500M-Instruct-q4_k_m.gguf",
        expectedSizeBytes = 280_000_000L,
        displaySizeMb     = 475,
        mmprojFilename    = "smolvlm-500m-mmproj-f16.gguf",
        mmprojUrl         = "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/" +
                            "mmproj-SmolVLM-500M-Instruct-f16.gguf",
    )

    // ── Moondream2 (compact VLM with strong image understanding) ──────────────
    // Repo:    cjpais/moondream2-llamafile (pure GGUF files, not the llamafile)
    // Params:  ~1.86B │  Q5_K ≈ 1.06 GB + 910 MB mmproj
    // RAM:     ~2.0 GB — fits Galaxy M31 (6 GB RAM)
    // Note:    No Q4_K_M exists for Moondream2; Q5_K is the smallest available quant.

    val MOONDREAM2 = CatalogModel(
        id                = "moondream2",
        displayName       = "Moondream2",
        description       = "Compact 1.86B vision-language model. Excellent at describing screens and answering questions about images with low RAM usage.",
        filename          = "moondream2-q5k.gguf",
        url               = "https://huggingface.co/cjpais/moondream2-llamafile/resolve/main/" +
                            "moondream2-050824-q5k.gguf",
        expectedSizeBytes = 1_000_000_000L,
        displaySizeMb     = 1900,
        mmprojFilename    = "moondream2-mmproj-f16.gguf",
        mmprojUrl         = "https://huggingface.co/cjpais/moondream2-llamafile/resolve/main/" +
                            "moondream2-mmproj-050824-f16.gguf",
    )

    // ── Qwen2.5-VL 3B (most capable on-device VLM) ────────────────────────────
    // Repo:    ggml-org/Qwen2.5-VL-3B-Instruct-GGUF   (Q4_K_M base)
    //          Mungert/Qwen2.5-VL-3B-Instruct-GGUF    (mmproj F16)
    // Params:  ~3B    │  Q4_K_M ≈ 1.93 GB + ~500 MB mmproj
    // RAM:     ~3 GB — requires 6 GB+ device (Galaxy M31 minimum)

    val QWEN25_VL_3B = CatalogModel(
        id                = "qwen2.5-vl-3b",
        displayName       = "Qwen2.5-VL 3B",
        description       = "Most powerful on-device vision model. Strong reasoning, document understanding, and screen interaction. Needs 6 GB+ RAM.",
        filename          = "qwen2.5-vl-3b-q4_k_m.gguf",
        url               = "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/" +
                            "qwen2.5-vl-3b-instruct-q4_k_m.gguf",
        expectedSizeBytes = 1_900_000_000L,
        displaySizeMb     = 2300,
        mmprojFilename    = "qwen2.5-vl-3b-mmproj-f16.gguf",
        mmprojUrl         = "https://huggingface.co/Mungert/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/" +
                            "Qwen2.5-VL-3B-Instruct-mmproj-f16.gguf",
    )

    // ── Llama 3.2 1B Instruct (text-only, fastest text LLM) ───────────────────
    // Repo:    bartowski/Llama-3.2-1B-Instruct-GGUF
    // Params:  ~1B    │  Q4_K_M ≈ 773 MB
    // RAM:     ~1.2 GB — runs on any Android phone, very fast inference
    // Note:    Text-only. SmolVLM helper handles vision if downloaded.

    val LLAMA32_1B = CatalogModel(
        id                = "llama3.2-1b",
        displayName       = "Llama 3.2 1B",
        description       = "Meta's fastest on-device text model. Excellent at instruction following and action planning. Pairs automatically with SmolVLM for screen vision. ~773 MB.",
        filename          = "llama-3.2-1b-instruct-q4_k_m.gguf",
        url               = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/" +
                            "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        expectedSizeBytes = 700_000_000L,
        displaySizeMb     = 773,
        // mmprojFilename / mmprojUrl intentionally null — text-only model
    )

    // ── Llama 3.2 3B Instruct (text-only, best text quality) ──────────────────
    // Repo:    bartowski/Llama-3.2-3B-Instruct-GGUF
    // Params:  ~3B    │  Q4_K_M ≈ 2.0 GB
    // RAM:     ~2.5 GB — comfortable on 6 GB devices

    val LLAMA32_3B = CatalogModel(
        id                = "llama3.2-3b",
        displayName       = "Llama 3.2 3B",
        description       = "Meta's strongest on-device text model. Better reasoning and fewer hallucinations than 1B. Pairs automatically with SmolVLM for screen vision. ~2.0 GB.",
        filename          = "llama-3.2-3b-instruct-q4_k_m.gguf",
        url               = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/" +
                            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        expectedSizeBytes = 1_900_000_000L,
        displaySizeMb     = 2020,
    )

    // ── Gemma 3 1B Instruct (text-only, Google) ───────────────────────────────
    // Repo:    ggml-org/gemma-3-1b-it-GGUF
    // Params:  ~1B    │  Q4_K_M ≈ 670 MB
    // RAM:     ~1.1 GB — fastest possible inference, great for older phones

    val GEMMA3_1B = CatalogModel(
        id                = "gemma3-1b",
        displayName       = "Gemma 3 1B",
        description       = "Google's ultra-fast 1B instruction model. Excellent at structured JSON output — ideal for ARIA's action format. Text-only, pairs with SmolVLM. ~670 MB.",
        filename          = "gemma-3-1b-it-q4_k_m.gguf",
        url               = "https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/" +
                            "gemma-3-1b-it-Q4_K_M.gguf",
        expectedSizeBytes = 600_000_000L,
        displaySizeMb     = 670,
    )

    // ── Gemma 3 4B Instruct (text-only, Google) ───────────────────────────────
    // Repo:    ggml-org/gemma-3-4b-it-GGUF
    // Params:  ~4B    │  Q4_K_M ≈ 2.5 GB
    // RAM:     ~3.0 GB — fits 6 GB devices, slightly tight on concurrent app use

    val GEMMA3_4B = CatalogModel(
        id                = "gemma3-4b",
        displayName       = "Gemma 3 4B",
        description       = "Google's dense 4B model — very high quality reasoning for its size. Text-only, pairs with SmolVLM for vision. ~2.5 GB. May feel tight on 6 GB phones during heavy app use.",
        filename          = "gemma-3-4b-it-q4_k_m.gguf",
        url               = "https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/" +
                            "gemma-3-4b-it-Q4_K_M.gguf",
        expectedSizeBytes = 2_200_000_000L,
        displaySizeMb     = 2500,
    )

    // ── Qwen2.5 1.5B Instruct (text-only, Alibaba) ────────────────────────────
    // Repo:    ggml-org/Qwen2.5-1.5B-Instruct-GGUF
    // Params:  ~1.5B  │  Q4_K_M ≈ 970 MB
    // RAM:     ~1.5 GB — fast and very capable at instruction following

    val QWEN25_1B5 = CatalogModel(
        id                = "qwen2.5-1.5b",
        displayName       = "Qwen2.5 1.5B",
        description       = "Alibaba's 1.5B instruction model — surprisingly capable at tool-use and structured output. Text-only, pairs with SmolVLM. ~970 MB.",
        filename          = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        url               = "https://huggingface.co/ggml-org/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/" +
                            "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        expectedSizeBytes = 850_000_000L,
        displaySizeMb     = 970,
    )

    // ── MiniCPM-V 2.6 Q4_0 (multimodal, 8B — best vision on mobile) ──────────
    // Repo:    openbmb/MiniCPM-V-2_6-gguf
    // Params:  ~8B    │  Q4_0 ≈ 4.5 GB base + ~450 MB mmproj
    // RAM:     ~5.5 GB — very tight on 6 GB phones, use Q4_0 only
    // Note:    Rivals GPT-4V on some vision tasks. Use Q4_0 (not Q4_K_M) for stability.

    val MINICPM_V26 = CatalogModel(
        id                = "minicpm-v2.6",
        displayName       = "MiniCPM-V 2.6",
        description       = "8B multimodal powerhouse — rivals GPT-4V on vision benchmarks. Q4_0 quantization for stability on 6 GB phones. Excellent at reading dense UI and complex screens. ~5.0 GB total.",
        filename          = "minicpm-v2.6-q4_0.gguf",
        url               = "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/" +
                            "ggml-model-Q4_0.gguf",
        expectedSizeBytes = 4_000_000_000L,
        displaySizeMb     = 5000,
        mmprojFilename    = "minicpm-v2.6-mmproj-f16.gguf",
        mmprojUrl         = "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/" +
                            "mmproj-model-f16.gguf",
    )

    // ── Llama 3.2 Vision 11B (NOT RECOMMENDED for 6 GB phones) ───────────────
    // Repo:    ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF
    // Params:  ~11B   │  Q4_K_M ≈ 7.0 GB base — exceeds 6 GB phone RAM
    // RAM:     8 GB+ required — will crash or run at <1 token/sec on 6 GB devices

    val LLAMA32_VISION_11B = CatalogModel(
        id                = "llama3.2-vision-11b",
        displayName       = "Llama 3.2 Vision 11B",
        description       = "⚠ NOT RECOMMENDED for 6 GB phones. 11B parameters require 8 GB+ RAM — will crash or run below 1 token/sec on most Android devices. Use Qwen2.5-VL-3B or MiniCPM-V 2.6 instead.",
        filename          = "llama-3.2-11b-vision-instruct-q4_k_m.gguf",
        url               = "https://huggingface.co/ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/" +
                            "Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf",
        expectedSizeBytes = 6_500_000_000L,
        displaySizeMb     = 8000,
        mmprojFilename    = "llama-3.2-11b-vision-mmproj-f16.gguf",
        mmprojUrl         = "https://huggingface.co/ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/" +
                            "mmproj-Llama-3.2-11B-Vision-Instruct-f16.gguf",
        notRecommended    = true,
    )

    // ── SmolLM2 135M (ultra-lightweight text — fastest possible) ──────────────
    // Repo:    bartowski/SmolLM2-135M-Instruct-GGUF
    // Params:  135M  │  Q8_0 ≈ 145 MB
    // RAM:     <400 MB — runs on anything, ideal specialist for quick routing/planning

    val SMOLLM2_135M = CatalogModel(
        id                = "smollm2-135m",
        displayName       = "SmolLM2 135M",
        description       = "World's fastest on-device text model. 135M parameters, under 400 MB RAM. Ideal as a planner or router model paired with a larger reasoning model.",
        filename          = "smollm2-135m-q8_0.gguf",
        url               = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/" +
                            "SmolLM2-135M-Instruct-Q8_0.gguf",
        expectedSizeBytes = 140_000_000L,
        displaySizeMb     = 145,
    )

    // ── SmolLM2 360M (fast text specialist) ──────────────────────────────────
    // Repo:    bartowski/SmolLM2-360M-Instruct-GGUF
    // Params:  360M  │  Q4_K_M ≈ 250 MB
    // RAM:     ~600 MB — great chat / routing model at near-zero RAM cost

    val SMOLLM2_360M = CatalogModel(
        id                = "smollm2-360m",
        displayName       = "SmolLM2 360M",
        description       = "360M text model — excellent response quality for its size. Great as a Chat or Tool-Use specialist leaving RAM for a larger vision model. ~250 MB.",
        filename          = "smollm2-360m-q4_k_m.gguf",
        url               = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/" +
                            "SmolLM2-360M-Instruct-Q4_K_M.gguf",
        expectedSizeBytes = 240_000_000L,
        displaySizeMb     = 250,
    )

    // ── SmolLM2 1.7B (strong compact text model) ─────────────────────────────
    // Repo:    bartowski/SmolLM2-1.7B-Instruct-GGUF
    // Params:  1.7B  │  Q4_K_M ≈ 1.06 GB
    // RAM:     ~1.3 GB — punches well above its weight class

    val SMOLLM2_1B7 = CatalogModel(
        id                = "smollm2-1.7b",
        displayName       = "SmolLM2 1.7B",
        description       = "HuggingFace's best 1.7B model. Remarkably strong reasoning and instruction following for its size. Pairs with SmolVLM for full vision coverage. ~1.06 GB.",
        filename          = "smollm2-1.7b-q4_k_m.gguf",
        url               = "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/" +
                            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        expectedSizeBytes = 1_000_000_000L,
        displaySizeMb     = 1060,
    )

    // ── Phi-3.5 Mini 3.8B (Microsoft — best tool-use on mobile) ─────────────
    // Repo:    bartowski/Phi-3.5-mini-instruct-GGUF
    // Params:  3.8B  │  Q4_K_M ≈ 2.2 GB
    // RAM:     ~2.8 GB — excellent at structured JSON output and function calling

    val PHI35_MINI = CatalogModel(
        id                = "phi3.5-mini",
        displayName       = "Phi-3.5 Mini 3.8B",
        description       = "Microsoft's top tool-use model. Best-in-class structured JSON output and function calling for action agents. Assign Tool Use role for best results. ~2.2 GB.",
        filename          = "phi-3.5-mini-instruct-q4_k_m.gguf",
        url               = "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/" +
                            "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        expectedSizeBytes = 2_100_000_000L,
        displaySizeMb     = 2200,
    )

    // ── Qwen2.5 0.5B (ultra-small text) ──────────────────────────────────────
    // Repo:    ggml-org/Qwen2.5-0.5B-Instruct-GGUF
    // Params:  0.5B  │  Q8_0 ≈ 530 MB
    // RAM:     ~700 MB — fastest Qwen; use as planner/router

    val QWEN25_0B5 = CatalogModel(
        id                = "qwen2.5-0.5b",
        displayName       = "Qwen2.5 0.5B",
        description       = "Alibaba's smallest instruction model. Surprisingly capable at short tasks and routing. Use as a Planning or Chat specialist — leaves most RAM for vision. ~530 MB.",
        filename          = "qwen2.5-0.5b-instruct-q8_0.gguf",
        url               = "https://huggingface.co/ggml-org/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/" +
                            "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
        expectedSizeBytes = 500_000_000L,
        displaySizeMb     = 530,
    )

    // ── Qwen2.5 3B Instruct (strong mid-range text) ───────────────────────────
    // Repo:    ggml-org/Qwen2.5-3B-Instruct-GGUF
    // Params:  3B    │  Q4_K_M ≈ 2.0 GB
    // RAM:     ~2.5 GB — strong reasoning and tool use, fits M31

    val QWEN25_3B = CatalogModel(
        id                = "qwen2.5-3b",
        displayName       = "Qwen2.5 3B",
        description       = "Alibaba's 3B model — strong structured output, tool use, and multilingual support. Excellent Reasoning or Tool-Use model on 6 GB devices. ~2.0 GB.",
        filename          = "qwen2.5-3b-instruct-q4_k_m.gguf",
        url               = "https://huggingface.co/ggml-org/Qwen2.5-3B-Instruct-GGUF/resolve/main/" +
                            "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        expectedSizeBytes = 1_900_000_000L,
        displaySizeMb     = 2000,
    )

    // ── Mistral 7B Instruct v0.3 (text-only, high quality) ───────────────────
    // Repo:    bartowski/Mistral-7B-Instruct-v0.3-GGUF
    // Params:  7B    │  Q4_K_M ≈ 4.1 GB
    // RAM:     ~5.0 GB — tight on 6 GB phones; use alone without other heavy models

    val MISTRAL_7B = CatalogModel(
        id                = "mistral-7b",
        displayName       = "Mistral 7B v0.3",
        description       = "Flagship 7B text model — top-tier reasoning and instruction following. Very tight on 6 GB phones; works best as the sole loaded model. ~4.1 GB.",
        filename          = "mistral-7b-instruct-v0.3-q4_k_m.gguf",
        url               = "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/" +
                            "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        expectedSizeBytes = 4_000_000_000L,
        displaySizeMb     = 4100,
        notRecommended    = true,
    )

    // ── InternVL3.5 2B (multimodal — strong vision + text, upgraded from InternVL2) ──────
    // Repo:    bartowski/OpenGVLab_InternVL3_5-2B-GGUF
    // Params:  2B    │  Q4_K_M ≈ 1.28 GB base + ~636 MB mmproj
    // RAM:     ~2.1 GB — excellent document/chart/UI understanding, improved over InternVL2
    // Note:    InternVL2-2B GGUF (ggml-org) repo was removed; this is the maintained successor.

    val INTERNVL2_2B = CatalogModel(
        id                = "internvl2-2b",
        displayName       = "InternVL3.5 2B",
        description       = "Shanghai AI Lab's upgraded 2B multimodal model. Excellent at chart, document, and complex UI understanding. Strong alternative to Moondream2 with higher accuracy. ~1.9 GB total.",
        filename          = "internvl3.5-2b-q4_k_m.gguf",
        url               = "https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-2B-GGUF/resolve/main/" +
                            "OpenGVLab_InternVL3_5-2B-Q4_K_M.gguf",
        expectedSizeBytes = 1_200_000_000L,
        displaySizeMb     = 1900,
        mmprojFilename    = "internvl3.5-2b-mmproj-f16.gguf",
        mmprojUrl         = "https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-2B-GGUF/resolve/main/" +
                            "mmproj-OpenGVLab_InternVL3_5-2B-f16.gguf",
    )

    // ── Registry ──────────────────────────────────────────────────────────────

    val ALL: List<CatalogModel> = listOf(
        // ── Multimodal (vision + text) ────────────────────────────────────────
        SMOLVLM_256M,           // 256M  — default, any phone
        SMOLVLM_500M,           // 500M  — better vision, still low RAM
        MOONDREAM2,             // 1.86B — compact VLM
        INTERNVL2_2B,           // 2B    — strong chart/document vision
        QWEN25_VL_3B,           // 3B    — best vision/text balance
        MINICPM_V26,            // 8B    — GPT-4V quality, tight on 6 GB
        LLAMA32_VISION_11B,     // 11B   — not recommended for 6 GB phones
        // ── Text-only (pair with SmolVLM for screen understanding) ────────────
        SMOLLM2_135M,           // 135M  — ultra-fast planner/router
        SMOLLM2_360M,           // 360M  — fast chat/tool specialist
        SMOLLM2_1B7,            // 1.7B  — strong compact text
        GEMMA3_1B,              // 1B    — Google, great structured output
        QWEN25_0B5,             // 0.5B  — ultra-small Alibaba
        QWEN25_1B5,             // 1.5B  — Alibaba, strong tool-use
        LLAMA32_1B,             // 1B    — fastest Llama text LLM
        LLAMA32_3B,             // 3B    — best Llama text quality / RAM balance
        QWEN25_3B,              // 3B    — Alibaba mid-range
        PHI35_MINI,             // 3.8B  — Microsoft, best tool-use
        GEMMA3_4B,              // 4B    — dense, high quality, ~3 GB RAM
        MISTRAL_7B,             // 7B    — flagship text, tight on 6 GB
    )

    /** Default is the smallest multimodal model — works on any device, fast first-boot. */
    const val DEFAULT_ID = ""  // no default — user must explicitly select a model

    fun findById(id: String): CatalogModel? = ALL.find { it.id == id }
}
