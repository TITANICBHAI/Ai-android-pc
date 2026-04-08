@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.ariaagent.mobile.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Label
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.ariaagent.mobile.core.ai.ModelCatalog
import com.ariaagent.mobile.core.ai.ModelDispatcher
import com.ariaagent.mobile.core.ai.ModelManager
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel
import com.ariaagent.mobile.ui.viewmodel.AppSkillItem
import com.ariaagent.mobile.ui.viewmodel.LlmRole
import com.ariaagent.mobile.ui.viewmodel.LoadedLlmEntry
import com.ariaagent.mobile.ui.theme.ARIAColors

/**
 * ModulesScreen — hardware/software module status panel.
 *
 * Shows:
 *  • LLM (Llama 3.2-1B Q4_K_M)
 *  • OCR (ML Kit)
 *  • Object Detector (EfficientDet-Lite0 INT8)
 *  • Vector Memory (ONNX MiniLM)
 *  • Object Label Store
 *  • Permissions (Accessibility + Screen Capture)
 *  • On-device Learning (LoRA version, policy steps, Adam optimizer metrics)
 *  • [Phase 15] App Skills — per-app success rates and learned elements
 *
 * Phase 11 — pure Compose. Phase 15 update: app skills + RL metrics.
 */
@Composable
fun ModulesScreen(
    vm: AgentViewModel = viewModel(),
    onRequestScreenCapture: () -> Unit = {},
    onGrantAccessibility: () -> Unit = {},
) {
    val context              = LocalContext.current
    val modules              by vm.moduleState.collectAsStateWithLifecycle()
    val learning             by vm.learningState.collectAsStateWithLifecycle()
    val appSkills            by vm.appSkills.collectAsStateWithLifecycle()
    val llmDownloading       by vm.llmDownloading.collectAsStateWithLifecycle()
    val loadedLlms           by vm.loadedLlms.collectAsStateWithLifecycle()
    val detectorDownloading  by vm.detectorDownloading.collectAsStateWithLifecycle()
    val embeddingDownloading by vm.embeddingDownloading.collectAsStateWithLifecycle()
    val visionDownloading    by vm.visionDownloading.collectAsStateWithLifecycle()
    val visionLoading        by vm.visionLoading.collectAsStateWithLifecycle()
    val sam2Downloading      by vm.sam2Downloading.collectAsStateWithLifecycle()
    val sam2Loading          by vm.sam2Loading.collectAsStateWithLifecycle()

    // Derived vision-auto-support status
    val anyLoadedHasVision = loadedLlms.values.any { entry ->
        ModelCatalog.findById(entry.modelId)?.isTextOnly == false
    }
    val allLoadedAreTextOnly = loadedLlms.values.isNotEmpty() && !anyLoadedHasVision
    val visionAutoActive = allLoadedAreTextOnly && modules.visionLoaded

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "MODULES",
                style = MaterialTheme.typography.headlineMedium.copy(
                    color = ARIAColors.Primary,
                    fontWeight = FontWeight.Bold
                )
            )
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                IconButton(onClick = { vm.refreshAppSkills() }) {
                    Icon(Icons.Default.Psychology, contentDescription = "Refresh skills",
                        tint = ARIAColors.Muted, modifier = Modifier.size(20.dp))
                }
                IconButton(onClick = { vm.refreshModuleState() }) {
                    Icon(Icons.Default.Refresh, contentDescription = "Refresh modules",
                        tint = ARIAColors.Muted)
                }
            }
        }

        // ── MODEL CATALOG ─────────────────────────────────────────────────────
        ARIACard {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        Icons.Default.Storage,
                        contentDescription = null,
                        tint     = ARIAColors.Primary,
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        "MODEL CATALOG",
                        style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                    )
                    if (loadedLlms.isNotEmpty()) {
                        val loadedCount = loadedLlms.values.count { it.isLoaded }
                        Text(
                            "$loadedCount/${loadedLlms.size} loaded",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Primary,
                                fontWeight = FontWeight.Bold
                            )
                        )
                    }
                }
                IconButton(onClick = { vm.refreshModuleState() }) {
                    Icon(Icons.Default.Refresh, contentDescription = "Refresh",
                        tint = ARIAColors.Muted, modifier = Modifier.size(18.dp))
                }
            }

            // Vision auto-support banner
            if (visionAutoActive) {
                Spacer(Modifier.height(8.dp))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(ARIAColors.Accent.copy(alpha = 0.12f))
                        .padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(Icons.Default.RemoveRedEye, contentDescription = null,
                        tint = ARIAColors.Accent, modifier = Modifier.size(14.dp))
                    Text(
                        "Vision auto-support active — SmolVLM handling screen reading for text-only model(s)",
                        style = MaterialTheme.typography.bodySmall.copy(
                            color = ARIAColors.Accent, fontSize = 11.sp
                        )
                    )
                }
            } else if (allLoadedAreTextOnly && !modules.visionLoaded) {
                Spacer(Modifier.height(8.dp))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(ARIAColors.Warning.copy(alpha = 0.10f))
                        .padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(Icons.Default.Warning, contentDescription = null,
                        tint = ARIAColors.Warning, modifier = Modifier.size(14.dp))
                    Column {
                        Text(
                            "Text-only models loaded — no vision coverage",
                            style = MaterialTheme.typography.bodySmall.copy(
                                color = ARIAColors.Warning, fontWeight = FontWeight.SemiBold, fontSize = 11.sp
                            )
                        )
                        Text(
                            "Download SmolVLM 256M below to auto-activate screen reading",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Muted, fontSize = 10.sp
                            )
                        )
                    }
                }
            }

            // ── Multi-model pipeline topology banner ───────────────────────────
            // Shown when 2+ slots are registered (loaded or not) so the user can
            // see how models are wired before starting the agent.
            val registeredCount = loadedLlms.size
            val loadedActiveCount = loadedLlms.values.count { it.isLoaded }
            if (registeredCount >= 2) {
                Spacer(Modifier.height(8.dp))
                val topologyLines = remember(loadedLlms) {
                    ModelDispatcher.topologySummary(loadedLlms).split("\n")
                }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(ARIAColors.Accent.copy(alpha = 0.08f))
                        .padding(horizontal = 10.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.Top,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        Icons.Default.AccountTree,
                        contentDescription = null,
                        tint = ARIAColors.Accent,
                        modifier = Modifier.size(14.dp).padding(top = 1.dp)
                    )
                    Column {
                        Text(
                            "MULTI-MODEL PIPELINE" +
                                if (loadedActiveCount < registeredCount) "  (${loadedActiveCount}/${registeredCount} loaded)" else "",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Accent,
                                fontWeight = FontWeight.Bold,
                                fontSize = 10.sp
                            )
                        )
                        Spacer(Modifier.height(3.dp))
                        topologyLines.forEach { line ->
                            Text(
                                line,
                                style = MaterialTheme.typography.labelSmall.copy(
                                    color = ARIAColors.TextSecondary,
                                    fontSize = 10.sp
                                )
                            )
                        }
                    }
                }
            }
        }

        // ── Per-model catalog cards ───────────────────────────────────────────
        val totalLoadedCount = loadedLlms.values.count { it.isLoaded }
        ModelCatalog.ALL.forEach { catalogEntry ->
            val slotEntry   = loadedLlms[catalogEntry.id]
            val isDownloaded = ModelManager.isModelDownloaded(context, catalogEntry.id)
            val isActiveRam  = modules.modelLoaded &&
                ModelManager.activeModelId(context) == catalogEntry.id
            CatalogModelCard(
                catalog      = catalogEntry,
                slotEntry    = slotEntry,
                isDownloaded = isDownloaded,
                isLoaded     = isActiveRam || (slotEntry?.isLoaded == true),
                isLoading    = slotEntry?.isLoading == true,
                totalLoadedCount   = totalLoadedCount,
                llmDownloading     = llmDownloading && slotEntry?.isDownloading == true,
                llmDownloadPercent = if (llmDownloading && ModelManager.activeModelId(context) == catalogEntry.id)
                    modules.llmDownloadPercent else 0,
                downloadError = slotEntry?.loadError
                    ?: if (ModelManager.activeModelId(context) == catalogEntry.id)
                        modules.llmDownloadError else null,
                onDownload   = if (!isDownloaded && !llmDownloading) {{
                    vm.downloadCatalogModel(catalogEntry.id)
                }} else null,
                onLoad       = if (isDownloaded && !isActiveRam && slotEntry?.isLoading != true) {{
                    vm.loadCatalogLlm(catalogEntry.id)
                }} else null,
                onUnload     = if (isActiveRam || slotEntry?.isLoaded == true) {{
                    vm.unloadCatalogLlm(catalogEntry.id)
                }} else null,
                onRoleChange = { role -> vm.setLlmRole(catalogEntry.id, role) },
                onPromptChange = { prompt -> vm.setLlmSystemPrompt(catalogEntry.id, prompt) },
            )
        }

        // ── OCR ───────────────────────────────────────────────────────────────
        ModuleCard(
            icon = Icons.Default.TextFields,
            title = "ML Kit OCR",
            subtitle = "On-device text recognition  •  bundled",
            status = if (modules.ocrReady) ModuleStatus.READY else ModuleStatus.MISSING
        )

        // ── Object Detector ───────────────────────────────────────────────────
        val detectorStatus = if (modules.detectorReady) ModuleStatus.READY else ModuleStatus.MISSING
        ModuleCard(
            icon = Icons.Default.Visibility,
            title = "EfficientDet-Lite0 INT8",
            subtitle = "Screen object detection  •  ~4.4 MB",
            status = detectorStatus,
            detail = if (modules.detectorSizeMb > 0)
                "${String.format("%.1f", modules.detectorSizeMb)} MB downloaded" else "Not downloaded",
            onDownload = if (detectorStatus == ModuleStatus.MISSING) {{ vm.downloadDetectorModel() }} else null,
            downloading = detectorDownloading,
        )

        // ── Vision Model (Phase 17) ───────────────────────────────────────────
        val visionStatus = when {
            modules.visionLoaded  -> ModuleStatus.ACTIVE
            modules.visionReady   -> ModuleStatus.READY
            else                  -> ModuleStatus.MISSING
        }
        val visionDetail = when {
            modules.visionLoaded ->
                "SmolVLM-256M active  •  ${String.format("%.0f", modules.visionModelDownloadedMb + modules.mmProjDownloadedMb)} MB"
            modules.visionReady  ->
                "Model ready  •  ${String.format("%.0f", modules.visionModelDownloadedMb)} MB + ${String.format("%.0f", modules.mmProjDownloadedMb)} MB mmproj"
            visionDownloading && modules.visionDownloadPercent > 0 ->
                "${modules.visionDownloadPercent}%  •  base + mmproj downloading"
            else -> "~200 MB total  •  not downloaded"
        }
        ModuleCard(
            icon = Icons.Default.RemoveRedEye,
            title = "SmolVLM-256M Multimodal",
            subtitle = "Pixel-level screen understanding  •  ~200 MB",
            status = visionStatus,
            detail = visionDetail,
            downloadProgress = if (visionDownloading && modules.visionDownloadPercent > 0)
                modules.visionDownloadPercent / 100f else null,
            downloadError = modules.visionDownloadError,
            onDownload = if (visionStatus == ModuleStatus.MISSING && !visionDownloading)
                {{ vm.downloadVisionModel() }} else null,
            downloading = visionDownloading,
            onLoad = if (modules.visionReady && !modules.visionLoaded && !visionLoading && !visionDownloading)
                {{ vm.loadVisionModel() }} else null,
            loading = visionLoading,
            onUnload = if (modules.visionLoaded) {{ vm.unloadVisionModel() }} else null,
        )

        // ── SAM2 / MobileSAM pixel segmentation (Phase 18) ───────────────────
        val sam2Status = when {
            modules.sam2Loaded -> ModuleStatus.ACTIVE
            modules.sam2Ready  -> ModuleStatus.READY
            else               -> ModuleStatus.MISSING
        }
        val sam2Detail = when {
            modules.sam2Loaded ->
                "MobileSAM active  •  ${String.format("%.0f", modules.sam2DownloadedMb)} MB"
            modules.sam2Ready  ->
                "Encoder ready  •  ${String.format("%.0f", modules.sam2DownloadedMb)} MB"
            sam2Downloading && modules.sam2DownloadPercent > 0 ->
                "${modules.sam2DownloadPercent}%  •  encoder downloading"
            modules.sam2DownloadedMb > 0 ->
                "${String.format("%.0f", modules.sam2DownloadedMb)} MB  •  download incomplete"
            else -> "~38 MB  •  not downloaded"
        }
        ModuleCard(
            icon = Icons.Default.CropFree,
            title = "MobileSAM (ViT-Tiny)",
            subtitle = "Pixel segmentation for game / Flutter screens  •  ~38 MB",
            status = sam2Status,
            detail = sam2Detail,
            downloadProgress = if (sam2Downloading && modules.sam2DownloadPercent > 0)
                modules.sam2DownloadPercent / 100f else null,
            downloadError = modules.sam2DownloadError,
            onDownload = if (sam2Status == ModuleStatus.MISSING && !sam2Downloading)
                {{ vm.downloadSam2Model() }} else null,
            downloading = sam2Downloading,
            onLoad = if (modules.sam2Ready && !modules.sam2Loaded && !sam2Loading && !sam2Downloading)
                {{ vm.loadSam2Model() }} else null,
            loading = sam2Loading,
            onUnload = if (modules.sam2Loaded) {{ vm.unloadSam2Model() }} else null,
        )

        // ── Vector Memory / Embedding model ──────────────────────────────────
        val embStatus = when {
            modules.embeddingReady && modules.embeddingVocabReady -> ModuleStatus.READY
            else                                                   -> ModuleStatus.MISSING
        }
        ModuleCard(
            icon = Icons.Default.DataObject,
            title = "MiniLM Embedding (ONNX)",
            subtitle = "Semantic memory search  •  ~23 MB",
            status = embStatus,
            detail = when {
                embStatus == ModuleStatus.READY ->
                    "${modules.embeddingCount} experiences  •  ${modules.episodesRun} episodes run"
                modules.embeddingDownloadedMb > 0 ->
                    "${String.format("%.1f", modules.embeddingDownloadedMb)} MB downloaded"
                else -> "Model not downloaded — memory search disabled"
            },
            onDownload = if (embStatus == ModuleStatus.MISSING && !embeddingDownloading) {{ vm.downloadEmbeddingModel() }} else null,
            downloading = embeddingDownloading,
        )

        // ── Object Label Store ────────────────────────────────────────────────
        ModuleCard(
            icon = Icons.AutoMirrored.Filled.Label,
            title = "Object Label Store",
            subtitle = "Custom screen element labels",
            status = ModuleStatus.READY,
            detail = "${modules.labelCount} label${if (modules.labelCount == 1) "" else "s"} defined"
        )

        // ── Permissions ───────────────────────────────────────────────────────
        ARIACard {
            Text("PERMISSIONS", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
            Spacer(Modifier.height(8.dp))
            PermissionRow(
                icon    = Icons.Default.Accessibility,
                label   = "Accessibility Service",
                granted = modules.accessibilityGranted,
                onGrant = if (!modules.accessibilityGranted) onGrantAccessibility else null,
            )
            Spacer(Modifier.height(6.dp))
            PermissionRow(
                icon = Icons.Default.Screenshot,
                label = "Screen Capture",
                granted = modules.screenCaptureGranted,
                onGrant = if (!modules.screenCaptureGranted) onRequestScreenCapture else null,
            )
        }

        // ── On-device learning ────────────────────────────────────────────────
        ARIACard {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "ON-DEVICE LEARNING",
                    style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                )
                Icon(
                    Icons.Default.School,
                    contentDescription = null,
                    tint = ARIAColors.Accent,
                    modifier = Modifier.size(20.dp)
                )
            }
            Spacer(Modifier.height(10.dp))

            // Main version metrics
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                RlMetric("LoRA",     "v${learning.loraVersion}")
                RlMetric("Policy",   "v${learning.policyVersion}")
                RlMetric("Adapter",  if (modules.adapterLoaded) "LOADED" else "NONE")
                RlMetric("Samples",  "${learning.untrainedSamples}")
            }

            if (learning.adamStep > 0) {
                Spacer(Modifier.height(10.dp))
                HorizontalDivider(color = ARIAColors.Divider)
                Spacer(Modifier.height(8.dp))
                Text(
                    "REINFORCE OPTIMIZER",
                    style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                )
                Spacer(Modifier.height(6.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    RlMetric("Adam Step", "${learning.adamStep}")
                    if (learning.lastPolicyLoss > 0.0) {
                        RlMetric("Policy Loss", String.format("%.5f", learning.lastPolicyLoss))
                    }
                    RlMetric("Episodes", "${modules.episodesRun}")
                }
            }
        }

        // ── Phase 15: App Skills ──────────────────────────────────────────────
        ARIACard {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Default.AppRegistration,
                        contentDescription = null,
                        tint = ARIAColors.Accent,
                        modifier = Modifier.size(16.dp)
                    )
                    Text(
                        "APP SKILLS",
                        style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                    )
                    if (appSkills.isNotEmpty()) {
                        Text(
                            "(${appSkills.size})",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Accent,
                                fontWeight = FontWeight.Bold
                            )
                        )
                    }
                }
                if (appSkills.isNotEmpty()) {
                    TextButton(
                        onClick = { vm.clearAppSkills() },
                        colors = ButtonDefaults.textButtonColors(contentColor = ARIAColors.Error)
                    ) {
                        Text("Reset", style = MaterialTheme.typography.labelSmall)
                    }
                }
            }

            if (appSkills.isEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    "No app skills yet. ARIA learns per-app knowledge after each completed task.",
                    style = MaterialTheme.typography.bodySmall.copy(
                        color = ARIAColors.Muted,
                        lineHeight = 18.sp
                    )
                )
            } else {
                Spacer(Modifier.height(8.dp))
                appSkills.take(8).forEachIndexed { index, skill ->
                    AppSkillRow(skill = skill)
                    if (index < appSkills.size - 1 && index < 7) {
                        HorizontalDivider(
                            color = ARIAColors.Divider.copy(alpha = 0.4f),
                            modifier = Modifier.padding(vertical = 6.dp)
                        )
                    }
                }
                if (appSkills.size > 8) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "… and ${appSkills.size - 8} more apps",
                        style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                    )
                }
            }
        }
    }
}

// ─── Private composables ──────────────────────────────────────────────────────

private enum class ModuleStatus { ACTIVE, READY, MISSING }

// ─── Role chip colour map ─────────────────────────────────────────────────────

/**
 * Returns a distinct, semantically meaningful colour for each [LlmRole] chip.
 * Colours are chosen to be legible on the dark ARIA surface (≥ 4.5:1 contrast).
 *
 *  EVERYTHING      amber   — "golden", handles all tasks
 *  EVERYTHING_ELSE indigo  — fallback / catch-all
 *  VISION          cyan    — screen/image input
 *  REASONING       green   — decision making
 *  PLANNING        orange  — forward planning
 *  TOOL_USE        red     — action / gesture dispatch
 *  CHAT            pink    — conversation
 *  CUSTOM          muted   — user-defined
 */
private fun roleChipColor(role: LlmRole): Color = when (role) {
    LlmRole.EVERYTHING      -> Color(0xFFF59E0B)   // amber
    LlmRole.EVERYTHING_ELSE -> Color(0xFF818CF8)   // indigo-400
    LlmRole.VISION          -> Color(0xFF22D3EE)   // cyan-400
    LlmRole.REASONING       -> Color(0xFF34D399)   // emerald-400
    LlmRole.PLANNING        -> Color(0xFFFB923C)   // orange-400
    LlmRole.TOOL_USE        -> Color(0xFFF87171)   // red-400
    LlmRole.CHAT            -> Color(0xFFF472B6)   // pink-400
    LlmRole.CUSTOM          -> Color(0xFF94A3B8)   // slate-400 (muted)
}

// ─── CatalogModelCard ─────────────────────────────────────────────────────────

@Composable
private fun CatalogModelCard(
    catalog:            com.ariaagent.mobile.core.ai.CatalogModel,
    slotEntry:          LoadedLlmEntry?,
    isDownloaded:       Boolean,
    isLoaded:           Boolean,
    isLoading:          Boolean,
    totalLoadedCount:   Int,
    llmDownloading:     Boolean,
    llmDownloadPercent: Int,
    downloadError:      String?,
    onDownload:         (() -> Unit)?,
    onLoad:             (() -> Unit)?,
    onUnload:           (() -> Unit)?,
    onRoleChange:       (LlmRole) -> Unit,
    onPromptChange:     (String) -> Unit,
) {
    var expanded       by remember { mutableStateOf(isLoaded) }
    var editingPrompt  by remember { mutableStateOf(false) }
    var promptDraft    by remember(slotEntry?.systemPrompt) {
        mutableStateOf(slotEntry?.systemPrompt ?: "")
    }
    val currentRole    = slotEntry?.role ?: LlmRole.REASONING

    val cardBorderColor = when {
        isLoaded      -> ARIAColors.Primary.copy(alpha = 0.5f)
        isDownloaded  -> ARIAColors.Success.copy(alpha = 0.3f)
        else          -> ARIAColors.Divider
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, cardBorderColor, RoundedCornerShape(12.dp)),
        shape  = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isLoaded)
                ARIAColors.Primary.copy(alpha = 0.06f) else ARIAColors.Surface
        ),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {

            // ── Header row ────────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null,
                        onClick    = { expanded = !expanded }
                    ),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment     = Alignment.Top
            ) {
                Box(
                    Modifier
                        .size(40.dp)
                        .background(
                            if (isLoaded) ARIAColors.Primary.copy(alpha = 0.15f)
                            else if (isDownloaded) ARIAColors.Success.copy(alpha = 0.10f)
                            else ARIAColors.Divider.copy(alpha = 0.15f),
                            RoundedCornerShape(10.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        if (!catalog.isTextOnly) Icons.Default.RemoveRedEye
                        else Icons.Default.Psychology,
                        contentDescription = null,
                        tint = if (isLoaded) ARIAColors.Primary
                               else if (isDownloaded) ARIAColors.Success
                               else ARIAColors.Muted,
                        modifier = Modifier.size(22.dp)
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(
                            catalog.displayName,
                            style = MaterialTheme.typography.bodyMedium.copy(
                                color      = if (isLoaded) ARIAColors.Primary else ARIAColors.OnSurface,
                                fontWeight = FontWeight.SemiBold
                            )
                        )
                        if (isLoaded) {
                            Box(
                                Modifier
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(ARIAColors.Primary.copy(alpha = 0.15f))
                                    .padding(horizontal = 5.dp, vertical = 1.dp)
                            ) {
                                Text(
                                    "IN RAM",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        color = ARIAColors.Primary, fontSize = 8.sp
                                    )
                                )
                            }
                        }
                    }
                    Spacer(Modifier.height(2.dp))
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalArrangement   = Arrangement.spacedBy(2.dp)
                    ) {
                        Text(
                            "~${catalog.displaySizeMb} MB",
                            style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                        )
                        Text(
                            if (catalog.isTextOnly) "• Text-only" else "• Vision+Text",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = if (catalog.isTextOnly) ARIAColors.Primary else ARIAColors.Accent
                            )
                        )
                        if (isDownloaded) {
                            Text(
                                "• On device",
                                style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Success)
                            )
                        }
                        if (catalog.notRecommended) {
                            Box(
                                Modifier
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(ARIAColors.Warning.copy(alpha = 0.15f))
                                    .padding(horizontal = 5.dp, vertical = 1.dp)
                            ) {
                                Text(
                                    "⚠ 8 GB+",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        color = ARIAColors.Warning, fontWeight = FontWeight.Bold, fontSize = 9.sp
                                    )
                                )
                            }
                        }
                        if (isLoaded && slotEntry != null) {
                            val badgeColor = roleChipColor(currentRole)
                            Box(
                                Modifier
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(badgeColor.copy(alpha = 0.15f))
                                    .padding(horizontal = 5.dp, vertical = 1.dp)
                            ) {
                                Text(
                                    currentRole.label,
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        color = badgeColor, fontSize = 9.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                )
                            }
                        }
                    }
                }
                Icon(
                    if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    contentDescription = null,
                    tint     = ARIAColors.Muted,
                    modifier = Modifier.size(18.dp)
                )
            }

            // ── Download progress ─────────────────────────────────────────
            if (llmDownloading && llmDownloadPercent > 0) {
                Spacer(Modifier.height(8.dp))
                Text(
                    "$llmDownloadPercent%  downloading…",
                    style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Primary)
                )
                Spacer(Modifier.height(4.dp))
                LinearProgressIndicator(
                    progress  = { llmDownloadPercent / 100f },
                    modifier  = Modifier.fillMaxWidth(),
                    color     = ARIAColors.Primary,
                    trackColor = ARIAColors.Divider,
                )
            }
            downloadError?.let {
                Spacer(Modifier.height(4.dp))
                Text(it, style = MaterialTheme.typography.bodySmall.copy(
                    color = ARIAColors.Error, fontSize = 11.sp))
            }

            // ── Action buttons (always visible) ──────────────────────────
            Spacer(Modifier.height(10.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                when {
                    onDownload != null -> {
                        Button(
                            onClick = onDownload,
                            modifier = Modifier.weight(1f),
                            shape  = RoundedCornerShape(8.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = ARIAColors.Accent),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp)
                        ) {
                            Icon(Icons.Default.Download, null, modifier = Modifier.size(14.dp),
                                tint = ARIAColors.Background)
                            Spacer(Modifier.width(4.dp))
                            Text("Download", fontSize = 12.sp, color = ARIAColors.Background,
                                fontWeight = FontWeight.Bold)
                        }
                    }
                    onLoad != null -> {
                        Button(
                            onClick  = onLoad,
                            enabled  = !isLoading,
                            modifier = Modifier.weight(1f),
                            shape    = RoundedCornerShape(8.dp),
                            colors   = ButtonDefaults.buttonColors(
                                containerColor         = ARIAColors.Primary,
                                disabledContainerColor = ARIAColors.Primary.copy(alpha = 0.4f)
                            ),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp)
                        ) {
                            if (isLoading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(12.dp),
                                    strokeWidth = 2.dp,
                                    color = Color.White
                                )
                                Spacer(Modifier.width(4.dp))
                                Text("Loading…", fontSize = 12.sp, color = Color.White)
                            } else {
                                Icon(Icons.Default.Memory, null, modifier = Modifier.size(14.dp),
                                    tint = Color.White)
                                Spacer(Modifier.width(4.dp))
                                Text("Load", fontSize = 12.sp, color = Color.White,
                                    fontWeight = FontWeight.Bold)
                            }
                        }
                        if (onUnload != null) {
                            OutlinedButton(
                                onClick = onUnload,
                                shape  = RoundedCornerShape(8.dp),
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Warning),
                                border = androidx.compose.foundation.BorderStroke(
                                    1.dp, ARIAColors.Warning.copy(alpha = 0.5f)
                                ),
                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp)
                            ) {
                                Text("Unload", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    }
                    isLoaded && onUnload != null -> {
                        OutlinedButton(
                            onClick = onUnload,
                            modifier = Modifier.weight(1f),
                            shape  = RoundedCornerShape(8.dp),
                            colors = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Warning),
                            border = androidx.compose.foundation.BorderStroke(
                                1.dp, ARIAColors.Warning.copy(alpha = 0.5f)
                            ),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp)
                        ) {
                            Icon(Icons.Default.Memory, null, modifier = Modifier.size(14.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("Unload from RAM", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                        }
                    }
                    !isDownloaded -> {
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .clip(RoundedCornerShape(8.dp))
                                .background(ARIAColors.Divider.copy(alpha = 0.15f))
                                .padding(vertical = 8.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                "Not downloaded",
                                style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                            )
                        }
                    }
                }
            }

            // ── Expanded section: role + system prompt ────────────────────
            AnimatedVisibility(
                visible = expanded && isDownloaded,
                enter   = expandVertically(),
                exit    = shrinkVertically()
            ) {
                Column {
                    Spacer(Modifier.height(12.dp))
                    HorizontalDivider(color = ARIAColors.Divider)
                    Spacer(Modifier.height(10.dp))

                    // Role selector — only shown when 2+ models are loaded
                    if (totalLoadedCount <= 1 && isLoaded) {
                        // Single model: handles everything — no role assignment needed
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(ARIAColors.Primary.copy(alpha = 0.08f))
                                .padding(horizontal = 10.dp, vertical = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                Icons.Default.AllInclusive,
                                contentDescription = null,
                                tint = ARIAColors.Primary,
                                modifier = Modifier.size(14.dp)
                            )
                            Column {
                                Text(
                                    "Handles everything",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        color = ARIAColors.Primary,
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 11.sp
                                    )
                                )
                                Text(
                                    "Load a 2nd model to assign specific roles",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        color = ARIAColors.Muted, fontSize = 10.sp
                                    )
                                )
                            }
                        }
                    } else if (totalLoadedCount >= 2) {
                        Text(
                            "ROLE",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Muted, fontSize = 10.sp
                            )
                        )
                        Spacer(Modifier.height(6.dp))
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalArrangement   = Arrangement.spacedBy(6.dp)
                        ) {
                            LlmRole.entries.forEach { role ->
                                val selected  = currentRole == role
                                val roleColor = roleChipColor(role)
                                Surface(
                                    shape  = RoundedCornerShape(8.dp),
                                    color  = if (selected) roleColor.copy(alpha = 0.15f)
                                             else         ARIAColors.Divider.copy(alpha = 0.10f),
                                    border = androidx.compose.foundation.BorderStroke(
                                        if (selected) 1.5.dp else 1.dp,
                                        if (selected) roleColor else ARIAColors.Divider
                                    ),
                                    modifier = Modifier.clickable(
                                        interactionSource = remember { MutableInteractionSource() },
                                        indication = null,
                                        onClick    = { onRoleChange(role) }
                                    )
                                ) {
                                    Text(
                                        role.label,
                                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp),
                                        style    = MaterialTheme.typography.labelSmall.copy(
                                            color      = if (selected) roleColor else ARIAColors.Muted,
                                            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                                            fontSize   = 11.sp
                                        )
                                    )
                                }
                            }
                        }
                        Text(
                            currentRole.description,
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Muted, fontSize = 10.sp
                            ),
                            modifier = Modifier.padding(top = 3.dp)
                        )
                    }

                    Spacer(Modifier.height(10.dp))

                    // System prompt editor
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment     = Alignment.CenterVertically
                    ) {
                        Text(
                            "SYSTEM PROMPT",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Muted, fontSize = 10.sp
                            )
                        )
                        if (!editingPrompt) {
                            TextButton(
                                onClick        = { editingPrompt = true },
                                contentPadding = PaddingValues(horizontal = 6.dp, vertical = 0.dp)
                            ) {
                                Text(
                                    if (promptDraft.isBlank()) "Add prompt" else "Edit",
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        color = ARIAColors.Primary, fontSize = 11.sp
                                    )
                                )
                            }
                        }
                    }

                    if (!editingPrompt) {
                        if (promptDraft.isBlank()) {
                            Text(
                                "No custom prompt — using default for ${currentRole.label} role",
                                style = MaterialTheme.typography.bodySmall.copy(
                                    color = ARIAColors.Muted, fontSize = 11.sp
                                )
                            )
                        } else {
                            Text(
                                promptDraft,
                                style = MaterialTheme.typography.bodySmall.copy(
                                    color = ARIAColors.OnSurface, fontSize = 11.sp
                                ),
                                maxLines = 3
                            )
                        }
                    } else {
                        Spacer(Modifier.height(4.dp))
                        OutlinedTextField(
                            value         = promptDraft,
                            onValueChange = { promptDraft = it },
                            modifier      = Modifier.fillMaxWidth(),
                            placeholder   = {
                                Text(
                                    "e.g. You are ARIA's screen-reading assistant. Describe the UI elements…",
                                    style = MaterialTheme.typography.bodySmall.copy(
                                        color = ARIAColors.Divider, fontSize = 10.sp
                                    )
                                )
                            },
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor   = ARIAColors.Primary,
                                unfocusedBorderColor = ARIAColors.Divider,
                                focusedTextColor     = ARIAColors.OnSurface,
                                unfocusedTextColor   = ARIAColors.OnSurface,
                            ),
                            singleLine = false,
                            maxLines   = 6
                        )
                        Spacer(Modifier.height(6.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(
                                onClick = {
                                    onPromptChange(promptDraft)
                                    editingPrompt = false
                                },
                                shape  = RoundedCornerShape(8.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = ARIAColors.Primary),
                                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                            ) {
                                Text("Save", fontSize = 12.sp, fontWeight = FontWeight.Bold,
                                    color = ARIAColors.Background)
                            }
                            OutlinedButton(
                                onClick = {
                                    promptDraft   = slotEntry?.systemPrompt ?: ""
                                    editingPrompt = false
                                },
                                shape  = RoundedCornerShape(8.dp),
                                colors = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Muted),
                                border = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Divider),
                                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                            ) {
                                Text("Cancel", fontSize = 12.sp)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ModuleCard(
    icon: ImageVector,
    title: String,
    subtitle: String,
    status: ModuleStatus,
    detail: String? = null,
    onDownload: (() -> Unit)? = null,
    downloading: Boolean = false,
    downloadProgress: Float? = null,
    downloadError: String? = null,
    onLoad: (() -> Unit)? = null,
    loading: Boolean = false,
    onUnload: (() -> Unit)? = null,
) {
    val (statusColor, statusLabel) = when (status) {
        ModuleStatus.ACTIVE  -> ARIAColors.Primary to "ACTIVE"
        ModuleStatus.READY   -> ARIAColors.Success  to "READY"
        ModuleStatus.MISSING -> ARIAColors.Error     to "MISSING"
    }
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    Modifier
                        .size(40.dp)
                        .background(statusColor.copy(alpha = 0.12f), RoundedCornerShape(10.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(icon, contentDescription = null, tint = statusColor, modifier = Modifier.size(22.dp))
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(title, style = MaterialTheme.typography.bodyMedium.copy(
                        color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold))
                    Text(subtitle, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
                    if (detail != null) {
                        Text(detail, style = MaterialTheme.typography.bodySmall.copy(
                            color = statusColor, fontSize = 11.sp))
                    }
                }
                Text(
                    statusLabel,
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = statusColor, fontWeight = FontWeight.Bold, letterSpacing = 0.5.sp
                    )
                )
            }
            // Download progress bar (real % from ModelDownloadService)
            if (downloadProgress != null) {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress          = { downloadProgress },
                    modifier          = Modifier.fillMaxWidth(),
                    color             = ARIAColors.Primary,
                    trackColor        = ARIAColors.Divider,
                )
            }
            // Download error
            if (downloadError != null) {
                Spacer(Modifier.height(4.dp))
                Text(
                    downloadError,
                    style = MaterialTheme.typography.bodySmall.copy(
                        color = ARIAColors.Error, fontSize = 11.sp
                    )
                )
            }
            // Download button (shown only when MISSING)
            if (onDownload != null) {
                Spacer(Modifier.height(10.dp))
                Button(
                    onClick   = onDownload,
                    enabled   = !downloading,
                    modifier  = Modifier.fillMaxWidth(),
                    shape     = RoundedCornerShape(8.dp),
                    colors    = ButtonDefaults.buttonColors(
                        containerColor         = ARIAColors.Error,
                        disabledContainerColor = ARIAColors.Error.copy(alpha = 0.4f),
                    ),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                ) {
                    if (downloading) {
                        CircularProgressIndicator(
                            modifier    = Modifier.size(14.dp),
                            strokeWidth = 2.dp,
                            color       = Color.White,
                        )
                        Spacer(Modifier.width(6.dp))
                        Text("Downloading…", fontSize = 12.sp, color = Color.White)
                    } else {
                        Icon(
                            Icons.Default.Download,
                            contentDescription = null,
                            modifier = Modifier.size(14.dp),
                            tint     = Color.White,
                        )
                        Spacer(Modifier.width(6.dp))
                        Text("Download", fontSize = 12.sp, color = Color.White, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
            // Load into RAM button (shown when downloaded but not yet loaded)
            if (onLoad != null) {
                Spacer(Modifier.height(10.dp))
                Button(
                    onClick   = onLoad,
                    enabled   = !loading,
                    modifier  = Modifier.fillMaxWidth(),
                    shape     = RoundedCornerShape(8.dp),
                    colors    = ButtonDefaults.buttonColors(
                        containerColor         = ARIAColors.Primary,
                        disabledContainerColor = ARIAColors.Primary.copy(alpha = 0.4f),
                    ),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                ) {
                    if (loading) {
                        CircularProgressIndicator(
                            modifier    = Modifier.size(14.dp),
                            strokeWidth = 2.dp,
                            color       = Color.White,
                        )
                        Spacer(Modifier.width(6.dp))
                        Text("Loading into RAM…", fontSize = 12.sp, color = Color.White)
                    } else {
                        Icon(
                            Icons.Default.Memory,
                            contentDescription = null,
                            modifier = Modifier.size(14.dp),
                            tint     = Color.White,
                        )
                        Spacer(Modifier.width(6.dp))
                        Text("Load into RAM", fontSize = 12.sp, color = Color.White, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
            // Unload button (shown only when ACTIVE/READY and model is loaded)
            if (onUnload != null) {
                Spacer(Modifier.height(6.dp))
                OutlinedButton(
                    onClick        = onUnload,
                    modifier       = Modifier.fillMaxWidth(),
                    shape          = RoundedCornerShape(8.dp),
                    colors         = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Warning),
                    border         = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Warning.copy(alpha = 0.5f)),
                    contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                ) {
                    Icon(Icons.Default.Memory, contentDescription = null, modifier = Modifier.size(14.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Free RAM (Unload)", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}

@Composable
private fun PermissionRow(
    icon: ImageVector,
    label: String,
    granted: Boolean,
    onGrant: (() -> Unit)? = null,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.weight(1f),
        ) {
            Icon(
                icon,
                contentDescription = null,
                tint = if (granted) ARIAColors.Success else ARIAColors.Error,
                modifier = Modifier.size(18.dp)
            )
            Text(label, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface))
        }
        if (!granted && onGrant != null) {
            TextButton(
                onClick        = onGrant,
                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
            ) {
                Text(
                    "GRANT",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = ARIAColors.Primary, fontWeight = FontWeight.Bold
                    )
                )
            }
        } else {
            Text(
                if (granted) "GRANTED" else "DENIED",
                style = MaterialTheme.typography.labelSmall.copy(
                    color = if (granted) ARIAColors.Success else ARIAColors.Error,
                    fontWeight = FontWeight.Bold
                )
            )
        }
    }
}

@Composable
private fun RlMetric(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            value,
            style = MaterialTheme.typography.bodySmall.copy(
                color = ARIAColors.Accent,
                fontWeight = FontWeight.Bold
            )
        )
        Text(
            label,
            style = MaterialTheme.typography.labelSmall.copy(
                color = ARIAColors.Muted,
                fontSize = 10.sp
            )
        )
    }
}

@Composable
private fun AppSkillRow(skill: AppSkillItem) {
    val totalTasks = skill.taskSuccess + skill.taskFailure
    val rateColor = when {
        skill.successRate >= 0.75f -> ARIAColors.Success
        skill.successRate >= 0.50f -> ARIAColors.Warning
        else                       -> ARIAColors.Error
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.Top
    ) {
        Box(
            Modifier
                .size(36.dp)
                .background(rateColor.copy(alpha = 0.12f), RoundedCornerShape(8.dp)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                Icons.Default.PhoneAndroid,
                contentDescription = null,
                tint = rateColor,
                modifier = Modifier.size(18.dp)
            )
        }
        Column(modifier = Modifier.weight(1f)) {
            Text(
                skill.appName,
                style = MaterialTheme.typography.bodySmall.copy(
                    color = ARIAColors.OnSurface,
                    fontWeight = FontWeight.SemiBold
                )
            )
            Text(
                skill.appPackage,
                style = MaterialTheme.typography.labelSmall.copy(
                    color = ARIAColors.Muted,
                    fontSize = 10.sp
                ),
                maxLines = 1
            )
            if (skill.learnedElements.isNotEmpty()) {
                Spacer(Modifier.height(2.dp))
                Text(
                    skill.learnedElements.take(3).joinToString(" · "),
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = ARIAColors.Muted,
                        fontSize = 10.sp
                    ),
                    maxLines = 1
                )
            }
        }
        Column(horizontalAlignment = Alignment.End) {
            Text(
                "${(skill.successRate * 100).toInt()}%",
                style = MaterialTheme.typography.bodySmall.copy(
                    color = rateColor,
                    fontWeight = FontWeight.Bold
                )
            )
            Text(
                "$totalTasks tasks",
                style = MaterialTheme.typography.labelSmall.copy(
                    color = ARIAColors.Muted,
                    fontSize = 10.sp
                )
            )
            if (skill.avgSteps > 0f) {
                Text(
                    "${String.format("%.1f", skill.avgSteps)} steps/task",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = ARIAColors.Muted,
                        fontSize = 10.sp
                    )
                )
            }
        }
    }
}
