@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.ariaagent.mobile.ui.screens

import android.content.Intent
import android.os.Build
import android.provider.Settings
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.NotificationManagerCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.ariaagent.mobile.core.ai.LlamaEngine
import com.ariaagent.mobile.core.ai.ModelManager
import com.ariaagent.mobile.core.config.AriaConfig
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel
import com.ariaagent.mobile.ui.theme.ARIAColors
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * SettingsScreen — full feature-parity with settings.tsx (852 lines).
 *
 * RN reference: app/(tabs)/settings.tsx
 * DO NOT DELETE settings.tsx until this screen is verified on emulator. (Migration Phase 2)
 *
 * Features matched:
 *   - Editable model path field
 *   - Quantization chip selector (Q4_K_M / Q4_0 / IQ2_S / Q5_K_M)
 *   - Context window chip selector (512 / 1024 / 2048 / 4096)
 *   - GPU layers chip selector (0=CPU / 8 / 16 / 24 / 32)
 *   - Temperature preset buttons (0.1 / 0.3 / 0.5 / 0.7 / 0.9)
 *   - RL enabled toggle
 *   - Editable LoRA adapter path field
 *   - Permissions section (Accessibility, Notifications, Screen Capture)
 *   - Save configuration button with success feedback
 *
 * Features added beyond RN:
 *   - System info row (device model + Android API level)
 *   - Clear Memory button (ExperienceStore.clearAll) with confirmation dialog
 *   - Reset Agent button (ProgressPersistence.clearProgress) with confirmation dialog
 */
@Composable
fun SettingsScreen(
    vm: AgentViewModel = viewModel(),
    onNavigateToSafety: (() -> Unit)? = null,
    onNavigateToKnowledge: (() -> Unit)? = null,
) {
    val context      = LocalContext.current
    val focusManager = LocalFocusManager.current

    val config       by vm.config.collectAsStateWithLifecycle()
    val moduleState  by vm.moduleState.collectAsStateWithLifecycle()
    val clipboardMgr = LocalClipboardManager.current

    // ── Local draft — applied only when Save is tapped ────────────────────────
    var modelPath       by remember(config.modelPath)        { mutableStateOf(config.modelPath) }
    var quantization    by remember(config.quantization)     { mutableStateOf(config.quantization) }
    var contextWindow   by remember(config.contextWindow)    { mutableStateOf(config.contextWindow) }
    var nGpuLayers      by remember(config.nGpuLayers)       { mutableStateOf(config.nGpuLayers) }
    var gpuBackend      by remember(config.gpuBackend)       { mutableStateOf(config.gpuBackend) }
    var temperatureX100     by remember(config.temperatureX100)       { mutableStateOf(config.temperatureX100) }
    var rlEnabled           by remember(config.rlEnabled)             { mutableStateOf(config.rlEnabled) }
    var loraPath            by remember(config.loraAdapterPath)       { mutableStateOf(config.loraAdapterPath ?: "") }
    var llmInferTimeoutSec  by remember(config.llmInferTimeoutSec)    { mutableStateOf(config.llmInferTimeoutSec) }

    // ── Permission state — checked live via DisposableEffect + moduleState ────
    val accessibilityGranted = moduleState.accessibilityGranted
    val screenCaptureGranted = moduleState.screenCaptureGranted
    var notificationsGranted by remember { mutableStateOf(false) }

    // ── Auto-detected .gguf files in the ARIA models directory ────────────────
    var detectedGgufs by remember { mutableStateOf<List<File>>(emptyList()) }

    // ── Backend probe — runs once on first composition, re-runs on Probe tap ──
    var backendStatus  by remember { mutableStateOf("Probing GPU backends…") }
    var probeRunning   by remember { mutableStateOf(true) }

    // Refresh notification permission state each time screen composes
    LaunchedEffect(Unit) {
        notificationsGranted = NotificationManagerCompat.from(context).areNotificationsEnabled()
        vm.refreshModuleState()
        // Scan models directory for any .gguf files
        detectedGgufs = ModelManager.modelDir(context)
            .listFiles { f -> f.isFile && f.name.endsWith(".gguf") }
            ?.sortedByDescending { it.length() }
            ?: emptyList()
        // If model path is blank and ARIA's own model is present, auto-fill
        if (modelPath.isBlank() && ModelManager.isModelReady(context)) {
            modelPath = ModelManager.modelPath(context).absolutePath
        }
        // Backend probe — runs on IO thread so it doesn't block UI
        backendStatus = withContext(Dispatchers.IO) { LlamaEngine.probeBackends() }
        probeRunning = false
    }

    // ── Save feedback ─────────────────────────────────────────────────────────
    var saveSuccess by remember { mutableStateOf(false) }

    // ── Engine reload dialog ──────────────────────────────────────────────────
    // Shown when the user saves settings that affect an already-loaded model.
    var showReloadDialog by remember { mutableStateOf(false) }

    // ── Danger-zone dialogs ───────────────────────────────────────────────────
    var showClearMemoryDialog  by remember { mutableStateOf(false) }
    var showResetAgentDialog   by remember { mutableStateOf(false) }

    if (showClearMemoryDialog) {
        ConfirmDialog(
            title   = "Clear Memory?",
            message = "This will permanently delete all experience store entries and embedding memories. The model weights are not affected.",
            confirmLabel = "Clear Memory",
            confirmColor = ARIAColors.Warning,
            onConfirm = {
                showClearMemoryDialog = false
                vm.clearMemory()
            },
            onDismiss = { showClearMemoryDialog = false }
        )
    }

    if (showResetAgentDialog) {
        ConfirmDialog(
            title   = "Reset Agent?",
            message = "This clears all learned progress, task history, app skill data, and LoRA references. LoRA adapter files on disk are preserved.",
            confirmLabel = "Reset Agent",
            confirmColor = ARIAColors.Destructive,
            onConfirm = {
                showResetAgentDialog = false
                vm.resetAgent()
            },
            onDismiss = { showResetAgentDialog = false }
        )
    }

    // ── Engine reload dialog ──────────────────────────────────────────────────
    if (showReloadDialog) {
        AlertDialog(
            onDismissRequest = { showReloadDialog = false },
            containerColor   = ARIAColors.Surface,
            title = {
                Text(
                    "Engine Reload Required",
                    style = MaterialTheme.typography.titleMedium.copy(
                        color = ARIAColors.Primary, fontWeight = FontWeight.Bold
                    )
                )
            },
            text = {
                Text(
                    "You changed GPU layers, context window, or quantization. " +
                    "These settings only take effect after the engine is reloaded. " +
                    "Reload now to apply the new configuration immediately.",
                    style = MaterialTheme.typography.bodyMedium.copy(color = ARIAColors.TextPrimary)
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        showReloadDialog = false
                        vm.reloadLlamaEngine()
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = ARIAColors.Primary)
                ) {
                    Text("Reload Now", fontWeight = FontWeight.SemiBold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showReloadDialog = false }) {
                    Text("Later", color = ARIAColors.Muted)
                }
            }
        )
    }

    // ── Layout ────────────────────────────────────────────────────────────────
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp)
            .padding(top = 16.dp, bottom = 80.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            "SETTINGS",
            style = MaterialTheme.typography.headlineMedium.copy(
                color      = ARIAColors.Primary,
                fontWeight = FontWeight.Bold
            )
        )

        // ── Safety Boundaries shortcut ────────────────────────────────────────
        if (onNavigateToSafety != null) {
            SettingsCard(
                modifier = Modifier.clickable(
                    interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                    indication = null,
                    onClick    = onNavigateToSafety
                )
            ) {
                Row(
                    modifier              = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment     = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .clip(androidx.compose.foundation.shape.RoundedCornerShape(10.dp))
                            .background(ARIAColors.Destructive.copy(alpha = 0.12f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.Default.Shield, contentDescription = null, tint = ARIAColors.Destructive, modifier = Modifier.size(20.dp))
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Safety Boundaries",
                            style = MaterialTheme.typography.bodyMedium.copy(
                                color = ARIAColors.Destructive, fontWeight = FontWeight.SemiBold
                            )
                        )
                        Text(
                            "Kill switch, app blocklist, confirmation mode",
                            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                        )
                    }
                    Icon(Icons.Default.ChevronRight, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(18.dp))
                }
            }
        }

        // ── Knowledge Base ────────────────────────────────────────────────────
        if (onNavigateToKnowledge != null) {
            SettingsCard(
                modifier = Modifier.clickable(
                    interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                    indication = null,
                    onClick    = onNavigateToKnowledge
                )
            ) {
                Row(
                    modifier              = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment     = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(38.dp)
                            .clip(androidx.compose.foundation.shape.RoundedCornerShape(10.dp))
                            .background(ARIAColors.Accent.copy(alpha = 0.12f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(Icons.AutoMirrored.Filled.MenuBook, contentDescription = null, tint = ARIAColors.Accent, modifier = Modifier.size(20.dp))
                    }
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Knowledge Base",
                            style = MaterialTheme.typography.bodyMedium.copy(
                                color = ARIAColors.Accent, fontWeight = FontWeight.SemiBold
                            )
                        )
                        Text(
                            "How ARIA works — AI engine, memory, learning, privacy",
                            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                        )
                    }
                    Icon(Icons.Default.ChevronRight, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(18.dp))
                }
            }
        }

        // ── Model Catalog (moved to Modules) ─────────────────────────────────
        SettingsCard(
            modifier = Modifier.clickable(
                interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                indication = null,
                onClick    = { /* Navigate handled by parent — user should go to Modules tab */ }
            )
        ) {
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment     = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(38.dp)
                        .clip(androidx.compose.foundation.shape.RoundedCornerShape(10.dp))
                        .background(ARIAColors.Primary.copy(alpha = 0.12f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(Icons.Default.Storage, contentDescription = null, tint = ARIAColors.Primary, modifier = Modifier.size(20.dp))
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Model Catalog — moved to Modules",
                        style = MaterialTheme.typography.bodyMedium.copy(
                            color = ARIAColors.Primary, fontWeight = FontWeight.SemiBold
                        )
                    )
                    Text(
                        "Download, load, unload and assign roles to LLMs from the Modules screen",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                }
                Icon(Icons.Default.ChevronRight, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(18.dp))
            }
        }

        // ── Model Configuration ───────────────────────────────────────────────
        SectionLabel("Inference Settings")

        SettingsCard {
            // Model path — editable with auto-detect (shows catalog files too)
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically
            ) {
                FieldLabel("GGUF Model Path")
                IconButton(
                    onClick = {
                        detectedGgufs = ModelManager.modelDir(context)
                            .listFiles { f -> f.isFile && f.name.endsWith(".gguf") }
                            ?.sortedByDescending { it.length() }
                            ?: emptyList()
                        if (detectedGgufs.isNotEmpty() && modelPath.isBlank()) {
                            modelPath = detectedGgufs.first().absolutePath
                        }
                    },
                    modifier = Modifier.size(32.dp)
                ) {
                    Icon(
                        Icons.Default.Search,
                        contentDescription = "Scan for models",
                        tint     = ARIAColors.Primary,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
            Text(
                "Internal storage path to your .gguf file  •  tap search to scan",
                style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted),
                modifier = Modifier.padding(bottom = 6.dp)
            )

            // Detected models chips — tap to quick-fill the path
            if (detectedGgufs.isNotEmpty()) {
                Text(
                    "DETECTED MODELS",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color    = ARIAColors.Muted,
                        fontSize = 10.sp,
                        letterSpacing = 0.5.sp
                    ),
                    modifier = Modifier.padding(bottom = 4.dp)
                )
                detectedGgufs.forEach { f ->
                    val isSelected = modelPath == f.absolutePath
                    val sizeMb     = f.length() / 1_048_576
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(
                                if (isSelected) ARIAColors.Primary.copy(alpha = 0.12f)
                                else ARIAColors.Divider.copy(alpha = 0.08f)
                            )
                            .clickable { modelPath = f.absolutePath }
                            .padding(horizontal = 12.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment     = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                f.name,
                                style = MaterialTheme.typography.bodySmall.copy(
                                    color      = if (isSelected) ARIAColors.Primary else ARIAColors.OnSurface,
                                    fontWeight = FontWeight.SemiBold
                                )
                            )
                            Text(
                                "${sizeMb} MB",
                                style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                            )
                        }
                        if (isSelected) {
                            Icon(
                                Icons.Default.Check,
                                contentDescription = null,
                                tint     = ARIAColors.Primary,
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    }
                    Spacer(Modifier.height(4.dp))
                }
                Spacer(Modifier.height(4.dp))
            }

            OutlinedTextField(
                value         = modelPath,
                onValueChange = { modelPath = it },
                modifier      = Modifier.fillMaxWidth(),
                label         = { Text("Custom path", fontSize = 11.sp) },
                placeholder   = {
                    Text(
                        "Paste a full .gguf path here",
                        style = MaterialTheme.typography.bodySmall.copy(
                            color    = ARIAColors.Divider,
                            fontSize = 11.sp
                        )
                    )
                },
                colors          = ariaTextFieldColors(),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                singleLine      = false,
                maxLines        = 3
            )

            CardDivider()

            // Quantization chip selector
            FieldLabel("Quantization")
            Spacer(Modifier.height(6.dp))
            Row(
                modifier              = Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                listOf("Q4_K_M", "Q4_0", "IQ2_S", "Q5_K_M").forEach { q ->
                    SelectableChip(
                        label    = q,
                        selected = quantization == q,
                        onClick  = { quantization = q }
                    )
                }
            }

            CardDivider()

            // Context window chip selector
            FieldLabel("Context Window")
            Spacer(Modifier.height(6.dp))
            Row(
                modifier              = Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                listOf(512, 1024, 2048, 4096).forEach { sz ->
                    SelectableChip(
                        label    = sz.toString(),
                        selected = contextWindow == sz,
                        onClick  = { contextWindow = sz }
                    )
                }
            }

            CardDivider()

            // GPU layers chip selector
            Row(
                modifier            = Modifier.fillMaxWidth(),
                verticalAlignment   = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    FieldLabel("GPU Layers (n_gpu_layers)")
                    Text(
                        "Mali-G72: 32 recommended  ·  0 = CPU-only",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                }
            }
            Spacer(Modifier.height(6.dp))
            Row(
                modifier              = Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                listOf(0, 8, 16, 24, 32).forEach { n ->
                    SelectableChip(
                        label    = if (n == 0) "CPU" else "$n L",
                        selected = nGpuLayers == n,
                        onClick  = { nGpuLayers = n }
                    )
                }
            }

            CardDivider()

            // GPU backend chip selector (OpenCL / CPU)
            Row(
                modifier            = Modifier.fillMaxWidth(),
                verticalAlignment   = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    FieldLabel("GPU Backend")
                    Text(
                        "OpenCL = Mali-G72 GPU acceleration  ·  CPU = safe fallback",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                }
            }
            Spacer(Modifier.height(6.dp))
            Row(
                modifier              = Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                listOf("opencl", "cpu").forEach { backend ->
                    SelectableChip(
                        label    = backend.uppercase(),
                        selected = gpuBackend == backend,
                        onClick  = { gpuBackend = backend }
                    )
                }
            }

            // ── Backend Status panel ───────────────────────────────────────────
            Spacer(Modifier.height(10.dp))
            val probeScope = rememberCoroutineScope()
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color    = ARIAColors.Surface.copy(alpha = 0.6f),
                shape    = RoundedCornerShape(8.dp),
                tonalElevation = 1.dp
            ) {
                Column(Modifier.padding(10.dp)) {
                    Row(
                        modifier              = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment     = Alignment.CenterVertically
                    ) {
                        Text(
                            "Backend Status",
                            style      = MaterialTheme.typography.labelMedium.copy(
                                color      = ARIAColors.Primary,
                                fontWeight = FontWeight.Bold
                            )
                        )
                        if (probeRunning) {
                            CircularProgressIndicator(
                                modifier  = Modifier.size(14.dp),
                                color     = ARIAColors.Primary,
                                strokeWidth = 2.dp
                            )
                        } else {
                            TextButton(
                                onClick = {
                                    probeRunning  = true
                                    backendStatus = "Probing…"
                                    probeScope.launch {
                                        backendStatus = withContext(Dispatchers.IO) {
                                            LlamaEngine.probeBackends()
                                        }
                                        probeRunning = false
                                    }
                                },
                                contentPadding = PaddingValues(horizontal = 6.dp, vertical = 0.dp)
                            ) {
                                Text("Re-probe", style = MaterialTheme.typography.labelSmall)
                            }
                        }
                    }
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text  = backendStatus,
                        style = MaterialTheme.typography.bodySmall.copy(
                            color      = if (backendStatus.contains("OK")) ARIAColors.Success
                                         else ARIAColors.Muted,
                            fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                        )
                    )
                }
            }

            CardDivider()

            // Temperature preset buttons
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically
            ) {
                FieldLabel("Temperature")
                Text(
                    "%.2f".format(temperatureX100 / 100f),
                    style = MaterialTheme.typography.titleLarge.copy(
                        color      = ARIAColors.Primary,
                        fontWeight = FontWeight.Bold
                    )
                )
            }
            Spacer(Modifier.height(6.dp))
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                listOf(0.1f, 0.3f, 0.5f, 0.7f, 0.9f).forEach { v ->
                    val targetX100 = Math.round(v * 100)
                    SelectableChip(
                        label    = "$v",
                        selected = temperatureX100 == targetX100,
                        onClick  = { temperatureX100 = targetX100 },
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        // ── LLM Inference Timeout ─────────────────────────────────────────────
        SettingsCard {
            FieldLabel("LLM Inference Timeout")
            FieldHint("Max seconds to wait for one LLM response before issuing a Wait action. 0 = no limit (not recommended).")
            Spacer(Modifier.height(6.dp))
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                listOf(0, 30, 60, 90, 120).forEach { secs ->
                    SelectableChip(
                        label    = if (secs == 0) "∞" else "${secs}s",
                        selected = llmInferTimeoutSec == secs,
                        onClick  = { llmInferTimeoutSec = secs },
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        // ── On-Device Learning ────────────────────────────────────────────────
        SectionLabel("On-Device Learning")

        SettingsCard {
            // RL toggle
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    FieldLabel("RL Module")
                    Text(
                        "Reinforcement learning from actions",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                }
                Switch(
                    checked         = rlEnabled,
                    onCheckedChange = { rlEnabled = it },
                    colors          = SwitchDefaults.colors(
                        checkedThumbColor   = ARIAColors.Background,
                        checkedTrackColor   = ARIAColors.Primary,
                        uncheckedThumbColor = ARIAColors.Muted,
                        uncheckedTrackColor = ARIAColors.Divider,
                    )
                )
            }

            CardDivider()

            // LoRA adapter path — editable (matches RN)
            FieldLabel("LoRA Adapter Path")
            Spacer(Modifier.height(6.dp))
            OutlinedTextField(
                value         = loraPath,
                onValueChange = { loraPath = it },
                modifier      = Modifier.fillMaxWidth(),
                placeholder   = {
                    Text(
                        "Optional — leave empty to use latest trained adapter",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Divider)
                    )
                },
                colors          = ariaTextFieldColors(),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                singleLine      = false,
                maxLines        = 2
            )
        }

        // ── Permissions ───────────────────────────────────────────────────────
        SectionLabel("Permissions")

        SettingsCard(padding = 0.dp) {
            // Accessibility Service
            PermissionRow(
                icon          = Icons.Default.Visibility,
                title         = "Accessibility Service",
                description   = "Reads the UI tree and dispatches gestures. ARIA cannot navigate apps without this.",
                granted       = accessibilityGranted,
                grantedLabel  = "ACTIVE",
                deniedLabel   = "REQUIRED",
                grantedColor  = ARIAColors.Success,
                deniedColor   = ARIAColors.Destructive,
                showDivider   = true,
                actionLabel   = if (!accessibilityGranted) "Open Accessibility Settings" else null,
                onAction      = {
                    context.startActivity(
                        Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    )
                }
            )

            // Notifications
            PermissionRow(
                icon          = Icons.Default.Notifications,
                title         = "Notifications",
                description   = "Download progress, training completion, and agent status alerts.",
                granted       = notificationsGranted,
                grantedLabel  = "GRANTED",
                deniedLabel   = "BLOCKED",
                grantedColor  = ARIAColors.Success,
                deniedColor   = ARIAColors.Warning,
                showDivider   = true,
                actionLabel   = if (!notificationsGranted) "Open Notification Settings" else null,
                onAction      = {
                    context.startActivity(
                        Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                            .putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    )
                }
            )

            // Screen Capture
            PermissionRow(
                icon          = Icons.Default.Fullscreen,
                title         = "Screen Capture",
                description   = "MediaProjection — Android shows a one-time consent dialog when you start the agent. No permanent grant needed.",
                granted       = screenCaptureGranted,
                grantedLabel  = "ACTIVE",
                deniedLabel   = "ON-DEMAND",
                grantedColor  = ARIAColors.Success,
                deniedColor   = ARIAColors.Primary,
                showDivider   = false,
                actionLabel   = null,
                onAction      = {}
            )
        }

        // ── Web Dashboard / Local Monitoring Server (Gap 6) ──────────────────
        SectionLabel("Web Dashboard")

        SettingsCard {
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    FieldLabel("Local Monitoring Server")
                    Text(
                        if (moduleState.localServerRunning)
                            "HTTP server active — push live agent state to browser"
                        else
                            "Start an HTTP server to monitor ARIA from your browser",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                }
                Switch(
                    checked         = moduleState.localServerRunning,
                    onCheckedChange = { vm.toggleLocalServer() },
                    colors          = SwitchDefaults.colors(
                        checkedThumbColor   = ARIAColors.Background,
                        checkedTrackColor   = ARIAColors.Primary,
                        uncheckedThumbColor = ARIAColors.Muted,
                        uncheckedTrackColor = ARIAColors.Divider,
                    )
                )
            }

            if (moduleState.localServerRunning && moduleState.localServerUrl.isNotEmpty()) {
                CardDivider()
                Row(
                    modifier              = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment     = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        FieldLabel("Server URL")
                        Text(
                            moduleState.localServerUrl,
                            style = MaterialTheme.typography.bodySmall.copy(
                                color      = ARIAColors.Accent,
                                fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                            )
                        )
                    }
                    IconButton(onClick = {
                        clipboardMgr.setText(AnnotatedString(moduleState.localServerUrl))
                    }) {
                        Icon(
                            Icons.Default.ContentCopy,
                            contentDescription = "Copy URL",
                            tint     = ARIAColors.Muted,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }
                Spacer(Modifier.height(4.dp))
                Text(
                    "Open this URL in a browser on the same Wi-Fi network to monitor ARIA in real time.",
                    style = MaterialTheme.typography.bodySmall.copy(
                        color      = ARIAColors.Muted,
                        lineHeight = 18.sp
                    )
                )
            }
        }

        // ── System Info (new — beyond RN) ─────────────────────────────────────
        SectionLabel("System Info")

        SettingsCard {
            InfoRow("Device",       Build.MANUFACTURER + " " + Build.MODEL)
            InfoRow("Android",      "API " + Build.VERSION.SDK_INT + "  (${Build.VERSION.RELEASE})")
            InfoRow("Package",      context.packageName)
            InfoRow("Architecture", Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown")
        }

        // ── Save button ───────────────────────────────────────────────────────
        Button(
            onClick  = {
                focusManager.clearFocus()
                // Detect whether the user changed any parameter that requires an engine
                // reload to take effect (GPU layers, context window, quantization).
                // Only relevant when LlamaEngine is already loaded — if it is not loaded,
                // the new params will be picked up automatically at next load() call.
                val engineParamsChanged = LlamaEngine.isLoaded() && (
                    nGpuLayers    != config.nGpuLayers    ||
                    gpuBackend    != config.gpuBackend    ||
                    contextWindow != config.contextWindow ||
                    quantization  != config.quantization
                )
                vm.saveConfig(
                    AriaConfig(
                        modelPath           = modelPath.trim(),
                        quantization        = quantization,
                        contextWindow       = contextWindow,
                        maxTokensPerTurn    = config.maxTokensPerTurn,
                        temperatureX100     = temperatureX100,
                        nGpuLayers          = nGpuLayers,
                        gpuBackend          = gpuBackend,
                        rlEnabled           = rlEnabled,
                        loraAdapterPath     = loraPath.trim(),
                        llmInferTimeoutSec  = llmInferTimeoutSec,
                        // Preserve fields managed elsewhere (ControlScreen Agent Mode card)
                        agentCombo          = config.agentCombo,
                        irlEnabled          = config.irlEnabled,
                        rlVisionMode        = config.rlVisionMode,
                        irlVisionMode       = config.irlVisionMode,
                        learningRate        = config.learningRate,
                    )
                )
                saveSuccess = true
                if (engineParamsChanged) showReloadDialog = true
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(50.dp),
            colors   = ButtonDefaults.buttonColors(
                containerColor = if (saveSuccess) ARIAColors.Success else ARIAColors.Primary
            ),
            shape    = RoundedCornerShape(12.dp)
        ) {
            Icon(
                if (saveSuccess) Icons.Default.CheckCircle else Icons.Default.Save,
                contentDescription = null,
                modifier = Modifier.size(18.dp)
            )
            Spacer(Modifier.width(8.dp))
            Text(
                if (saveSuccess) "SAVED!" else "SAVE CONFIGURATION",
                fontWeight = FontWeight.Bold
            )
        }

        if (saveSuccess) {
            LaunchedEffect(saveSuccess) {
                kotlinx.coroutines.delay(3_000)
                saveSuccess = false
            }
        }

        // ── Danger Zone (new — beyond RN) ─────────────────────────────────────
        SectionLabel("Danger Zone")

        SettingsCard {
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                // Clear Memory
                OutlinedButton(
                    onClick  = { showClearMemoryDialog = true },
                    modifier = Modifier.weight(1f),
                    colors   = ButtonDefaults.outlinedButtonColors(
                        contentColor = ARIAColors.Warning
                    ),
                    border   = androidx.compose.foundation.BorderStroke(
                        1.dp, ARIAColors.Warning.copy(alpha = 0.5f)
                    ),
                    shape    = RoundedCornerShape(10.dp)
                ) {
                    Icon(Icons.Default.DeleteSweep, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Clear Memory", fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                }

                // Reset Agent
                OutlinedButton(
                    onClick  = { showResetAgentDialog = true },
                    modifier = Modifier.weight(1f),
                    colors   = ButtonDefaults.outlinedButtonColors(
                        contentColor = ARIAColors.Destructive
                    ),
                    border   = androidx.compose.foundation.BorderStroke(
                        1.dp, ARIAColors.Destructive.copy(alpha = 0.5f)
                    ),
                    shape    = RoundedCornerShape(10.dp)
                ) {
                    Icon(Icons.Default.RestartAlt, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Reset Agent", fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                }
            }

            Spacer(Modifier.height(6.dp))
            Text(
                "Clear Memory removes experience entries and embeddings. " +
                "Reset Agent clears all learned progress, skills, and task history.",
                style = MaterialTheme.typography.bodySmall.copy(
                    color = ARIAColors.Muted,
                    lineHeight = 18.sp
                )
            )
        }

        // ── On-device privacy note ─────────────────────────────────────────────
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
            verticalAlignment     = Alignment.CenterVertically
        ) {
            Icon(
                Icons.Default.Lock,
                contentDescription = null,
                tint     = ARIAColors.Muted,
                modifier = Modifier.size(12.dp)
            )
            Spacer(Modifier.width(6.dp))
            Text(
                "Strictly on-device. No data ever leaves this phone.",
                style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
            )
        }
    }
}

// ─── Sub-composables ──────────────────────────────────────────────────────────

@Composable
private fun SettingsCard(
    modifier: Modifier = Modifier,
    padding: androidx.compose.ui.unit.Dp = 16.dp,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(16.dp),
        colors   = CardDefaults.cardColors(containerColor = ARIAColors.Surface)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(padding),
            verticalArrangement = Arrangement.spacedBy(if (padding > 0.dp) 10.dp else 0.dp),
            content = content
        )
    }
}

@Composable
private fun SectionLabel(text: String) {
    Text(
        text.uppercase(),
        style = MaterialTheme.typography.labelSmall.copy(
            color       = ARIAColors.Muted,
            fontWeight  = FontWeight.Bold,
            letterSpacing = 1.sp
        ),
        modifier = Modifier.padding(start = 4.dp, bottom = 2.dp)
    )
}

@Composable
private fun FieldLabel(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.bodyMedium.copy(
            color      = ARIAColors.OnSurface,
            fontWeight = FontWeight.Medium
        )
    )
}

@Composable
private fun FieldHint(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.bodySmall.copy(
            color = ARIAColors.OnSurface.copy(alpha = 0.6f)
        )
    )
}

@Composable
private fun CardDivider() {
    HorizontalDivider(
        modifier  = Modifier.padding(vertical = 6.dp),
        color     = ARIAColors.Divider,
        thickness = 0.5.dp
    )
}

@Composable
private fun SelectableChip(
    label    : String,
    selected : Boolean,
    onClick  : () -> Unit,
    modifier : Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(8.dp))
            .background(if (selected) ARIAColors.Primary else ARIAColors.SurfaceVariant)
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 8.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelMedium.copy(
                color      = if (selected) ARIAColors.Background else ARIAColors.OnSurface,
                fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
            )
        )
    }
}

@Composable
private fun PermissionRow(
    icon          : ImageVector,
    title         : String,
    description   : String,
    granted       : Boolean,
    grantedLabel  : String,
    deniedLabel   : String,
    grantedColor  : Color,
    deniedColor   : Color,
    showDivider   : Boolean,
    actionLabel   : String?,
    onAction      : () -> Unit,
) {
    Column {
        Row(
            modifier  = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Icon circle
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(
                        if (granted) grantedColor.copy(alpha = 0.13f)
                        else deniedColor.copy(alpha = 0.13f)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    icon,
                    contentDescription = null,
                    tint     = if (granted) grantedColor else deniedColor,
                    modifier = Modifier.size(18.dp)
                )
            }

            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                // Title + badge
                Row(
                    verticalAlignment      = Alignment.CenterVertically,
                    horizontalArrangement  = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        title,
                        style = MaterialTheme.typography.bodyMedium.copy(
                            color      = ARIAColors.OnSurface,
                            fontWeight = FontWeight.SemiBold
                        )
                    )
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(
                                if (granted) grantedColor.copy(alpha = 0.13f)
                                else deniedColor.copy(alpha = 0.13f)
                            )
                            .padding(horizontal = 7.dp, vertical = 2.dp)
                    ) {
                        Text(
                            if (granted) grantedLabel else deniedLabel,
                            style = MaterialTheme.typography.labelSmall.copy(
                                color      = if (granted) grantedColor else deniedColor,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 0.6.sp
                            )
                        )
                    }
                }

                // Description
                Text(
                    description,
                    style = MaterialTheme.typography.bodySmall.copy(
                        color      = ARIAColors.Muted,
                        lineHeight = 18.sp
                    )
                )

                // Action button — only when not granted and actionLabel != null
                if (!granted && actionLabel != null) {
                    Spacer(Modifier.height(4.dp))
                    OutlinedButton(
                        onClick = onAction,
                        colors  = ButtonDefaults.outlinedButtonColors(
                            contentColor = deniedColor
                        ),
                        border  = androidx.compose.foundation.BorderStroke(
                            1.dp, deniedColor.copy(alpha = 0.4f)
                        ),
                        shape   = RoundedCornerShape(8.dp),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                    ) {
                        Icon(
                            Icons.AutoMirrored.Filled.OpenInNew,
                            contentDescription = null,
                            modifier = Modifier.size(13.dp)
                        )
                        Spacer(Modifier.width(6.dp))
                        Text(actionLabel, fontSize = 13.sp)
                    }
                }
            }
        }

        if (showDivider) {
            HorizontalDivider(color = ARIAColors.Divider, thickness = 0.5.dp)
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier              = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment     = Alignment.CenterVertically
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
        )
        Text(
            value,
            style    = MaterialTheme.typography.bodySmall.copy(
                color      = ARIAColors.OnSurface,
                fontWeight = FontWeight.Medium
            ),
            maxLines = 1
        )
    }
}

@Composable
private fun ConfirmDialog(
    title        : String,
    message      : String,
    confirmLabel : String,
    confirmColor : Color,
    onConfirm    : () -> Unit,
    onDismiss    : () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor   = ARIAColors.Surface,
        title = {
            Text(
                title,
                style = MaterialTheme.typography.titleMedium.copy(
                    color      = ARIAColors.OnSurface,
                    fontWeight = FontWeight.Bold
                )
            )
        },
        text = {
            Text(
                message,
                style = MaterialTheme.typography.bodySmall.copy(
                    color      = ARIAColors.Muted,
                    lineHeight = 20.sp
                )
            )
        },
        confirmButton = {
            Button(
                onClick = onConfirm,
                colors  = ButtonDefaults.buttonColors(containerColor = confirmColor),
                shape   = RoundedCornerShape(8.dp)
            ) {
                Text(confirmLabel, fontWeight = FontWeight.Bold)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = ARIAColors.Muted)
            }
        }
    )
}

@Composable
private fun ariaTextFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor   = ARIAColors.Primary,
    unfocusedBorderColor = ARIAColors.Divider,
    focusedTextColor     = ARIAColors.OnSurface,
    unfocusedTextColor   = ARIAColors.OnSurface,
    cursorColor          = ARIAColors.Primary,
)
