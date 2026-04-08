@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.ariaagent.mobile.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Label
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.automirrored.filled.PlaylistPlay
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.ariaagent.mobile.core.ai.ModelCatalog
import com.ariaagent.mobile.core.ai.ModelManager
import com.ariaagent.mobile.core.config.AgentCombo
import com.ariaagent.mobile.core.config.ModuleVisionMode
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel
import com.ariaagent.mobile.ui.viewmodel.ChainedTaskItem
import com.ariaagent.mobile.ui.viewmodel.QueuedTaskItem
import com.ariaagent.mobile.ui.theme.ARIAColors
import java.text.SimpleDateFormat
import java.util.*

/**
 * ControlScreen — start/pause/stop the agent and manage the task queue.
 *
 * Phase 4 gap-fill over the existing stub:
 *   ✓ Chained task notification banner  (chainedTask + dismissChainNotification)
 *   ✓ Learn-only mode toggle            (vm.startLearnOnly vs vm.startAgent)
 *   ✓ LLM Load Gate card               (when !moduleState.modelLoaded)
 *   ✓ Active task display              (agentState.currentTask green box)
 *   ✓ Separate queue-goal + queue-app fields (split from main goal field)
 *   ✓ "Teach the Agent" → LabelerScreen entry point
 *   ✓ Status dot in header
 *
 * Pure Compose — calls AgentViewModel which calls AgentLoop / AgentForegroundService directly.
 * No bridge, no React Native, no JS.
 */

private val PRESET_TASKS = listOf(
    "Open YouTube and play the trending video",
    "Open Settings and check available storage",
    "Open Gallery and find the latest photo",
    "Open Chrome and go to google.com",
    "Open WhatsApp and read the latest message",
    "Open Maps and search for nearby restaurants",
    "Open Calculator and compute 15% tip on 85",
    "Check battery level in Settings",
)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ControlScreen(
    vm: AgentViewModel = viewModel(),
    onNavigateToLabeler: (() -> Unit)? = null,
    onNavigateToGoals: (() -> Unit)? = null,
) {
    val context      = LocalContext.current
    val agentState   by vm.agentState.collectAsStateWithLifecycle()
    val moduleState  by vm.moduleState.collectAsStateWithLifecycle()
    val taskQueue    by vm.taskQueue.collectAsStateWithLifecycle()
    val chainedTask  by vm.chainedTask.collectAsStateWithLifecycle()
    val loadedLlms   by vm.loadedLlms.collectAsStateWithLifecycle()
    val config       by vm.config.collectAsStateWithLifecycle()
    val focusManager = LocalFocusManager.current
    val activeModel  = remember { ModelManager.activeEntry(context) }  // nullable — null when no model selected
    val loadedCount  = loadedLlms.values.count { it.isLoaded }

    var goalText    by remember { mutableStateOf("") }
    var targetApp   by remember { mutableStateOf("") }
    var queueGoal   by remember { mutableStateOf("") }
    var queueApp    by remember { mutableStateOf("") }
    var learnOnly   by remember { mutableStateOf(false) }

    val selectedCombo   = config.agentCombo
    val rlVisionMode    = config.rlVisionMode
    val irlVisionMode   = config.irlVisionMode

    val isRunning  = agentState.status == "running"
    val isPaused   = agentState.status == "paused"
    val isIdle     = agentState.status == "idle"
            || agentState.status == "done"
            || agentState.status == "error"
    val llmNeeded  = selectedCombo.usesLlm
    val canStart   = isIdle && goalText.isNotBlank()
            && (!llmNeeded || moduleState.modelLoaded)
            && moduleState.accessibilityGranted

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {

        // ── Header with status dot ──────────────────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "CONTROL",
                style = MaterialTheme.typography.headlineMedium.copy(
                    color = ARIAColors.Primary,
                    fontWeight = FontWeight.Bold
                )
            )
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Goals shortcut button
                if (onNavigateToGoals != null) {
                    IconButton(
                        onClick  = onNavigateToGoals,
                        modifier = Modifier
                            .size(32.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(ARIAColors.Primary.copy(alpha = 0.10f))
                    ) {
                        Icon(
                            Icons.Default.Queue,
                            contentDescription = "Manage Goals",
                            tint     = ARIAColors.Primary,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
                val dotColor = when (agentState.status) {
                    "running" -> ARIAColors.Success
                    "paused"  -> ARIAColors.Warning
                    "error"   -> ARIAColors.Error
                    else      -> ARIAColors.Muted
                }
                Box(
                    modifier = Modifier
                        .size(9.dp)
                        .clip(CircleShape)
                        .background(dotColor)
                )
                Text(
                    agentState.status.replaceFirstChar { it.uppercase() },
                    style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                )
            }
        }

        // ── Chained task notification banner ───────────────────────────────────
        chainedTask?.let { chain ->
            ARIACard(
                modifier = Modifier.border(
                    1.dp,
                    ARIAColors.Primary.copy(alpha = 0.35f),
                    RoundedCornerShape(12.dp)
                ),
                containerColor = ARIAColors.Primary.copy(alpha = 0.08f)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.Top
                ) {
                    Icon(
                        Icons.Default.Link,
                        contentDescription = null,
                        tint = ARIAColors.Primary,
                        modifier = Modifier
                            .size(15.dp)
                            .padding(top = 2.dp)
                    )
                    Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(
                            "AUTO-CHAINED TO NEXT TASK",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Primary, fontWeight = FontWeight.Bold
                            )
                        )
                        Text(
                            chain.goal,
                            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface),
                            maxLines = 2
                        )
                        val sub = buildString {
                            if (chain.appPackage.isNotBlank()) append(chain.appPackage)
                            if (chain.queueSize > 0) {
                                if (isNotEmpty()) append(" · ")
                                append("${chain.queueSize} more in queue")
                            }
                        }
                        if (sub.isNotBlank()) {
                            Text(
                                sub,
                                style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                            )
                        }
                    }
                    IconButton(
                        onClick = { vm.dismissChainNotification() },
                        modifier = Modifier.size(28.dp)
                    ) {
                        Icon(
                            Icons.Default.Close,
                            contentDescription = "Dismiss",
                            tint = ARIAColors.Muted,
                            modifier = Modifier.size(14.dp)
                        )
                    }
                }
            }
        }

        // ── LLM Load Gate ──────────────────────────────────────────────────────
        if (loadedCount > 0) {
            // ── Loaded models summary ─────────────────────────────────────────
            ARIACard(
                modifier = Modifier.border(
                    1.dp,
                    ARIAColors.Primary.copy(alpha = 0.40f),
                    RoundedCornerShape(12.dp)
                ),
                containerColor = ARIAColors.Primary.copy(alpha = 0.06f)
            ) {
                Row(
                    modifier              = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment     = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.Memory, null,
                        tint = ARIAColors.Primary, modifier = Modifier.size(20.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "$loadedCount model${if (loadedCount == 1) "" else "s"} in RAM",
                            style = MaterialTheme.typography.bodyMedium.copy(
                                color = ARIAColors.Primary, fontWeight = FontWeight.SemiBold
                            )
                        )
                        loadedLlms.values.filter { it.isLoaded }.forEach { entry ->
                            val cat = com.ariaagent.mobile.core.ai.ModelCatalog.findById(entry.modelId)
                            Text(
                                "• ${cat?.displayName ?: entry.modelId}  ·  ${entry.role.label}",
                                style = MaterialTheme.typography.bodySmall.copy(
                                    color = ARIAColors.Muted, fontSize = 11.sp
                                )
                            )
                        }
                    }
                }
            }
        } else if (!moduleState.modelLoaded && llmNeeded) {
            // ── No model loaded gate (only shown when combo uses LLM) ─────────
            ARIACard(
                modifier = Modifier.border(
                    1.dp,
                    ARIAColors.Accent.copy(alpha = 0.40f),
                    RoundedCornerShape(12.dp)
                ),
                containerColor = ARIAColors.Accent.copy(alpha = 0.08f)
            ) {
                // Quick-load current active model
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null,
                            onClick    = { vm.loadModel() }
                        ),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalAlignment     = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.Memory, null,
                        tint = ARIAColors.Accent, modifier = Modifier.size(22.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "Load ${activeModel?.displayName ?: "Model"}",
                            style = MaterialTheme.typography.bodyMedium.copy(
                                color = ARIAColors.Accent, fontWeight = FontWeight.SemiBold
                            )
                        )
                        Text(
                            when {
                                !moduleState.modelReady ->
                                    "Not downloaded — go to Modules to pick & download a model"
                                else ->
                                    "Tap to load into RAM  ·  or go to Modules to load multiple"
                            },
                            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                        )
                    }
                    Icon(Icons.Default.ChevronRight, null,
                        tint = ARIAColors.Accent, modifier = Modifier.size(18.dp))
                }
            }
        }

        // ── Readiness checks ──────────────────────────────────────────────────
        ARIACard {
            Text("READINESS", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
            Spacer(Modifier.height(8.dp))
            ReadinessRow(
                label    = if (llmNeeded) "Model downloaded" else "Model downloaded (optional)",
                ok       = moduleState.modelReady || !llmNeeded
            )
            ReadinessRow(
                label    = if (llmNeeded) "Model loaded" else "Model loaded (optional — LLM off)",
                ok       = moduleState.modelLoaded || !llmNeeded
            )
            ReadinessRow("Accessibility",  ok = moduleState.accessibilityGranted)
            ReadinessRow("Screen capture", ok = moduleState.screenCaptureGranted)
        }

        // ── Goal input ────────────────────────────────────────────────────────
        ARIACard {
            Text("TASK GOAL", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = goalText,
                onValueChange = { goalText = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text(
                        "e.g. Open YouTube and search for cooking tutorials",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor   = ARIAColors.Primary,
                    unfocusedBorderColor = ARIAColors.Divider,
                    focusedTextColor     = ARIAColors.OnSurface,
                    unfocusedTextColor   = ARIAColors.OnSurface,
                    cursorColor          = ARIAColors.Primary,
                ),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
                keyboardActions = KeyboardActions(onNext = { focusManager.clearFocus() }),
                maxLines = 3,
                enabled = isIdle
            )
            Spacer(Modifier.height(6.dp))
            OutlinedTextField(
                value = targetApp,
                onValueChange = { targetApp = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text(
                        "Target app package (optional)",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor   = ARIAColors.Primary,
                    unfocusedBorderColor = ARIAColors.Divider,
                    focusedTextColor     = ARIAColors.OnSurface,
                    unfocusedTextColor   = ARIAColors.OnSurface,
                    cursorColor          = ARIAColors.Primary,
                ),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { focusManager.clearFocus() }),
                maxLines = 1,
                enabled = isIdle
            )
        }

        // ── Preset task chips ─────────────────────────────────────────────────
        ARIACard {
            Text(
                "QUICK GOALS",
                style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
            )
            Spacer(Modifier.height(8.dp))
            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                PRESET_TASKS.forEach { preset ->
                    SuggestionChip(
                        onClick = {
                            goalText = preset
                            focusManager.clearFocus()
                        },
                        label = {
                            Text(
                                preset.take(32) + if (preset.length > 32) "…" else "",
                                style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp)
                            )
                        },
                        colors = SuggestionChipDefaults.suggestionChipColors(
                            containerColor = ARIAColors.Primary.copy(alpha = 0.08f),
                            labelColor = ARIAColors.Primary
                        ),
                        border = SuggestionChipDefaults.suggestionChipBorder(
                            enabled = true,
                            borderColor = ARIAColors.Primary.copy(alpha = 0.25f)
                        )
                    )
                }
            }
        }

        // ── Learn-only mode toggle ────────────────────────────────────────────
        ARIACard(
            containerColor = if (learnOnly)
                ARIAColors.Accent.copy(alpha = 0.08f)
            else
                ARIAColors.Surface,
            modifier = if (learnOnly) Modifier.border(
                1.dp,
                ARIAColors.Accent.copy(alpha = 0.35f),
                RoundedCornerShape(12.dp)
            ) else Modifier
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.AutoMirrored.Filled.MenuBook,
                    contentDescription = null,
                    tint = if (learnOnly) ARIAColors.Accent else ARIAColors.Muted,
                    modifier = Modifier.size(20.dp)
                )
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Learn-Only Mode",
                        style = MaterialTheme.typography.bodyMedium.copy(
                            color = if (learnOnly) ARIAColors.Accent else ARIAColors.OnSurface,
                            fontWeight = FontWeight.SemiBold
                        )
                    )
                    Text(
                        if (learnOnly)
                            "Observing & reasoning — no gestures dispatched"
                        else
                            "Observe, reason, and act on the device",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                }
                Switch(
                    checked = learnOnly,
                    onCheckedChange = { learnOnly = it },
                    enabled = isIdle,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = ARIAColors.Accent,
                        checkedTrackColor = ARIAColors.Accent.copy(alpha = 0.45f),
                    )
                )
            }
        }

        // ── Agent Mode — combo + vision mode selectors ────────────────────────
        AgentModeCard(
            selectedCombo  = selectedCombo,
            rlVisionMode   = rlVisionMode,
            irlVisionMode  = irlVisionMode,
            enabled        = isIdle,
            onComboChange        = { vm.setAgentCombo(it) },
            onBothVisionChange   = { vm.setBothVisionModes(it) },
            onRlVisionChange     = { vm.setRlVisionMode(it) },
            onIrlVisionChange    = { vm.setIrlVisionMode(it) },
        )

        // ── Control buttons ───────────────────────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = {
                    focusManager.clearFocus()
                    if (learnOnly) vm.startLearnOnly(goalText.trim(), targetApp.trim())
                    else           vm.startAgent(goalText.trim(), targetApp.trim())
                },
                modifier = Modifier.weight(1f),
                enabled = canStart,
                colors = ButtonDefaults.buttonColors(
                    containerColor = ARIAColors.Primary,
                    disabledContainerColor = ARIAColors.Divider
                ),
                shape = RoundedCornerShape(10.dp)
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(4.dp))
                Text(
                    if (learnOnly) "START LEARNING" else "START AGENT",
                    fontWeight = FontWeight.Bold
                )
            }

            OutlinedButton(
                onClick = {
                    if (isPaused) vm.startAgent(agentState.currentTask, agentState.currentApp)
                    else          vm.pauseAgent()
                },
                modifier = Modifier.weight(1f),
                enabled = isRunning || isPaused,
                colors = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Warning),
                shape = RoundedCornerShape(10.dp)
            ) {
                Icon(
                    if (isPaused) Icons.Default.PlayArrow else Icons.Default.Pause,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(Modifier.width(4.dp))
                Text(if (isPaused) "RESUME" else "PAUSE", fontWeight = FontWeight.Bold)
            }

            OutlinedButton(
                onClick = { vm.stopAgent() },
                modifier = Modifier.weight(1f),
                enabled = isRunning || isPaused,
                colors = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Error),
                shape = RoundedCornerShape(10.dp)
            ) {
                Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(4.dp))
                Text("STOP", fontWeight = FontWeight.Bold)
            }
        }

        // ── Active task display ───────────────────────────────────────────────
        if (agentState.currentTask.isNotBlank()) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(ARIAColors.Success.copy(alpha = 0.10f), RoundedCornerShape(10.dp))
                    .border(1.dp, ARIAColors.Success.copy(alpha = 0.28f), RoundedCornerShape(10.dp))
                    .padding(12.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.Top
            ) {
                Icon(
                    Icons.Default.RadioButtonChecked,
                    contentDescription = null,
                    tint = ARIAColors.Success,
                    modifier = Modifier
                        .size(14.dp)
                        .padding(top = 2.dp)
                )
                Text(
                    agentState.currentTask,
                    style = MaterialTheme.typography.bodySmall.copy(
                        color = ARIAColors.OnSurface,
                        lineHeight = 18.sp
                    ),
                    modifier = Modifier.weight(1f)
                )
            }
        }

        // ── Task Queue ────────────────────────────────────────────────────────
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
                        Icons.AutoMirrored.Filled.PlaylistPlay,
                        contentDescription = null,
                        tint = ARIAColors.Accent,
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        "TASK QUEUE",
                        style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                    )
                    if (taskQueue.isNotEmpty()) {
                        Text(
                            "(${taskQueue.size})",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Accent, fontWeight = FontWeight.Bold
                            )
                        )
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    if (taskQueue.isNotEmpty()) {
                        TextButton(
                            onClick = { vm.clearTaskQueue() },
                            colors = ButtonDefaults.textButtonColors(contentColor = ARIAColors.Error)
                        ) {
                            Text("Clear all", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                    IconButton(
                        onClick = { vm.refreshTaskQueue() },
                        modifier = Modifier.size(28.dp)
                    ) {
                        Icon(
                            Icons.Default.Refresh,
                            contentDescription = "Refresh",
                            tint = ARIAColors.Muted,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }

            Spacer(Modifier.height(10.dp))

            // ── Separate queue-goal + queue-app fields ──────────────────────
            OutlinedTextField(
                value = queueGoal,
                onValueChange = { queueGoal = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text(
                        "Goal to queue after current task…",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor   = ARIAColors.Accent,
                    unfocusedBorderColor = ARIAColors.Divider,
                    focusedTextColor     = ARIAColors.OnSurface,
                    unfocusedTextColor   = ARIAColors.OnSurface,
                    cursorColor          = ARIAColors.Accent,
                ),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
                keyboardActions = KeyboardActions(onNext = { focusManager.clearFocus() }),
                maxLines = 2,
            )
            Spacer(Modifier.height(6.dp))
            OutlinedTextField(
                value = queueApp,
                onValueChange = { queueApp = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text(
                        "App package (optional, e.g. com.google.android.youtube)",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                    )
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor   = ARIAColors.Accent,
                    unfocusedBorderColor = ARIAColors.Divider,
                    focusedTextColor     = ARIAColors.OnSurface,
                    unfocusedTextColor   = ARIAColors.OnSurface,
                    cursorColor          = ARIAColors.Accent,
                ),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { focusManager.clearFocus() }),
                maxLines = 1,
            )
            Spacer(Modifier.height(8.dp))
            OutlinedButton(
                onClick = {
                    if (queueGoal.isNotBlank()) {
                        focusManager.clearFocus()
                        vm.enqueueTask(queueGoal.trim(), queueApp.trim())
                        queueGoal = ""
                        queueApp  = ""
                    }
                },
                enabled = queueGoal.isNotBlank(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Accent),
                shape = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Default.AddCircleOutline, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text("Add to Queue", fontWeight = FontWeight.SemiBold)
            }

            if (taskQueue.isNotEmpty()) {
                Spacer(Modifier.height(10.dp))
                HorizontalDivider(color = ARIAColors.Divider)
                Spacer(Modifier.height(8.dp))

                taskQueue.take(10).forEachIndexed { index, task ->
                    QueuedTaskRow(
                        index  = index + 1,
                        task   = task,
                        onRemove = { vm.removeQueuedTask(task.id) }
                    )
                    if (index < taskQueue.size - 1 && index < 9) {
                        HorizontalDivider(
                            color = ARIAColors.Divider.copy(alpha = 0.4f),
                            modifier = Modifier.padding(vertical = 4.dp)
                        )
                    }
                }
                if (taskQueue.size > 10) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "… and ${taskQueue.size - 10} more tasks",
                        style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                    )
                }
            } else {
                Spacer(Modifier.height(6.dp))
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.AutoMirrored.Filled.List,
                        contentDescription = null,
                        tint = ARIAColors.Muted,
                        modifier = Modifier.size(14.dp)
                    )
                    Text(
                        "No tasks queued · Add a goal above to chain automatically after the current task finishes",
                        style = MaterialTheme.typography.bodySmall.copy(
                            color = ARIAColors.Muted,
                            lineHeight = 18.sp
                        )
                    )
                }
            }
        }

        // ── Teach the Agent entry point ───────────────────────────────────────
        ARIACard {
            Text(
                "TEACH THE AGENT",
                style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
            )
            Spacer(Modifier.height(10.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication = null,
                        onClick = { onNavigateToLabeler?.invoke() }
                    ),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(ARIAColors.Accent.copy(alpha = 0.15f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        Icons.AutoMirrored.Filled.Label,
                        contentDescription = null,
                        tint = ARIAColors.Accent,
                        modifier = Modifier.size(22.dp)
                    )
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "Object Labeler",
                        style = MaterialTheme.typography.bodyMedium.copy(
                            color = ARIAColors.OnSurface,
                            fontWeight = FontWeight.SemiBold
                        )
                    )
                    Text(
                        "Annotate UI elements · LLM enriches context · Agent learns",
                        style = MaterialTheme.typography.bodySmall.copy(
                            color = ARIAColors.Muted,
                            lineHeight = 16.sp
                        )
                    )
                }
                Icon(
                    Icons.Default.ChevronRight,
                    contentDescription = null,
                    tint = ARIAColors.Muted,
                    modifier = Modifier.size(18.dp)
                )
            }
        }

        // ── Status banners ────────────────────────────────────────────────────
        if (!moduleState.modelReady && llmNeeded) {
            InfoBanner("Model not downloaded. Go to Modules to download it.", ARIAColors.Warning)
        }
        if (!moduleState.accessibilityGranted) {
            InfoBanner("Accessibility permission not granted. Enable ARIA in Accessibility Settings.", ARIAColors.Error)
        }
        if (agentState.lastError.isNotBlank()) {
            InfoBanner("Last error: ${agentState.lastError}", ARIAColors.Error)
        }
    }
}

// ─── Private composables ──────────────────────────────────────────────────────

@Composable
private fun QueuedTaskRow(
    index: Int,
    task: QueuedTaskItem,
    onRemove: () -> Unit
) {
    val fmt = remember { SimpleDateFormat("HH:mm", Locale.getDefault()) }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(6.dp))
                .background(ARIAColors.Primary.copy(alpha = 0.14f))
                .padding(horizontal = 6.dp, vertical = 3.dp)
        ) {
            Text(
                "#$index",
                style = MaterialTheme.typography.labelSmall.copy(
                    color = ARIAColors.Primary, fontWeight = FontWeight.Bold, fontSize = 10.sp
                )
            )
        }
        Column(modifier = Modifier.weight(1f)) {
            Text(
                task.goal,
                style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface),
                maxLines = 2
            )
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                if (task.appPackage.isNotBlank()) {
                    Text(
                        task.appPackage.substringAfterLast('.'),
                        style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp)
                    )
                }
                Text(
                    fmt.format(Date(task.enqueuedAt)),
                    style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp)
                )
                if (task.priority != 0) {
                    Text(
                        "p${task.priority}",
                        style = MaterialTheme.typography.labelSmall.copy(
                            color = ARIAColors.Warning, fontSize = 10.sp
                        )
                    )
                }
            }
        }
        IconButton(onClick = onRemove, modifier = Modifier.size(28.dp)) {
            Icon(
                Icons.Default.RemoveCircleOutline,
                contentDescription = "Remove",
                tint = ARIAColors.Error,
                modifier = Modifier.size(15.dp)
            )
        }
    }
}

@Composable
private fun ReadinessRow(label: String, ok: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface))
        Icon(
            if (ok) Icons.Default.CheckCircle else Icons.Default.Cancel,
            contentDescription = null,
            tint = if (ok) ARIAColors.Success else ARIAColors.Error,
            modifier = Modifier.size(16.dp)
        )
    }
}

// ─── Agent Mode Card ──────────────────────────────────────────────────────────

@Composable
private fun AgentModeCard(
    selectedCombo:      AgentCombo,
    rlVisionMode:       ModuleVisionMode,
    irlVisionMode:      ModuleVisionMode,
    enabled:            Boolean,
    onComboChange:      (AgentCombo) -> Unit,
    onBothVisionChange: (ModuleVisionMode) -> Unit,
    onRlVisionChange:   (ModuleVisionMode) -> Unit,
    onIrlVisionChange:  (ModuleVisionMode) -> Unit,
) {
    var showCustom       by remember { mutableStateOf(false) }
    var showVisionConfig by remember { mutableStateOf(true) }
    // Show per-module overrides when RL and IRL differ (user had set them differently before)
    var showPerModule    by remember { mutableStateOf(rlVisionMode != irlVisionMode) }

    // Local toggles for custom mode — mirror the selected combo
    var customLlm by remember(selectedCombo) { mutableStateOf(selectedCombo.usesLlm) }
    var customRl  by remember(selectedCombo) { mutableStateOf(selectedCombo.usesRl) }
    var customIrl by remember(selectedCombo) { mutableStateOf(selectedCombo.usesIrl) }

    // Resolve three booleans to a combo; if all are false (meaningless), fall back to
    // LLM_ONLY (safest single-engine default) rather than silently turning everything on.
    fun resolveCombo(llm: Boolean, rl: Boolean, irl: Boolean): AgentCombo =
        AgentCombo.entries.firstOrNull {
            it.usesLlm == llm && it.usesRl == rl && it.usesIrl == irl
        } ?: AgentCombo.LLM_ONLY

    ARIACard {
        // ── Header ──────────────────────────────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "AGENT MODE",
                style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
            )
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                // Active module badges
                if (selectedCombo.usesLlm) ModuleBadge("LLM", ARIAColors.Primary)
                if (selectedCombo.usesRl)  ModuleBadge("RL",  ARIAColors.Accent)
                if (selectedCombo.usesIrl) ModuleBadge("IRL", ARIAColors.Warning)
            }
        }

        Spacer(Modifier.height(10.dp))

        // ── Preset combo chips ─────────────────────────────────────────────
        Text(
            "Preset",
            style = MaterialTheme.typography.labelSmall.copy(
                color = ARIAColors.Muted, fontSize = 10.sp
            )
        )
        Spacer(Modifier.height(6.dp))
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            AgentCombo.entries.forEach { combo ->
                val isSelected = combo == selectedCombo && !showCustom
                FilterChip(
                    selected = isSelected,
                    onClick  = {
                        if (enabled) {
                            showCustom = false
                            onComboChange(combo)
                        }
                    },
                    label = {
                        Text(
                            combo.label,
                            style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp)
                        )
                    },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = ARIAColors.Primary.copy(alpha = 0.18f),
                        selectedLabelColor     = ARIAColors.Primary,
                        containerColor         = ARIAColors.Surface,
                        labelColor             = ARIAColors.OnSurface,
                    ),
                    border = FilterChipDefaults.filterChipBorder(
                        enabled       = true,
                        selected      = isSelected,
                        borderColor          = ARIAColors.Divider,
                        selectedBorderColor  = ARIAColors.Primary.copy(alpha = 0.50f),
                    ),
                    enabled = enabled
                )
            }
            // Custom chip
            FilterChip(
                selected = showCustom,
                onClick  = { if (enabled) showCustom = !showCustom },
                label    = {
                    Text(
                        "Custom…",
                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp)
                    )
                },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = ARIAColors.Accent.copy(alpha = 0.18f),
                    selectedLabelColor     = ARIAColors.Accent,
                    containerColor         = ARIAColors.Surface,
                    labelColor             = ARIAColors.OnSurface,
                ),
                border = FilterChipDefaults.filterChipBorder(
                    enabled      = true,
                    selected     = showCustom,
                    borderColor         = ARIAColors.Divider,
                    selectedBorderColor = ARIAColors.Accent.copy(alpha = 0.50f),
                ),
                enabled = enabled
            )
        }

        // ── Custom toggles (expanded when Custom is active) ────────────────
        if (showCustom) {
            Spacer(Modifier.height(10.dp))
            HorizontalDivider(color = ARIAColors.Divider)
            Spacer(Modifier.height(10.dp))
            Text(
                "CUSTOM COMBINATION",
                style = MaterialTheme.typography.labelSmall.copy(
                    color = ARIAColors.Accent, fontSize = 10.sp, fontWeight = FontWeight.Bold
                )
            )
            Spacer(Modifier.height(8.dp))
            // At least one module must remain on — prevent all-off (meaningless combo).
            val activeCount = listOf(customLlm, customRl, customIrl).count { it }
            ModuleToggleRow(
                label    = "LLM  (language model reasoning)",
                checked  = customLlm,
                // Disable the switch when turning it off would leave nothing active
                enabled  = enabled && (customLlm || activeCount < 3),
                color    = ARIAColors.Primary,
                onChange = { v ->
                    if (!v && activeCount <= 1) return@ModuleToggleRow  // block last-off
                    customLlm = v
                    onComboChange(resolveCombo(v, customRl, customIrl))
                }
            )
            ModuleToggleRow(
                label    = "RL  (policy network — fast actions)",
                checked  = customRl,
                enabled  = enabled && (customRl || activeCount < 3),
                color    = ARIAColors.Accent,
                onChange = { v ->
                    if (!v && activeCount <= 1) return@ModuleToggleRow
                    customRl = v
                    onComboChange(resolveCombo(customLlm, v, customIrl))
                }
            )
            ModuleToggleRow(
                label    = "IRL  (learn from screen recordings)",
                checked  = customIrl,
                enabled  = enabled && (customIrl || activeCount < 3),
                color    = ARIAColors.Warning,
                onChange = { v ->
                    if (!v && activeCount <= 1) return@ModuleToggleRow
                    customIrl = v
                    onComboChange(resolveCombo(customLlm, customRl, v))
                }
            )
            // Show the resolved preset name
            val resolved = resolveCombo(customLlm, customRl, customIrl)
            Spacer(Modifier.height(6.dp))
            Text(
                "→ Maps to preset: ${resolved.label}",
                style = MaterialTheme.typography.labelSmall.copy(
                    color = ARIAColors.Muted, fontSize = 10.sp
                )
            )
        }

        // ── IRL_ONLY warning ──────────────────────────────────────────────────
        // IRL only runs when the user imports a screen recording — it never fires
        // automatically in the agent loop. With IRL_ONLY selected there is no action
        // engine (no LLM, no policy net), so the agent cannot act on the device.
        if (!selectedCombo.usesLlm && !selectedCombo.usesRl && selectedCombo.usesIrl) {
            Spacer(Modifier.height(8.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(ARIAColors.Warning.copy(alpha = 0.10f))
                    .padding(8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment     = Alignment.Top
            ) {
                Icon(
                    Icons.Default.Info,
                    contentDescription = null,
                    tint = ARIAColors.Warning,
                    modifier = Modifier.size(14.dp).padding(top = 1.dp)
                )
                Text(
                    "IRL Only has no action engine — the agent cannot tap or swipe. " +
                    "IRL runs when you import a screen recording, not during live sessions. " +
                    "Consider RL + IRL or All modules instead.",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = ARIAColors.Warning, lineHeight = 16.sp
                    )
                )
            }
        }

        // ── Vision mode sub-section ────────────────────────────────────────
        Spacer(Modifier.height(10.dp))
        HorizontalDivider(color = ARIAColors.Divider)
        Spacer(Modifier.height(8.dp))
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clickable(
                    interactionSource = remember { MutableInteractionSource() },
                    indication        = null,
                    onClick           = { showVisionConfig = !showVisionConfig }
                ),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically
        ) {
            Text(
                "VISION MODE",
                style = MaterialTheme.typography.labelSmall.copy(
                    color = ARIAColors.Muted, fontSize = 10.sp
                )
            )
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Show current effective mode (use rlVisionMode as canonical when both are synced)
                val effectiveLabel = if (rlVisionMode == irlVisionMode)
                    rlVisionMode.label.substringBefore(" —")
                else "Custom"
                Text(
                    effectiveLabel,
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = ARIAColors.Primary, fontSize = 10.sp
                    )
                )
                Icon(
                    if (showVisionConfig) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    contentDescription = null,
                    tint = ARIAColors.Muted,
                    modifier = Modifier.size(16.dp)
                )
            }
        }

        if (showVisionConfig) {
            Spacer(Modifier.height(8.dp))

            // Single unified selector — applies to both RL reward and IRL processing
            VisionModeRow(
                label    = "Vision mode",
                sublabel = "How RL reward scoring and IRL video processing use the vision model",
                selected = if (rlVisionMode == irlVisionMode) rlVisionMode else ModuleVisionMode.AUTO,
                enabled  = enabled,
                onChange = { mode ->
                    onBothVisionChange(mode)
                    showPerModule = false
                },
            )

            Spacer(Modifier.height(4.dp))

            // "Customize per-module" toggle row
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable(
                        interactionSource = remember { MutableInteractionSource() },
                        indication        = null,
                        onClick           = { showPerModule = !showPerModule }
                    )
                    .padding(vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically
            ) {
                Text(
                    "Different settings per module",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = if (showPerModule) ARIAColors.Primary else ARIAColors.Muted,
                        fontSize = 10.sp
                    )
                )
                Icon(
                    if (showPerModule) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    contentDescription = null,
                    tint = if (showPerModule) ARIAColors.Primary else ARIAColors.Muted,
                    modifier = Modifier.size(14.dp)
                )
            }

            // Per-module overrides (advanced)
            if (showPerModule) {
                Spacer(Modifier.height(4.dp))
                VisionModeRow(
                    label    = "RL reward + PolicyNetwork",
                    sublabel = "How RL grades action quality and whether PolicyNetwork calls the vision model",
                    selected = rlVisionMode,
                    enabled  = enabled,
                    onChange = onRlVisionChange,
                )
                Spacer(Modifier.height(6.dp))
                VisionModeRow(
                    label    = "IRL video processing",
                    sublabel = "How imported screen recordings are analysed (does not affect the live agent loop)",
                    selected = irlVisionMode,
                    enabled  = enabled,
                    onChange = onIrlVisionChange,
                )
            }
        }
    }
}

@Composable
private fun ModuleBadge(label: String, color: Color) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(4.dp))
            .background(color.copy(alpha = 0.14f))
            .padding(horizontal = 5.dp, vertical = 2.dp)
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelSmall.copy(
                color = color, fontSize = 9.sp, fontWeight = FontWeight.Bold
            )
        )
    }
}

@Composable
private fun ModuleToggleRow(
    label:    String,
    checked:  Boolean,
    enabled:  Boolean,
    color:    Color,
    onChange: (Boolean) -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment     = Alignment.CenterVertically
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface),
            modifier = Modifier.weight(1f)
        )
        Switch(
            checked         = checked,
            onCheckedChange = onChange,
            enabled         = enabled,
            modifier        = Modifier.height(24.dp),
            colors          = SwitchDefaults.colors(
                checkedThumbColor = color,
                checkedTrackColor = color.copy(alpha = 0.40f),
            )
        )
    }
}

@Composable
private fun VisionModeRow(
    label:    String,
    sublabel: String,
    selected: ModuleVisionMode,
    enabled:  Boolean,
    onChange: (ModuleVisionMode) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall.copy(
                color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold
            )
        )
        Text(
            sublabel,
            style = MaterialTheme.typography.labelSmall.copy(
                color = ARIAColors.Muted, fontSize = 10.sp
            )
        )
        FlowRow(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            ModuleVisionMode.entries.forEach { mode ->
                val isSelected = mode == selected
                FilterChip(
                    selected = isSelected,
                    onClick  = { if (enabled) onChange(mode) },
                    label    = {
                        Text(
                            mode.label.substringBefore(" —"),
                            style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp)
                        )
                    },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = ARIAColors.Primary.copy(alpha = 0.16f),
                        selectedLabelColor     = ARIAColors.Primary,
                        containerColor         = ARIAColors.Surface,
                        labelColor             = ARIAColors.OnSurface,
                    ),
                    border = FilterChipDefaults.filterChipBorder(
                        enabled      = true,
                        selected     = isSelected,
                        borderColor         = ARIAColors.Divider,
                        selectedBorderColor = ARIAColors.Primary.copy(alpha = 0.40f),
                    ),
                    enabled = enabled
                )
            }
        }
    }
}

@Composable
private fun InfoBanner(message: String, color: Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(color.copy(alpha = 0.12f), RoundedCornerShape(8.dp))
            .padding(10.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(Icons.Default.Info, contentDescription = null, tint = color, modifier = Modifier.size(16.dp))
        Text(message, style = MaterialTheme.typography.bodySmall.copy(color = color))
    }
}
