package com.ariaagent.mobile.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.PlaylistPlay
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel
import com.ariaagent.mobile.ui.viewmodel.SuggestionBannerItem
import com.ariaagent.mobile.ui.theme.ARIAColors

/**
 * DashboardScreen — at-a-glance status panel.
 *
 * Shows:
 *  • Agent status pill (IDLE / RUNNING / PAUSED / DONE / ERROR)
 *  • Current task & target app
 *  • Step count + token rate
 *  • Thermal level banners (severe / critical)
 *  • Live step activity bar (observe → reason → act → store)
 *  • Live LLM token stream
 *  • Last LoRA + policy learning stats
 *  • [Phase 15] Chained task notification banner (dismissible)
 *  • [Phase 6/8] Game loop metrics card (when in game mode)
 *
 * Phase 11 — pure Compose. Phase 15 update: chained task + game loop.
 */
@Composable
fun DashboardScreen(vm: AgentViewModel = viewModel()) {
    val agentState         by vm.agentState.collectAsStateWithLifecycle()
    val thermalState       by vm.thermalState.collectAsStateWithLifecycle()
    val stepState          by vm.stepState.collectAsStateWithLifecycle()
    val learningState      by vm.learningState.collectAsStateWithLifecycle()
    val streamBuffer       by vm.streamBuffer.collectAsStateWithLifecycle()
    val chainedTask        by vm.chainedTask.collectAsStateWithLifecycle()
    val gameLoopState      by vm.gameLoopState.collectAsStateWithLifecycle()
    val pendingSuggestions by vm.pendingSuggestions.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) { vm.refreshPendingSuggestions() }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // ── Header ────────────────────────────────────────────────────────────
        Text(
            "ARIA AGENT",
            style = MaterialTheme.typography.headlineMedium.copy(
                color = ARIAColors.Primary,
                fontWeight = FontWeight.Bold,
                letterSpacing = 4.sp
            )
        )

        // ── T005: Proactive Goal Surfacing — suggestion banner ────────────────────
        val topSuggestion = pendingSuggestions.firstOrNull()
        if (topSuggestion != null) {
            SuggestionBanner(
                suggestion = topSuggestion,
                onAccept   = { vm.acceptSuggestion(topSuggestion) },
                onDismiss  = { vm.dismissSuggestion(topSuggestion.id) }
            )
        }

        // ── Phase 15: Chained task notification banner ─────────────────────────
        chainedTask?.let { chain ->
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(10.dp),
                colors = CardDefaults.cardColors(
                    containerColor = ARIAColors.Accent.copy(alpha = 0.12f)
                ),
                elevation = CardDefaults.cardElevation(0.dp)
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.AutoMirrored.Filled.PlaylistPlay,
                        contentDescription = null,
                        tint = ARIAColors.Accent,
                        modifier = Modifier.size(20.dp)
                    )
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "TASK CHAINED",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Accent,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        )
                        Text(
                            chain.goal,
                            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface),
                            maxLines = 2
                        )
                        if (chain.queueSize > 0) {
                            Text(
                                "${chain.queueSize} task${if (chain.queueSize == 1) "" else "s"} remaining",
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
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }
        }

        // ── Thermal banners ───────────────────────────────────────────────────
        if (thermalState.level == "severe") {
            ThermalBanner(
                "Cooling down — agent throttled",
                ARIAColors.Error,
                Icons.Default.Thermostat
            )
        }
        if (thermalState.level == "critical") {
            ThermalBanner(
                "Device critical — inference suspended",
                Color(0xFFDC2626),
                Icons.Default.Warning
            )
        }

        // ── Status card ───────────────────────────────────────────────────────
        ARIACard {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    StatusPill(status = agentState.status)
                    if (agentState.currentTask.isNotBlank()) {
                        Spacer(Modifier.height(6.dp))
                        Text(
                            agentState.currentTask,
                            style = MaterialTheme.typography.bodyMedium.copy(color = ARIAColors.OnSurface),
                            maxLines = 2
                        )
                    }
                    if (agentState.currentApp.isNotBlank()) {
                        Text(
                            "▸ ${agentState.currentApp}",
                            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                        )
                    }
                }
                Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    MetricChip(Icons.Default.Speed, "${String.format("%.1f", agentState.tokenRate)} t/s")
                    MetricChip(Icons.Default.Loop, "Step ${agentState.stepCount}")
                }
            }

            if (agentState.status == "running") {
                Spacer(Modifier.height(12.dp))
                StepActivityBar(activity = stepState.activity, stepNumber = stepState.stepNumber)
            }
        }

        // ── Live token stream ─────────────────────────────────────────────────
        if (agentState.status == "running" && streamBuffer.isNotBlank()) {
            ARIACard {
                Text(
                    "THINKING…",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = ARIAColors.Primary, letterSpacing = 1.sp
                    )
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    streamBuffer,
                    style = MaterialTheme.typography.bodySmall.copy(
                        fontFamily = FontFamily.Monospace,
                        color = ARIAColors.OnSurface,
                        lineHeight = 18.sp
                    ),
                    maxLines = 8
                )
            }
        }

        // ── Phase 6/8: Game loop card ─────────────────────────────────────────
        val gl = gameLoopState
        if (gl != null && (gl.isActive || agentState.gameMode != "none")) {
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
                            Icons.Default.SportsEsports,
                            contentDescription = null,
                            tint = ARIAColors.Accent,
                            modifier = Modifier.size(18.dp)
                        )
                        Text(
                            "GAME MODE — ${gl.gameType.uppercase()}",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Accent,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        )
                    }
                    if (gl.isGameOver) {
                        Text(
                            "GAME OVER",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Error, fontWeight = FontWeight.Bold
                            )
                        )
                    }
                }
                Spacer(Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    GameStat("Score",   String.format("%.0f", gl.currentScore))
                    GameStat("Best",    String.format("%.0f", gl.highScore))
                    GameStat("Episode", "${gl.episodeCount}")
                    GameStat("Reward",  String.format("%.1f", gl.totalReward))
                }
                if (gl.lastAction.isNotBlank()) {
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "Last: ${gl.lastAction}",
                        style = MaterialTheme.typography.labelSmall.copy(
                            color = ARIAColors.Muted,
                            fontFamily = FontFamily.Monospace
                        )
                    )
                }
            }
        }

        // ── System status ─────────────────────────────────────────────────────
        ARIACard {
            Text("SYSTEM", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
            Spacer(Modifier.height(10.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                StatusDotRow("Accessibility", agentState.accessibilityActive)
                StatusDotRow("Screen Cap",    agentState.screenCaptureActive)
                StatusDotRow("Model",         agentState.modelLoaded)
            }
            if (thermalState.level != "safe") {
                Spacer(Modifier.height(8.dp))
                Row(
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        Modifier.size(6.dp).clip(CircleShape)
                            .background(thermalColor(thermalState.level))
                    )
                    Text(
                        "Thermal: ${thermalState.level.uppercase()}",
                        style = MaterialTheme.typography.labelSmall.copy(
                            color = thermalColor(thermalState.level)
                        )
                    )
                }
            }
        }

        // ── Learning stats ────────────────────────────────────────────────────
        ARIACard {
            Text("ON-DEVICE LEARNING", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
            Spacer(Modifier.height(10.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                LearningStat("LoRA", "v${learningState.loraVersion}")
                LearningStat("Policy", "v${learningState.policyVersion}")
                LearningStat("Adam Steps", "${learningState.adamStep}")
                if (learningState.lastPolicyLoss > 0.0) {
                    LearningStat("Loss", String.format("%.4f", learningState.lastPolicyLoss))
                }
            }
            if (learningState.untrainedSamples > 0) {
                Spacer(Modifier.height(6.dp))
                Text(
                    "${learningState.untrainedSamples} samples pending training",
                    style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)
                )
            }
        }
    }
}

// ─── Shared composable helpers (used across screens) ──────────────────────────

@Composable
fun ARIACard(
    modifier: Modifier = Modifier,
    containerColor: Color = ARIAColors.Surface,
    content: @Composable ColumnScope.() -> Unit,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = containerColor),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp), content = content)
    }
}

@Composable
fun StatusPill(status: String) {
    val (bg, fg) = when (status) {
        "running" -> ARIAColors.Primary.copy(alpha = 0.18f) to ARIAColors.Primary
        "paused"  -> ARIAColors.Warning.copy(alpha = 0.18f) to ARIAColors.Warning
        "done"    -> ARIAColors.Success.copy(alpha = 0.18f) to ARIAColors.Success
        "error"   -> ARIAColors.Error.copy(alpha = 0.18f)   to ARIAColors.Error
        else      -> ARIAColors.Muted.copy(alpha = 0.14f)   to ARIAColors.Muted
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        modifier = Modifier
            .clip(CircleShape)
            .background(bg)
            .padding(horizontal = 10.dp, vertical = 4.dp)
    ) {
        Box(Modifier.size(6.dp).clip(CircleShape).background(fg))
        Text(
            status.uppercase(),
            style = MaterialTheme.typography.labelMedium.copy(
                color = fg, fontWeight = FontWeight.Bold, letterSpacing = 1.sp
            )
        )
    }
}

@Composable
fun MetricChip(icon: ImageVector, label: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Icon(icon, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(14.dp))
        Text(label, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
    }
}

@Composable
fun StatusIndicator(ok: Boolean, label: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        Box(
            Modifier.size(6.dp).clip(CircleShape)
                .background(if (ok) ARIAColors.Success else ARIAColors.Error)
        )
        Text(label, style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
    }
}

@Composable
fun StepActivityBar(activity: String, stepNumber: Int) {
    val steps = listOf("observe", "reason", "act", "store")
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        steps.forEach { step ->
            val active = step == activity
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Box(
                    Modifier
                        .size(width = 48.dp, height = 4.dp)
                        .clip(RoundedCornerShape(2.dp))
                        .background(if (active) ARIAColors.Primary else ARIAColors.Divider)
                )
                Spacer(Modifier.height(3.dp))
                Text(
                    step.uppercase(),
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = if (active) ARIAColors.Primary else ARIAColors.Muted,
                        fontSize = 9.sp,
                        fontWeight = if (active) FontWeight.Bold else FontWeight.Normal
                    )
                )
            }
        }
        Text(
            "#$stepNumber",
            style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 9.sp)
        )
    }
}

@Composable
private fun ThermalBanner(message: String, color: Color, icon: ImageVector) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(color.copy(alpha = 0.12f), RoundedCornerShape(8.dp))
            .padding(10.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(16.dp))
        Text(message, style = MaterialTheme.typography.bodySmall.copy(color = color))
    }
}

@Composable
private fun StatusDotRow(label: String, ok: Boolean) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            Modifier.size(6.dp).clip(CircleShape)
                .background(if (ok) ARIAColors.Success else ARIAColors.Error)
        )
        Text(label, style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
    }
}

@Composable
private fun GameStat(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, style = MaterialTheme.typography.bodyMedium.copy(
            color = ARIAColors.Primary, fontWeight = FontWeight.Bold))
        Text(label, style = MaterialTheme.typography.labelSmall.copy(
            color = ARIAColors.Muted, fontSize = 10.sp))
    }
}

@Composable
private fun LearningStat(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, style = MaterialTheme.typography.bodySmall.copy(
            color = ARIAColors.Accent, fontWeight = FontWeight.Bold))
        Text(label, style = MaterialTheme.typography.labelSmall.copy(
            color = ARIAColors.Muted, fontSize = 10.sp))
    }
}

@Composable
private fun SuggestionBanner(
    suggestion: SuggestionBannerItem,
    onAccept: () -> Unit,
    onDismiss: () -> Unit,
) {
    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(10.dp),
        colors    = CardDefaults.cardColors(
            containerColor = ARIAColors.Success.copy(alpha = 0.10f)
        ),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp)) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment     = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.Default.AutoAwesome,
                    contentDescription = null,
                    tint     = ARIAColors.Success,
                    modifier = Modifier.size(18.dp)
                )
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "SUGGESTION · seen ${suggestion.repeatCount}×",
                        style = MaterialTheme.typography.labelSmall.copy(
                            color      = ARIAColors.Success,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 0.5.sp
                        )
                    )
                    Text(
                        suggestion.suggestionText.ifBlank { suggestion.goalText },
                        style   = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface),
                        maxLines = 2
                    )
                }
                IconButton(
                    onClick  = onDismiss,
                    modifier = Modifier.size(28.dp)
                ) {
                    Icon(
                        Icons.Default.Close,
                        contentDescription = "Dismiss",
                        tint     = ARIAColors.Muted,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = onAccept,
                    colors  = ButtonDefaults.buttonColors(containerColor = ARIAColors.Success),
                    shape   = RoundedCornerShape(8.dp),
                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp),
                    modifier = Modifier.height(34.dp)
                ) {
                    Icon(
                        Icons.Default.PlayArrow,
                        contentDescription = null,
                        modifier = Modifier.size(14.dp)
                    )
                    Spacer(Modifier.width(4.dp))
                    Text(
                        "Automate it",
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                    )
                }
                TextButton(
                    onClick = onDismiss,
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 6.dp),
                    modifier = Modifier.height(34.dp)
                ) {
                    Text(
                        "Not now",
                        style = MaterialTheme.typography.labelMedium.copy(color = ARIAColors.Muted)
                    )
                }
            }
        }
    }
}

fun thermalColor(level: String): Color = when (level) {
    "light"    -> ARIAColors.Warning
    "moderate" -> Color(0xFFF97316)
    "severe"   -> ARIAColors.Error
    "critical" -> Color(0xFFDC2626)
    else       -> ARIAColors.Success
}
