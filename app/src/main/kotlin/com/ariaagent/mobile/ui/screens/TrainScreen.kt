package com.ariaagent.mobile.ui.screens

import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.ariaagent.mobile.ui.theme.ARIAColors
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel

/**
 * TrainScreen — Migration Phase 6.
 *
 * Pure Kotlin + Jetpack Compose replacement for train.tsx (692 lines).
 * All calls go through AgentViewModel → IrlModule / LoraTrainer / PolicyNetwork.
 * No bridge. No JS.
 *
 * Features:
 *  - RL Status card: LoRA version, adapter path, untrained samples, adam step, policy loss, refresh
 *  - Run RL Cycle card: run button with CircularProgressIndicator, result display
 *  - Video Training (IRL) card: video picker, goal field, app package field, run button, results
 *  - Navigate to Labeler: nav card
 *  - NEW: Auto-schedule RL toggle (triggers when untrainedSamples > 50)
 *  - NEW: Last trained timestamp
 */
@Composable
fun TrainScreen(
    vm: AgentViewModel,
    onNavigateToLabeler: (() -> Unit)? = null,
) {
    val statusUi         by vm.learningStatusUi.collectAsState()
    val rlRunning        by vm.rlRunning.collectAsState()
    val rlResult         by vm.rlResult.collectAsState()
    val irlRunning       by vm.irlRunning.collectAsState()
    val irlResult        by vm.irlResult.collectAsState()
    val irlExtracting    by vm.irlExtracting.collectAsState()
    val irlFramePaths    by vm.irlFramePaths.collectAsState()
    val autoSchedule     by vm.autoScheduleRl.collectAsState()
    val schedulerActive  by vm.schedulerActive.collectAsState()
    val loraHistory      by vm.loraHistory.collectAsState()
    val loraProgress     by vm.loraTrainingProgress.collectAsState()

    var videoUri      by remember { mutableStateOf<Uri?>(null) }
    var videoName     by remember { mutableStateOf("") }
    var irlGoal       by remember { mutableStateOf("") }
    var irlApp        by remember { mutableStateOf("") }

    var showAnnotationDialog by remember { mutableStateOf(false) }

    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            videoUri  = uri
            videoName = uri.lastPathSegment ?: "video.mp4"
        }
    }

    LaunchedEffect(Unit) { vm.refreshLearningStatus(); vm.refreshLoraHistory() }

    // Open annotation dialog once frame extraction finishes
    LaunchedEffect(irlFramePaths) {
        if (irlFramePaths.isNotEmpty()) showAnnotationDialog = true
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp)
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(0.dp),
    ) {
        Spacer(Modifier.height(20.dp))

        TrainHeader(
            loadingStatus = false,
            onRefresh     = { vm.refreshLearningStatus() },
        )

        Spacer(Modifier.height(24.dp))

        SectionLabel("On-Device RL Status")
        Spacer(Modifier.height(8.dp))

        RlStatusCard(
            status          = statusUi,
            rlRunning       = rlRunning,
            rlResult        = rlResult,
            autoSchedule    = autoSchedule,
            schedulerActive = schedulerActive,
            onRunRl         = { vm.runRlCycle() },
            onAutoSchedule  = { vm.setAutoScheduleRl(it) },
        )

        Spacer(Modifier.height(24.dp))

        SectionLabel("Video Training (IRL)")
        Spacer(Modifier.height(8.dp))

        IrlCard(
            videoUri      = videoUri,
            videoName     = videoName,
            irlGoal       = irlGoal,
            irlApp        = irlApp,
            irlRunning    = irlRunning,
            irlExtracting = irlExtracting,
            irlResult     = irlResult,
            onPickVideo   = { videoPicker.launch("video/*") },
            onClearVideo  = { videoUri = null; videoName = "" },
            onGoalChange  = { irlGoal = it },
            onAppChange   = { irlApp  = it },
            onAnnotateFrames = {
                videoUri?.let { uri -> vm.extractIrlFrames(uri.toString()) }
            },
            onRunIrl      = {
                videoUri?.let { uri ->
                    vm.processIrlVideo(
                        videoUri   = uri.toString(),
                        goal       = irlGoal.trim(),
                        appPackage = irlApp.trim().ifBlank { "com.android.chrome" },
                    )
                }
            },
        )

        // Per-frame annotation dialog — shown after frame extraction completes
        if (showAnnotationDialog && irlFramePaths.isNotEmpty()) {
            IrlFrameAnnotationDialog(
                framePaths = irlFramePaths,
                onFinish = { annotations ->
                    showAnnotationDialog = false
                    vm.clearIrlFrames()
                    videoUri?.let { uri ->
                        vm.processIrlVideo(
                            videoUri         = uri.toString(),
                            goal             = irlGoal.trim(),
                            appPackage       = irlApp.trim().ifBlank { "com.android.chrome" },
                            frameAnnotations = annotations,
                        )
                    }
                },
                onDismiss = {
                    showAnnotationDialog = false
                    vm.clearIrlFrames()
                },
            )
        }

        Spacer(Modifier.height(24.dp))

        SectionLabel("Screenshot Labeling")
        Spacer(Modifier.height(8.dp))

        LabelerShortcutCard(onNavigate = onNavigateToLabeler)

        // ── Active training progress ──────────────────────────────────────────
        if (loraProgress != null || rlRunning) {
            Spacer(Modifier.height(24.dp))
            SectionLabel("Active Training")
            Spacer(Modifier.height(8.dp))
            LoraProgressCard(progress = loraProgress, rlRunning = rlRunning)
        }

        // ── Checkpoint history ────────────────────────────────────────────────
        Spacer(Modifier.height(24.dp))
        SectionLabel("LoRA Checkpoints")
        Spacer(Modifier.height(8.dp))
        LoraHistoryCard(checkpoints = loraHistory, onRefresh = { vm.refreshLoraHistory() })

        Spacer(Modifier.height(32.dp))
    }
}

// ─── Header ───────────────────────────────────────────────────────────────────

@Composable
private fun TrainHeader(loadingStatus: Boolean, onRefresh: () -> Unit) {
    Row(
        modifier              = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment     = Alignment.Top,
    ) {
        Column {
            Text("Training", color = ARIAColors.TextPrimary, fontSize = 26.sp, fontWeight = FontWeight.Bold)
            Text("Teach ARIA from video or on-device RL", color = ARIAColors.TextMuted, fontSize = 13.sp)
        }
        IconButton(
            onClick  = onRefresh,
            enabled  = !loadingStatus,
            modifier = Modifier
                .size(36.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(ARIAColors.Surface2),
        ) {
            if (loadingStatus) {
                CircularProgressIndicator(Modifier.size(14.dp), color = ARIAColors.TextMuted, strokeWidth = 2.dp)
            } else {
                Icon(Icons.Default.Refresh, contentDescription = "Refresh", tint = ARIAColors.TextMuted, modifier = Modifier.size(16.dp))
            }
        }
    }
}

// ─── Section label ────────────────────────────────────────────────────────────

@Composable
private fun SectionLabel(title: String) {
    Text(
        text       = title.uppercase(),
        color      = ARIAColors.TextMuted,
        fontSize   = 10.sp,
        fontWeight = FontWeight.SemiBold,
        fontFamily = FontFamily.Monospace,
        letterSpacing = 1.sp,
    )
}

// ─── RL Status card ───────────────────────────────────────────────────────────

@Composable
private fun RlStatusCard(
    status: com.ariaagent.mobile.ui.viewmodel.LearningStatusUi?,
    rlRunning: Boolean,
    rlResult: com.ariaagent.mobile.ui.viewmodel.RlResultUi?,
    autoSchedule: Boolean,
    schedulerActive: Boolean,
    onRunRl: () -> Unit,
    onAutoSchedule: (Boolean) -> Unit,
) {
    TrainCard {
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            StatBadge(
                label  = "LoRA Version",
                value  = status?.loraVersion?.toString() ?: "—",
                accent = status?.adapterExists == true,
                modifier = Modifier.weight(1f),
            )
            StatBadge(
                label  = "Untrained",
                value  = status?.untrainedSamples?.toString() ?: "—",
                accent = (status?.untrainedSamples ?: 0) > 0,
                modifier = Modifier.weight(1f),
            )
            StatBadge(
                label  = "Policy",
                value  = if (status?.policyReady == true) "Ready" else "Not ready",
                accent = status?.policyReady == true,
                modifier = Modifier.weight(1f),
            )
        }

        // Bug #7 fix: capture stable local val before the null check so a concurrent
        // recomposition setting status = null can't cause a NullPointerException on !!
        val adapterPath = status?.latestAdapterPath
        if (!adapterPath.isNullOrBlank()) {
            Text(
                text     = adapterPath.substringAfterLast('/'),
                color    = ARIAColors.TextMuted,
                fontSize = 11.sp,
                maxLines = 1,
            )
        }

        if (status != null && status.adamStep > 0) {
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Text("Adam step: ${status.adamStep}", color = ARIAColors.TextMuted, fontSize = 11.sp)
                Text("Policy loss: %.4f".format(status.lastPolicyLoss), color = ARIAColors.TextMuted, fontSize = 11.sp)
            }
        }

        if (status != null && status.lastTrainedAt > 0L) {
            val ago = formatTimeAgo(status.lastTrainedAt)
            Text("Last trained: $ago", color = ARIAColors.TextMuted, fontSize = 11.sp)
        }

        rlResult?.let { result ->
            ResultBox(success = result.success) {
                if (result.success) {
                    Text("✓ RL Cycle complete", color = ARIAColors.Success, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    Text("${result.samplesUsed} samples used · LoRA v${result.loraVersion}", color = ARIAColors.TextPrimary, fontSize = 12.sp)
                    if (result.adapterPath.isNotBlank()) {
                        Text(result.adapterPath.substringAfterLast('/'), color = ARIAColors.TextMuted, fontSize = 11.sp, maxLines = 1)
                    }
                } else {
                    Text("✗ RL Cycle failed", color = ARIAColors.Error, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    Text(result.errorMessage.ifBlank { "Unknown error" }, color = ARIAColors.TextMuted, fontSize = 12.sp)
                }
            }
        }

        Button(
            onClick  = onRunRl,
            enabled  = !rlRunning,
            modifier = Modifier.fillMaxWidth(),
            shape    = RoundedCornerShape(12.dp),
            colors   = ButtonDefaults.buttonColors(
                containerColor = ARIAColors.Primary,
                disabledContainerColor = ARIAColors.Surface2,
            ),
        ) {
            if (rlRunning) {
                CircularProgressIndicator(Modifier.size(16.dp), color = ARIAColors.TextPrimary, strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Text("Running RL cycle…", color = ARIAColors.TextMuted)
            } else {
                Icon(Icons.Default.FlashOn, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(8.dp))
                Text("Run RL Cycle Now", color = ARIAColors.Background, fontWeight = FontWeight.SemiBold)
            }
        }

        Text(
            text     = "Triggers REINFORCE + Adam update on PolicyNetwork using stored experience tuples. Best run when charging and idle.",
            color    = ARIAColors.TextMuted,
            fontSize = 11.sp,
            lineHeight = 16.sp,
        )

        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically,
        ) {
            Column {
                Text("Auto-schedule RL", color = ARIAColors.TextPrimary, fontSize = 13.sp)
                Text("Runs when untrained samples > 50", color = ARIAColors.TextMuted, fontSize = 11.sp)
            }
            Switch(
                checked         = autoSchedule,
                onCheckedChange = onAutoSchedule,
                colors          = SwitchDefaults.colors(
                    checkedThumbColor       = ARIAColors.Background,
                    checkedTrackColor       = ARIAColors.Primary,
                    uncheckedThumbColor     = ARIAColors.TextMuted,
                    uncheckedTrackColor     = ARIAColors.Surface3,
                ),
            )
        }

        // ── Auto-schedule status pill ─────────────────────────────────────────
        // Shows whether LearningScheduler is currently running a training cycle.
        // Driven by scheduler_training_started / scheduler_training_stopped events.
        Row(
            modifier              = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .background(
                    if (schedulerActive) ARIAColors.Accent.copy(alpha = 0.10f)
                    else if (autoSchedule) ARIAColors.Primary.copy(alpha = 0.07f)
                    else ARIAColors.Surface2
                )
                .padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment     = Alignment.CenterVertically,
        ) {
            if (schedulerActive) {
                CircularProgressIndicator(
                    modifier    = Modifier.size(12.dp),
                    color       = ARIAColors.Accent,
                    strokeWidth = 2.dp,
                )
            } else {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(androidx.compose.foundation.shape.CircleShape)
                        .background(if (autoSchedule) ARIAColors.Primary else ARIAColors.Surface3)
                )
            }
            Column {
                Text(
                    text = when {
                        schedulerActive -> "Auto-training in progress…"
                        autoSchedule    -> "Waiting — will train when charging + idle + samples > 50"
                        else            -> "Auto-training disabled"
                    },
                    color    = when {
                        schedulerActive -> ARIAColors.Accent
                        autoSchedule    -> ARIAColors.TextPrimary
                        else            -> ARIAColors.TextMuted
                    },
                    fontSize = 12.sp,
                )
                if (autoSchedule && !schedulerActive) {
                    Text(
                        "LearningScheduler is monitoring battery state",
                        color    = ARIAColors.TextMuted,
                        fontSize = 10.sp,
                    )
                }
            }
        }
    }
}

// ─── IRL Video card ───────────────────────────────────────────────────────────

@Composable
private fun IrlCard(
    videoUri: Uri?,
    videoName: String,
    irlGoal: String,
    irlApp: String,
    irlRunning: Boolean,
    irlExtracting: Boolean,
    irlResult: com.ariaagent.mobile.ui.viewmodel.IrlResultUi?,
    onPickVideo: () -> Unit,
    onClearVideo: () -> Unit,
    onGoalChange: (String) -> Unit,
    onAppChange: (String) -> Unit,
    onAnnotateFrames: () -> Unit,
    onRunIrl: () -> Unit,
) {
    TrainCard {
        Text(
            text       = "Record yourself completing a task on your device, then feed the video here. ARIA extracts (state→action) tuples using OCR + accessibility tree.",
            color      = ARIAColors.TextMuted,
            fontSize   = 13.sp,
            lineHeight = 19.sp,
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(ARIAColors.Surface2)
                .border(
                    width = 1.5.dp,
                    color = if (videoUri != null) ARIAColors.Primary.copy(alpha = 0.6f) else ARIAColors.Surface3,
                    shape = RoundedCornerShape(10.dp),
                )
        ) {
            Row(
                modifier              = Modifier
                    .fillMaxWidth()
                    .padding(14.dp),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Icon(
                    imageVector = if (videoUri != null) Icons.Default.Movie else Icons.Default.Upload,
                    contentDescription = null,
                    tint     = if (videoUri != null) ARIAColors.Primary else ARIAColors.TextMuted,
                    modifier = Modifier.size(22.dp),
                )
                Text(
                    text     = if (videoUri != null) videoName else "Tap to pick a video from gallery",
                    color    = if (videoUri != null) ARIAColors.TextPrimary else ARIAColors.TextMuted,
                    fontSize = 13.sp,
                    modifier = Modifier.weight(1f),
                    maxLines = 1,
                )
                if (videoUri != null) {
                    IconButton(onClick = onClearVideo, modifier = Modifier.size(24.dp)) {
                        Icon(Icons.Default.Close, contentDescription = "Clear", tint = ARIAColors.TextMuted, modifier = Modifier.size(16.dp))
                    }
                } else {
                    TextButton(onClick = onPickVideo) {
                        Text("Pick", color = ARIAColors.Primary, fontSize = 13.sp)
                    }
                }
            }
        }

        Text("Task Goal", color = ARIAColors.TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.Medium)
        OutlinedTextField(
            value         = irlGoal,
            onValueChange = onGoalChange,
            modifier      = Modifier.fillMaxWidth(),
            placeholder   = { Text("e.g. Open YouTube and play trending video", color = ARIAColors.TextMuted, fontSize = 13.sp) },
            maxLines      = 2,
            colors        = outlinedFieldColors(),
            shape         = RoundedCornerShape(8.dp),
        )

        Text(
            text = "App Package (optional)",
            color = ARIAColors.TextPrimary,
            fontSize = 13.sp,
            fontWeight = FontWeight.Medium,
        )
        OutlinedTextField(
            value         = irlApp,
            onValueChange = onAppChange,
            modifier      = Modifier.fillMaxWidth(),
            placeholder   = { Text("e.g. com.google.android.youtube", color = ARIAColors.TextMuted, fontSize = 13.sp) },
            singleLine    = true,
            colors        = outlinedFieldColors(),
            shape         = RoundedCornerShape(8.dp),
        )

        irlResult?.let { result ->
            ResultBox(success = result.errorMessage.isBlank()) {
                if (result.errorMessage.isBlank()) {
                    Text("✓ Video processed", color = ARIAColors.Success, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    Row(horizontalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.padding(top = 2.dp)) {
                        Text("${result.framesProcessed} frames", color = ARIAColors.TextPrimary, fontSize = 12.sp)
                        Text("${result.tuplesExtracted} tuples", color = ARIAColors.TextPrimary, fontSize = 12.sp)
                        Text("${result.llmAssistedCount} LLM-assisted", color = ARIAColors.TextPrimary, fontSize = 12.sp)
                    }
                    Text("Tuples stored — run an RL cycle to train on them.", color = ARIAColors.TextMuted, fontSize = 11.sp)
                } else {
                    Text("✗ Processing failed", color = ARIAColors.Error, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    Text(result.errorMessage, color = ARIAColors.TextMuted, fontSize = 12.sp)
                }
            }
        }

        val busy   = irlRunning || irlExtracting
        val canRun = !busy && videoUri != null && irlGoal.isNotBlank()

        // Annotate Frames — extract key frames first, then let user explain each one
        OutlinedButton(
            onClick  = onAnnotateFrames,
            enabled  = canRun,
            modifier = Modifier.fillMaxWidth(),
            shape    = RoundedCornerShape(12.dp),
            border   = BorderStroke(1.5.dp, if (canRun) ARIAColors.Primary.copy(alpha = 0.7f) else ARIAColors.Surface3),
        ) {
            if (irlExtracting) {
                CircularProgressIndicator(Modifier.size(14.dp), color = ARIAColors.Primary, strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Text("Extracting frames…", color = ARIAColors.TextMuted, fontSize = 13.sp)
            } else {
                Icon(Icons.Default.Edit, contentDescription = null, modifier = Modifier.size(15.dp), tint = if (canRun) ARIAColors.Primary else ARIAColors.TextMuted)
                Spacer(Modifier.width(6.dp))
                Text("Annotate Frames", color = if (canRun) ARIAColors.Primary else ARIAColors.TextMuted, fontSize = 13.sp, fontWeight = FontWeight.Medium)
            }
        }

        // Process Video — run directly without per-frame annotations
        Button(
            onClick  = onRunIrl,
            enabled  = canRun,
            modifier = Modifier.fillMaxWidth(),
            shape    = RoundedCornerShape(12.dp),
            colors   = ButtonDefaults.buttonColors(
                containerColor         = ARIAColors.Primary,
                disabledContainerColor = ARIAColors.Surface2,
            ),
        ) {
            if (irlRunning) {
                CircularProgressIndicator(Modifier.size(16.dp), color = ARIAColors.TextPrimary, strokeWidth = 2.dp)
                Spacer(Modifier.width(8.dp))
                Text("Extracting tuples…", color = ARIAColors.TextMuted)
            } else {
                Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(8.dp))
                Text("Process Video", color = if (canRun) ARIAColors.Background else ARIAColors.TextMuted, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

// ─── LoRA Training Progress Card ─────────────────────────────────────────────

@Composable
private fun LoraProgressCard(
    progress: com.ariaagent.mobile.ui.viewmodel.LoraTrainingProgress?,
    rlRunning: Boolean,
) {
    TrainCard {
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment     = Alignment.CenterVertically
        ) {
            CircularProgressIndicator(
                modifier    = androidx.compose.ui.Modifier.size(22.dp),
                color       = ARIAColors.Accent,
                strokeWidth = 2.5.dp,
            )
            Column(modifier = androidx.compose.ui.Modifier.weight(1f)) {
                Text(
                    progress?.phase?.replaceFirstChar { it.uppercaseChar() } ?: "Initialising…",
                    color      = ARIAColors.TextPrimary,
                    fontSize   = 13.sp,
                    fontWeight = FontWeight.SemiBold,
                )
                if (progress != null) {
                    Text(
                        "${progress.samplesUsed} samples · loss ${String.format("%.4f", progress.currentLoss)}",
                        color    = ARIAColors.TextMuted,
                        fontSize = 11.sp,
                    )
                }
            }
            if (progress != null) {
                Text(
                    "${progress.percentComplete}%",
                    color      = ARIAColors.Accent,
                    fontSize   = 14.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace,
                )
            }
        }
        if (progress != null && progress.percentComplete > 0) {
            LinearProgressIndicator(
                progress    = { progress.percentComplete / 100f },
                modifier    = androidx.compose.ui.Modifier
                    .fillMaxWidth()
                    .height(5.dp)
                    .clip(androidx.compose.foundation.shape.RoundedCornerShape(3.dp)),
                color       = ARIAColors.Accent,
            )
        }
    }
}

// ─── LoRA Checkpoint History Card ─────────────────────────────────────────────

@Composable
private fun LoraHistoryCard(
    checkpoints: List<com.ariaagent.mobile.ui.viewmodel.LoraCheckpointItem>,
    onRefresh: () -> Unit,
) {
    TrainCard {
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically,
        ) {
            Text(
                "${checkpoints.size} adapter${if (checkpoints.size != 1) "s" else ""} on disk",
                color    = ARIAColors.TextMuted,
                fontSize = 12.sp,
            )
            IconButton(onClick = onRefresh, modifier = androidx.compose.ui.Modifier.size(28.dp)) {
                Icon(Icons.Default.Refresh, contentDescription = "Refresh", tint = ARIAColors.TextMuted, modifier = androidx.compose.ui.Modifier.size(14.dp))
            }
        }

        if (checkpoints.isEmpty()) {
            Text(
                "No adapters found yet. Run an RL cycle to create the first checkpoint.",
                color      = ARIAColors.TextMuted,
                fontSize   = 12.sp,
            )
        } else {
            checkpoints.forEach { ckpt ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(androidx.compose.foundation.shape.RoundedCornerShape(8.dp))
                        .background(if (ckpt.isLatest) ARIAColors.Accent.copy(alpha = 0.08f) else ARIAColors.Surface2)
                        .border(
                            1.dp,
                            if (ckpt.isLatest) ARIAColors.Accent.copy(alpha = 0.4f) else ARIAColors.Surface3,
                            androidx.compose.foundation.shape.RoundedCornerShape(8.dp)
                        )
                        .padding(horizontal = 10.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment     = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Default.Memory,
                        contentDescription = null,
                        tint     = if (ckpt.isLatest) ARIAColors.Accent else ARIAColors.TextMuted,
                        modifier = androidx.compose.ui.Modifier.size(15.dp),
                    )
                    Column(modifier = androidx.compose.ui.Modifier.weight(1f)) {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment     = Alignment.CenterVertically,
                        ) {
                            Text(
                                "v${ckpt.version}",
                                color      = if (ckpt.isLatest) ARIAColors.Accent else ARIAColors.TextPrimary,
                                fontSize   = 13.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace,
                            )
                            if (ckpt.isLatest) {
                                Box(
                                    modifier = androidx.compose.ui.Modifier
                                        .clip(androidx.compose.foundation.shape.RoundedCornerShape(4.dp))
                                        .background(ARIAColors.Accent.copy(alpha = 0.2f))
                                        .padding(horizontal = 5.dp, vertical = 1.dp)
                                ) {
                                    Text("ACTIVE", color = ARIAColors.Accent, fontSize = 9.sp, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                        Text(
                            "${ckpt.sizeKb} KB · ${formatTimeAgo(ckpt.createdAt)}",
                            color    = ARIAColors.TextMuted,
                            fontSize = 10.sp,
                        )
                    }
                }
            }
        }
    }
}

// ─── Labeler shortcut card ────────────────────────────────────────────────────

@Composable
private fun LabelerShortcutCard(onNavigate: (() -> Unit)?) {
    Card(
        onClick   = { onNavigate?.invoke() },
        enabled   = onNavigate != null,
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(16.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface1),
        border    = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Surface3),
    ) {
        Row(
            modifier              = Modifier.padding(16.dp),
            verticalAlignment     = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(ARIAColors.Primary.copy(alpha = 0.13f)),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.AutoMirrored.Filled.Label, contentDescription = null, tint = ARIAColors.Primary, modifier = Modifier.size(22.dp))
            }
            Column(modifier = Modifier.weight(1f)) {
                Text("Object Labeler", color = ARIAColors.TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                Text(
                    "Capture the screen, tap UI elements to annotate them, enrich with the LLM. Labels are injected into every future agent prompt.",
                    color      = ARIAColors.TextMuted,
                    fontSize   = 12.sp,
                    lineHeight = 17.sp,
                )
            }
            Icon(Icons.Default.ChevronRight, contentDescription = null, tint = ARIAColors.TextMuted)
        }
    }
}

// ─── Shared composables ───────────────────────────────────────────────────────

@Composable
private fun TrainCard(content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(16.dp),
        colors   = CardDefaults.cardColors(containerColor = ARIAColors.Surface1),
        border   = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Surface3),
    ) {
        Column(
            modifier            = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            content             = content,
        )
    }
}

@Composable
private fun StatBadge(label: String, value: String, accent: Boolean, modifier: Modifier = Modifier) {
    Column(
        modifier            = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(if (accent) ARIAColors.Primary.copy(alpha = 0.11f) else ARIAColors.Surface2)
            .border(1.dp, if (accent) ARIAColors.Primary.copy(alpha = 0.4f) else ARIAColors.Surface3, RoundedCornerShape(10.dp))
            .padding(horizontal = 8.dp, vertical = 10.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(value, color = if (accent) ARIAColors.Primary else ARIAColors.TextPrimary, fontSize = 16.sp, fontWeight = FontWeight.Bold)
        Text(label, color = ARIAColors.TextMuted, fontSize = 10.sp, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
    }
}

@Composable
private fun ResultBox(success: Boolean, content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier            = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(if (success) ARIAColors.Success.copy(alpha = 0.07f) else ARIAColors.Error.copy(alpha = 0.07f))
            .border(1.dp, if (success) ARIAColors.Success.copy(alpha = 0.4f) else ARIAColors.Error.copy(alpha = 0.4f), RoundedCornerShape(10.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
        content             = content,
    )
}

@Composable
private fun outlinedFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor      = ARIAColors.Primary,
    unfocusedBorderColor    = ARIAColors.Surface3,
    focusedContainerColor   = ARIAColors.Surface2,
    unfocusedContainerColor = ARIAColors.Surface2,
    focusedTextColor        = ARIAColors.TextPrimary,
    unfocusedTextColor      = ARIAColors.TextPrimary,
)

// ─── IRL Frame Annotation Dialog ──────────────────────────────────────────────
//
// Shown after key-frame extraction completes. The user steps through each frame,
// optionally writes an explanation of the action logic, then taps Finish & Train.
//
// Controls:
//   Same as before — copies the annotation from the previous frame
//   Skip           — leaves this frame un-annotated (no entry in the map)
//   Next →         — saves current text and advances; on last frame becomes "Finish & Train"

@Composable
private fun IrlFrameAnnotationDialog(
    framePaths: List<String>,
    onFinish: (Map<Int, String>) -> Unit,
    onDismiss: () -> Unit,
) {
    val annotations = remember { mutableStateMapOf<Int, String>() }
    var currentIdx  by remember { mutableStateOf(0) }
    var currentText by remember { mutableStateOf("") }

    val total  = framePaths.size
    val isLast = currentIdx == total - 1

    fun saveAndAdvance() {
        if (currentText.isNotBlank()) annotations[currentIdx] = currentText.trim()
        if (isLast) {
            onFinish(annotations.toMap())
        } else {
            currentIdx++
            currentText = annotations[currentIdx] ?: ""
        }
    }

    fun skip() {
        annotations.remove(currentIdx)
        if (isLast) {
            onFinish(annotations.toMap())
        } else {
            currentIdx++
            currentText = annotations[currentIdx] ?: ""
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.95f)
                .wrapContentHeight()
                .clip(RoundedCornerShape(16.dp))
                .background(ARIAColors.Surface1)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // ── Header ──────────────────────────────────────────────────────────
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment     = Alignment.CenterVertically,
            ) {
                Text(
                    text       = "Annotate Frames",
                    color      = ARIAColors.TextPrimary,
                    fontSize   = 16.sp,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text       = "${currentIdx + 1} / $total",
                    color      = ARIAColors.TextMuted,
                    fontSize   = 13.sp,
                    fontFamily = FontFamily.Monospace,
                )
            }

            // ── Frame image ─────────────────────────────────────────────────────
            val framePath = framePaths.getOrNull(currentIdx)
            val bitmap = remember(framePath) {
                framePath?.let { runCatching { BitmapFactory.decodeFile(it) }.getOrNull() }
            }
            if (bitmap != null) {
                Image(
                    bitmap             = bitmap.asImageBitmap(),
                    contentDescription = "Frame ${currentIdx + 1}",
                    contentScale       = ContentScale.Fit,
                    modifier           = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 260.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(ARIAColors.Surface2),
                )
            } else {
                Box(
                    modifier         = Modifier
                        .fillMaxWidth()
                        .height(180.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(ARIAColors.Surface2),
                    contentAlignment = Alignment.Center,
                ) {
                    Text("Frame ${currentIdx + 1}", color = ARIAColors.TextMuted, fontSize = 13.sp)
                }
            }

            // ── Annotation text field ────────────────────────────────────────────
            OutlinedTextField(
                value         = currentText,
                onValueChange = { currentText = it },
                modifier      = Modifier.fillMaxWidth(),
                placeholder   = { Text("Explain the action logic for this frame…", color = ARIAColors.TextMuted, fontSize = 13.sp) },
                label         = { Text("Explanation (optional)", color = ARIAColors.TextMuted, fontSize = 11.sp) },
                minLines      = 2,
                maxLines      = 4,
                colors        = outlinedFieldColors(),
                shape         = RoundedCornerShape(8.dp),
            )

            // ── Button row ───────────────────────────────────────────────────────
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // Same as before
                OutlinedButton(
                    onClick  = {
                        if (currentIdx > 0) {
                            currentText = annotations[currentIdx - 1] ?: ""
                        }
                    },
                    enabled  = currentIdx > 0,
                    modifier = Modifier.weight(1f),
                    shape    = RoundedCornerShape(8.dp),
                    border   = BorderStroke(1.dp, ARIAColors.Surface3),
                ) {
                    Text("Same as before", color = ARIAColors.TextMuted, fontSize = 11.sp, maxLines = 1)
                }

                // Skip
                OutlinedButton(
                    onClick  = { skip() },
                    modifier = Modifier.weight(1f),
                    shape    = RoundedCornerShape(8.dp),
                    border   = BorderStroke(1.dp, ARIAColors.Surface3),
                ) {
                    Text("Skip", color = ARIAColors.TextMuted, fontSize = 11.sp)
                }

                // Next / Finish & Train
                Button(
                    onClick  = { saveAndAdvance() },
                    modifier = Modifier.weight(1f),
                    shape    = RoundedCornerShape(8.dp),
                    colors   = ButtonDefaults.buttonColors(containerColor = ARIAColors.Primary),
                ) {
                    Text(
                        text       = if (isLast) "Train" else "Next →",
                        color      = ARIAColors.Background,
                        fontSize   = 12.sp,
                        fontWeight = FontWeight.SemiBold,
                        maxLines   = 1,
                    )
                }
            }
        }
    }
}

private fun formatTimeAgo(ts: Long): String {
    val diff = System.currentTimeMillis() - ts
    return when {
        diff < 60_000L        -> "just now"
        diff < 3_600_000L     -> "${diff / 60_000L} min ago"
        diff < 86_400_000L    -> "${diff / 3_600_000L} hr ago"
        else                  -> "${diff / 86_400_000L} day(s) ago"
    }
}
