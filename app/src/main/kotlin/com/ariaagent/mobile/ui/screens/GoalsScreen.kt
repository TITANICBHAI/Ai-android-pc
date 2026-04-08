@file:OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)

package com.ariaagent.mobile.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.ariaagent.mobile.core.triggers.TriggerItem
import com.ariaagent.mobile.core.triggers.TriggerType
import com.ariaagent.mobile.ui.theme.ARIAColors
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel
import com.ariaagent.mobile.ui.viewmodel.QueuedTaskItem
import java.text.SimpleDateFormat
import java.util.*

/**
 * GoalsScreen — full-screen task queue management.
 *
 * Accessible from ControlScreen header "Goals →" button.
 * No bottom-nav entry — it's a focused full-screen editor.
 *
 * Tabs:
 *   Queue     — live task queue with priority badges, remove, clear-all
 *   Templates — preset task library tiles; tap to enqueue quickly
 *   Triggers  — placeholder for future scheduled/event-based triggers
 */

private val GOAL_TEMPLATES = listOf(
    Triple("Open YouTube",            "Watch trending videos",                "com.google.android.youtube"),
    Triple("Check Gmail",             "Read and summarize new emails",        "com.google.android.gm"),
    Triple("Open Settings",           "Check available storage",              "com.android.settings"),
    Triple("Open Google Maps",        "Find nearby restaurants",              "com.google.android.apps.maps"),
    Triple("Open Chrome",             "Go to google.com",                     "com.android.chrome"),
    Triple("Open WhatsApp",           "Read the latest message",              "com.whatsapp"),
    Triple("Open Camera",             "Take a photo",                         "com.android.camera2"),
    Triple("Check Battery",           "Open Settings and check battery level","com.android.settings"),
    Triple("Open Spotify",            "Play a recommended playlist",          "com.spotify.music"),
    Triple("Open Play Store",         "Check for pending app updates",        "com.android.vending"),
    Triple("Open Calendar",           "Review today's events",                "com.google.android.calendar"),
    Triple("Open Calculator",         "Compute 15% tip on 85",               "com.android.calculator2"),
)

private enum class GoalsTab(val label: String, val icon: ImageVector) {
    Queue("Queue",       Icons.Default.Queue),
    Templates("Templates", Icons.Default.GridView),
    Triggers("Triggers",   Icons.Default.Schedule),
    Skills("Skills",       Icons.Default.School),
}

@Composable
fun GoalsScreen(
    vm: AgentViewModel,
    onBack: () -> Unit,
) {
    val taskQueue    by vm.taskQueue.collectAsStateWithLifecycle()
    val focusManager = LocalFocusManager.current

    var activeTab      by remember { mutableStateOf(GoalsTab.Queue) }
    var goalText       by remember { mutableStateOf("") }
    var appText        by remember { mutableStateOf("") }
    var priorityLevel  by remember { mutableIntStateOf(0) }
    var showClearConfirm by remember { mutableStateOf(false) }

    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            containerColor   = ARIAColors.Surface,
            title = { Text("Clear queue?", style = MaterialTheme.typography.titleMedium.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.Bold)) },
            text  = { Text("All ${taskQueue.size} queued tasks will be permanently removed.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
            confirmButton = {
                Button(onClick = { vm.clearTaskQueue(); showClearConfirm = false },
                    colors = ButtonDefaults.buttonColors(containerColor = ARIAColors.Destructive),
                    shape  = RoundedCornerShape(8.dp)) {
                    Text("Clear All", fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = { TextButton(onClick = { showClearConfirm = false }) { Text("Cancel", color = ARIAColors.Muted) } }
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
    ) {
        // ── Header ────────────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(ARIAColors.Surface)
                .padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = ARIAColors.OnSurface)
            }
            Column(modifier = Modifier.weight(1f)) {
                Text("GOALS", style = MaterialTheme.typography.titleLarge.copy(color = ARIAColors.Primary, fontWeight = FontWeight.Bold))
                Text("${taskQueue.size} task${if (taskQueue.size != 1) "s" else ""} queued", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
            }
            if (activeTab == GoalsTab.Queue && taskQueue.isNotEmpty()) {
                IconButton(onClick = { showClearConfirm = true }) {
                    Icon(Icons.Default.DeleteSweep, contentDescription = "Clear all", tint = ARIAColors.Destructive)
                }
            }
        }

        // ── Tab bar ───────────────────────────────────────────────────────────
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(ARIAColors.Surface)
                .padding(horizontal = 16.dp, vertical = 0.dp),
        ) {
            GoalsTab.entries.forEach { tab ->
                val selected = tab == activeTab
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null,
                            onClick = { activeTab = tab }
                        )
                        .padding(vertical = 10.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    Icon(
                        tab.icon,
                        contentDescription = tab.label,
                        tint = if (selected) ARIAColors.Primary else ARIAColors.Muted,
                        modifier = Modifier.size(18.dp)
                    )
                    Text(
                        tab.label,
                        style = MaterialTheme.typography.labelSmall.copy(
                            color = if (selected) ARIAColors.Primary else ARIAColors.Muted,
                            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
                        )
                    )
                    Box(
                        modifier = Modifier
                            .height(2.dp)
                            .fillMaxWidth(0.6f)
                            .clip(RoundedCornerShape(1.dp))
                            .background(if (selected) ARIAColors.Primary else ARIAColors.Background)
                    )
                }
            }
        }

        HorizontalDivider(color = ARIAColors.Divider, thickness = 0.5.dp)

        // ── Tab content ───────────────────────────────────────────────────────
        when (activeTab) {
            GoalsTab.Queue -> QueueTab(
                taskQueue   = taskQueue,
                goalText    = goalText,
                appText     = appText,
                priorityLevel = priorityLevel,
                onGoalChange  = { goalText = it },
                onAppChange   = { appText  = it },
                onPriorityChange = { priorityLevel = it },
                onEnqueue     = {
                    if (goalText.isNotBlank()) {
                        vm.enqueueTask(goalText.trim(), appText.trim(), priorityLevel)
                        goalText = ""; appText = ""; priorityLevel = 0
                        focusManager.clearFocus()
                    }
                },
                onRemove      = { vm.removeQueuedTask(it) },
            )
            GoalsTab.Templates -> TemplatesTab(
                onEnqueue = { goal, app ->
                    vm.enqueueTask(goal, app, 0)
                    activeTab = GoalsTab.Queue
                }
            )
            GoalsTab.Triggers -> TriggersTab(vm = vm)
            GoalsTab.Skills   -> SkillsTab(vm = vm)
        }
    }
}

// ─── Queue tab ────────────────────────────────────────────────────────────────

@Composable
private fun QueueTab(
    taskQueue: List<QueuedTaskItem>,
    goalText: String,
    appText: String,
    priorityLevel: Int,
    onGoalChange: (String) -> Unit,
    onAppChange: (String) -> Unit,
    onPriorityChange: (Int) -> Unit,
    onEnqueue: () -> Unit,
    onRemove: (String) -> Unit,
) {
    LazyColumn(
        modifier        = Modifier.fillMaxSize(),
        contentPadding  = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Add goal card
        item {
            GoalComposerCard(
                goalText      = goalText,
                appText       = appText,
                priorityLevel = priorityLevel,
                onGoalChange  = onGoalChange,
                onAppChange   = onAppChange,
                onPriorityChange = onPriorityChange,
                onEnqueue     = onEnqueue,
            )
        }

        if (taskQueue.isEmpty()) {
            item {
                Box(
                    modifier = Modifier.fillMaxWidth().padding(top = 40.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(Icons.Default.PlaylistAdd, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(48.dp))
                        Text("Queue is empty", style = MaterialTheme.typography.bodyLarge.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold))
                        Text("Add a goal above or pick from the Templates tab", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted), textAlign = androidx.compose.ui.text.style.TextAlign.Center)
                    }
                }
            }
        } else {
            item {
                Text(
                    "NEXT UP",
                    style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontFamily = FontFamily.Monospace, letterSpacing = 1.sp)
                )
            }
            items(taskQueue, key = { it.id }) { task ->
                QueueTaskRow(task = task, onRemove = { onRemove(task.id) })
            }
        }
    }
}

@Composable
private fun GoalComposerCard(
    goalText: String,
    appText: String,
    priorityLevel: Int,
    onGoalChange: (String) -> Unit,
    onAppChange: (String) -> Unit,
    onPriorityChange: (Int) -> Unit,
    onEnqueue: () -> Unit,
) {
    val focusManager = LocalFocusManager.current
    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(14.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text("ADD GOAL", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontFamily = FontFamily.Monospace, letterSpacing = 1.sp))

            OutlinedTextField(
                value         = goalText,
                onValueChange = onGoalChange,
                modifier      = Modifier.fillMaxWidth(),
                placeholder   = { Text("What should ARIA do?", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
                colors        = goalsFieldColors(),
                shape         = RoundedCornerShape(8.dp),
                minLines      = 2,
                maxLines      = 4,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
            )

            OutlinedTextField(
                value         = appText,
                onValueChange = onAppChange,
                modifier      = Modifier.fillMaxWidth(),
                placeholder   = { Text("App package (optional)", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
                colors        = goalsFieldColors(),
                shape         = RoundedCornerShape(8.dp),
                singleLine    = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { focusManager.clearFocus() }),
                leadingIcon   = { Icon(Icons.Default.Apps, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(16.dp)) },
            )

            // Priority selector
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("Priority:", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
                listOf(0 to "Normal", 1 to "High", 2 to "Critical").forEach { (level, label) ->
                    val selected = priorityLevel == level
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(
                                if (selected) when (level) {
                                    2 -> ARIAColors.Destructive.copy(alpha = 0.2f)
                                    1 -> ARIAColors.Warning.copy(alpha = 0.2f)
                                    else -> ARIAColors.Success.copy(alpha = 0.2f)
                                } else ARIAColors.SurfaceVariant
                            )
                            .border(
                                1.dp,
                                if (selected) when (level) {
                                    2 -> ARIAColors.Destructive.copy(alpha = 0.6f)
                                    1 -> ARIAColors.Warning.copy(alpha = 0.6f)
                                    else -> ARIAColors.Success.copy(alpha = 0.6f)
                                } else ARIAColors.Divider,
                                RoundedCornerShape(6.dp)
                            )
                            .clickable(
                                interactionSource = remember { MutableInteractionSource() },
                                indication = null,
                                onClick = { onPriorityChange(level) }
                            )
                            .padding(horizontal = 10.dp, vertical = 5.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            label,
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = if (selected) when (level) {
                                    2 -> ARIAColors.Destructive
                                    1 -> ARIAColors.Warning
                                    else -> ARIAColors.Success
                                } else ARIAColors.Muted,
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
                            )
                        )
                    }
                }
            }

            Button(
                onClick  = onEnqueue,
                enabled  = goalText.isNotBlank(),
                modifier = Modifier.fillMaxWidth(),
                shape    = RoundedCornerShape(8.dp),
                colors   = ButtonDefaults.buttonColors(
                    containerColor         = ARIAColors.Primary,
                    disabledContainerColor = ARIAColors.SurfaceVariant,
                )
            ) {
                Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(6.dp))
                Text("Add to Queue", fontWeight = FontWeight.SemiBold, color = if (goalText.isNotBlank()) ARIAColors.Background else ARIAColors.Muted)
            }
        }
    }
}

@Composable
private fun QueueTaskRow(task: QueuedTaskItem, onRemove: () -> Unit) {
    val fmt = remember { SimpleDateFormat("HH:mm", Locale.getDefault()) }
    val priorityColor = when (task.priority) {
        2 -> ARIAColors.Destructive
        1 -> ARIAColors.Warning
        else -> ARIAColors.Success
    }
    val priorityLabel = when (task.priority) {
        2 -> "CRITICAL"
        1 -> "HIGH"
        else -> "NORMAL"
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(ARIAColors.Surface)
            .padding(12.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.Top
    ) {
        // Priority indicator
        Box(
            modifier = Modifier
                .size(6.dp)
                .offset(y = 6.dp)
                .clip(CircleShape)
                .background(priorityColor)
        )
        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(task.goal, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface), maxLines = 3)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                if (task.appPackage.isNotBlank()) {
                    Text(task.appPackage.substringAfterLast('.'), style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Primary, fontSize = 10.sp))
                }
                Text(fmt.format(Date(task.enqueuedAt)), style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp))
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(3.dp))
                        .background(priorityColor.copy(alpha = 0.15f))
                        .padding(horizontal = 4.dp, vertical = 1.dp)
                ) {
                    Text(priorityLabel, style = MaterialTheme.typography.labelSmall.copy(color = priorityColor, fontWeight = FontWeight.Bold, fontSize = 9.sp))
                }
            }
        }
        IconButton(onClick = onRemove, modifier = Modifier.size(28.dp)) {
            Icon(Icons.Default.RemoveCircleOutline, contentDescription = "Remove", tint = ARIAColors.Destructive, modifier = Modifier.size(16.dp))
        }
    }
}

// ─── Templates tab ────────────────────────────────────────────────────────────

@Composable
private fun TemplatesTab(onEnqueue: (goal: String, app: String) -> Unit) {
    var enqueuedLabel by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(enqueuedLabel) {
        if (enqueuedLabel != null) {
            kotlinx.coroutines.delay(1500)
            enqueuedLabel = null
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        enqueuedLabel?.let { label ->
            Row(
                modifier = Modifier.fillMaxWidth().background(ARIAColors.Success.copy(alpha = 0.12f)).padding(horizontal = 16.dp, vertical = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = ARIAColors.Success, modifier = Modifier.size(16.dp))
                Text("\"$label\" added to queue", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Success))
            }
        }

        LazyVerticalGrid(
            columns         = GridCells.Fixed(2),
            modifier        = Modifier.fillMaxSize(),
            contentPadding  = PaddingValues(12.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalArrangement   = Arrangement.spacedBy(10.dp)
        ) {
            items(GOAL_TEMPLATES) { (title, goal, app) ->
                TemplateCard(title = title, goal = goal, app = app, onClick = {
                    onEnqueue(goal, app)
                    enqueuedLabel = title
                })
            }
        }
    }
}

@Composable
private fun TemplateCard(title: String, goal: String, app: String, onClick: () -> Unit) {
    Card(
        onClick   = onClick,
        modifier  = Modifier.fillMaxWidth().aspectRatio(1.2f),
        shape     = RoundedCornerShape(12.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        border    = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Divider),
    ) {
        Column(
            modifier = Modifier.fillMaxSize().padding(12.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Icon(Icons.Default.PlayArrow, contentDescription = null, tint = ARIAColors.Primary, modifier = Modifier.size(20.dp))
            Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Text(title, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold), maxLines = 2)
                Text(app.substringAfterLast('.'), style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp))
            }
        }
    }
}

// ─── Triggers tab ─────────────────────────────────────────────────────────────

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TriggersTab(vm: AgentViewModel) {
    val triggers by vm.triggers.collectAsStateWithLifecycle()
    val focusManager = LocalFocusManager.current

    LaunchedEffect(Unit) { vm.loadTriggers() }

    var showCreateForm by remember { mutableStateOf(false) }
    var selectedType  by remember { mutableStateOf(TriggerType.TIME_DAILY) }
    var triggerGoal   by remember { mutableStateOf("") }
    var goalApp       by remember { mutableStateOf("") }
    var watchApp      by remember { mutableStateOf("") }
    var hourStr       by remember { mutableStateOf("09") }
    var minuteStr     by remember { mutableStateOf("00") }
    var dayOfWeek     by remember { mutableIntStateOf(2) }
    var deleteConfirm by remember { mutableStateOf<String?>(null) }

    // Bug #9 fix: capture a stable local val — dismiss sets deleteConfirm = null and a
    // concurrent recomposition would crash on the !! dereference inside the dialog lambdas.
    deleteConfirm?.let { dc ->
        AlertDialog(
            onDismissRequest = { deleteConfirm = null },
            containerColor   = ARIAColors.Surface,
            title = { Text("Delete trigger?", style = MaterialTheme.typography.titleMedium.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.Bold)) },
            text  = { Text("This trigger will be permanently removed.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
            confirmButton = {
                Button(onClick = { vm.deleteTrigger(dc); deleteConfirm = null },
                    colors = ButtonDefaults.buttonColors(containerColor = ARIAColors.Destructive),
                    shape  = RoundedCornerShape(8.dp)) {
                    Text("Delete", fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = { TextButton(onClick = { deleteConfirm = null }) { Text("Cancel", color = ARIAColors.Muted) } }
        )
    }

    LazyColumn(
        modifier        = Modifier.fillMaxSize(),
        contentPadding  = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {

        // ── Create trigger card ────────────────────────────────────────────────
        item {
            Card(
                modifier  = Modifier.fillMaxWidth(),
                shape     = RoundedCornerShape(14.dp),
                colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
                elevation = CardDefaults.cardElevation(0.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("ADD TRIGGER",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color = ARIAColors.Muted, fontFamily = FontFamily.Monospace, letterSpacing = 1.sp))
                        IconButton(
                            onClick = { showCreateForm = !showCreateForm },
                            modifier = Modifier.size(28.dp)
                        ) {
                            Icon(
                                if (showCreateForm) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                                contentDescription = if (showCreateForm) "Collapse" else "Expand",
                                tint = ARIAColors.Muted,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }

                    if (!showCreateForm) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(ARIAColors.SurfaceVariant)
                                .clickable(
                                    interactionSource = remember { MutableInteractionSource() },
                                    indication        = null,
                                    onClick           = { showCreateForm = true }
                                )
                                .padding(12.dp),
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(Icons.Default.Add, contentDescription = null, tint = ARIAColors.Primary, modifier = Modifier.size(16.dp))
                            Text("Tap to create a new trigger",
                                style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
                        }
                    } else {
                        // Trigger type chips
                        Text("Type", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            listOf(
                                TriggerType.TIME_DAILY  to "Daily",
                                TriggerType.TIME_ONCE   to "Once",
                                TriggerType.TIME_WEEKLY to "Weekly",
                                TriggerType.APP_LAUNCH  to "App Launch",
                                TriggerType.CHARGING    to "Charging",
                            ).forEach { (type, label) ->
                                val sel = selectedType == type
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(6.dp))
                                        .background(if (sel) ARIAColors.Primary.copy(alpha = 0.18f) else ARIAColors.SurfaceVariant)
                                        .border(1.dp, if (sel) ARIAColors.Primary.copy(alpha = 0.6f) else ARIAColors.Divider, RoundedCornerShape(6.dp))
                                        .clickable(
                                            interactionSource = remember { MutableInteractionSource() },
                                            indication = null,
                                            onClick = { selectedType = type }
                                        )
                                        .padding(horizontal = 10.dp, vertical = 5.dp)
                                ) {
                                    Text(label, style = MaterialTheme.typography.labelSmall.copy(
                                        color = if (sel) ARIAColors.Primary else ARIAColors.Muted,
                                        fontWeight = if (sel) FontWeight.Bold else FontWeight.Normal
                                    ))
                                }
                            }
                        }

                        // Goal input
                        OutlinedTextField(
                            value         = triggerGoal,
                            onValueChange = { triggerGoal = it },
                            modifier      = Modifier.fillMaxWidth(),
                            placeholder   = { Text("Goal for ARIA to run", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
                            colors        = goalsFieldColors(),
                            shape         = RoundedCornerShape(8.dp),
                            minLines      = 2,
                            maxLines      = 3,
                            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
                            label = { Text("Goal", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)) }
                        )

                        // Goal app package (optional)
                        OutlinedTextField(
                            value         = goalApp,
                            onValueChange = { goalApp = it },
                            modifier      = Modifier.fillMaxWidth(),
                            placeholder   = { Text("com.example.app (optional)", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
                            colors        = goalsFieldColors(),
                            shape         = RoundedCornerShape(8.dp),
                            singleLine    = true,
                            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
                            label = { Text("App package for goal", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)) },
                            leadingIcon = { Icon(Icons.Default.Apps, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(16.dp)) }
                        )

                        // Time-based fields
                        if (selectedType == TriggerType.TIME_DAILY ||
                            selectedType == TriggerType.TIME_ONCE  ||
                            selectedType == TriggerType.TIME_WEEKLY) {
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedTextField(
                                    value         = hourStr,
                                    onValueChange = { if (it.length <= 2 && it.all { c -> c.isDigit() }) hourStr = it },
                                    modifier      = Modifier.weight(1f),
                                    label = { Text("Hour (0-23)", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)) },
                                    colors        = goalsFieldColors(),
                                    shape         = RoundedCornerShape(8.dp),
                                    singleLine    = true,
                                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number, imeAction = ImeAction.Next),
                                )
                                OutlinedTextField(
                                    value         = minuteStr,
                                    onValueChange = { if (it.length <= 2 && it.all { c -> c.isDigit() }) minuteStr = it },
                                    modifier      = Modifier.weight(1f),
                                    label = { Text("Minute (0-59)", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)) },
                                    colors        = goalsFieldColors(),
                                    shape         = RoundedCornerShape(8.dp),
                                    singleLine    = true,
                                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number, imeAction = ImeAction.Next),
                                )
                            }
                        }

                        // Weekly day selector
                        if (selectedType == TriggerType.TIME_WEEKLY) {
                            Text("Day of week", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
                            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                                listOf(1 to "Su", 2 to "Mo", 3 to "Tu", 4 to "We", 5 to "Th", 6 to "Fr", 7 to "Sa")
                                    .forEach { (d, label) ->
                                        val sel = dayOfWeek == d
                                        Box(
                                            modifier = Modifier
                                                .size(36.dp)
                                                .clip(CircleShape)
                                                .background(if (sel) ARIAColors.Primary else ARIAColors.SurfaceVariant)
                                                .clickable(
                                                    interactionSource = remember { MutableInteractionSource() },
                                                    indication = null,
                                                    onClick = { dayOfWeek = d }
                                                ),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Text(label, style = MaterialTheme.typography.labelSmall.copy(
                                                color = if (sel) ARIAColors.Background else ARIAColors.Muted,
                                                fontWeight = if (sel) FontWeight.Bold else FontWeight.Normal,
                                                fontSize = 10.sp
                                            ))
                                        }
                                    }
                            }
                        }

                        // App launch watcher field
                        if (selectedType == TriggerType.APP_LAUNCH) {
                            OutlinedTextField(
                                value         = watchApp,
                                onValueChange = { watchApp = it },
                                modifier      = Modifier.fillMaxWidth(),
                                placeholder   = { Text("com.whatsapp", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
                                colors        = goalsFieldColors(),
                                shape         = RoundedCornerShape(8.dp),
                                singleLine    = true,
                                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                                keyboardActions = KeyboardActions(onDone = { focusManager.clearFocus() }),
                                label = { Text("Watch for app launch", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted)) },
                                leadingIcon = { Icon(Icons.Default.Visibility, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(16.dp)) }
                            )
                        }

                        // Save button
                        Button(
                            onClick = {
                                if (triggerGoal.isBlank()) return@Button
                                val h = hourStr.toIntOrNull()?.coerceIn(0, 23) ?: 9
                                val m = minuteStr.toIntOrNull()?.coerceIn(0, 59) ?: 0
                                vm.addTrigger(TriggerItem(
                                    type           = selectedType,
                                    goal           = triggerGoal.trim(),
                                    goalAppPackage = goalApp.trim(),
                                    watchPackage   = watchApp.trim(),
                                    hourOfDay      = h,
                                    minuteOfHour   = m,
                                    dayOfWeek      = dayOfWeek,
                                    enabled        = true,
                                ))
                                triggerGoal = ""; goalApp = ""; watchApp = ""; hourStr = "09"; minuteStr = "00"; dayOfWeek = 2
                                showCreateForm = false
                                focusManager.clearFocus()
                            },
                            enabled  = triggerGoal.isNotBlank(),
                            modifier = Modifier.fillMaxWidth(),
                            shape    = RoundedCornerShape(8.dp),
                            colors   = ButtonDefaults.buttonColors(
                                containerColor         = ARIAColors.Primary,
                                disabledContainerColor = ARIAColors.SurfaceVariant,
                            )
                        ) {
                            Icon(Icons.Default.Schedule, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(Modifier.width(6.dp))
                            Text("Save Trigger", fontWeight = FontWeight.SemiBold,
                                color = if (triggerGoal.isNotBlank()) ARIAColors.Background else ARIAColors.Muted)
                        }
                    }
                }
            }
        }

        // ── Active triggers list ───────────────────────────────────────────────
        if (triggers.isEmpty()) {
            item {
                Box(
                    modifier = Modifier.fillMaxWidth().padding(top = 32.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Icon(Icons.Default.Schedule, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(48.dp))
                        Text("No triggers yet", style = MaterialTheme.typography.bodyLarge.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold))
                        Text("Create a trigger above to automate tasks", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted), textAlign = TextAlign.Center)
                    }
                }
            }
        } else {
            item {
                Text("ACTIVE TRIGGERS",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color = ARIAColors.Muted, fontFamily = FontFamily.Monospace, letterSpacing = 1.sp))
            }
            items(triggers, key = { it.id }) { trigger ->
                TriggerRow(
                    trigger   = trigger,
                    onToggle  = { vm.toggleTrigger(trigger.id) },
                    onDelete  = { deleteConfirm = trigger.id },
                )
            }
        }
    }
}

@Composable
private fun TriggerRow(
    trigger: TriggerItem,
    onToggle: () -> Unit,
    onDelete: () -> Unit,
) {
    val typeIcon = when (trigger.type) {
        TriggerType.TIME_ONCE, TriggerType.TIME_DAILY, TriggerType.TIME_WEEKLY -> Icons.Default.Schedule
        TriggerType.APP_LAUNCH -> Icons.Default.Launch
        TriggerType.CHARGING   -> Icons.Default.BatteryFull
    }
    val typeLabel = when (trigger.type) {
        TriggerType.TIME_ONCE    -> "Once · ${trigger.displayTime}"
        TriggerType.TIME_DAILY   -> "Daily · ${trigger.displayTime}"
        TriggerType.TIME_WEEKLY  -> "${trigger.dayLabel} · ${trigger.displayTime}"
        TriggerType.APP_LAUNCH   -> "App launch · ${trigger.watchPackage.substringAfterLast('.').ifBlank { trigger.watchPackage }}"
        TriggerType.CHARGING     -> "On charging"
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(ARIAColors.Surface)
            .border(
                1.dp,
                if (trigger.enabled) ARIAColors.Primary.copy(alpha = 0.25f) else ARIAColors.Divider,
                RoundedCornerShape(10.dp)
            )
            .padding(12.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.Top
    ) {
        Box(
            modifier = Modifier
                .size(34.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(
                    if (trigger.enabled) ARIAColors.Primary.copy(alpha = 0.15f)
                    else ARIAColors.Muted.copy(alpha = 0.1f)
                ),
            contentAlignment = Alignment.Center
        ) {
            Icon(typeIcon, contentDescription = null,
                tint = if (trigger.enabled) ARIAColors.Primary else ARIAColors.Muted,
                modifier = Modifier.size(16.dp))
        }

        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(trigger.goal,
                style = MaterialTheme.typography.bodySmall.copy(
                    color = if (trigger.enabled) ARIAColors.OnSurface else ARIAColors.Muted),
                maxLines = 2)
            Text(typeLabel,
                style = MaterialTheme.typography.labelSmall.copy(
                    color = if (trigger.enabled) ARIAColors.Primary else ARIAColors.Muted,
                    fontSize = 10.sp))
            if (trigger.goalAppPackage.isNotBlank()) {
                Text(trigger.goalAppPackage.substringAfterLast('.'),
                    style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp))
            }
        }

        Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Switch(
                checked  = trigger.enabled,
                onCheckedChange = { onToggle() },
                modifier = Modifier.height(24.dp),
                colors   = SwitchDefaults.colors(
                    checkedThumbColor       = ARIAColors.Background,
                    checkedTrackColor       = ARIAColors.Primary,
                    uncheckedThumbColor     = ARIAColors.Muted,
                    uncheckedTrackColor     = ARIAColors.SurfaceVariant,
                )
            )
            IconButton(onClick = onDelete, modifier = Modifier.size(24.dp)) {
                Icon(Icons.Default.Delete, contentDescription = "Delete",
                    tint = ARIAColors.Destructive, modifier = Modifier.size(14.dp))
            }
        }
    }
}

// ─── Skills tab ───────────────────────────────────────────────────────────────

@Composable
private fun SkillsTab(vm: AgentViewModel) {
    val skills by vm.appSkills.collectAsStateWithLifecycle()
    var clearConfirm    by remember { mutableStateOf(false) }
    var deleteConfirm   by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) { vm.refreshAppSkills() }

    if (clearConfirm) {
        AlertDialog(
            onDismissRequest = { clearConfirm = false },
            containerColor   = ARIAColors.Surface,
            title = { Text("Clear all skills?", style = MaterialTheme.typography.titleMedium.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.Bold)) },
            text  = { Text("All ${skills.size} learned app skills will be permanently removed. ARIA will start fresh for every app.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
            confirmButton = {
                Button(onClick = { vm.clearAppSkills(); clearConfirm = false },
                    colors = ButtonDefaults.buttonColors(containerColor = ARIAColors.Destructive),
                    shape  = RoundedCornerShape(8.dp)) {
                    Text("Clear All", fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = { TextButton(onClick = { clearConfirm = false }) { Text("Cancel", color = ARIAColors.Muted) } }
        )
    }

    // Bug #9 fix: same race pattern as TriggersTab — stable local val prevents NPE.
    deleteConfirm?.let { dc ->
        AlertDialog(
            onDismissRequest = { deleteConfirm = null },
            containerColor   = ARIAColors.Surface,
            title = { Text("Remove skill?", style = MaterialTheme.typography.titleMedium.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.Bold)) },
            text  = { Text("ARIA's learned knowledge for ${dc.substringAfterLast('.')} will be removed.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
            confirmButton = {
                Button(onClick = { vm.deleteAppSkill(dc); deleteConfirm = null },
                    colors = ButtonDefaults.buttonColors(containerColor = ARIAColors.Destructive),
                    shape  = RoundedCornerShape(8.dp)) {
                    Text("Remove", fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = { TextButton(onClick = { deleteConfirm = null }) { Text("Cancel", color = ARIAColors.Muted) } }
        )
    }

    LazyColumn(
        modifier        = Modifier.fillMaxSize(),
        contentPadding  = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        item {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text("LEARNED SKILLS", style = MaterialTheme.typography.labelSmall.copy(
                        color = ARIAColors.Muted, fontFamily = FontFamily.Monospace, letterSpacing = 1.sp))
                    Text("${skills.size} app${if (skills.size != 1) "s" else ""} in registry",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
                }
                if (skills.isNotEmpty()) {
                    IconButton(onClick = { clearConfirm = true }) {
                        Icon(Icons.Default.DeleteSweep, contentDescription = "Clear all skills", tint = ARIAColors.Destructive)
                    }
                }
            }
        }

        if (skills.isEmpty()) {
            item {
                Box(modifier = Modifier.fillMaxWidth().padding(top = 40.dp), contentAlignment = Alignment.Center) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Icon(Icons.Default.School, contentDescription = null, tint = ARIAColors.Muted, modifier = Modifier.size(48.dp))
                        Text("No skills yet", style = MaterialTheme.typography.bodyLarge.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold))
                        Text("Run ARIA on any app to start learning", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted), textAlign = TextAlign.Center)
                    }
                }
            }
        } else {
            items(skills, key = { it.appPackage }) { skill ->
                AppSkillRow(skill = skill, onDelete = { deleteConfirm = skill.appPackage })
            }
        }
    }
}

@Composable
private fun AppSkillRow(
    skill: com.ariaagent.mobile.ui.viewmodel.AppSkillItem,
    onDelete: () -> Unit,
) {
    val successPct = (skill.successRate * 100).toInt()
    val rateColor  = when {
        successPct >= 80 -> ARIAColors.Success
        successPct >= 50 -> ARIAColors.Warning
        else             -> ARIAColors.Destructive
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(ARIAColors.Surface)
            .border(1.dp, ARIAColors.Divider, RoundedCornerShape(10.dp))
            .padding(12.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Box(
            modifier = Modifier
                .size(38.dp)
                .clip(RoundedCornerShape(8.dp))
                .background(ARIAColors.Primary.copy(alpha = 0.12f)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                skill.appName.firstOrNull()?.uppercaseChar()?.toString() ?: "?",
                style = MaterialTheme.typography.titleMedium.copy(color = ARIAColors.Primary, fontWeight = FontWeight.Bold)
            )
        }

        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(skill.appName, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold))
            Text(skill.appPackage.substringAfterLast('.'), style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp))

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(4.dp))
                        .background(rateColor.copy(alpha = 0.15f))
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text("$successPct% success", style = MaterialTheme.typography.labelSmall.copy(color = rateColor, fontWeight = FontWeight.Bold, fontSize = 9.sp))
                }
                Text("${skill.taskSuccess}✓ / ${skill.taskFailure}✗", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp))
                if (skill.avgSteps > 0f) {
                    Text("~${"%.1f".format(skill.avgSteps)} steps", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp))
                }
            }

            if (skill.learnedElements.isNotEmpty()) {
                Text(
                    skill.learnedElements.take(5).joinToString(", "),
                    style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Primary.copy(alpha = 0.8f), fontSize = 10.sp),
                    maxLines = 1,
                )
            }
        }

        IconButton(onClick = onDelete, modifier = Modifier.size(28.dp)) {
            Icon(Icons.Default.RemoveCircleOutline, contentDescription = "Remove skill", tint = ARIAColors.Destructive, modifier = Modifier.size(16.dp))
        }
    }
}

// ─── Field colors ─────────────────────────────────────────────────────────────

@Composable
private fun goalsFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor      = ARIAColors.Primary,
    unfocusedBorderColor    = ARIAColors.Divider,
    focusedContainerColor   = ARIAColors.SurfaceVariant,
    unfocusedContainerColor = ARIAColors.SurfaceVariant,
    focusedTextColor        = ARIAColors.OnSurface,
    unfocusedTextColor      = ARIAColors.OnSurface,
    cursorColor             = ARIAColors.Primary,
)
