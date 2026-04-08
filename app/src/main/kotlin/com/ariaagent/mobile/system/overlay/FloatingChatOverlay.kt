package com.ariaagent.mobile.system.overlay

import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.ariaagent.mobile.core.events.AgentEventBus
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.mapNotNull

// ARIA colour palette (replicated here so the overlay compiles without the main theme dependency)
private val ARIAPrimary    = Color(0xFF6C63FF)
private val ARIABackground = Color(0xFF0E0E1A)
private val ARIASurface    = Color(0xFF1A1A2E)
private val ARIASuccess    = Color(0xFF4CAF50)
private val ARIAWarning    = Color(0xFFFFC107)
private val ARIAMuted      = Color(0xFF888899)
private val ARIAOnSurface  = Color(0xFFEEEEFF)

/**
 * FloatingChatOverlay — the actual Compose UI drawn over every app.
 *
 * Layout (collapsed):
 *   ┌─────────────────────────────────────┐
 *   │  ● ARIA  [status pill]  [↕] [✕]    │  ← header bar
 *   └─────────────────────────────────────┘
 *
 * Layout (expanded):
 *   ┌─────────────────────────────────────┐
 *   │  ● ARIA  [RUNNING 12 steps]  [↕][✕]│
 *   ├─────────────────────────────────────┤
 *   │  ACTION: TapNode #5                 │
 *   │  REASON: "Tapping Submit button"    │
 *   ├─────────────────────────────────────┤
 *   │  [ Draw gesture on screen ]         │  ← gesture canvas hint
 *   ├─────────────────────────────────────┤
 *   │  ╔══════════════════════════╗ [→]  │  ← text instruction input
 *   │  ║ type instruction...      ║       │
 *   │  ╚══════════════════════════╝       │
 *   └─────────────────────────────────────┘
 *
 * Keyboard fix
 * ────────────
 * [onInputFocused] is called with true when the text field gains focus (so
 * FloatingChatService can clear FLAG_NOT_FOCUSABLE and let the keyboard attach),
 * and with false when focus is lost (flag is restored so taps pass through).
 */
@Composable
fun FloatingChatOverlay(
    onInstruction:       (String) -> Unit,
    onGestureAnnotation: (String) -> Unit,
    onDismiss:           () -> Unit,
    onInputFocused:      (Boolean) -> Unit = {}
) {
    var expanded     by remember { mutableStateOf(true) }
    var inputText    by remember { mutableStateOf("") }
    var lastAction   by remember { mutableStateOf("Waiting…") }
    var lastReason   by remember { mutableStateOf("") }
    var status       by remember { mutableStateOf("idle") }
    var stepCount    by remember { mutableStateOf(0) }
    var gestureHint  by remember { mutableStateOf("") }
    var gestureStart by remember { mutableStateOf(Offset.Zero) }

    // Subscribe to AgentEventBus for live step updates.
    // Single collector with a when() dispatcher replaces three separate LaunchedEffect(Unit)
    // blocks that each opened their own hot-flow subscription — reducing overhead from
    // 3 concurrent coroutines + 3 flow collectors to 1.
    LaunchedEffect(Unit) {
        AgentEventBus.flow.collect { (name, data) ->
            when (name) {
                "action_performed" -> {
                    val tool   = data["tool"]   as? String ?: ""
                    val node   = ((data["nodeId"] as? String) ?: "").take(60)
                    val reason = data["reason"] as? String ?: ""
                    lastAction = if (node.isNotBlank()) "$tool → $node" else tool
                    if (reason.isNotBlank()) lastReason = reason.take(80)
                }
                "agent_status_changed" -> {
                    status    = data["status"]    as? String ?: status
                    stepCount = (data["stepCount"] as? Int)  ?: stepCount
                }
            }
        }
    }

    val statusColor = when (status) {
        "running" -> ARIASuccess
        "paused"  -> ARIAWarning
        "error"   -> Color(0xFFFF5252)
        else      -> ARIAMuted
    }

    Box(
        modifier = Modifier
            .width(300.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(ARIABackground.copy(alpha = 0.95f))
            .border(1.dp, ARIAPrimary.copy(alpha = 0.4f), RoundedCornerShape(16.dp))
    ) {
        Column {
            // ── Header ────────────────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(ARIASurface.copy(alpha = 0.9f))
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(
                    verticalAlignment     = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(
                        Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(statusColor)
                    )
                    Text(
                        "ARIA",
                        style = MaterialTheme.typography.labelLarge.copy(
                            color      = ARIAPrimary,
                            fontWeight = FontWeight.Bold
                        )
                    )
                    Box(
                        Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(statusColor.copy(alpha = 0.18f))
                            .padding(horizontal = 6.dp, vertical = 2.dp)
                    ) {
                        Text(
                            if (status == "running") "${status.uppercase()} ·$stepCount" else status.uppercase(),
                            style = MaterialTheme.typography.labelSmall.copy(
                                color      = statusColor,
                                fontWeight = FontWeight.Bold,
                                fontSize   = 9.sp
                            )
                        )
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    IconButton(
                        onClick  = { expanded = !expanded },
                        modifier = Modifier.size(28.dp)
                    ) {
                        Icon(
                            if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                            contentDescription = "Collapse",
                            tint     = ARIAMuted,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                    IconButton(
                        onClick  = onDismiss,
                        modifier = Modifier.size(28.dp)
                    ) {
                        Icon(
                            Icons.Default.Close,
                            contentDescription = "Close",
                            tint     = ARIAMuted,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }

            // ── Expanded body ─────────────────────────────────────────────────
            AnimatedVisibility(
                visible = expanded,
                enter   = fadeIn() + expandVertically(),
                exit    = fadeOut() + shrinkVertically()
            ) {
                Column(
                    modifier            = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    // Live action card
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(ARIASurface)
                            .padding(10.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            "ACTION",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color    = ARIAPrimary,
                                fontSize = 9.sp
                            )
                        )
                        Text(
                            lastAction,
                            style = MaterialTheme.typography.bodySmall.copy(
                                color      = ARIAOnSurface,
                                fontFamily = FontFamily.Monospace,
                                fontSize   = 11.sp
                            )
                        )
                        if (lastReason.isNotBlank()) {
                            Text(
                                lastReason,
                                style = MaterialTheme.typography.bodySmall.copy(
                                    color    = ARIAMuted,
                                    fontSize = 10.sp
                                )
                            )
                        }
                    }

                    // Gesture canvas — user drags here to annotate a screen region
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(ARIASurface)
                            .border(
                                1.dp,
                                if (gestureHint.isNotBlank()) ARIASuccess.copy(alpha = 0.6f)
                                else ARIAMuted.copy(alpha = 0.3f),
                                RoundedCornerShape(10.dp)
                            )
                            .pointerInput(Unit) {
                                detectDragGestures(
                                    onDragStart = { offset ->
                                        gestureStart = offset
                                        gestureHint  = ""
                                    },
                                    onDragEnd = {
                                        if (gestureHint.isNotBlank()) {
                                            onGestureAnnotation(gestureHint)
                                        }
                                    },
                                    onDrag = { change, dragAmount ->
                                        change.consume()
                                        val dir = when {
                                            dragAmount.y < -5f -> "upward"
                                            dragAmount.y > 5f  -> "downward"
                                            dragAmount.x < -5f -> "leftward"
                                            dragAmount.x > 5f  -> "rightward"
                                            else               -> "tap"
                                        }
                                        val region = when {
                                            gestureStart.y < 150f -> "top"
                                            gestureStart.y > 350f -> "bottom"
                                            else                  -> "center"
                                        } + when {
                                            gestureStart.x < 100f -> "-left"
                                            gestureStart.x > 250f -> "-right"
                                            else                  -> ""
                                        }
                                        gestureHint = "User gestured $dir in the $region of the screen"
                                    }
                                )
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            if (gestureHint.isNotBlank()) gestureHint
                            else "✦  draw here to annotate a screen region",
                            style = MaterialTheme.typography.labelSmall.copy(
                                color    = if (gestureHint.isNotBlank()) ARIASuccess else ARIAMuted,
                                fontSize = 10.sp
                            )
                        )
                    }

                    // Text instruction input
                    // onFocusChanged → notifies FloatingChatService to toggle
                    // FLAG_NOT_FOCUSABLE so the soft keyboard can attach/detach.
                    Row(
                        modifier              = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment     = Alignment.CenterVertically
                    ) {
                        OutlinedTextField(
                            value         = inputText,
                            onValueChange = { inputText = it },
                            modifier      = Modifier
                                .weight(1f)
                                .onFocusChanged { state ->
                                    onInputFocused(state.isFocused)
                                },
                            placeholder = {
                                Text(
                                    "instruction to ARIA…",
                                    style = MaterialTheme.typography.bodySmall.copy(
                                        color    = ARIAMuted,
                                        fontSize = 11.sp
                                    )
                                )
                            },
                            textStyle = MaterialTheme.typography.bodySmall.copy(
                                color    = ARIAOnSurface,
                                fontSize = 11.sp
                            ),
                            singleLine = true,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor   = ARIAPrimary,
                                unfocusedBorderColor = ARIAMuted.copy(alpha = 0.4f),
                                cursorColor          = ARIAPrimary
                            ),
                            shape = RoundedCornerShape(10.dp)
                        )
                        IconButton(
                            onClick = {
                                if (inputText.isNotBlank()) {
                                    onInstruction(inputText.trim())
                                    inputText = ""
                                }
                            },
                            modifier = Modifier
                                .size(38.dp)
                                .clip(RoundedCornerShape(10.dp))
                                .background(ARIAPrimary)
                        ) {
                            Icon(
                                Icons.AutoMirrored.Filled.Send,
                                contentDescription = "Send",
                                tint     = Color.White,
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}
