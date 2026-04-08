package com.ariaagent.mobile.ui.screens

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.ariaagent.mobile.ui.theme.ARIAColors
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel
import com.ariaagent.mobile.ui.viewmodel.ChatMessageItem
import kotlinx.coroutines.launch

/**
 * ChatScreen — Migration Phase 5.
 *
 * Pure Kotlin + Jetpack Compose replacement for chat.tsx (601 lines).
 * Inference via AgentViewModel.sendChatMessage() → ChatContextBuilder → LlamaEngine.infer().
 * No bridge. No JS.
 *
 * Features:
 *  - LazyColumn message list, auto-scrolls to bottom on new message
 *  - User bubble (right-aligned, primary colour)
 *  - AI bubble (left-aligned, surface colour)
 *  - System message (centred, muted)
 *  - Welcome message on first open
 *  - Typing indicator — three animated dots (InfiniteTransition)
 *  - Context line at top — agent task / current app
 *  - OutlinedTextField input + Send button
 *  - Clear conversation button (with confirmation dialog)
 *  - Preset prompt chips (horizontal LazyRow)
 *  - LLM status pill in header
 *  - Token/sec rate shown beneath each AI response (NEW beyond RN)
 */
@Composable
fun ChatScreen(vm: AgentViewModel) {
    val agentState   by vm.agentState.collectAsState()
    val taskQueue    by vm.taskQueue.collectAsState()
    val appSkills    by vm.appSkills.collectAsState()
    val messages     by vm.chatMessages.collectAsState()
    val thinking     by vm.chatThinking.collectAsState()

    var input by remember { mutableStateOf("") }
    var showClearDialog by remember { mutableStateOf(false) }

    val listState    = rememberLazyListState()
    val scope        = rememberCoroutineScope()
    val llmLoaded    = agentState.modelLoaded

    val contextLine = remember(agentState, taskQueue, appSkills) {
        val parts = mutableListOf<String>()
        if (agentState.status != "idle") parts += "agent: ${agentState.status}"
        if (taskQueue.isNotEmpty()) parts += "${taskQueue.size} queued"
        if (appSkills.isNotEmpty()) parts += "${appSkills.size} skills"
        parts.joinToString(" · ").ifBlank { null }
    }

    LaunchedEffect(messages.size, thinking) {
        if (messages.isNotEmpty()) {
            scope.launch {
                listState.animateScrollToItem(
                    if (thinking) messages.size else (messages.size - 1).coerceAtLeast(0)
                )
            }
        }
    }

    if (showClearDialog) {
        AlertDialog(
            onDismissRequest  = { showClearDialog = false },
            title             = { Text("Clear conversation?", color = ARIAColors.TextPrimary) },
            text              = { Text("All messages will be removed.", color = ARIAColors.TextSecondary) },
            confirmButton     = {
                TextButton(onClick = { vm.clearChat(); showClearDialog = false }) {
                    Text("Clear", color = ARIAColors.Error)
                }
            },
            dismissButton     = { TextButton(onClick = { showClearDialog = false }) { Text("Cancel") } },
            containerColor    = ARIAColors.Surface1,
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
    ) {
        ChatHeader(
            llmLoaded     = llmLoaded,
            contextLine   = contextLine,
            hasMessages   = messages.size > 1,
            onClear       = { showClearDialog = true },
        )

        ContextTagBar(taskQueueCount = taskQueue.size, appSkillsCount = appSkills.size)

        LazyColumn(
            state                   = listState,
            modifier                = Modifier.weight(1f),
            contentPadding          = PaddingValues(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement     = Arrangement.spacedBy(8.dp),
        ) {
            items(messages, key = { it.id }) { msg ->
                ChatBubble(msg)
            }
            if (thinking) {
                item(key = "typing") { TypingIndicator() }
            }
        }

        PresetChips(
            enabled  = !thinking && llmLoaded,
            onSelect = { chip ->
                vm.sendChatMessage(chip)
            }
        )

        ChatInputBar(
            input     = input,
            onInput   = { input = it },
            canSend   = input.isNotBlank() && !thinking && llmLoaded,
            thinking  = thinking,
            llmLoaded = llmLoaded,
            onSend    = {
                val text = input.trim()
                if (text.isNotEmpty()) {
                    input = ""
                    vm.sendChatMessage(text)
                }
            },
        )
    }
}

// ─── Header ──────────────────────────────────────────────────────────────────

@Composable
private fun ChatHeader(
    llmLoaded: Boolean,
    contextLine: String?,
    hasMessages: Boolean,
    onClear: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(ARIAColors.Background)
            .padding(horizontal = 20.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.weight(1f)) {
            Box(
                modifier = Modifier
                    .size(34.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(ARIAColors.Primary.copy(alpha = 0.13f)),
                contentAlignment = Alignment.Center
            ) {
                Text("⚡", fontSize = 16.sp)
            }
            Column {
                Text(
                    text       = "Chat with ARIA",
                    color      = ARIAColors.TextPrimary,
                    fontWeight = FontWeight.Bold,
                    fontSize   = 17.sp,
                )
                Text(
                    text     = contextLine ?: if (llmLoaded) "LLM ready · context injected" else "LLM not loaded",
                    color    = ARIAColors.TextMuted,
                    fontSize = 11.sp,
                )
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            LlmStatusPill(llmLoaded)
            if (hasMessages) {
                IconButton(onClick = onClear, modifier = Modifier.size(34.dp)) {
                    Icon(Icons.Default.Clear, contentDescription = "Clear", tint = ARIAColors.TextMuted, modifier = Modifier.size(16.dp))
                }
            }
        }
    }
    HorizontalDivider(color = ARIAColors.Surface3, thickness = 0.5.dp)
}

@Composable
private fun LlmStatusPill(llmLoaded: Boolean) {
    val bgColor  = if (llmLoaded) ARIAColors.Success.copy(alpha = 0.13f) else ARIAColors.Error.copy(alpha = 0.11f)
    val dotColor = if (llmLoaded) ARIAColors.Success else ARIAColors.Error
    val txtColor = if (llmLoaded) ARIAColors.Success else ARIAColors.Error
    val label    = if (llmLoaded) "LLM ON" else "LLM OFF"

    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(bgColor)
            .padding(horizontal = 8.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(5.dp),
    ) {
        Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(dotColor))
        Text(label, color = txtColor, fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 0.4.sp)
    }
}

// ─── Context tag bar ──────────────────────────────────────────────────────────

@Composable
private fun ContextTagBar(taskQueueCount: Int, appSkillsCount: Int) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(ARIAColors.Surface1)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        ContextTag("Agent State")
        ContextTag("Memory")
        ContextTag("Task Queue${if (taskQueueCount > 0) " ($taskQueueCount)" else ""}")
        ContextTag("App Skills${if (appSkillsCount > 0) " ($appSkillsCount)" else ""}")
    }
    HorizontalDivider(color = ARIAColors.Surface3, thickness = 0.5.dp)
}

@Composable
private fun ContextTag(label: String) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(6.dp))
            .background(ARIAColors.Surface3)
            .padding(horizontal = 7.dp, vertical = 4.dp),
    ) {
        Text(label, color = ARIAColors.TextMuted, fontSize = 10.sp, fontWeight = FontWeight.Medium)
    }
}

// ─── Message bubbles ──────────────────────────────────────────────────────────

@Composable
private fun ChatBubble(msg: ChatMessageItem) {
    when (msg.role) {
        "system" -> {
            Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                Text(
                    text      = msg.text,
                    color     = ARIAColors.TextMuted,
                    fontSize  = 12.sp,
                    textAlign = TextAlign.Center,
                    modifier  = Modifier.fillMaxWidth(0.88f).padding(vertical = 6.dp),
                    lineHeight = 18.sp,
                )
            }
        }
        "user" -> {
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
            ) {
                Column(horizontalAlignment = Alignment.End) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(0.78f)
                            .clip(RoundedCornerShape(topStart = 18.dp, topEnd = 18.dp, bottomStart = 18.dp, bottomEnd = 4.dp))
                            .background(ARIAColors.Primary)
                            .padding(horizontal = 14.dp, vertical = 10.dp),
                    ) {
                        Text(msg.text, color = ARIAColors.Background, fontSize = 14.sp, lineHeight = 20.sp)
                    }
                }
            }
        }
        else -> {
            Row(
                modifier             = Modifier.fillMaxWidth(),
                verticalAlignment    = Alignment.Bottom,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Box(
                    modifier = Modifier
                        .size(30.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .background(ARIAColors.Primary.copy(alpha = 0.13f)),
                    contentAlignment = Alignment.Center,
                ) {
                    Text("⚡", fontSize = 13.sp)
                }
                Column {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(0.78f)
                            .clip(RoundedCornerShape(topStart = 14.dp, topEnd = 14.dp, bottomStart = 4.dp, bottomEnd = 14.dp))
                            .background(ARIAColors.Surface1)
                            .padding(horizontal = 14.dp, vertical = 10.dp),
                    ) {
                        Text(msg.text, color = ARIAColors.TextPrimary, fontSize = 14.sp, lineHeight = 20.sp)
                    }
                    if (msg.tps > 0.0) {
                        Text(
                            text     = "%.1f tok/s".format(msg.tps),
                            color    = ARIAColors.TextMuted,
                            fontSize = 10.sp,
                            modifier = Modifier.padding(start = 2.dp, top = 2.dp),
                        )
                    }
                }
            }
        }
    }
}

// ─── Typing indicator ─────────────────────────────────────────────────────────

@Composable
private fun TypingIndicator() {
    val transition = rememberInfiniteTransition(label = "typing")
    Row(
        modifier              = Modifier.fillMaxWidth(),
        verticalAlignment     = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(
            modifier = Modifier
                .size(30.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(ARIAColors.Primary.copy(alpha = 0.13f)),
            contentAlignment = Alignment.Center,
        ) { Text("⚡", fontSize = 13.sp) }

        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(topStart = 14.dp, topEnd = 14.dp, bottomStart = 4.dp, bottomEnd = 14.dp))
                .background(ARIAColors.Surface1)
                .padding(horizontal = 18.dp, vertical = 14.dp),
            contentAlignment = Alignment.Center,
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically) {
                repeat(3) { i ->
                    val alpha by transition.animateFloat(
                        initialValue   = 0.2f,
                        targetValue    = 1f,
                        animationSpec  = infiniteRepeatable(
                            animation  = tween(durationMillis = 600, delayMillis = i * 200, easing = LinearEasing),
                            repeatMode = RepeatMode.Reverse,
                        ),
                        label = "dot$i",
                    )
                    Box(
                        modifier = Modifier
                            .size(7.dp)
                            .clip(CircleShape)
                            .background(ARIAColors.Primary.copy(alpha = alpha))
                    )
                }
            }
        }
    }
}

// ─── Preset prompt chips ──────────────────────────────────────────────────────

private val PRESET_PROMPTS = listOf(
    "What are you doing?",
    "Pause and explain",
    "How is the model performing?",
    "What did you learn?",
    "Show memory summary",
)

@Composable
private fun PresetChips(enabled: Boolean, onSelect: (String) -> Unit) {
    LazyRow(
        contentPadding          = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement   = Arrangement.spacedBy(8.dp),
    ) {
        items(PRESET_PROMPTS) { prompt ->
            FilterChip(
                selected = false,
                onClick  = { if (enabled) onSelect(prompt) },
                label    = { Text(prompt, fontSize = 12.sp) },
                enabled  = enabled,
                colors   = FilterChipDefaults.filterChipColors(
                    containerColor         = ARIAColors.Surface1,
                    labelColor             = ARIAColors.TextSecondary,
                    selectedContainerColor = ARIAColors.PrimaryContainer,
                    selectedLabelColor     = ARIAColors.Primary,
                ),
                border = FilterChipDefaults.filterChipBorder(
                    enabled         = true,
                    selected        = false,
                    borderColor     = ARIAColors.Surface3,
                    selectedBorderColor = ARIAColors.Primary,
                ),
            )
        }
    }
}

// ─── Input bar ────────────────────────────────────────────────────────────────

@Composable
private fun ChatInputBar(
    input: String,
    onInput: (String) -> Unit,
    canSend: Boolean,
    thinking: Boolean,
    llmLoaded: Boolean,
    onSend: () -> Unit,
) {
    HorizontalDivider(color = ARIAColors.Surface3, thickness = 0.5.dp)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(ARIAColors.Background)
            .padding(horizontal = 16.dp, vertical = 10.dp)
            .navigationBarsPadding(),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedTextField(
            value         = input,
            onValueChange = onInput,
            modifier      = Modifier.weight(1f),
            placeholder   = {
                Text(
                    if (llmLoaded) "Ask ARIA anything…" else "Load the LLM first (Control tab)",
                    color    = ARIAColors.TextMuted,
                    fontSize = 14.sp,
                )
            },
            enabled       = !thinking,
            maxLines      = 4,
            textStyle     = LocalTextStyle.current.copy(color = ARIAColors.TextPrimary, fontSize = 14.sp),
            shape         = RoundedCornerShape(24.dp),
            colors        = OutlinedTextFieldDefaults.colors(
                focusedBorderColor   = ARIAColors.Primary,
                unfocusedBorderColor = ARIAColors.Surface3,
                focusedContainerColor   = ARIAColors.Surface1,
                unfocusedContainerColor = ARIAColors.Surface1,
            ),
        )
        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(if (canSend) ARIAColors.Primary else ARIAColors.Surface2),
            contentAlignment = Alignment.Center,
        ) {
            if (thinking) {
                CircularProgressIndicator(
                    modifier  = Modifier.size(20.dp),
                    color     = ARIAColors.Background,
                    strokeWidth = 2.dp,
                )
            } else {
                IconButton(onClick = onSend, enabled = canSend) {
                    Icon(
                        Icons.AutoMirrored.Filled.Send,
                        contentDescription = "Send",
                        tint   = if (canSend) ARIAColors.Background else ARIAColors.TextMuted,
                        modifier = Modifier.size(18.dp),
                    )
                }
            }
        }
    }
}
