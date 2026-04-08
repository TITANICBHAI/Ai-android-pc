package com.ariaagent.mobile.ui.screens

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ScreenShare
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.ariaagent.mobile.core.memory.ObjectLabelStore
import com.ariaagent.mobile.ui.theme.ARIAColors
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel

/**
 * LabelerScreen — Migration Phase 7.
 *
 * Pure Kotlin + Jetpack Compose replacement for labeler.tsx (1,017 lines).
 * Most complex screen in the app.
 *
 * Features:
 *  - Capture button → calls ScreenCaptureService → saves JPEG, runs OCR, gets a11y tree
 *  - Screenshot display with Box overlay of pins (absolute positioned by normalised coords)
 *  - pointerInput tap handler: tap = place pin at normalised 0.0–1.0 coords
 *  - Pin overlay: coloured circle dot + label initial; selected pin has ring highlight
 *  - Tap pin to select / deselect
 *  - Long-press pin to quick-delete (NEW beyond RN)
 *  - Drag pin to reposition (NEW beyond RN)
 *  - Auto-detect button: ObjectDetectorEngine places pins at detected centres
 *  - Enrich All button: LlamaEngine enriches all pins with JSON metadata
 *  - Save button: ObjectLabelStore.saveAll() then navigate back
 *  - Back button with unsaved-changes warning dialog
 *  - Pin editor panel (visible when pin selected):
 *      Name OutlinedTextField
 *      Context OutlinedTextField
 *      Element type chip row (button/text/input/icon/image/container/toggle/link/unknown)
 *      Delete label button with confirmation
 *      Importance score display (read-only after enrichment)
 *  - Stats bar: total labels, enriched count, OCR text preview (expandable)
 */
@Composable
fun LabelerScreen(
    vm: AgentViewModel,
    onBack: () -> Unit,
    onRequestScreenCapture: () -> Unit = {},
) {
    val capture       by vm.labelerCapture.collectAsState()
    val labels        by vm.labelerLabels.collectAsState()
    val capturing     by vm.labelerCapturing.collectAsState()
    val detecting     by vm.labelerDetecting.collectAsState()
    val enriching     by vm.labelerEnriching.collectAsState()
    val saving        by vm.labelerSaving.collectAsState()
    val error         by vm.labelerError.collectAsState()
    val saveSuccess   by vm.labelerSaveSuccess.collectAsState()
    val agentState    by vm.agentState.collectAsState()
    val captureActive = agentState.screenCaptureActive

    // Gallery image picker (Fix 3)
    val galleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { vm.loadImageFromGallery(it) }
    }

    var selectedId       by remember { mutableStateOf<String?>(null) }
    var showDeleteDialog by remember { mutableStateOf<String?>(null) }
    var showBackDialog   by remember { mutableStateOf(false) }
    var hasEdits         by remember { mutableStateOf(false) }

    val selectedLabel = labels.find { it.id == selectedId }

    LaunchedEffect(saveSuccess) {
        if (saveSuccess) {
            vm.clearLabelerCapture()
            onBack()
        }
    }

    LaunchedEffect(labels.size) {
        if (labels.isNotEmpty()) hasEdits = true
    }

    error?.let { msg ->
        LaunchedEffect(msg) { vm.dismissLabelerError() }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(ARIAColors.Error.copy(alpha = 0.12f))
                .padding(horizontal = 16.dp, vertical = 8.dp)
        ) {
            Text(msg, color = ARIAColors.Error, fontSize = 12.sp)
        }
    }

    if (showBackDialog) {
        AlertDialog(
            onDismissRequest  = { showBackDialog = false },
            title             = { Text("Unsaved changes", color = ARIAColors.TextPrimary) },
            text              = { Text("You have unsaved label changes. Discard them?", color = ARIAColors.TextSecondary) },
            confirmButton     = {
                TextButton(onClick = { vm.clearLabelerCapture(); onBack() }) {
                    Text("Discard", color = ARIAColors.Error)
                }
            },
            dismissButton     = { TextButton(onClick = { showBackDialog = false }) { Text("Keep editing") } },
            containerColor    = ARIAColors.Surface1,
        )
    }

    showDeleteDialog?.let { deleteId ->
        AlertDialog(
            onDismissRequest  = { showDeleteDialog = null },
            title             = { Text("Delete pin?", color = ARIAColors.TextPrimary) },
            text              = { Text("This label will be permanently removed.", color = ARIAColors.TextSecondary) },
            confirmButton     = {
                TextButton(onClick = {
                    vm.deleteLabelerLabel(deleteId)
                    if (selectedId == deleteId) selectedId = null
                    showDeleteDialog = null
                }) { Text("Delete", color = ARIAColors.Error) }
            },
            dismissButton     = { TextButton(onClick = { showDeleteDialog = null }) { Text("Cancel") } },
            containerColor    = ARIAColors.Surface1,
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
    ) {
        LabelerHeader(
            capture   = capture,
            labels    = labels,
            capturing = capturing,
            onBack    = {
                if (hasEdits && labels.isNotEmpty() && capture != null) showBackDialog = true
                else { vm.clearLabelerCapture(); onBack() }
            },
            onCapture = { vm.captureScreenForLabeling() },
        )

        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
        ) {
            // Bug #8 fix: snapshot capture into a stable local val so a concurrent
            // recomposition can't null it out between the != null check and the !! dereference.
            val cap = capture
            if (cap != null) {
                ScreenshotArea(
                    imagePath    = cap.imagePath,
                    labels       = labels,
                    selectedId   = selectedId,
                    onTap        = { normX, normY ->
                        vm.addLabelerPin(normX, normY)
                        selectedId = labels.lastOrNull()?.id
                        hasEdits   = true
                    },
                    onSelectPin  = { id -> selectedId = if (selectedId == id) null else id },
                    onDeletePin  = { id -> showDeleteDialog = id },
                    onDragPin    = { id, normX, normY ->
                        val lbl = labels.find { it.id == id } ?: return@ScreenshotArea
                        vm.updateLabelerLabel(lbl.copy(x = normX, y = normY))
                    },
                )

                StatsBar(labels = labels, ocrText = cap.ocrText)
            } else {
                LabelerEmptyState(
                    capturing              = capturing,
                    captureActive          = captureActive,
                    onCapture              = { vm.captureScreenForLabeling() },
                    onRequestScreenCapture = onRequestScreenCapture,
                    onGallery              = { galleryLauncher.launch("image/*") },
                )
            }

            selectedLabel?.let { lbl ->
                PinEditor(
                    label    = lbl,
                    onUpdate = { updated ->
                        vm.updateLabelerLabel(updated)
                        hasEdits = true
                    },
                    onDelete = { showDeleteDialog = lbl.id },
                )
            }

            if (capture != null) {
                LabelerToolbar(
                    labels    = labels,
                    detecting = detecting,
                    enriching = enriching,
                    saving    = saving,
                    onDetect  = { vm.autoDetectLabelerPins() },
                    onEnrich  = { vm.enrichAllLabelerPins() },
                    onSave    = {
                        vm.saveLabelerLabels(
                            onSuccess = {},
                            onError   = {},
                        )
                    },
                )
            }

            Spacer(Modifier.height(32.dp))
        }
    }
}

// ─── Header ───────────────────────────────────────────────────────────────────

@Composable
private fun LabelerHeader(
    capture: com.ariaagent.mobile.ui.viewmodel.ScreenCaptureUi?,
    labels: List<ObjectLabelStore.ObjectLabel>,
    capturing: Boolean,
    onBack: () -> Unit,
    onCapture: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(ARIAColors.Surface1)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment     = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        IconButton(onClick = onBack) {
            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = ARIAColors.TextPrimary)
        }
        Column(modifier = Modifier.weight(1f)) {
            Text("Object Labeler", color = ARIAColors.TextPrimary, fontSize = 17.sp, fontWeight = FontWeight.Bold)
            Text(
                text     = if (capture != null) "${capture.appPackage.substringAfterLast('.')} · ${labels.size} pin${if (labels.size != 1) "s" else ""}"
                           else "Teach the agent about UI elements",
                color    = ARIAColors.TextMuted,
                fontSize = 11.sp,
            )
        }
        Button(
            onClick  = onCapture,
            enabled  = !capturing,
            shape    = RoundedCornerShape(12.dp),
            colors   = ButtonDefaults.buttonColors(containerColor = ARIAColors.Primary, disabledContainerColor = ARIAColors.Surface2),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
        ) {
            if (capturing) {
                CircularProgressIndicator(Modifier.size(16.dp), color = ARIAColors.TextMuted, strokeWidth = 2.dp)
            } else {
                Icon(Icons.Default.Camera, contentDescription = "Capture", modifier = Modifier.size(18.dp))
            }
        }
    }
    HorizontalDivider(color = ARIAColors.Surface3, thickness = 0.5.dp)
}

// ─── Screenshot area ──────────────────────────────────────────────────────────

@Composable
private fun ScreenshotArea(
    imagePath: String,
    labels: List<ObjectLabelStore.ObjectLabel>,
    selectedId: String?,
    onTap: (Float, Float) -> Unit,
    onSelectPin: (String) -> Unit,
    onDeletePin: (String) -> Unit,
    onDragPin: (String, Float, Float) -> Unit,
) {
    var boxSize by remember { mutableStateOf(IntSize.Zero) }

    Column(modifier = Modifier.padding(16.dp)) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .border(1.dp, ARIAColors.Surface3, RoundedCornerShape(12.dp))
                .onSizeChanged { boxSize = it }
                .pointerInput(boxSize) {
                    detectTapGestures { offset ->
                        if (boxSize.width > 0 && boxSize.height > 0) {
                            onTap(offset.x / boxSize.width, offset.y / boxSize.height)
                        }
                    }
                }
        ) {
            AsyncImage(
                model             = "file://$imagePath",
                contentDescription = "Screen capture",
                contentScale      = ContentScale.Fit,
                modifier          = Modifier.fillMaxWidth(),
            )

            labels.forEach { label ->
                val pinX = label.x * boxSize.width
                val pinY = label.y * boxSize.height
                val pinSizePx = with(LocalDensity.current) { 28.dp.toPx() }
                val offsetX = (pinX - pinSizePx / 2).coerceAtLeast(0f)
                val offsetY = (pinY - pinSizePx / 2).coerceAtLeast(0f)
                val isSelected = label.id == selectedId

                Box(
                    modifier = Modifier
                        .offset(
                            x = with(LocalDensity.current) { offsetX.toDp() },
                            y = with(LocalDensity.current) { offsetY.toDp() },
                        )
                        .size(28.dp)
                        .clip(CircleShape)
                        .background(
                            when {
                                isSelected    -> ARIAColors.Primary
                                label.isEnriched -> ARIAColors.Success
                                else          -> ARIAColors.Accent
                            }
                        )
                        .then(
                            if (isSelected) Modifier.border(2.dp, Color.White, CircleShape)
                            else Modifier
                        )
                        .pointerInput(label.id, boxSize) {
                            detectTapGestures(
                                onTap      = { onSelectPin(label.id) },
                                onLongPress = { onDeletePin(label.id) },
                            )
                        }
                        .pointerInput(label.id, boxSize) {
                            detectDragGestures { _, dragAmount ->
                                if (boxSize.width > 0 && boxSize.height > 0) {
                                    val newX = (label.x + dragAmount.x / boxSize.width).coerceIn(0f, 1f)
                                    val newY = (label.y + dragAmount.y / boxSize.height).coerceIn(0f, 1f)
                                    onDragPin(label.id, newX, newY)
                                }
                            }
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text     = if (label.name.isNotBlank()) label.name.first().uppercaseChar().toString() else "?",
                        color    = Color.White,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }

        Spacer(Modifier.height(8.dp))
        Text(
            text     = "Tap screenshot to place pin · Tap pin to edit · Long-press to delete · Drag to reposition",
            color    = ARIAColors.TextMuted,
            fontSize = 11.sp,
        )
    }
}

// ─── Stats bar ────────────────────────────────────────────────────────────────

@Composable
private fun StatsBar(labels: List<ObjectLabelStore.ObjectLabel>, ocrText: String) {
    var ocrExpanded by remember { mutableStateOf(false) }
    val enriched = labels.count { it.isEnriched }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(ARIAColors.Surface2)
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("Labels: ${labels.size}", color = ARIAColors.TextSecondary, fontSize = 12.sp)
            Text("Enriched: $enriched", color = if (enriched > 0) ARIAColors.Success else ARIAColors.TextMuted, fontSize = 12.sp)
        }
        if (ocrText.isNotBlank()) {
            TextButton(
                onClick          = { ocrExpanded = !ocrExpanded },
                contentPadding   = PaddingValues(0.dp),
            ) {
                Text(
                    text     = if (ocrExpanded) "▲ Hide OCR text" else "▼ Show OCR text",
                    color    = ARIAColors.TextMuted,
                    fontSize = 11.sp,
                )
            }
            if (ocrExpanded) {
                Text(
                    text       = ocrText.take(400),
                    color      = ARIAColors.TextMuted,
                    fontSize   = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    lineHeight = 16.sp,
                )
            }
        }
    }
}

// ─── Empty state ──────────────────────────────────────────────────────────────

@Composable
private fun LabelerEmptyState(
    capturing: Boolean,
    captureActive: Boolean,
    onCapture: () -> Unit,
    onRequestScreenCapture: () -> Unit,
    onGallery: () -> Unit,
) {
    Column(
        modifier            = Modifier
            .fillMaxWidth()
            .padding(40.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Icon(
            Icons.Default.PhotoCamera,
            contentDescription = null,
            tint     = ARIAColors.TextMuted,
            modifier = Modifier.size(56.dp),
        )
        Text(
            "No screen captured yet",
            color      = ARIAColors.TextPrimary,
            fontSize   = 16.sp,
            fontWeight = FontWeight.SemiBold,
        )

        if (captureActive) {
            // Service is running — let them capture normally
            Text(
                text       = "Press Capture to grab the current foreground app screen,\nthen tap anywhere to place annotation pins.",
                color      = ARIAColors.TextMuted,
                fontSize   = 13.sp,
                textAlign  = TextAlign.Center,
                lineHeight = 19.sp,
            )
            Button(
                onClick = onCapture,
                enabled = !capturing,
                shape   = RoundedCornerShape(12.dp),
                colors  = ButtonDefaults.buttonColors(containerColor = ARIAColors.Primary),
            ) {
                if (capturing) {
                    CircularProgressIndicator(Modifier.size(16.dp), color = ARIAColors.Background, strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text("Capturing…", color = ARIAColors.Background)
                } else {
                    Icon(Icons.Default.Camera, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Capture Screen", color = ARIAColors.Background, fontWeight = FontWeight.SemiBold)
                }
            }
        } else {
            // Service not active — offer to grant permission or use gallery
            Text(
                text       = "Screen capture permission is required to annotate live screens.\nYou can also import an image from your gallery.",
                color      = ARIAColors.TextMuted,
                fontSize   = 13.sp,
                textAlign  = TextAlign.Center,
                lineHeight = 19.sp,
            )
            Button(
                onClick = onRequestScreenCapture,
                shape   = RoundedCornerShape(12.dp),
                colors  = ButtonDefaults.buttonColors(containerColor = ARIAColors.Primary),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(Icons.AutoMirrored.Filled.ScreenShare, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(8.dp))
                Text("Grant Screen Capture", color = ARIAColors.Background, fontWeight = FontWeight.SemiBold)
            }
            OutlinedButton(
                onClick  = onGallery,
                shape    = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Icon(Icons.Default.Photo, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(Modifier.width(8.dp))
                Text("Import from Gallery", fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

// ─── Pin editor ───────────────────────────────────────────────────────────────

private val ELEMENT_TYPES = ObjectLabelStore.ElementType.values().toList()

@Composable
private fun PinEditor(
    label: ObjectLabelStore.ObjectLabel,
    onUpdate: (ObjectLabelStore.ObjectLabel) -> Unit,
    onDelete: () -> Unit,
) {
    Column(
        modifier            = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(ARIAColors.Surface1)
            .border(1.dp, ARIAColors.Surface3, RoundedCornerShape(16.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically,
        ) {
            Text("Edit Pin", color = ARIAColors.TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
            IconButton(onClick = onDelete) {
                Icon(Icons.Default.Delete, contentDescription = "Delete", tint = ARIAColors.Error)
            }
        }

        Text("Element Name *", color = ARIAColors.TextMuted, fontSize = 12.sp)
        OutlinedTextField(
            value         = label.name,
            onValueChange = { onUpdate(label.copy(name = it, updatedAt = System.currentTimeMillis())) },
            modifier      = Modifier.fillMaxWidth(),
            placeholder   = { Text("e.g. Checkout Button", color = ARIAColors.TextMuted, fontSize = 13.sp) },
            singleLine    = true,
            colors        = labelerFieldColors(),
            shape         = RoundedCornerShape(8.dp),
        )

        Text("Purpose / Context", color = ARIAColors.TextMuted, fontSize = 12.sp)
        OutlinedTextField(
            value         = label.context,
            onValueChange = { onUpdate(label.copy(context = it, updatedAt = System.currentTimeMillis())) },
            modifier      = Modifier.fillMaxWidth(),
            placeholder   = { Text("What does this element do?", color = ARIAColors.TextMuted, fontSize = 13.sp) },
            maxLines      = 2,
            colors        = labelerFieldColors(),
            shape         = RoundedCornerShape(8.dp),
        )

        Text("Element Type", color = ARIAColors.TextMuted, fontSize = 12.sp)
        LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            items(ELEMENT_TYPES) { type ->
                val selected = label.elementType == type
                FilterChip(
                    selected = selected,
                    onClick  = { onUpdate(label.copy(elementType = type, updatedAt = System.currentTimeMillis())) },
                    label    = { Text(type.name.lowercase(), fontSize = 12.sp) },
                    colors   = FilterChipDefaults.filterChipColors(
                        containerColor         = ARIAColors.Surface2,
                        labelColor             = ARIAColors.TextMuted,
                        selectedContainerColor = ARIAColors.Primary,
                        selectedLabelColor     = ARIAColors.Background,
                    ),
                    border = FilterChipDefaults.filterChipBorder(
                        enabled = true, selected = selected,
                        borderColor = ARIAColors.Surface3,
                        selectedBorderColor = ARIAColors.Primary,
                    ),
                )
            }
        }

        if (label.isEnriched) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Importance: ${label.importanceScore}/10 (enriched by LLM)", color = ARIAColors.TextMuted, fontSize = 12.sp)
                if (label.meaning.isNotBlank()) Text("Meaning: ${label.meaning}", color = ARIAColors.TextSecondary, fontSize = 12.sp, lineHeight = 17.sp)
                if (label.interactionHint.isNotBlank()) Text("Hint: ${label.interactionHint}", color = ARIAColors.TextSecondary, fontSize = 12.sp, lineHeight = 17.sp)
            }
        } else {
            Text("Importance: ${label.importanceScore}/10", color = ARIAColors.TextMuted, fontSize = 12.sp)
            LazyRow(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                items((1..10).toList()) { n ->
                    val sel = label.importanceScore == n
                    FilterChip(
                        selected = sel,
                        onClick  = { onUpdate(label.copy(importanceScore = n, updatedAt = System.currentTimeMillis())) },
                        label    = { Text("$n", fontSize = 12.sp) },
                        colors   = FilterChipDefaults.filterChipColors(
                            containerColor         = ARIAColors.Surface2,
                            labelColor             = ARIAColors.TextMuted,
                            selectedContainerColor = ARIAColors.Accent,
                            selectedLabelColor     = Color.White,
                        ),
                        border = FilterChipDefaults.filterChipBorder(
                            enabled = true, selected = sel,
                            borderColor = ARIAColors.Surface3,
                            selectedBorderColor = ARIAColors.Accent,
                        ),
                    )
                }
            }
        }
    }
    Spacer(Modifier.height(12.dp))
}

// ─── Toolbar ──────────────────────────────────────────────────────────────────

@Composable
private fun LabelerToolbar(
    labels: List<ObjectLabelStore.ObjectLabel>,
    detecting: Boolean,
    enriching: Boolean,
    saving: Boolean,
    onDetect: () -> Unit,
    onEnrich: () -> Unit,
    onSave: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedButton(
            onClick   = onDetect,
            enabled   = !detecting && !enriching && !saving,
            modifier  = Modifier.weight(1f),
            shape     = RoundedCornerShape(10.dp),
            colors    = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.TextMuted),
            border    = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Surface3),
        ) {
            if (detecting) {
                CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = ARIAColors.TextMuted)
            } else {
                Icon(Icons.Default.AutoAwesome, contentDescription = null, modifier = Modifier.size(14.dp))
            }
            Spacer(Modifier.width(4.dp))
            Text(if (detecting) "Detecting…" else "Auto-detect", fontSize = 12.sp)
        }

        OutlinedButton(
            onClick   = onEnrich,
            enabled   = !enriching && !saving && labels.isNotEmpty(),
            modifier  = Modifier.weight(1f),
            shape     = RoundedCornerShape(10.dp),
            colors    = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Accent),
            border    = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Accent.copy(alpha = 0.4f)),
        ) {
            if (enriching) {
                CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = ARIAColors.Accent)
            } else {
                Icon(Icons.Default.Memory, contentDescription = null, modifier = Modifier.size(14.dp))
            }
            Spacer(Modifier.width(4.dp))
            Text(if (enriching) "Enriching…" else "Enrich All", fontSize = 12.sp)
        }

        Button(
            onClick   = onSave,
            enabled   = !saving && !enriching && labels.isNotEmpty(),
            modifier  = Modifier.weight(1f),
            shape     = RoundedCornerShape(10.dp),
            colors    = ButtonDefaults.buttonColors(
                containerColor         = ARIAColors.Primary,
                disabledContainerColor = ARIAColors.Surface2,
            ),
        ) {
            if (saving) {
                CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp, color = ARIAColors.TextMuted)
            } else {
                Icon(Icons.Default.Save, contentDescription = null, modifier = Modifier.size(14.dp))
            }
            Spacer(Modifier.width(4.dp))
            Text(
                text  = if (saving) "Saving…" else "Save Labels",
                color = if (saving) ARIAColors.TextMuted else ARIAColors.Background,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold,
            )
        }
    }
}

// ─── Shared field colors ──────────────────────────────────────────────────────

@Composable
private fun labelerFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor      = ARIAColors.Primary,
    unfocusedBorderColor    = ARIAColors.Surface3,
    focusedContainerColor   = ARIAColors.Surface2,
    unfocusedContainerColor = ARIAColors.Surface2,
    focusedTextColor        = ARIAColors.TextPrimary,
    unfocusedTextColor      = ARIAColors.TextPrimary,
)
