package com.ariaagent.mobile.ui.screens

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.ariaagent.mobile.ui.theme.ARIAColors
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel

/**
 * OnboardingScreen — first-run guided setup wizard.
 *
 * Shown automatically on first launch before the main app.
 * ComposeMainActivity checks SharedPreferences for "onboarding_complete" and
 * routes here if not set. ARIAComposeApp uses vm.onboardingComplete to decide
 * the start destination.
 *
 * Steps (updated Phase 17):
 *   0 — Welcome              : ARIA intro, "Get Started" button
 *   1 — Download AI Model    : Llama 3.2-1B (~700 MB) — required
 *   2 — Enable Vision        : SmolVLM-256M (~200 MB) — skippable
 *   3 — Accessibility        : GRANT button + explanation
 *   4 — Screen Capture       : REQUEST button — skippable
 *   5 — Ready                : summary checklist, "Start Using ARIA"
 *
 * Vision step (step 2) is the new addition for Phase 17. It explains what
 * SmolVLM does and lets the user download it right inside the wizard.
 * If the user skips it they can always download from the Modules screen later.
 */

private data class OnboardingStep(
    val index: Int,
    val title: String,
    val subtitle: String,
    val icon: ImageVector,
    val iconTint: Color,
    val skippable: Boolean = false,
)

private val ONBOARDING_STEPS = listOf(
    OnboardingStep(0, "Welcome to ARIA",
        "Your 100% on-device autonomous Android agent. No cloud. No subscriptions. Fully private.",
        Icons.Default.SmartToy, ARIAColors.Primary),
    OnboardingStep(1, "Download the AI Model",
        "ARIA runs a local Llama 3.2-1B model (~700 MB). Download it once and it lives on your device forever.",
        Icons.Default.Download, ARIAColors.Accent),
    OnboardingStep(2, "Enable Vision (Optional)",
        "SmolVLM-256M lets ARIA visually understand your screen — useful for games, Flutter apps, and complex UIs where text alone isn't enough. ~200 MB, runs fully on-device.",
        Icons.Default.RemoveRedEye, ARIAColors.Primary, skippable = true),
    OnboardingStep(3, "Grant Accessibility",
        "ARIA needs the Accessibility Service to see and interact with other apps on your behalf.",
        Icons.Default.Accessibility, ARIAColors.Success),
    OnboardingStep(4, "Grant Screen Capture",
        "ARIA uses screen capture to understand what's on screen before taking any action.",
        Icons.Default.Screenshot, ARIAColors.Warning, skippable = true),
    OnboardingStep(5, "You're ready!",
        "ARIA is set up and ready. Go to the Control tab to give it its first task.",
        Icons.Default.CheckCircle, ARIAColors.Success),
)

@Composable
fun OnboardingScreen(
    vm: AgentViewModel,
    onFinish: () -> Unit,
    onGrantAccessibility: () -> Unit,
    onRequestScreenCapture: () -> Unit,
) {
    val moduleState      by vm.moduleState.collectAsStateWithLifecycle()
    val llmDownloading   by vm.llmDownloading.collectAsStateWithLifecycle()
    val visionDownloading by vm.visionDownloading.collectAsStateWithLifecycle()

    var currentStep by remember { mutableIntStateOf(0) }
    val step = ONBOARDING_STEPS[currentStep]
    val isLast = currentStep == ONBOARDING_STEPS.lastIndex

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(Modifier.height(60.dp))

            // ── Step indicator dots ───────────────────────────────────────────
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment     = Alignment.CenterVertically
            ) {
                ONBOARDING_STEPS.forEachIndexed { i, _ ->
                    val active = i == currentStep
                    val done   = i < currentStep
                    Box(
                        modifier = Modifier
                            .size(if (active) 10.dp else 7.dp)
                            .clip(CircleShape)
                            .background(when {
                                active -> ARIAColors.Primary
                                done   -> ARIAColors.Success
                                else   -> ARIAColors.SurfaceVariant
                            })
                    )
                }
            }

            Spacer(Modifier.height(48.dp))

            // ── Animated step content ─────────────────────────────────────────
            AnimatedContent(
                targetState = currentStep,
                transitionSpec = {
                    val enter = fadeIn() + slideInHorizontally { w -> if (targetState > initialState) w / 3 else -w / 3 }
                    val exit  = fadeOut() + slideOutHorizontally { w -> if (targetState > initialState) -w / 3 else w / 3 }
                    enter togetherWith exit
                },
                label = "onboarding_step"
            ) { stepIdx ->
                val s = ONBOARDING_STEPS[stepIdx]
                Column(
                    modifier            = Modifier.fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    // Big icon
                    Box(
                        modifier = Modifier
                            .size(96.dp)
                            .clip(RoundedCornerShape(24.dp))
                            .background(s.iconTint.copy(alpha = 0.12f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(s.icon, contentDescription = null, tint = s.iconTint, modifier = Modifier.size(48.dp))
                    }

                    Text(
                        s.title,
                        style = MaterialTheme.typography.headlineSmall.copy(
                            color = ARIAColors.OnSurface, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center
                        ),
                        textAlign = TextAlign.Center
                    )

                    Text(
                        s.subtitle,
                        style = MaterialTheme.typography.bodyMedium.copy(
                            color = ARIAColors.Muted, textAlign = TextAlign.Center, lineHeight = 22.sp
                        ),
                        textAlign = TextAlign.Center
                    )

                    // ── Step-specific content ─────────────────────────────────
                    when (stepIdx) {
                        1 -> ModelDownloadStep(
                            modelReady  = moduleState.modelReady,
                            downloading = llmDownloading,
                            downloadPct = moduleState.llmDownloadPercent,
                            onDownload  = { vm.downloadLlmModel() }
                        )
                        2 -> VisionDownloadStep(
                            visionReady  = moduleState.visionReady,
                            downloading  = visionDownloading,
                            downloadPct  = moduleState.visionDownloadPercent,
                            onDownload   = { vm.downloadVisionModel() }
                        )
                        3 -> AccessibilityStep(
                            granted = moduleState.accessibilityGranted,
                            onGrant = onGrantAccessibility
                        )
                        4 -> ScreenCaptureStep(
                            granted   = moduleState.screenCaptureGranted,
                            onRequest = onRequestScreenCapture
                        )
                        5 -> ReadySummary(
                            modelReady          = moduleState.modelReady,
                            visionReady         = moduleState.visionReady,
                            accessibilityGranted = moduleState.accessibilityGranted,
                            screenCaptureGranted = moduleState.screenCaptureGranted,
                        )
                        else -> {}
                    }
                }
            }

            Spacer(Modifier.weight(1f))

            // ── Nav buttons ───────────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 48.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                if (currentStep > 0) {
                    OutlinedButton(
                        onClick = { currentStep-- },
                        modifier = Modifier.weight(1f),
                        shape    = RoundedCornerShape(10.dp),
                        colors   = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Muted),
                        border   = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Divider)
                    ) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("Back")
                    }
                }

                Button(
                    onClick  = {
                        if (isLast) {
                            vm.setOnboardingComplete()
                            onFinish()
                        } else {
                            currentStep++
                        }
                    },
                    modifier = Modifier.weight(if (currentStep == 0) 2f else 1f),
                    shape    = RoundedCornerShape(10.dp),
                    colors   = ButtonDefaults.buttonColors(containerColor = if (isLast) ARIAColors.Success else ARIAColors.Primary)
                ) {
                    Text(
                        when {
                            isLast           -> "Start Using ARIA"
                            currentStep == 0 -> "Get Started"
                            step.skippable   -> "Next"
                            else             -> "Continue"
                        },
                        fontWeight = FontWeight.Bold,
                        color = ARIAColors.Background
                    )
                    Spacer(Modifier.width(4.dp))
                    Icon(
                        if (isLast) Icons.Default.CheckCircle else Icons.AutoMirrored.Filled.ArrowForward,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = ARIAColors.Background
                    )
                }
            }

            // Skip link for skippable steps
            if (!isLast && step.skippable) {
                TextButton(
                    onClick  = { currentStep++ },
                    modifier = Modifier.padding(bottom = 12.dp)
                ) {
                    Text("Skip for now", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
                }
            }
        }
    }
}

// ─── Step-specific content composables ───────────────────────────────────────

@Composable
private fun ModelDownloadStep(
    modelReady: Boolean,
    downloading: Boolean,
    downloadPct: Int,
    onDownload: () -> Unit,
) {
    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(12.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (modelReady) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = ARIAColors.Success, modifier = Modifier.size(20.dp))
                    Text("Model ready", style = MaterialTheme.typography.bodyMedium.copy(color = ARIAColors.Success, fontWeight = FontWeight.Bold))
                }
                Text("Llama 3.2-1B is downloaded and ready.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted, textAlign = TextAlign.Center), textAlign = TextAlign.Center)
            } else if (downloading) {
                Text("Downloading model…", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Accent))
                LinearProgressIndicator(
                    progress = { downloadPct / 100f },
                    modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
                    color    = ARIAColors.Accent,
                )
                Text("$downloadPct% complete", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontFamily = FontFamily.Monospace))
            } else {
                Button(
                    onClick  = onDownload,
                    modifier = Modifier.fillMaxWidth(),
                    shape    = RoundedCornerShape(8.dp),
                    colors   = ButtonDefaults.buttonColors(containerColor = ARIAColors.Accent)
                ) {
                    Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Download Model (~700 MB)", fontWeight = FontWeight.SemiBold, color = Color.White)
                }
                Text("Requires a Wi-Fi connection. The download happens once.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted, textAlign = TextAlign.Center), textAlign = TextAlign.Center)
            }
        }
    }
}

/**
 * Vision model step (Phase 17) — SmolVLM-256M + mmproj (~200 MB total).
 *
 * Explains what vision adds before asking the user to download. Skippable —
 * the user can always come back to it in the Modules screen later.
 *
 * When active, it shows:
 *   • A two-row capability list (what vision is used for)
 *   • Download button or live progress bar
 *   • "Already downloaded" confirmation row if ready
 */
@Composable
private fun VisionDownloadStep(
    visionReady: Boolean,
    downloading: Boolean,
    downloadPct: Int,
    onDownload: () -> Unit,
) {
    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(12.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Capability hints shown regardless of state
            Column(
                modifier            = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                VisionCapabilityRow(Icons.Default.VideogameAsset,  "Understand games & Flutter apps (no a11y tree)")
                VisionCapabilityRow(Icons.Default.TouchApp,        "Identify buttons and icons by what they look like")
                VisionCapabilityRow(Icons.Default.Speed,           "Caches unchanged frames — ~0 ms when screen stays still")
                VisionCapabilityRow(Icons.Default.Psychology,      "Goal-aware — answers 'what helps me achieve this task?'")
            }

            HorizontalDivider(color = ARIAColors.Divider)

            when {
                visionReady -> {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.CheckCircle, contentDescription = null, tint = ARIAColors.Success, modifier = Modifier.size(20.dp))
                        Text("SmolVLM-256M ready", style = MaterialTheme.typography.bodyMedium.copy(color = ARIAColors.Success, fontWeight = FontWeight.Bold))
                    }
                    Text("Vision model downloaded. ARIA can see the screen.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted, textAlign = TextAlign.Center), textAlign = TextAlign.Center)
                }
                downloading -> {
                    Text("Downloading vision model…", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Primary))
                    LinearProgressIndicator(
                        progress = { downloadPct / 100f },
                        modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
                        color    = ARIAColors.Primary,
                    )
                    Text("$downloadPct% complete (base + mmproj)", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontFamily = FontFamily.Monospace))
                }
                else -> {
                    Button(
                        onClick  = onDownload,
                        modifier = Modifier.fillMaxWidth(),
                        shape    = RoundedCornerShape(8.dp),
                        colors   = ButtonDefaults.buttonColors(containerColor = ARIAColors.Primary)
                    ) {
                        Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("Download Vision Model (~200 MB)", fontWeight = FontWeight.SemiBold, color = Color.White)
                    }
                    Text(
                        "Optional but recommended. Wi-Fi advised. You can download later from the Modules screen.",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted, textAlign = TextAlign.Center),
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Composable
private fun VisionCapabilityRow(icon: ImageVector, text: String) {
    Row(
        modifier            = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment   = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = ARIAColors.Primary, modifier = Modifier.size(16.dp))
        Text(text, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface, lineHeight = 18.sp))
    }
}

@Composable
private fun AccessibilityStep(granted: Boolean, onGrant: () -> Unit) {
    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(12.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (granted) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = ARIAColors.Success, modifier = Modifier.size(20.dp))
                    Text("Accessibility granted", style = MaterialTheme.typography.bodyMedium.copy(color = ARIAColors.Success, fontWeight = FontWeight.Bold))
                }
            } else {
                Button(
                    onClick  = onGrant,
                    modifier = Modifier.fillMaxWidth(),
                    shape    = RoundedCornerShape(8.dp),
                    colors   = ButtonDefaults.buttonColors(containerColor = ARIAColors.Success)
                ) {
                    Icon(Icons.Default.Accessibility, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Grant Accessibility Access", fontWeight = FontWeight.SemiBold, color = Color.White)
                }
                Text(
                    "In the settings screen that opens: find \"ARIA Agent\" and toggle it on.",
                    style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted, textAlign = TextAlign.Center),
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

@Composable
private fun ScreenCaptureStep(granted: Boolean, onRequest: () -> Unit) {
    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(12.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (granted) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = ARIAColors.Success, modifier = Modifier.size(20.dp))
                    Text("Screen capture granted", style = MaterialTheme.typography.bodyMedium.copy(color = ARIAColors.Success, fontWeight = FontWeight.Bold))
                }
            } else {
                Button(
                    onClick  = onRequest,
                    modifier = Modifier.fillMaxWidth(),
                    shape    = RoundedCornerShape(8.dp),
                    colors   = ButtonDefaults.buttonColors(containerColor = ARIAColors.Warning)
                ) {
                    Icon(Icons.Default.Screenshot, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Request Screen Capture", fontWeight = FontWeight.SemiBold, color = ARIAColors.Background)
                }
                Text(
                    "Tap \"Start now\" in the dialog that appears. ARIA only captures frames during active tasks.",
                    style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted, textAlign = TextAlign.Center),
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

@Composable
private fun ReadySummary(
    modelReady: Boolean,
    visionReady: Boolean,
    accessibilityGranted: Boolean,
    screenCaptureGranted: Boolean,
) {
    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(12.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            OnboardingCheckRow("AI Model (Llama 3.2-1B)",       modelReady,          required = true)
            OnboardingCheckRow("Vision Model (SmolVLM-256M)",   visionReady,         required = false)
            OnboardingCheckRow("Accessibility Service",          accessibilityGranted, required = true)
            OnboardingCheckRow("Screen Capture Permission",      screenCaptureGranted, required = false)
        }
    }
}

@Composable
private fun OnboardingCheckRow(label: String, ok: Boolean, required: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            if (ok) Icons.Default.CheckCircle else Icons.Default.RadioButtonUnchecked,
            contentDescription = null,
            tint = when {
                ok       -> ARIAColors.Success
                required -> ARIAColors.Warning
                else     -> ARIAColors.Muted
            },
            modifier = Modifier.size(18.dp)
        )
        Column {
            Text(
                label,
                style = MaterialTheme.typography.bodyMedium.copy(color = if (ok) ARIAColors.OnSurface else ARIAColors.Muted)
            )
            if (!ok && !required) {
                Text("Optional — can be set up later", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted))
            }
        }
    }
}
