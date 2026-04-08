package com.ariaagent.mobile.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.ariaagent.mobile.ui.theme.ARIAColors
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel
import com.ariaagent.mobile.ui.viewmodel.SafetyConfig

/**
 * SafetyScreen — full-screen safety and permission boundary panel.
 *
 * Accessible from Settings → "Safety Boundaries" button.
 * No bottom-nav entry — it's a focused security configuration screen.
 *
 * Sections:
 *   1. Emergency Kill Switch — one tap stops all agent actions globally
 *   2. Action Confirmation Mode — ask user before each action is executed
 *   3. App Blocklist — ARIA will never interact with these apps
 *   4. Allowlist Mode — ARIA can ONLY interact with explicitly listed apps
 *   5. Sensitive App Defaults — pre-detected sensitive app categories
 */

private val SENSITIVE_APP_PRESETS = listOf(
    Triple("Banking & Finance",  Icons.Default.AccountBalance,
        listOf("com.chase.sig.android", "com.bankofamerica.digitalwallet", "com.wellsfargo.mobile", "com.citi.mobile", "com.venmo", "com.paypal.android.p2pmobile")),
    Triple("Medical & Health",   Icons.Default.LocalHospital,
        listOf("com.epic.mychart.android", "com.dexcom.g6", "com.medtronic.sync", "com.withings.wiscale2")),
    Triple("Passwords & Auth",   Icons.Default.Lock,
        listOf("com.lastpass.lpandroid", "com.onepassword.android", "com.google.android.apps.authenticator2", "org.keepass2android.keepass2android")),
    Triple("Work & Enterprise",  Icons.Default.Work,
        listOf("com.microsoft.intune.mam.managedbrowser", "com.microsoft.teams", "com.slack", "com.zoom.us")),
)

@Composable
fun SafetyScreen(
    vm: AgentViewModel,
    onBack: () -> Unit,
) {
    val safetyConfig by vm.safetyConfig.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) { vm.loadSafetyConfig() }

    LazyColumn(
        modifier        = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background),
        contentPadding  = PaddingValues(bottom = 40.dp),
    ) {
        // ── Header ────────────────────────────────────────────────────────────
        item {
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
                    Text("SAFETY BOUNDARIES", style = MaterialTheme.typography.titleLarge.copy(color = ARIAColors.Destructive, fontWeight = FontWeight.Bold))
                    Text("Control what ARIA can and cannot do", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
                }
            }
        }

        // ── 1. Emergency Kill Switch ──────────────────────────────────────────
        item {
            Spacer(Modifier.height(16.dp))
            SafetySectionLabel("Emergency Kill Switch", Icons.Default.StopCircle, ARIAColors.Destructive)
            Spacer(Modifier.height(8.dp))
            Box(modifier = Modifier.padding(horizontal = 16.dp)) {
                KillSwitchCard(
                    active   = safetyConfig.globalKillActive,
                    onToggle = { vm.toggleGlobalKill() }
                )
            }
        }

        // ── 2. Action Confirmation Mode ───────────────────────────────────────
        item {
            Spacer(Modifier.height(16.dp))
            SafetySectionLabel("Action Confirmation", Icons.Default.TouchApp, ARIAColors.Warning)
            Spacer(Modifier.height(8.dp))
            Box(modifier = Modifier.padding(horizontal = 16.dp)) {
                SafetyCard {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Column(modifier = Modifier.weight(1f).padding(end = 8.dp)) {
                            Text("Confirm before each action", style = MaterialTheme.typography.bodyMedium.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold))
                            Text(
                                "ARIA will pause and show a dialog before executing each click, swipe, or type. Useful for verifying sensitive tasks.",
                                style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted, lineHeight = 16.sp)
                            )
                        }
                        Switch(
                            checked  = safetyConfig.confirmMode,
                            onCheckedChange = { vm.setConfirmMode(it) },
                            colors = SwitchDefaults.colors(
                                checkedThumbColor  = ARIAColors.Background,
                                checkedTrackColor  = ARIAColors.Warning,
                                uncheckedThumbColor = ARIAColors.Muted,
                                uncheckedTrackColor = ARIAColors.SurfaceVariant,
                            )
                        )
                    }
                    if (safetyConfig.confirmMode) {
                        SafetyInfoRow(Icons.Default.Info, "Confirmation dialogs add latency. Not recommended for automated multi-step tasks.", ARIAColors.Warning)
                    }
                }
            }
        }

        // ── 3. Blocked Apps ───────────────────────────────────────────────────
        item {
            Spacer(Modifier.height(16.dp))
            SafetySectionLabel("Blocked Apps", Icons.Default.Block, ARIAColors.Destructive)
            Spacer(Modifier.height(8.dp))
        }
        item {
            Box(modifier = Modifier.padding(horizontal = 16.dp)) {
                BlocklistSection(
                    packages  = safetyConfig.blockedPackages,
                    onAdd     = { vm.addBlockedPackage(it) },
                    onRemove  = { vm.removeBlockedPackage(it) },
                )
            }
        }

        // ── 4. Allowlist Mode ─────────────────────────────────────────────────
        item {
            Spacer(Modifier.height(16.dp))
            SafetySectionLabel("Allowlist Mode", Icons.Default.CheckCircle, ARIAColors.Success)
            Spacer(Modifier.height(8.dp))
            Box(modifier = Modifier.padding(horizontal = 16.dp)) {
                AllowlistSection(
                    allowlistMode = safetyConfig.allowlistMode,
                    packages      = safetyConfig.allowedPackages,
                    onToggleMode  = { vm.setAllowlistMode(it) },
                    onAdd         = { vm.addAllowedPackage(it) },
                    onRemove      = { vm.removeAllowedPackage(it) },
                )
            }
        }

        // ── 5. Sensitive App Presets ──────────────────────────────────────────
        item {
            Spacer(Modifier.height(16.dp))
            SafetySectionLabel("Sensitive App Categories", Icons.Default.Shield, ARIAColors.Accent)
            Spacer(Modifier.height(8.dp))
            Box(modifier = Modifier.padding(horizontal = 16.dp)) {
                SafetyCard {
                    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(
                            "Tap a category to block all its apps at once.",
                            style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
                        )
                        SENSITIVE_APP_PRESETS.forEach { (label, icon, packages) ->
                            SensitiveCategoryRow(
                                label    = label,
                                icon     = icon,
                                packages = packages,
                                blockedPackages = safetyConfig.blockedPackages,
                                onBlockAll   = { packages.forEach { p -> vm.addBlockedPackage(p) } },
                                onUnblockAll = { packages.forEach { p -> vm.removeBlockedPackage(p) } },
                            )
                        }
                    }
                }
            }
        }
    }
}

// ─── Kill Switch Card ─────────────────────────────────────────────────────────

@Composable
private fun KillSwitchCard(active: Boolean, onToggle: () -> Unit) {
    val bgColor = if (active) ARIAColors.Destructive.copy(alpha = 0.12f) else ARIAColors.Surface
    val borderColor = if (active) ARIAColors.Destructive.copy(alpha = 0.6f) else ARIAColors.Divider

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(14.dp))
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Icon(
                    Icons.Default.StopCircle,
                    contentDescription = null,
                    tint = ARIAColors.Destructive,
                    modifier = Modifier.size(28.dp)
                )
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        if (active) "KILL SWITCH ACTIVE" else "Kill Switch",
                        style = MaterialTheme.typography.bodyMedium.copy(
                            color = ARIAColors.Destructive, fontWeight = FontWeight.Bold
                        )
                    )
                    Text(
                        if (active)
                            "All ARIA actions are blocked. The agent cannot click, swipe, or type anything."
                        else
                            "One tap to immediately stop all agent actions on any app.",
                        style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted, lineHeight = 16.sp)
                    )
                }
                Switch(
                    checked  = active,
                    onCheckedChange = { onToggle() },
                    colors = SwitchDefaults.colors(
                        checkedThumbColor   = ARIAColors.Background,
                        checkedTrackColor   = ARIAColors.Destructive,
                        uncheckedThumbColor = ARIAColors.Muted,
                        uncheckedTrackColor = ARIAColors.SurfaceVariant,
                    )
                )
            }
            if (!active) {
                Button(
                    onClick   = onToggle,
                    modifier  = Modifier.fillMaxWidth(),
                    shape     = RoundedCornerShape(8.dp),
                    colors    = ButtonDefaults.buttonColors(containerColor = ARIAColors.Destructive),
                ) {
                    Icon(Icons.Default.StopCircle, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Activate Kill Switch", fontWeight = FontWeight.Bold)
                }
            } else {
                Button(
                    onClick   = onToggle,
                    modifier  = Modifier.fillMaxWidth(),
                    shape     = RoundedCornerShape(8.dp),
                    colors    = ButtonDefaults.buttonColors(containerColor = ARIAColors.Success),
                ) {
                    Icon(Icons.Default.CheckCircle, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(6.dp))
                    Text("Resume Agent Actions", fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

// ─── Blocklist Section ────────────────────────────────────────────────────────

@Composable
private fun BlocklistSection(
    packages: Set<String>,
    onAdd: (String) -> Unit,
    onRemove: (String) -> Unit,
) {
    val focusManager = LocalFocusManager.current
    var input by remember { mutableStateOf("") }

    SafetyCard {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                "ARIA will never interact with apps in this list.",
                style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value         = input,
                    onValueChange = { input = it },
                    modifier      = Modifier.weight(1f),
                    placeholder   = { Text("com.example.app", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
                    singleLine    = true,
                    shape         = RoundedCornerShape(8.dp),
                    colors        = safetyFieldColors(),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(onDone = {
                        if (input.isNotBlank()) { onAdd(input.trim()); input = "" }
                        focusManager.clearFocus()
                    })
                )
                IconButton(
                    onClick  = { if (input.isNotBlank()) { onAdd(input.trim()); input = "" } },
                    modifier = Modifier
                        .size(42.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(ARIAColors.Destructive.copy(alpha = 0.15f))
                ) {
                    Icon(Icons.Default.Block, contentDescription = "Block", tint = ARIAColors.Destructive)
                }
            }
            if (packages.isEmpty()) {
                Text("No apps blocked — ARIA can operate on any app.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
            } else {
                packages.sorted().forEach { pkg ->
                    PackageRow(pkg = pkg, color = ARIAColors.Destructive, onRemove = { onRemove(pkg) })
                }
            }
        }
    }
}

// ─── Allowlist Section ────────────────────────────────────────────────────────

@Composable
private fun AllowlistSection(
    allowlistMode: Boolean,
    packages: Set<String>,
    onToggleMode: (Boolean) -> Unit,
    onAdd: (String) -> Unit,
    onRemove: (String) -> Unit,
) {
    val focusManager = LocalFocusManager.current
    var input by remember { mutableStateOf("") }

    SafetyCard {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column(modifier = Modifier.weight(1f).padding(end = 8.dp)) {
                    Text("Allowlist mode", style = MaterialTheme.typography.bodyMedium.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold))
                    Text("ARIA can ONLY operate on apps in the list below.", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted))
                }
                Switch(
                    checked  = allowlistMode,
                    onCheckedChange = onToggleMode,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor  = ARIAColors.Background,
                        checkedTrackColor  = ARIAColors.Success,
                        uncheckedThumbColor = ARIAColors.Muted,
                        uncheckedTrackColor = ARIAColors.SurfaceVariant,
                    )
                )
            }

            if (allowlistMode) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value         = input,
                        onValueChange = { input = it },
                        modifier      = Modifier.weight(1f),
                        placeholder   = { Text("com.example.trusted", style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.Muted)) },
                        singleLine    = true,
                        shape         = RoundedCornerShape(8.dp),
                        colors        = safetyFieldColors(),
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                        keyboardActions = KeyboardActions(onDone = {
                            if (input.isNotBlank()) { onAdd(input.trim()); input = "" }
                            focusManager.clearFocus()
                        })
                    )
                    IconButton(
                        onClick  = { if (input.isNotBlank()) { onAdd(input.trim()); input = "" } },
                        modifier = Modifier
                            .size(42.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(ARIAColors.Success.copy(alpha = 0.15f))
                    ) {
                        Icon(Icons.Default.AddCircle, contentDescription = "Allow", tint = ARIAColors.Success)
                    }
                }
                if (packages.isEmpty()) {
                    SafetyInfoRow(Icons.Default.Warning, "Allowlist is empty — ARIA is currently fully blocked in allowlist mode.", ARIAColors.Warning)
                } else {
                    packages.sorted().forEach { pkg ->
                        PackageRow(pkg = pkg, color = ARIAColors.Success, onRemove = { onRemove(pkg) })
                    }
                }
            }
        }
    }
}

// ─── Sensitive Category Row ───────────────────────────────────────────────────

@Composable
private fun SensitiveCategoryRow(
    label: String,
    icon: ImageVector,
    packages: List<String>,
    blockedPackages: Set<String>,
    onBlockAll: () -> Unit,
    onUnblockAll: () -> Unit,
) {
    val allBlocked = packages.all { it in blockedPackages }
    val someBlocked = packages.any { it in blockedPackages }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(ARIAColors.SurfaceVariant)
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null,
                onClick = if (allBlocked) onUnblockAll else onBlockAll
            )
            .padding(10.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = if (allBlocked) ARIAColors.Destructive else ARIAColors.Muted, modifier = Modifier.size(18.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(label, style = MaterialTheme.typography.bodySmall.copy(color = ARIAColors.OnSurface, fontWeight = FontWeight.SemiBold))
            Text("${packages.size} apps", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Muted, fontSize = 10.sp))
        }
        if (someBlocked && !allBlocked) {
            Box(modifier = Modifier.clip(RoundedCornerShape(4.dp)).background(ARIAColors.Warning.copy(alpha = 0.2f)).padding(horizontal = 6.dp, vertical = 2.dp)) {
                Text("PARTIAL", style = MaterialTheme.typography.labelSmall.copy(color = ARIAColors.Warning, fontWeight = FontWeight.Bold, fontSize = 9.sp))
            }
        }
        Icon(
            if (allBlocked) Icons.Default.Block else Icons.Default.AddCircleOutline,
            contentDescription = if (allBlocked) "Unblock" else "Block all",
            tint = if (allBlocked) ARIAColors.Destructive else ARIAColors.Muted,
            modifier = Modifier.size(18.dp)
        )
    }
}

// ─── Shared sub-composables ───────────────────────────────────────────────────

@Composable
private fun PackageRow(pkg: String, color: androidx.compose.ui.graphics.Color, onRemove: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(6.dp))
            .background(color.copy(alpha = 0.08f))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(Icons.Default.Apps, contentDescription = null, tint = color, modifier = Modifier.size(14.dp))
        Text(pkg, style = MaterialTheme.typography.labelMedium.copy(color = ARIAColors.OnSurface, fontFamily = FontFamily.Monospace, fontSize = 11.sp), modifier = Modifier.weight(1f))
        IconButton(onClick = onRemove, modifier = Modifier.size(24.dp)) {
            Icon(Icons.Default.Close, contentDescription = "Remove", tint = ARIAColors.Muted, modifier = Modifier.size(13.dp))
        }
    }
}

@Composable
private fun SafetyInfoRow(icon: ImageVector, text: String, color: androidx.compose.ui.graphics.Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(6.dp))
            .background(color.copy(alpha = 0.08f))
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.Top
    ) {
        Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(13.dp).padding(top = 1.dp))
        Text(text, style = MaterialTheme.typography.bodySmall.copy(color = color, lineHeight = 15.sp))
    }
}

@Composable
private fun SafetySectionLabel(title: String, icon: ImageVector, color: androidx.compose.ui.graphics.Color) {
    Row(
        modifier = Modifier.padding(horizontal = 16.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(icon, contentDescription = null, tint = color, modifier = Modifier.size(13.dp))
        Text(
            title.uppercase(),
            style = MaterialTheme.typography.labelSmall.copy(
                color = color, fontFamily = FontFamily.Monospace, letterSpacing = 1.sp, fontWeight = FontWeight.SemiBold
            )
        )
    }
}

@Composable
private fun SafetyCard(content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(12.dp),
        colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
        elevation = CardDefaults.cardElevation(0.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp), content = content)
    }
}

@Composable
private fun safetyFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor      = ARIAColors.Primary,
    unfocusedBorderColor    = ARIAColors.Divider,
    focusedContainerColor   = ARIAColors.SurfaceVariant,
    unfocusedContainerColor = ARIAColors.SurfaceVariant,
    focusedTextColor        = ARIAColors.OnSurface,
    unfocusedTextColor      = ARIAColors.OnSurface,
    cursorColor             = ARIAColors.Primary,
)
