package com.ariaagent.mobile.ui.theme

import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * ARIA Design System — Jetpack Compose Material3 theme.
 *
 * Matches the React Native color system in constants/colors.ts:
 *   Background: #0A0F1E  (deep navy — dark cyber AI feel)
 *   Primary:    #00D4FF  (cyan — interactive elements)
 *   Secondary:  #1E2A3A  (muted panel backgrounds)
 *   Accent:     #7C3AED  (violet — LLM/AI operations)
 *   Success:    #10B981  (green — ready/running states)
 *   Error:      #EF4444  (red — errors)
 *   Warning:    #F59E0B  (amber — thermal/caution)
 *   Text:       #E2E8F0  (light slate)
 *   Muted:      #64748B  (dim secondary text)
 *
 * Phase: 11 (Jetpack Compose UI)
 */

// ─── ARIA Color Tokens ────────────────────────────────────────────────────────
object ARIAColors {
    val Background       = Color(0xFF0A0F1E)
    val BackgroundDark   = Color(0xFF06080F)   // status bar / nav bar
    val Surface1         = Color(0xFF111827)   // card backgrounds
    val Surface2         = Color(0xFF1A2332)   // elevated surfaces
    val Surface3         = Color(0xFF1E2A3A)   // borders / dividers

    val Primary          = Color(0xFF00D4FF)   // cyan
    val PrimaryDim       = Color(0x3300D4FF)   // 20% primary for backgrounds
    val PrimaryContainer = Color(0xFF0E3A4A)

    val Accent           = Color(0xFF7C3AED)   // violet — LLM
    val AccentDim        = Color(0x337C3AED)
    val AccentContainer  = Color(0xFF2D1B69)

    val Success          = Color(0xFF10B981)   // green — running/ready
    val SuccessDim       = Color(0x3310B981)
    val Warning          = Color(0xFFF59E0B)   // amber — thermal
    val WarningDim       = Color(0x33F59E0B)
    val Error            = Color(0xFFEF4444)   // red — error/stop
    val ErrorDim         = Color(0x33EF4444)

    val TextPrimary      = Color(0xFFE2E8F0)   // main text
    val TextSecondary    = Color(0xFF94A3B8)   // secondary labels
    val TextMuted        = Color(0xFF64748B)   // disabled / hints

    // ── Short aliases used by Compose screens ──────────────────────────────
    val Surface        = Surface1        // default card background
    val SurfaceVariant = Surface2        // chip / selectable backgrounds
    val OnSurface      = TextPrimary     // text on cards
    val Muted          = TextMuted       // dim labels / hints
    val Divider        = Surface3        // separator lines / input borders
    val Destructive    = Error           // destructive action buttons
}

// ─── Material3 dark color scheme ─────────────────────────────────────────────
private val AriaColorScheme = darkColorScheme(
    primary             = ARIAColors.Primary,
    onPrimary           = ARIAColors.Background,
    primaryContainer    = ARIAColors.PrimaryContainer,
    onPrimaryContainer  = ARIAColors.Primary,

    secondary           = ARIAColors.Accent,
    onSecondary         = Color.White,
    secondaryContainer  = ARIAColors.AccentContainer,
    onSecondaryContainer = ARIAColors.Accent,

    tertiary            = ARIAColors.Success,
    onTertiary          = Color.White,

    background          = ARIAColors.Background,
    onBackground        = ARIAColors.TextPrimary,

    surface             = ARIAColors.Surface1,
    onSurface           = ARIAColors.TextPrimary,
    surfaceVariant      = ARIAColors.Surface2,
    onSurfaceVariant    = ARIAColors.TextSecondary,

    outline             = ARIAColors.Surface3,
    outlineVariant      = ARIAColors.Surface2,

    error               = ARIAColors.Error,
    onError             = Color.White,
    errorContainer      = Color(0xFF3A0A0A),
    onErrorContainer    = ARIAColors.Error,
)

// ─── Typography ───────────────────────────────────────────────────────────────
private val AriaTypography = androidx.compose.material3.Typography(
    headlineLarge  = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Bold,   fontSize = 28.sp, color = ARIAColors.TextPrimary),
    headlineMedium = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Bold,   fontSize = 22.sp, color = ARIAColors.TextPrimary),
    headlineSmall  = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.SemiBold, fontSize = 18.sp, color = ARIAColors.TextPrimary),
    titleLarge     = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.SemiBold, fontSize = 16.sp, color = ARIAColors.TextPrimary),
    titleMedium    = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Medium,  fontSize = 14.sp, color = ARIAColors.TextPrimary),
    bodyLarge      = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Normal,  fontSize = 14.sp, color = ARIAColors.TextPrimary),
    bodyMedium     = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Normal,  fontSize = 13.sp, color = ARIAColors.TextSecondary),
    bodySmall      = TextStyle(fontFamily = FontFamily.Default, fontWeight = FontWeight.Normal,  fontSize = 11.sp, color = ARIAColors.TextMuted),
    labelLarge     = TextStyle(fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Medium, fontSize = 12.sp, color = ARIAColors.TextSecondary),
    labelMedium    = TextStyle(fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Normal, fontSize = 11.sp, color = ARIAColors.TextMuted),
)

// ─── Theme composable ─────────────────────────────────────────────────────────

@Composable
fun ARIATheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = AriaColorScheme,
        typography  = AriaTypography,
        content     = content
    )
}

// ─── Status color helpers ─────────────────────────────────────────────────────

/** Map an agent status string to its display color. */
fun statusColor(status: String): Color = when (status) {
    "running"  -> ARIAColors.Success
    "paused"   -> ARIAColors.Warning
    "error"    -> ARIAColors.Error
    "done"     -> ARIAColors.Primary
    else       -> ARIAColors.TextMuted     // idle
}

/** Map a thermal level string to its display color. */
fun thermalColor(level: String): Color = when (level) {
    "light"    -> ARIAColors.Success
    "moderate" -> ARIAColors.Warning
    "severe"   -> Color(0xFFFF6B35)        // orange
    "critical" -> ARIAColors.Error
    else       -> ARIAColors.TextMuted     // safe
}
