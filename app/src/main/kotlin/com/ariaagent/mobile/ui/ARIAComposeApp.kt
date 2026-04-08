package com.ariaagent.mobile.ui

import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.ariaagent.mobile.ui.screens.*
import com.ariaagent.mobile.ui.theme.ARIAColors
import com.ariaagent.mobile.ui.theme.ARIATheme
import com.ariaagent.mobile.ui.viewmodel.AgentViewModel

/**
 * ARIAComposeApp — root composable for the pure-Kotlin native UI.
 *
 * Migration Phase 5/6/7 update: adds Chat, Train tabs and Labeler full-screen route.
 *
 * Navigation destinations (bottom nav — 7 tabs):
 *   dashboard  → DashboardScreen  — at-a-glance status
 *   control    → ControlScreen    — start/pause/stop + goal input
 *   chat       → ChatScreen       — on-device LLM chat (Phase 5)
 *   activity   → ActivityScreen   — live action log + token stream
 *   train      → TrainScreen      — RL cycle + IRL video training (Phase 6)
 *   modules    → ModulesScreen    — hardware/model readiness
 *   settings   → SettingsScreen   — inference parameter editing
 *
 * Full-screen routes (no bottom nav):
 *   labeler    → LabelerScreen    — screenshot annotation + LLM enrichment (Phase 7)
 */

private sealed class Screen(
    val route: String,
    val label: String,
    val icon: ImageVector,
) {
    object Dashboard : Screen("dashboard", "Dashboard", Icons.Default.Dashboard)
    object Control   : Screen("control",   "Control",   Icons.Default.SmartToy)
    object Chat      : Screen("chat",      "Chat",      Icons.AutoMirrored.Filled.Chat)
    object Activity  : Screen("activity",  "Activity",  Icons.Default.Timeline)
    object Train     : Screen("train",     "Train",     Icons.Default.FitnessCenter)
    object Modules   : Screen("modules",   "Modules",   Icons.Default.Memory)
    object Settings  : Screen("settings",  "Settings",  Icons.Default.Settings)
}

private val bottomNavScreens = listOf(
    Screen.Dashboard,
    Screen.Control,
    Screen.Chat,
    Screen.Activity,
    Screen.Train,
    Screen.Modules,
    Screen.Settings,
)

private const val ROUTE_LABELER          = "labeler"
private const val ROUTE_GOALS            = "goals"
private const val ROUTE_SAFETY           = "safety"
private const val ROUTE_ONBOARDING       = "onboarding"
private const val ROUTE_KNOWLEDGE_WIZARD = "knowledge_wizard"

@Composable
fun ARIAComposeApp() {
    ARIATheme {
        val navController = rememberNavController()
        val vm: AgentViewModel = viewModel()
        val agentState by vm.agentState.collectAsState()
        val context = LocalContext.current

        // ── Screen capture permission (Fix 1) ─────────────────────────────────
        val screenCaptureLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.StartActivityForResult()
        ) { result ->
            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                vm.onScreenCaptureResult(result.resultCode, result.data!!)
            }
        }

        LaunchedEffect(Unit) {
            vm.screenCaptureRequestFlow.collect {
                val manager = context.getSystemService(MediaProjectionManager::class.java)
                screenCaptureLauncher.launch(manager.createScreenCaptureIntent())
            }
        }

        val onRequestScreenCapture: () -> Unit = { vm.requestScreenCapturePermission() }

        val onGrantAccessibility: () -> Unit = {
            context.startActivity(
                Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            )
        }

        val onboardingComplete by vm.onboardingComplete.collectAsState()

        val navBackStackEntry by navController.currentBackStackEntryAsState()
        val currentRoute = navBackStackEntry?.destination?.route

        val showBottomBar = currentRoute != ROUTE_LABELER
                && currentRoute != ROUTE_GOALS
                && currentRoute != ROUTE_SAFETY
                && currentRoute != ROUTE_ONBOARDING
                && currentRoute != ROUTE_KNOWLEDGE_WIZARD

        Scaffold(
            containerColor = ARIAColors.Background,
            bottomBar = {
                if (showBottomBar) {
                    NavigationBar(
                        containerColor = ARIAColors.Surface,
                        tonalElevation = androidx.compose.ui.unit.Dp(0f),
                        modifier       = Modifier.navigationBarsPadding()
                    ) {
                        val currentDestination = navBackStackEntry?.destination

                        bottomNavScreens.forEach { screen ->
                            val selected = currentDestination?.hierarchy
                                ?.any { it.route == screen.route } == true

                            NavigationBarItem(
                                icon = {
                                    BadgedBox(
                                        badge = {
                                            if (screen is Screen.Control && agentState.status == "running") {
                                                Badge(containerColor = ARIAColors.Success)
                                            }
                                        }
                                    ) {
                                        Icon(screen.icon, contentDescription = screen.label)
                                    }
                                },
                                label = {
                                    Text(
                                        screen.label,
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
                                        )
                                    )
                                },
                                selected = selected,
                                onClick = {
                                    navController.navigate(screen.route) {
                                        popUpTo(navController.graph.findStartDestination().id) {
                                            saveState = true
                                        }
                                        launchSingleTop = true
                                        restoreState    = true
                                    }
                                },
                                colors = NavigationBarItemDefaults.colors(
                                    selectedIconColor   = ARIAColors.Primary,
                                    selectedTextColor   = ARIAColors.Primary,
                                    unselectedIconColor = ARIAColors.Muted,
                                    unselectedTextColor = ARIAColors.Muted,
                                    indicatorColor      = ARIAColors.Primary.copy(alpha = 0.15f),
                                )
                            )
                        }
                    }
                }
            }
        ) { innerPadding ->
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(ARIAColors.Background)
                    .padding(innerPadding)
            ) {
                NavHost(
                    navController    = navController,
                    startDestination = if (onboardingComplete) Screen.Dashboard.route else ROUTE_ONBOARDING
                ) {
                    // ── Onboarding (first-run) ────────────────────────────────────────
                    composable(ROUTE_ONBOARDING) {
                        OnboardingScreen(
                            vm                     = vm,
                            onFinish               = {
                                navController.navigate(Screen.Dashboard.route) {
                                    popUpTo(ROUTE_ONBOARDING) { inclusive = true }
                                }
                            },
                            onGrantAccessibility   = onGrantAccessibility,
                            onRequestScreenCapture = onRequestScreenCapture,
                        )
                    }

                    composable(Screen.Dashboard.route) { DashboardScreen(vm) }

                    composable(Screen.Control.route) {
                        ControlScreen(
                            vm                  = vm,
                            onNavigateToLabeler = { navController.navigate(ROUTE_LABELER) },
                            onNavigateToGoals   = { navController.navigate(ROUTE_GOALS) },
                        )
                    }

                    composable(Screen.Chat.route) {
                        ChatScreen(vm)
                    }

                    composable(Screen.Activity.route) { ActivityScreen(vm) }

                    composable(Screen.Train.route) {
                        TrainScreen(
                            vm                  = vm,
                            onNavigateToLabeler = { navController.navigate(ROUTE_LABELER) },
                        )
                    }

                    composable(Screen.Modules.route) {
                        ModulesScreen(
                            vm                     = vm,
                            onRequestScreenCapture = onRequestScreenCapture,
                            onGrantAccessibility   = onGrantAccessibility,
                        )
                    }
                    composable(Screen.Settings.route) {
                        SettingsScreen(
                            vm                       = vm,
                            onNavigateToSafety       = { navController.navigate(ROUTE_SAFETY) },
                            onNavigateToKnowledge    = { navController.navigate(ROUTE_KNOWLEDGE_WIZARD) },
                        )
                    }

                    composable(ROUTE_LABELER) {
                        LabelerScreen(
                            vm                    = vm,
                            onBack                = { navController.popBackStack() },
                            onRequestScreenCapture = onRequestScreenCapture,
                        )
                    }

                    // ── Full-screen feature routes ────────────────────────────────────
                    composable(ROUTE_GOALS) {
                        GoalsScreen(
                            vm     = vm,
                            onBack = { navController.popBackStack() },
                        )
                    }

                    composable(ROUTE_SAFETY) {
                        SafetyScreen(
                            vm     = vm,
                            onBack = { navController.popBackStack() },
                        )
                    }

                    composable(ROUTE_KNOWLEDGE_WIZARD) {
                        KnowledgeWizardScreen(
                            onBack = { navController.popBackStack() },
                        )
                    }
                }
            }
        }
    }
}
