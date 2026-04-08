package com.ariaagent.mobile.ui.viewmodel

import android.app.Application
import android.content.Intent
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ariaagent.mobile.core.agent.AgentLoop
import com.ariaagent.mobile.core.agent.AppSkillRegistry
import com.ariaagent.mobile.core.agent.TaskQueueManager
import com.ariaagent.mobile.core.ai.ChatContextBuilder
import com.ariaagent.mobile.core.ai.LlamaEngine
import com.ariaagent.mobile.core.ai.ModelDownloadService
import com.ariaagent.mobile.core.ai.ModelManager
import com.ariaagent.mobile.core.ai.Sam2Engine
import com.ariaagent.mobile.core.ai.VisionEngine
import com.ariaagent.mobile.core.config.AgentCombo
import com.ariaagent.mobile.core.config.AriaConfig
import com.ariaagent.mobile.core.config.ConfigStore
import com.ariaagent.mobile.core.config.ModuleVisionMode
import com.ariaagent.mobile.core.config.SafetyConfigStore
import com.ariaagent.mobile.core.events.AgentEventBus
import com.ariaagent.mobile.core.memory.EmbeddingModelManager
import com.ariaagent.mobile.core.memory.ExperienceStore
import com.ariaagent.mobile.core.memory.ObjectLabelStore
import com.ariaagent.mobile.core.monitoring.LocalDeviceServer
import com.ariaagent.mobile.core.monitoring.MonitoringPusher
import com.ariaagent.mobile.core.ocr.OcrEngine
import com.ariaagent.mobile.core.perception.ObjectDetectorEngine
import com.ariaagent.mobile.core.perception.ScreenObserver
import com.ariaagent.mobile.core.persistence.ProgressPersistence
import com.ariaagent.mobile.core.memory.SessionReplayStore
import com.ariaagent.mobile.core.patterns.SuggestionStore
import com.ariaagent.mobile.core.rl.IrlModule
import com.ariaagent.mobile.core.rl.LoraTrainer
import com.ariaagent.mobile.core.rl.PolicyNetwork
import com.ariaagent.mobile.system.AgentForegroundService
import com.ariaagent.mobile.system.accessibility.AgentAccessibilityService
import com.ariaagent.mobile.system.overlay.FloatingChatService
import com.ariaagent.mobile.system.screen.ScreenCaptureService
import org.json.JSONObject
import kotlinx.coroutines.Dispatchers
import com.ariaagent.mobile.core.triggers.TriggerItem
import com.ariaagent.mobile.core.triggers.TriggerStore
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

// ─── UI state data classes ────────────────────────────────────────────────────

data class AgentUiState(
    val status: String               = "idle",
    val currentTask: String          = "",
    val currentApp: String           = "",
    val stepCount: Int               = 0,
    val lastAction: String           = "",
    val lastError: String            = "",
    val gameMode: String             = "none",
    val tokenRate: Double            = 0.0,
    val modelReady: Boolean          = false,
    val modelLoaded: Boolean         = false,
    val accessibilityActive: Boolean = false,
    val screenCaptureActive: Boolean = false,
)

data class ActionLogEntry(
    val id: Long,
    val tool: String,
    val nodeId: String,
    val success: Boolean,
    val reward: Double,
    val stepCount: Int,
    val appPackage: String,
    val timestamp: Long,
)

data class ThermalUiState(
    val level: String          = "safe",
    val inferenceSafe: Boolean = true,
    val trainingSafe: Boolean  = true,
    val emergency: Boolean     = false,
)

data class LearningUiState(
    val loraVersion: Int      = 0,
    val policyVersion: Int    = 0,
    val adamStep: Int         = 0,
    val lastPolicyLoss: Double = 0.0,
    val untrainedSamples: Int = 0,
)

data class StepUiState(
    val stepNumber: Int  = 0,
    val activity: String = "idle",
)

data class ModuleUiState(
    val modelReady: Boolean           = false,
    val modelLoaded: Boolean          = false,
    val tokensPerSecond: Double       = 0.0,
    val ocrReady: Boolean             = false,
    val detectorReady: Boolean        = false,
    val detectorSizeMb: Float         = 0f,
    val embeddingCount: Int           = 0,
    val labelCount: Int               = 0,
    val accessibilityGranted: Boolean = false,
    val screenCaptureGranted: Boolean = false,
    val episodesRun: Int              = 0,
    val adapterLoaded: Boolean        = false,
    val loraVersion: Int              = 0,
    // LLM download progress (driven by ModelDownloadService via AgentEventBus)
    val llmDownloadPercent: Int       = 0,
    val llmDownloadMb: Double         = 0.0,
    val llmDownloadTotalMb: Double    = 0.0,
    val llmDownloadSpeedMbps: Double  = 0.0,
    val llmDownloadError: String?     = null,
    // Embedding model (MiniLM ONNX ~23 MB)
    val embeddingReady: Boolean       = false,
    val embeddingVocabReady: Boolean  = false,
    val embeddingDownloadedMb: Float  = 0f,
    // Local monitoring server
    val localServerRunning: Boolean   = false,
    val localServerUrl: String        = "",
    // Vision model — SmolVLM-256M + mmproj (Phase 17)
    val visionReady: Boolean          = false,
    val visionLoaded: Boolean         = false,
    val visionModelDownloadedMb: Float = 0f,
    val mmProjDownloadedMb: Float     = 0f,
    val visionDownloadPercent: Int    = 0,
    val visionDownloadError: String?  = null,
    // SAM2 / MobileSAM pixel segmentation encoder (Phase 18)
    val sam2Ready: Boolean            = false,
    val sam2Loaded: Boolean           = false,
    val sam2DownloadedMb: Float       = 0f,
    val sam2DownloadPercent: Int      = 0,
    val sam2DownloadError: String?    = null,
)

/** ExperienceStore breakdown for ActivityScreen. */
data class MemoryStatsUi(
    val total: Int       = 0,
    val success: Int     = 0,
    val failure: Int     = 0,
    val edgeCase: Int    = 0,
    val untrained: Int   = 0,
    val enrichedLabels: Int = 0,
    val totalLabels: Int    = 0,
)

/** Phase 15: mirrors TaskQueueManager.QueuedTask for Compose UI. */
data class QueuedTaskItem(
    val id: String,
    val goal: String,
    val appPackage: String,
    val priority: Int,
    val enqueuedAt: Long,
)

/** Phase 15: mirrors AppSkillRegistry.AppSkill for Compose UI. */
data class AppSkillItem(
    val appPackage: String,
    val appName: String,
    val taskSuccess: Int,
    val taskFailure: Int,
    val totalSteps: Int,
    val successRate: Float,
    val avgSteps: Float,
    val learnedElements: List<String>,
    val promptHint: String,
    val lastSeen: Long,
)

/** Phase 6/8: game loop metrics for Dashboard. */
data class GameLoopUiState(
    val isActive: Boolean    = false,
    val gameType: String     = "none",
    val episodeCount: Int    = 0,
    val stepCount: Int       = 0,
    val currentScore: Double = 0.0,
    val highScore: Double    = 0.0,
    val totalReward: Double  = 0.0,
    val lastAction: String   = "",
    val isGameOver: Boolean  = false,
)

/** Memory entry shown in ActivityScreen Memory tab. Mapped from ExperienceStore.ExperienceTuple. */
data class MemoryEntry(
    val id         : String,
    val summary    : String,
    val app        : String,
    val taskType   : String,
    val result     : String,   // "success" | "failure"
    val reward     : Double,
    val isEdgeCase : Boolean,
    val timestamp  : Long,
)

// ── Migration Phase 5: Chat ───────────────────────────────────────────────────

/** A single message in the chat conversation. role: "user" | "aria" | "system" */
data class ChatMessageItem(
    val id: String,
    val role: String,
    val text: String,
    val tps: Double = 0.0,
    val ts: Long = System.currentTimeMillis(),
)

// ── Migration Phase 6: Train ──────────────────────────────────────────────────

data class LearningStatusUi(
    val loraVersion: Int,
    val latestAdapterPath: String,
    val adapterExists: Boolean,
    val untrainedSamples: Int,
    val policyReady: Boolean,
    val adamStep: Int,
    val lastPolicyLoss: Double,
    val lastTrainedAt: Long,
)

data class RlResultUi(
    val success: Boolean,
    val samplesUsed: Int,
    val adapterPath: String,
    val loraVersion: Int,
    val errorMessage: String,
)

data class IrlResultUi(
    val framesProcessed: Int,
    val tuplesExtracted: Int,
    val llmAssistedCount: Int,
    val errorMessage: String,
)

// ── Migration Phase 7: Labeler ────────────────────────────────────────────────

data class ScreenCaptureUi(
    val imagePath: String,
    val appPackage: String,
    val screenHash: String,
    val ocrText: String,
    val a11yTree: String,
)

/** Safety configuration — persisted via SafetyConfigStore. */
data class SafetyConfig(
    val globalKillActive: Boolean  = false,
    val confirmMode: Boolean       = false,
    val blockedPackages: Set<String> = emptySet(),
    val allowlistMode: Boolean     = false,
    val allowedPackages: Set<String> = emptySet(),
)

/**
 * State for the safety gate warning banner.
 * Populated when [ConfidenceGate] blocks a destructive/uncertain action.
 * Dismissed via [AgentViewModel.dismissSafetyBanner].
 */
data class SafetyGateBannerState(
    val blockedAction: String,   // original action JSON that was blocked
    val safeAction:    String,   // replacement Wait action ARIA used instead
    val stepCount:     Int,      // agent step at which the block occurred
    val appPackage:    String,   // app that was active when blocked
)

/** One LoRA adapter checkpoint found on-disk. */
data class LoraCheckpointItem(
    val version: Int,
    val adapterPath: String,
    val sizeKb: Long,
    val createdAt: Long,
    val isLatest: Boolean,
)

// ── Multi-LLM Management (Phase 20) ──────────────────────────────────────────

/**
 * The functional role assigned to a loaded LLM.
 * ARIA routes different subsystems to whichever model holds each role.
 */
enum class LlmRole(val label: String, val description: String) {
    EVERYTHING(     "Everything",      "Single model handles all tasks — vision, reasoning, chat, tools"),
    EVERYTHING_ELSE("Everything Else", "Fallback — handles any task not claimed by a specialist model"),
    REASONING(      "Reasoning",       "Action planning, decision making, task decomposition"),
    VISION(         "Vision",          "Screen reading, image understanding, UI parsing — chains output to Reasoning model"),
    PLANNING(       "Planning",        "Multi-step goal planning and sub-task generation — runs before Reasoning"),
    TOOL_USE(       "Tool Use",        "Function calling, gesture dispatch, app control"),
    CHAT(           "Chat",            "Natural language conversation and user interaction"),
    CUSTOM(         "Custom",          "Custom role — define your own system prompt"),
}

/**
 * Tracks one catalog/custom LLM entry in the multi-LLM slot system.
 * Models are registered in this map as soon as the user taps Load.
 * [isLoaded] = the model weights are in RAM and ready for inference.
 */
data class LoadedLlmEntry(
    val modelId:        String,
    val role:           LlmRole = LlmRole.REASONING,
    val systemPrompt:   String  = "",
    val isLoaded:       Boolean = false,
    val isLoading:      Boolean = false,
    val isDownloading:  Boolean = false,
    val downloadPercent: Int    = 0,
    val downloadError:  String? = null,
    val loadError:      String? = null,
)

/** Live progress during a LoRA training run. */
data class LoraTrainingProgress(
    val percentComplete: Int = 0,
    val currentLoss: Double  = 0.0,
    val samplesUsed: Int     = 0,
    val phase: String        = "preparing",
)

/** Phase 15: notification shown when ARIA auto-chains to the next queued task. */
data class ChainedTaskItem(
    val taskId: String,
    val goal: String,
    val appPackage: String,
    val queueSize: Int,
    val timestamp: Long,
)

/** T002: Session Replay — summary of one recorded agent session. */
data class ReplaySessionItem(
    val sessionId: String,
    val goal: String,
    val stepCount: Int,
    val succeeded: Int,
    val failed: Int,
    val startTime: Long,
    val endTime: Long,
)

/** T002: Session Replay — one step within a recorded session. */
data class ReplayStepItem(
    val stepIdx: Int,
    val screenHash: String,
    val actionJson: String,
    val reason: String,
    val result: String,
    val appPackage: String,
    val timestamp: Long,
)

/** T005: Proactive Goal Surfacing — pending automation suggestion for DashboardScreen banner. */
data class SuggestionBannerItem(
    val id: Long,
    val appPackage: String,
    val goalText: String,
    val suggestionText: String,
    val repeatCount: Int,
)

/**
 * AgentViewModel — primary state holder for the Jetpack Compose UI.
 *
 * Phase 11: initial implementation.
 * Phase 15 update: adds task queue, app skills, game loop, and chained-task state.
 *
 * Subscribes to AgentEventBus (Kotlin SharedFlow) — NO React Native bridge, NO JS.
 *
 * Event wiring:
 *   agent_status_changed    → agentState
 *   action_performed        → actionLogs (prepend, max 200)
 *   token_generated         → agentState.tokenRate + streamBuffer
 *   step_started            → stepState
 *   thermal_status_changed  → thermalState
 *   learning_cycle_complete → learningState + refreshModuleState
 *   model_download_complete → refreshModuleState
 *   game_loop_status        → gameLoopState          [Phase 15]
 *   skill_updated           → appSkills merge        [Phase 15]
 *   task_chain_advanced     → chainedTask + refresh  [Phase 15]
 *   config_updated          → config (via ConfigStore flow)
 */
class AgentViewModel(app: Application) : AndroidViewModel(app) {

    private val context = app.applicationContext

    // ─── Observable state ─────────────────────────────────────────────────────

    private val _agentState   = MutableStateFlow(AgentUiState())
    val agentState: StateFlow<AgentUiState> = _agentState.asStateFlow()

    private val _actionLogs   = MutableStateFlow<List<ActionLogEntry>>(emptyList())
    val actionLogs: StateFlow<List<ActionLogEntry>> = _actionLogs.asStateFlow()

    // ── Action log internal buffer ────────────────────────────────────────────
    // ArrayDeque avoids the triple-allocation pattern in the old update:
    //   listOf(entry) + prev.take(199)  ← 2 list allocations per action
    // Now: addFirst() + removeLast() + single toList() snapshot.
    // Thread safety: only written from viewModelScope coroutines (sequential).
    private val actionLogsDeque = ArrayDeque<ActionLogEntry>(200)

    // ── refreshModuleState concurrency guard ──────────────────────────────────
    // refreshModuleState() is called from 30+ callsites. Without a guard,
    // a burst of events (model load + download complete + status change) can
    // spawn dozens of concurrent IO coroutines, all querying the same SQLite DB.
    // Solution: the last call always wins — any in-progress refresh is cancelled
    // and a new one starts immediately (no artificial delay).
    private var refreshModuleJob: kotlinx.coroutines.Job? = null

    private val _thermalState = MutableStateFlow(ThermalUiState())
    val thermalState: StateFlow<ThermalUiState> = _thermalState.asStateFlow()

    private val _learningState = MutableStateFlow(LearningUiState())
    val learningState: StateFlow<LearningUiState> = _learningState.asStateFlow()

    private val _stepState    = MutableStateFlow(StepUiState())
    val stepState: StateFlow<StepUiState> = _stepState.asStateFlow()

    private val _moduleState  = MutableStateFlow(ModuleUiState())
    val moduleState: StateFlow<ModuleUiState> = _moduleState.asStateFlow()

    private val _streamBuffer = MutableStateFlow("")
    val streamBuffer: StateFlow<String> = _streamBuffer.asStateFlow()

    // Phase 15: task queue
    private val _taskQueue = MutableStateFlow<List<QueuedTaskItem>>(emptyList())
    val taskQueue: StateFlow<List<QueuedTaskItem>> = _taskQueue.asStateFlow()

    // Phase 15: app skills
    private val _appSkills = MutableStateFlow<List<AppSkillItem>>(emptyList())
    val appSkills: StateFlow<List<AppSkillItem>> = _appSkills.asStateFlow()

    // Phase 6/8: game loop
    private val _gameLoopState = MutableStateFlow<GameLoopUiState?>(null)
    val gameLoopState: StateFlow<GameLoopUiState?> = _gameLoopState.asStateFlow()

    // Phase 15: chained task notification
    private val _chainedTask = MutableStateFlow<ChainedTaskItem?>(null)
    val chainedTask: StateFlow<ChainedTaskItem?> = _chainedTask.asStateFlow()

    // Migration Phase 3: memory entries for ActivityScreen Memory tab
    private val _memoryEntries = MutableStateFlow<List<MemoryEntry>>(emptyList())
    val memoryEntries: StateFlow<List<MemoryEntry>> = _memoryEntries.asStateFlow()

    // ── Migration Phase 5: Chat ───────────────────────────────────────────────

    private val welcomeChatMsg = ChatMessageItem(
        id   = "welcome",
        role = "system",
        text = "Chat with ARIA — your messages are sent to the on-device LLM with full context: agent state, memory, task queue and app skills injected automatically.",
        ts   = 0L,
    )

    private val _chatMessages = MutableStateFlow<List<ChatMessageItem>>(listOf(welcomeChatMsg))
    val chatMessages: StateFlow<List<ChatMessageItem>> = _chatMessages.asStateFlow()

    private val _chatThinking = MutableStateFlow(false)
    val chatThinking: StateFlow<Boolean> = _chatThinking.asStateFlow()

    // ── Migration Phase 6: Train ──────────────────────────────────────────────

    private val _learningStatusUi = MutableStateFlow<LearningStatusUi?>(null)
    val learningStatusUi: StateFlow<LearningStatusUi?> = _learningStatusUi.asStateFlow()

    private val _rlRunning = MutableStateFlow(false)
    val rlRunning: StateFlow<Boolean> = _rlRunning.asStateFlow()

    private val _rlResult = MutableStateFlow<RlResultUi?>(null)
    val rlResult: StateFlow<RlResultUi?> = _rlResult.asStateFlow()

    private val _irlRunning = MutableStateFlow(false)
    val irlRunning: StateFlow<Boolean> = _irlRunning.asStateFlow()

    private val _irlResult = MutableStateFlow<IrlResultUi?>(null)
    val irlResult: StateFlow<IrlResultUi?> = _irlResult.asStateFlow()

    private val _irlExtracting = MutableStateFlow(false)
    val irlExtracting: StateFlow<Boolean> = _irlExtracting.asStateFlow()

    private val _irlFramePaths = MutableStateFlow<List<String>>(emptyList())
    val irlFramePaths: StateFlow<List<String>> = _irlFramePaths.asStateFlow()

    private val _autoScheduleRl = MutableStateFlow(false)
    val autoScheduleRl: StateFlow<Boolean> = _autoScheduleRl.asStateFlow()

    // True while LearningScheduler is actively running a training cycle.
    // Set by "scheduler_training_started" / "scheduler_training_stopped" events.
    private val _schedulerActive = MutableStateFlow(false)
    val schedulerActive: StateFlow<Boolean> = _schedulerActive.asStateFlow()

    // ── Migration Phase 7: Labeler ────────────────────────────────────────────

    private val _labelerCapture = MutableStateFlow<ScreenCaptureUi?>(null)
    val labelerCapture: StateFlow<ScreenCaptureUi?> = _labelerCapture.asStateFlow()

    private val _labelerLabels = MutableStateFlow<List<ObjectLabelStore.ObjectLabel>>(emptyList())
    val labelerLabels: StateFlow<List<ObjectLabelStore.ObjectLabel>> = _labelerLabels.asStateFlow()

    private val _labelerCapturing = MutableStateFlow(false)
    val labelerCapturing: StateFlow<Boolean> = _labelerCapturing.asStateFlow()

    private val _labelerDetecting = MutableStateFlow(false)
    val labelerDetecting: StateFlow<Boolean> = _labelerDetecting.asStateFlow()

    private val _labelerEnriching = MutableStateFlow(false)
    val labelerEnriching: StateFlow<Boolean> = _labelerEnriching.asStateFlow()

    private val _labelerSaving = MutableStateFlow(false)
    val labelerSaving: StateFlow<Boolean> = _labelerSaving.asStateFlow()

    private val _labelerError = MutableStateFlow<String?>(null)
    val labelerError: StateFlow<String?> = _labelerError.asStateFlow()

    private val _labelerSaveSuccess = MutableStateFlow(false)
    val labelerSaveSuccess: StateFlow<Boolean> = _labelerSaveSuccess.asStateFlow()

    // ── Screen capture permission request (Fix 1) ─────────────────────────────

    private val _screenCaptureRequestFlow = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
    val screenCaptureRequestFlow: SharedFlow<Unit> = _screenCaptureRequestFlow.asSharedFlow()

    // ── Model download state (Fix 2) ──────────────────────────────────────────

    private val _llmDownloading = MutableStateFlow(false)
    val llmDownloading: StateFlow<Boolean> = _llmDownloading.asStateFlow()

    private val _detectorDownloading = MutableStateFlow(false)
    val detectorDownloading: StateFlow<Boolean> = _detectorDownloading.asStateFlow()

    private val _embeddingDownloading = MutableStateFlow(false)
    val embeddingDownloading: StateFlow<Boolean> = _embeddingDownloading.asStateFlow()

    private val _visionDownloading = MutableStateFlow(false)
    val visionDownloading: StateFlow<Boolean> = _visionDownloading.asStateFlow()

    private val _visionLoading = MutableStateFlow(false)
    val visionLoading: StateFlow<Boolean> = _visionLoading.asStateFlow()

    private val _sam2Downloading = MutableStateFlow(false)
    val sam2Downloading: StateFlow<Boolean> = _sam2Downloading.asStateFlow()

    private val _sam2Loading = MutableStateFlow(false)
    val sam2Loading: StateFlow<Boolean> = _sam2Loading.asStateFlow()

    private val _memoryStats = MutableStateFlow(MemoryStatsUi())
    val memoryStats: StateFlow<MemoryStatsUi> = _memoryStats.asStateFlow()

    private val _allLabels = MutableStateFlow<List<ObjectLabelStore.ObjectLabel>>(emptyList())
    val allLabels: StateFlow<List<ObjectLabelStore.ObjectLabel>> = _allLabels.asStateFlow()

    // ── Onboarding ────────────────────────────────────────────────────────────
    private val _onboardingComplete = MutableStateFlow(false)
    val onboardingComplete: StateFlow<Boolean> = _onboardingComplete.asStateFlow()

    // ── Safety config ─────────────────────────────────────────────────────────
    private val _safetyConfig = MutableStateFlow(SafetyConfig())
    val safetyConfig: StateFlow<SafetyConfig> = _safetyConfig.asStateFlow()

    /**
     * Non-null while the confidence gate has blocked a destructive/uncertain action.
     * The UI should show a dismissible warning banner with the blocked action details.
     * Cleared by [dismissSafetyBanner].
     */
    private val _safetyGateBanner = MutableStateFlow<SafetyGateBannerState?>(null)
    val safetyGateBanner: StateFlow<SafetyGateBannerState?> = _safetyGateBanner.asStateFlow()

    // ── Triggers ──────────────────────────────────────────────────────────────
    private val _triggers = MutableStateFlow<List<TriggerItem>>(emptyList())
    val triggers: StateFlow<List<TriggerItem>> = _triggers.asStateFlow()

    // ── LoRA training history + live progress ─────────────────────────────────
    private val _loraHistory = MutableStateFlow<List<LoraCheckpointItem>>(emptyList())
    val loraHistory: StateFlow<List<LoraCheckpointItem>> = _loraHistory.asStateFlow()

    private val _loraTrainingProgress = MutableStateFlow<LoraTrainingProgress?>(null)
    val loraTrainingProgress: StateFlow<LoraTrainingProgress?> = _loraTrainingProgress.asStateFlow()

    // ── Multi-LLM slot state (Phase 20) ──────────────────────────────────────
    // Key = modelId.  Each entry tracks the model's role, system prompt, and RAM state.
    private val _loadedLlms = MutableStateFlow<Map<String, LoadedLlmEntry>>(emptyMap())
    val loadedLlms: StateFlow<Map<String, LoadedLlmEntry>> = _loadedLlms.asStateFlow()

    // ── T002: Session Replay ──────────────────────────────────────────────────
    private val _replaySessions = MutableStateFlow<List<ReplaySessionItem>>(emptyList())
    val replaySessions: StateFlow<List<ReplaySessionItem>> = _replaySessions.asStateFlow()

    private val _replaySteps = MutableStateFlow<List<ReplayStepItem>>(emptyList())
    val replaySteps: StateFlow<List<ReplayStepItem>> = _replaySteps.asStateFlow()

    // ── T005: Proactive Goal Surfacing ────────────────────────────────────────
    private val _pendingSuggestions = MutableStateFlow<List<SuggestionBannerItem>>(emptyList())
    val pendingSuggestions: StateFlow<List<SuggestionBannerItem>> = _pendingSuggestions.asStateFlow()

    /** Config — reactive DataStore flow, auto-updates on any config change. */
    val config: StateFlow<AriaConfig> = ConfigStore.flow(context)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000L), AriaConfig())

    // ─── Init ─────────────────────────────────────────────────────────────────

    init {
        restoreLlmSlots()
        refreshModuleState()
        refreshTaskQueue()
        refreshAppSkills()
        checkOnboardingComplete()
        viewModelScope.launch(Dispatchers.IO) {
            _safetyConfig.value = SafetyConfigStore.load(context)
        }
        refreshLoraHistory()
        refreshReplaySessions()
        refreshPendingSuggestions()

        // Auto-load vision model on startup if files are already on disk.
        // This pre-warms the ~200 MB model so the first agent step does not
        // pay the cold-start latency cost (~2–4 s on Exynos 9611).
        viewModelScope.launch(Dispatchers.IO) {
            val visionEngine = com.ariaagent.mobile.core.ai.VisionEngine
            if (visionEngine.isVisionModelReady(context) && !LlamaEngine.isVisionLoaded()) {
                android.util.Log.i("AgentViewModel", "Auto-loading vision model (files present on disk)…")
                _visionLoading.value = true
                runCatching {
                    val cfg = ConfigStore.getBlocking(context)
                    visionEngine.ensureLoaded(context, cfg.nGpuLayers, cfg.gpuBackend)
                }
                _visionLoading.value = false
                refreshModuleState()
            }
        }

        // ── Wire loaded-LLM slots into AgentLoop so ModelDispatcher can route ────
        // Every time _loadedLlms changes (load / unload / role change), push the
        // full map to AgentLoop.setSlots().  AgentLoop reads activeSlots in its
        // REASON step to compute the dispatch plan via ModelDispatcher.plan().
        viewModelScope.launch {
            _loadedLlms.collect { slots ->
                AgentLoop.setSlots(slots)
            }
        }

        viewModelScope.launch {
            AgentEventBus.flow.collect { (name, data) ->
                when (name) {
                    "agent_status_changed"    -> handleStatusChanged(data)
                    "action_performed"        -> handleActionPerformed(data)
                    "token_generated"         -> handleTokenGenerated(data)
                    "step_started"            -> handleStepStarted(data)
                    "thermal_status_changed"  -> handleThermalChanged(data)
                    "learning_cycle_complete"      -> handleLearningComplete(data)
                    "scheduler_training_started"  -> _schedulerActive.value = true
                    "scheduler_training_stopped"  -> _schedulerActive.value = false
                    "model_download_progress"      -> handleLlmDownloadProgress(data)
                    "model_download_complete"  -> {
                        _llmDownloading.value = false
                        _moduleState.update { it.copy(llmDownloadPercent = 100, llmDownloadError = null) }
                        refreshModuleState()
                        val downloadedPath = data["path"] as? String
                        if (!downloadedPath.isNullOrBlank()) {
                            val current = config.value
                            if (current.modelPath.isBlank() || current.modelPath != downloadedPath) {
                                saveConfig(current.copy(modelPath = downloadedPath))
                            }
                        }
                    }
                    "model_download_error"     -> {
                        _llmDownloading.value = false
                        _moduleState.update { it.copy(llmDownloadError = data["error"] as? String ?: "Download failed") }
                    }
                    "game_loop_status"         -> handleGameLoopStatus(data)
                    "skill_updated"           -> handleSkillUpdated(data)
                    "task_chain_advanced"     -> handleTaskChainAdvanced(data)
                    "config_updated"          -> { /* handled by ConfigStore.flow() above */ }
                    "action_review_needed"    -> handleActionReviewNeeded(data)
                }
            }
        }
    }

    // ─── Event handlers ───────────────────────────────────────────────────────

    private fun handleStatusChanged(data: Map<String, Any>) {
        val status = data["status"] as? String ?: return
        _agentState.update { prev -> prev.copy(
            status      = status,
            currentTask = data["currentTask"] as? String ?: prev.currentTask,
            currentApp  = data["currentApp"]  as? String ?: prev.currentApp,
            stepCount   = (data["stepCount"] as? Int) ?: prev.stepCount,
            lastAction  = data["lastAction"]  as? String ?: prev.lastAction,
            lastError   = data["lastError"]   as? String ?: prev.lastError,
            gameMode    = data["gameMode"]    as? String ?: prev.gameMode,
        )}
        if (status == "idle" || status == "done" || status == "error") {
            _streamBuffer.value = ""
            _stepState.value = StepUiState()
            stopFloatingChat()
        }
    }

    private fun handleActionPerformed(data: Map<String, Any>) {
        val entry = ActionLogEntry(
            id         = (data["timestamp"] as? Long) ?: System.currentTimeMillis(),
            tool       = data["tool"]       as? String  ?: "unknown",
            nodeId     = data["nodeId"]     as? String  ?: "",
            success    = data["success"]    as? Boolean ?: false,
            reward     = (data["reward"]    as? Double) ?: 0.0,
            stepCount  = (data["stepCount"] as? Int)    ?: 0,
            appPackage = data["appPackage"] as? String  ?: "",
            timestamp  = (data["timestamp"] as? Long)   ?: System.currentTimeMillis(),
        )
        actionLogsDeque.addFirst(entry)
        while (actionLogsDeque.size > 200) actionLogsDeque.removeLast()
        _actionLogs.value = actionLogsDeque.toList()
        _streamBuffer.value = ""
    }

    private fun handleTokenGenerated(data: Map<String, Any>) {
        val token = data["token"] as? String ?: return
        val tps   = (data["tokensPerSecond"] as? Double) ?: 0.0
        _agentState.update { it.copy(tokenRate = tps) }
        _streamBuffer.update { it + token }
    }

    private fun handleStepStarted(data: Map<String, Any>) {
        _stepState.value = StepUiState(
            stepNumber = (data["stepNumber"] as? Int) ?: 0,
            activity   = data["activity"] as? String ?: "observe",
        )
        _streamBuffer.value = ""
    }

    private fun handleThermalChanged(data: Map<String, Any>) {
        _thermalState.value = ThermalUiState(
            level         = data["level"]         as? String  ?: "safe",
            inferenceSafe = data["inferenceSafe"] as? Boolean ?: true,
            trainingSafe  = data["trainingSafe"]  as? Boolean ?: true,
            emergency     = data["emergency"]     as? Boolean ?: false,
        )
    }

    private fun handleLearningComplete(data: Map<String, Any>) {
        _learningState.update { prev -> prev.copy(
            loraVersion   = (data["loraVersion"]   as? Int) ?: prev.loraVersion,
            policyVersion = (data["policyVersion"] as? Int) ?: prev.policyVersion,
            adamStep      = PolicyNetwork.adamStepCount,
            lastPolicyLoss = PolicyNetwork.lastPolicyLoss,
        )}
        refreshModuleState()
    }

    /**
     * Confidence gate blocked a destructive action — show a warning banner in the UI.
     * The banner includes the blocked action JSON and the reason it was held.
     * Cleared when the user dismisses it via [dismissSafetyBanner].
     */
    private fun handleActionReviewNeeded(data: Map<String, Any>) {
        val original  = data["original_action"] as? String ?: return
        val safe      = data["safe_action"]     as? String ?: ""
        val stepCount = (data["stepCount"]      as? Int)   ?: 0
        val app       = data["currentApp"]      as? String ?: ""
        _safetyGateBanner.value = SafetyGateBannerState(
            blockedAction = original,
            safeAction    = safe,
            stepCount     = stepCount,
            appPackage    = app,
        )
        android.util.Log.w("AgentViewModel", "Safety gate banner shown — blocked: $original")
    }

    /** Dismiss the confidence gate warning banner. */
    fun dismissSafetyBanner() {
        _safetyGateBanner.value = null
    }

    /** Phase 6/8: game loop event from GameLoop.kt via AgentEventBus. */
    private fun handleGameLoopStatus(data: Map<String, Any>) {
        _gameLoopState.value = GameLoopUiState(
            isActive     = data["isActive"]     as? Boolean ?: false,
            gameType     = data["gameType"]     as? String  ?: "none",
            episodeCount = (data["episodeCount"] as? Int)   ?: 0,
            stepCount    = (data["stepCount"]   as? Int)    ?: 0,
            currentScore = (data["currentScore"] as? Double) ?: 0.0,
            highScore    = (data["highScore"]   as? Double)  ?: 0.0,
            totalReward  = (data["totalReward"] as? Double)  ?: 0.0,
            lastAction   = data["lastAction"]   as? String  ?: "",
            isGameOver   = data["isGameOver"]   as? Boolean ?: false,
        )
    }

    /** Phase 15: skill_updated — merge changed entry into list without full reload. */
    private fun handleSkillUpdated(data: Map<String, Any>) {
        val pkg     = data["appPackage"]  as? String ?: return
        val success = (data["taskSuccess"] as? Int)  ?: 0
        val failure = (data["taskFailure"] as? Int)  ?: 0
        val rate    = (data["successRate"] as? Double)?.toFloat() ?: 0f
        _appSkills.update { prev ->
            if (prev.any { it.appPackage == pkg }) {
                prev.map { s ->
                    if (s.appPackage == pkg) s.copy(
                        taskSuccess = success,
                        taskFailure = failure,
                        successRate = rate,
                    ) else s
                }
            } else prev
        }
        refreshAppSkills()
    }

    /** Phase 15: task_chain_advanced — show banner and refresh task queue. */
    private fun handleTaskChainAdvanced(data: Map<String, Any>) {
        _chainedTask.value = ChainedTaskItem(
            taskId     = data["taskId"]     as? String ?: "",
            goal       = data["goal"]       as? String ?: "",
            appPackage = data["appPackage"] as? String ?: "",
            queueSize  = (data["queueSize"] as? Int)   ?: 0,
            timestamp  = System.currentTimeMillis(),
        )
        refreshTaskQueue()
    }

    // ─── Module state refresh ─────────────────────────────────────────────────

    private fun handleLlmDownloadProgress(data: Map<String, Any>) {
        val pct     = (data["percent"]      as? Int)    ?: 0
        val dlMb    = (data["downloadedMb"] as? Double) ?: 0.0
        val totalMb = (data["totalMb"]      as? Double) ?: 0.0
        val speed   = (data["speedMbps"]    as? Double) ?: 0.0
        _moduleState.update { it.copy(
            llmDownloadPercent   = pct,
            llmDownloadMb        = dlMb,
            llmDownloadTotalMb   = totalMb,
            llmDownloadSpeedMbps = speed,
            llmDownloadError     = null,
        )}
        _llmDownloading.value = true
    }

    fun refreshModuleState() {
        refreshModuleJob?.cancel()
        refreshModuleJob = viewModelScope.launch(Dispatchers.IO) {
            val store = ExperienceStore.getInstance(context)
            val loraVer = LoraTrainer.currentVersion(context)
            val ocrAvailable = runCatching { OcrEngine.isAvailable(context) }.getOrDefault(true)
            val embReady     = EmbeddingModelManager.isModelReady(context)
            val embVocab     = EmbeddingModelManager.isVocabReady(context)
            val embMb        = EmbeddingModelManager.downloadedBytes(context).toFloat() / 1_048_576f
            _moduleState.update { prev -> prev.copy(
                modelReady           = ModelManager.isModelReady(context),
                modelLoaded          = LlamaEngine.isLoaded(),
                tokensPerSecond      = LlamaEngine.lastToksPerSec,
                ocrReady             = ocrAvailable,
                detectorReady        = ObjectDetectorEngine.isModelReady(context),
                detectorSizeMb       = ObjectDetectorEngine.downloadedBytes(context).toFloat() / 1_048_576f,
                embeddingCount       = store.count(),
                labelCount           = ObjectLabelStore.getInstance(context).count(),
                accessibilityGranted = AgentAccessibilityService.isActive,
                screenCaptureGranted = ScreenCaptureService.isActive,
                episodesRun          = store.countByResult("success") + store.countByResult("failure"),
                adapterLoaded        = LoraTrainer.latestAdapterPath(context) != null,
                loraVersion          = loraVer,
                embeddingReady       = embReady,
                embeddingVocabReady  = embVocab,
                embeddingDownloadedMb = embMb,
                localServerRunning    = LocalDeviceServer.running,
                localServerUrl       = if (LocalDeviceServer.running) LocalDeviceServer.serverUrl() else "",
                visionReady          = com.ariaagent.mobile.core.ai.VisionEngine.isVisionModelReady(context),
                visionLoaded         = LlamaEngine.isVisionLoaded(),
                visionModelDownloadedMb = com.ariaagent.mobile.core.ai.VisionEngine.visionModelDownloadedBytes(context).toFloat() / 1_048_576f,
                mmProjDownloadedMb   = com.ariaagent.mobile.core.ai.VisionEngine.mmProjDownloadedBytes(context).toFloat() / 1_048_576f,
                sam2Ready            = com.ariaagent.mobile.core.ai.Sam2Engine.isModelReady(context),
                sam2Loaded           = com.ariaagent.mobile.core.ai.Sam2Engine.isLoaded(),
                sam2DownloadedMb     = com.ariaagent.mobile.core.ai.Sam2Engine.downloadedBytes(context).toFloat() / 1_048_576f,
            )}
            _agentState.update { it.copy(
                modelReady          = ModelManager.isModelReady(context),
                modelLoaded         = LlamaEngine.isLoaded(),
                accessibilityActive = AgentAccessibilityService.isActive,
                screenCaptureActive = ScreenCaptureService.isActive,
            )}
            _learningState.update { it.copy(
                adamStep       = PolicyNetwork.adamStepCount,
                lastPolicyLoss = PolicyNetwork.lastPolicyLoss,
                untrainedSamples = store.getUntrainedSuccesses(1000).size,
                loraVersion    = loraVer,
            )}
        }
    }

    // ─── Fix 1: Screen capture permission ────────────────────────────────────

    /** Signal the Activity-level composable to launch the MediaProjection chooser. */
    fun requestScreenCapturePermission() {
        _screenCaptureRequestFlow.tryEmit(Unit)
    }

    /**
     * Called by the Activity after the user grants the MediaProjection permission.
     * Starts ScreenCaptureService with the resultCode + projection Intent.
     */
    fun onScreenCaptureResult(resultCode: Int, projectionData: Intent) {
        val serviceIntent = Intent(context, ScreenCaptureService::class.java).apply {
            putExtra("resultCode", resultCode)
            putExtra("projectionData", projectionData)
        }
        context.startForegroundService(serviceIntent)
        viewModelScope.launch(Dispatchers.IO) {
            refreshModuleState()
        }
    }

    // ─── Fix 2: Model downloads ───────────────────────────────────────────────

    /** Start the LLM foreground download service. Safe to call when already downloaded. */
    fun downloadLlmModel() {
        if (_llmDownloading.value) return
        if (ModelManager.isModelReady(context)) {
            refreshModuleState()
            return
        }
        _llmDownloading.value = true
        _moduleState.update { it.copy(llmDownloadPercent = 0, llmDownloadError = null) }
        val intent = Intent(context, ModelDownloadService::class.java)
        context.startForegroundService(intent)
        // State is now driven entirely by model_download_progress / model_download_complete /
        // model_download_error events from ModelDownloadService via AgentEventBus.
    }

    /** Download the EfficientDet-Lite0 INT8 model (~4.4 MB) in the background. */
    fun downloadDetectorModel() {
        if (_detectorDownloading.value) return
        if (ObjectDetectorEngine.isModelReady(context)) {
            refreshModuleState()
            return
        }
        _detectorDownloading.value = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                ObjectDetectorEngine.ensureModel(context)
                refreshModuleState()
            } finally {
                _detectorDownloading.value = false
            }
        }
    }

    // ─── Gap 4: Embedding model download ─────────────────────────────────────

    /** Download the MiniLM ONNX embedding model + vocab (~23 MB total). */
    fun downloadEmbeddingModel() {
        if (_embeddingDownloading.value) return
        if (EmbeddingModelManager.isModelReady(context) && EmbeddingModelManager.isVocabReady(context)) {
            refreshModuleState()
            return
        }
        _embeddingDownloading.value = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                EmbeddingModelManager.download(context)
                refreshModuleState()
            } finally {
                _embeddingDownloading.value = false
            }
        }
    }

    // ─── Phase 17: Vision model download ────────────────────────────────────

    /**
     * Download SmolVLM-256M-Instruct-Q4_K_M.gguf (~150 MB) and its mmproj
     * (~50 MB) to internal storage. Reports progress via moduleState.visionDownloadPercent.
     * Safe to call when already downloaded — exits immediately.
     */
    fun downloadVisionModel() {
        if (_visionDownloading.value) return
        val visionEngine = com.ariaagent.mobile.core.ai.VisionEngine
        if (visionEngine.isVisionModelReady(context)) {
            refreshModuleState()
            return
        }
        _visionDownloading.value = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val totalBytes =
                    visionEngine.VISION_MODEL_MIN_BYTES + visionEngine.MMPROJ_MIN_BYTES

                // Download vision base model
                if (!visionEngine.isVisionModelFileReady(context)) {
                    var lastPercent = 0
                    visionEngine.downloadFile(
                        url     = visionEngine.VISION_MODEL_URL,
                        dest    = visionEngine.visionModelPath(context),
                        partial = visionEngine.visionModelPartial(context),
                        onProgress = { dl, _ ->
                            val pct = ((dl.toDouble() / totalBytes) * 50).toInt().coerceIn(0, 50)
                            if (pct != lastPercent) {
                                lastPercent = pct
                                _moduleState.update { it.copy(visionDownloadPercent = pct) }
                            }
                        }
                    )
                }

                // Download mmproj
                if (!visionEngine.isMmProjReady(context)) {
                    var lastPercent = 50
                    visionEngine.downloadFile(
                        url     = visionEngine.MMPROJ_URL,
                        dest    = visionEngine.mmProjPath(context),
                        partial = visionEngine.mmProjPartial(context),
                        onProgress = { dl, _ ->
                            val pct = (50 + (dl.toDouble() / totalBytes) * 50).toInt().coerceIn(50, 100)
                            if (pct != lastPercent) {
                                lastPercent = pct
                                _moduleState.update { it.copy(visionDownloadPercent = pct) }
                            }
                        }
                    )
                }

                _moduleState.update { it.copy(visionDownloadPercent = 100, visionDownloadError = null) }
                refreshModuleState()
            } catch (e: Exception) {
                _moduleState.update { it.copy(visionDownloadError = e.message ?: "Vision download failed") }
            } finally {
                _visionDownloading.value = false
            }
        }
    }

    /** Free the ~200 MB of RAM used by the loaded vision model + mmproj. */
    fun unloadVisionModel() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { com.ariaagent.mobile.core.ai.VisionEngine.unload() }
            refreshModuleState()
        }
    }

    /**
     * Explicitly load the vision model files into RAM.
     * Called when the user taps "Load Vision" on the Modules screen.
     * The model is also loaded lazily by VisionEngine.describe() during the
     * agent loop, but this lets users pre-warm it before starting a task.
     */
    fun loadVisionModel() {
        if (_visionLoading.value || LlamaEngine.isVisionLoaded()) return
        val visionEngine = com.ariaagent.mobile.core.ai.VisionEngine
        if (!visionEngine.isVisionModelReady(context)) return
        _visionLoading.value = true
        viewModelScope.launch(Dispatchers.IO) {
            val cfg = ConfigStore.getBlocking(context)
            runCatching { visionEngine.ensureLoaded(context, cfg.nGpuLayers, cfg.gpuBackend) }
            _visionLoading.value = false
            refreshModuleState()
        }
    }

    // ── Catalog model management ───────────────────────────────────────────────

    /**
     * Persist [modelId] as the active model without downloading anything.
     * The new active model will be loaded on the next ARIA start / reload.
     */
    fun selectActiveModel(modelId: String) {
        ModelManager.setActiveModelId(context, modelId)
        refreshModuleState()
    }

    /**
     * Download a specific catalog model by [modelId] via the foreground
     * ModelDownloadService. If the model is already fully downloaded this
     * returns immediately (same behaviour as downloadLlmModel).
     *
     * Progress and completion are delivered through the same AgentEventBus
     * events ("model_download_progress", "model_download_complete",
     * "model_download_error") already handled by the ViewModel's event loop.
     */
    fun downloadCatalogModel(modelId: String) {
        if (_llmDownloading.value) return
        if (ModelManager.isModelDownloaded(context, modelId)) {
            refreshModuleState()
            return
        }
        _llmDownloading.value = true
        _moduleState.update { it.copy(llmDownloadPercent = 0, llmDownloadError = null) }
        val intent = Intent(context, ModelDownloadService::class.java)
            .putExtra(ModelDownloadService.EXTRA_MODEL_ID, modelId)
        context.startForegroundService(intent)
    }

    // ── Multi-LLM slot management (Phase 20) ─────────────────────────────────

    /**
     * Set the functional [role] for [modelId].
     * Persists the change in the in-memory slot map; no RAM operation.
     */
    fun setLlmRole(modelId: String, role: LlmRole) {
        _loadedLlms.update { map ->
            val existing = map[modelId] ?: LoadedLlmEntry(modelId = modelId)
            map + (modelId to existing.copy(role = role))
        }
        persistLlmSlots()
    }

    /**
     * Set a custom [systemPrompt] for [modelId] that will be injected
     * at the beginning of every inference session for that model.
     */
    fun setLlmSystemPrompt(modelId: String, systemPrompt: String) {
        _loadedLlms.update { map ->
            val existing = map[modelId] ?: LoadedLlmEntry(modelId = modelId)
            map + (modelId to existing.copy(systemPrompt = systemPrompt))
        }
        persistLlmSlots()
    }

    /**
     * Load [modelId] into RAM.
     *
     * - If the model is already loaded this is a no-op.
     * - If [modelId] is text-only and no other loaded model has vision
     *   capability, the SmolVLM vision helper is auto-activated.
     * - LlamaEngine supports one primary model at a time on mobile.
     *   Calling this when another non-vision LLM is already loaded will
     *   first unload the previous primary before loading the new one.
     */
    fun loadCatalogLlm(modelId: String) {
        val catalog = com.ariaagent.mobile.core.ai.ModelCatalog.findById(modelId) ?: return
        if (!ModelManager.isModelDownloaded(context, modelId)) return

        // Register the slot entry if it doesn't exist yet; clear any previous load error
        _loadedLlms.update { map ->
            val existing = map[modelId] ?: LoadedLlmEntry(modelId = modelId)
            map + (modelId to existing.copy(isLoading = true, downloadError = null, loadError = null))
        }

        viewModelScope.launch(Dispatchers.IO) {
            val loadResult = runCatching {
                val cfg = ConfigStore.getBlocking(context)
                // Select this model as the active one so loadModel() picks it up
                ModelManager.setActiveModelId(context, modelId)

                val mmProjPath: String? = catalog.mmprojFilename?.let { fname ->
                    java.io.File(ModelManager.modelDir(context), fname)
                        .takeIf { it.exists() && it.length() > 0 }
                        ?.absolutePath
                }

                if (mmProjPath != null) {
                    LlamaEngine.loadUnified(
                        modelPath   = ModelManager.modelPath(context).absolutePath,
                        mmProjPath  = mmProjPath,
                        contextSize = cfg.contextWindow,
                        nGpuLayers  = cfg.nGpuLayers,
                        gpuBackend  = cfg.gpuBackend,
                    )
                } else {
                    LlamaEngine.load(
                        path        = ModelManager.modelPath(context).absolutePath,
                        contextSize = cfg.contextWindow,
                        nGpuLayers  = cfg.nGpuLayers,
                        gpuBackend  = cfg.gpuBackend,
                    )
                }
            }
            val loaded   = LlamaEngine.isLoaded()
            val loadErr  = if (!loaded) loadResult.exceptionOrNull()?.message
                               ?.takeIf { it.isNotBlank() }
                               ?: if (loadResult.isFailure) "Failed to load model — check available RAM" else null
                           else null
            android.util.Log.d("AgentViewModel", "loadCatalogLlm $modelId: loaded=$loaded err=$loadErr")
            _loadedLlms.update { map ->
                val existing = map[modelId] ?: LoadedLlmEntry(modelId = modelId)
                map + (modelId to existing.copy(isLoaded = loaded, isLoading = false, loadError = loadErr))
            }
            persistLlmSlots()
            refreshModuleState()
            // Auto-activate vision support if the loaded model is text-only
            // and no other model with vision is in the loaded set
            autoEnsureVisionSupport()
        }
    }

    /**
     * Unload [modelId] from RAM.
     * If [modelId] is the primary LLM, LlamaEngine.unload() is called.
     * The slot entry is kept so the user can reload quickly.
     */
    fun unloadCatalogLlm(modelId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            // Only unload from LlamaEngine if this is the currently active model
            val activeId = ModelManager.activeModelId(context)
            if (activeId == modelId && LlamaEngine.isLoaded()) {
                runCatching { LlamaEngine.unload() }
            }
            _loadedLlms.update { map ->
                val existing = map[modelId] ?: LoadedLlmEntry(modelId = modelId)
                map + (modelId to existing.copy(isLoaded = false, isLoading = false))
            }
            persistLlmSlots()
            refreshModuleState()
        }
    }

    /**
     * Remove [modelId] from the loaded-LLM set entirely.
     * Same as unload but also removes the slot entry from the map.
     */
    fun removeLlmSlot(modelId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val activeId = ModelManager.activeModelId(context)
            if (activeId == modelId && LlamaEngine.isLoaded()) {
                runCatching { LlamaEngine.unload() }
            }
            _loadedLlms.update { map -> map - modelId }
            persistLlmSlots()
            refreshModuleState()
        }
    }

    /**
     * Auto-activate SmolVLM vision helper if:
     *   a) There are loaded text-only LLMs, AND
     *   b) No loaded LLM has built-in vision, AND
     *   c) SmolVLM helper files are present on disk.
     */
    private suspend fun autoEnsureVisionSupport() {
        val slots = _loadedLlms.value
        // Use the native LlamaEngine.isLoaded() as ground truth.
        //
        // Problem: after process restoration the ViewModel state may have all slots
        // with isLoaded = false (stale ViewModel bool) even though LlamaEngine has
        // already reloaded weights from disk and is actively serving inference.
        // The old guard `slots.values.none { it.isLoaded }` would incorrectly skip
        // the vision-support check on every startup, leaving the agent without SmolVLM.
        //
        // Fix: check native state first. If LlamaEngine confirms it IS loaded but the
        // ViewModel slots all say false, the native state wins and we proceed.
        // Only skip if BOTH the native engine is unloaded AND no slot claims loaded.
        val nativeLoaded = LlamaEngine.isLoaded()
        if (!nativeLoaded && slots.values.none { it.isLoaded }) return
        val anyVisionModel = slots.values.any { entry ->
            val cat = com.ariaagent.mobile.core.ai.ModelCatalog.findById(entry.modelId)
            cat != null && !cat.isTextOnly
        }
        if (anyVisionModel) return  // already have a vision-capable primary model
        // Check if SmolVLM helper is ready and not yet loaded
        if (!LlamaEngine.isVisionLoaded() && !_visionLoading.value) {
            // Check that SmolVLM files exist
            val smolId = com.ariaagent.mobile.core.ai.ModelCatalog.DEFAULT_ID
            val smolEntry = com.ariaagent.mobile.core.ai.ModelCatalog.findById(smolId) ?: return
            val modelFile = java.io.File(ModelManager.modelDir(context), smolEntry.filename)
            val mmprojFile = smolEntry.mmprojFilename?.let {
                java.io.File(ModelManager.modelDir(context), it)
            }
            if (modelFile.exists() && mmprojFile?.exists() == true) {
                _visionLoading.value = true
                val cfg = ConfigStore.getBlocking(context)
                runCatching { VisionEngine.ensureLoaded(context, cfg.nGpuLayers, cfg.gpuBackend) }
                _visionLoading.value = false
                refreshModuleState()
            }
        }
    }

    /**
     * Persist the current slot configuration to SharedPreferences as a JSON string.
     *
     * Old approach: index-based keys (slot_0_id, slot_1_id, …) — broke silently when
     * a slot was removed from the middle of the list, because restoreLlmSlots read by
     * sequential index and stopped at the first missing entry.
     *
     * New approach: single "slots_json" key containing a JSON array of objects, each
     * with { id, role, prompt }. Any slot can be removed without corrupting the others.
     *
     * Format: [{"id":"<modelId>","role":"REASONING","prompt":"..."},…]
     */
    private fun persistLlmSlots() {
        val slots = _loadedLlms.value
        val arr = org.json.JSONArray()
        slots.entries.forEach { (id, entry) ->
            arr.put(org.json.JSONObject().apply {
                put("id",     id)
                put("role",   entry.role.name)
                put("prompt", entry.systemPrompt)
            })
        }
        context.getSharedPreferences("aria_llm_slots", android.content.Context.MODE_PRIVATE)
            .edit()
            .putString("slots_json", arr.toString())
            .apply()
    }

    /** Restore the slot configuration from SharedPreferences on ViewModel init. */
    private fun restoreLlmSlots() {
        val prefs = context.getSharedPreferences("aria_llm_slots", android.content.Context.MODE_PRIVATE)
        val map = mutableMapOf<String, LoadedLlmEntry>()

        // ── New path: JSON array format ───────────────────────────────────────
        val json = prefs.getString("slots_json", null)
        if (!json.isNullOrBlank()) {
            runCatching {
                val arr = org.json.JSONArray(json)
                for (i in 0 until arr.length()) {
                    val obj    = arr.getJSONObject(i)
                    val id     = obj.optString("id"); if (id.isBlank()) continue
                    val role   = runCatching {
                        LlmRole.valueOf(obj.optString("role", "REASONING"))
                    }.getOrDefault(LlmRole.REASONING)
                    val prompt = obj.optString("prompt", "")
                    map[id] = LoadedLlmEntry(modelId = id, role = role, systemPrompt = prompt)
                }
            }.onFailure { android.util.Log.w("AgentViewModel", "restoreLlmSlots JSON parse failed: ${it.message}") }
            _loadedLlms.value = map
            return
        }

        // ── Legacy path: index-based keys (migration read, never written again) ──
        val count = prefs.getInt("slot_count", 0)
        for (i in 0 until count) {
            val id     = prefs.getString("slot_${i}_id",     null) ?: continue
            val role   = runCatching {
                LlmRole.valueOf(prefs.getString("slot_${i}_role", "REASONING") ?: "REASONING")
            }.getOrDefault(LlmRole.REASONING)
            val prompt = prefs.getString("slot_${i}_prompt", "") ?: ""
            map[id] = LoadedLlmEntry(modelId = id, role = role, systemPrompt = prompt)
        }
        _loadedLlms.value = map
        // Migrate to JSON format immediately
        if (map.isNotEmpty()) persistLlmSlots()
    }

    /**
     * Save [path] as the user's custom local GGUF path and clear the active
     * catalog selection so ARIA will load this file instead.
     * Pass null or blank to remove the custom path override.
     */
    fun setLocalModelPath(path: String?) {
        ModelManager.setCustomModelPath(context, path)
        if (path == null) {
            ModelManager.setCustomModelType(context, ModelManager.CustomModelType.TEXT_LLM)
            ModelManager.setCustomMmProjPath(context, null)
        }
        refreshModuleState()
    }

    /**
     * Save all three pieces of metadata for a user-supplied custom GGUF in one call.
     * Called from SettingsScreen when the user taps "Use this file".
     */
    fun setCustomModelMeta(
        path:       String?,
        type:       ModelManager.CustomModelType,
        mmProjPath: String?
    ) {
        ModelManager.setCustomModelPath(context, path)
        ModelManager.setCustomModelType(context, type)
        ModelManager.setCustomMmProjPath(context, mmProjPath)
        refreshModuleState()
    }

    // ─── Phase 18: SAM2 / MobileSAM pixel segmentation ──────────────────────

    /**
     * Download the MobileSAM ViT-Tiny encoder ONNX (~22 MB) to internal storage.
     * Reports download progress via moduleState.sam2DownloadPercent.
     * Safe to call when already downloaded — exits immediately.
     */
    fun downloadSam2Model() {
        if (_sam2Downloading.value) return
        val sam2 = com.ariaagent.mobile.core.ai.Sam2Engine
        if (sam2.isModelReady(context)) {
            refreshModuleState()
            return
        }
        _sam2Downloading.value = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                var lastPct = 0
                sam2.downloadModel(context) { downloaded, total ->
                    if (total > 0) {
                        val pct = ((downloaded.toDouble() / total) * 100).toInt().coerceIn(0, 100)
                        if (pct != lastPct) {
                            lastPct = pct
                            _moduleState.update { it.copy(sam2DownloadPercent = pct, sam2DownloadError = null) }
                        }
                    }
                }
                _moduleState.update { it.copy(sam2DownloadPercent = 100, sam2DownloadError = null) }
                refreshModuleState()
            } catch (e: Exception) {
                _moduleState.update { it.copy(sam2DownloadError = e.message ?: "SAM2 download failed") }
            } finally {
                _sam2Downloading.value = false
            }
        }
    }

    /** Load MobileSAM encoder ONNX session into RAM (~22 MB). */
    fun loadSam2Model() {
        if (_sam2Loading.value || com.ariaagent.mobile.core.ai.Sam2Engine.isLoaded()) return
        if (!com.ariaagent.mobile.core.ai.Sam2Engine.isModelReady(context)) return
        _sam2Loading.value = true
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { com.ariaagent.mobile.core.ai.Sam2Engine.ensureLoaded(context) }
            _sam2Loading.value = false
            refreshModuleState()
        }
    }

    /** Release MobileSAM encoder ONNX session and free ~22 MB of RAM. */
    fun unloadSam2Model() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { com.ariaagent.mobile.core.ai.Sam2Engine.unload() }
            refreshModuleState()
        }
    }

    // ─── Gap 5: Unload LLM from RAM ──────────────────────────────────────────

    /** Free the ~800 MB of RAM used by the loaded LLM when ARIA is not in use. */
    fun unloadLlmModel() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { LlamaEngine.unload() }
            refreshModuleState()
        }
    }

    // ─── Gap 6: Local monitoring server ──────────────────────────────────────

    /** Start or stop the embedded HTTP monitoring server + live pusher. */
    fun toggleLocalServer() {
        viewModelScope.launch(Dispatchers.IO) {
            if (LocalDeviceServer.running) {
                MonitoringPusher.stop()
                LocalDeviceServer.stop()
            } else {
                LocalDeviceServer.start()
                MonitoringPusher.start(context)
            }
            refreshModuleState()
        }
    }

    // ─── Gap 7: Object label management ──────────────────────────────────────

    /** Load all enriched labels for the ActivityScreen labels tab. */
    fun refreshAllLabels() {
        viewModelScope.launch(Dispatchers.IO) {
            _allLabels.value = ObjectLabelStore.getInstance(context).getAllEnriched()
        }
    }

    /** Wipe every stored UI element label across all apps. */
    fun clearAllLabels() {
        viewModelScope.launch(Dispatchers.IO) {
            ObjectLabelStore.getInstance(context).clearAll()
            _allLabels.value = emptyList()
            refreshModuleState()
        }
    }

    // ─── Gap 8: ExperienceStore breakdown stats ───────────────────────────────

    /** Compute success / failure / edge-case counts for the ActivityScreen stats bar. */
    fun refreshMemoryStats() {
        viewModelScope.launch(Dispatchers.IO) {
            val store = ExperienceStore.getInstance(context)
            val labelStore = ObjectLabelStore.getInstance(context)
            _memoryStats.value = MemoryStatsUi(
                total       = store.count(),
                success     = store.countByResult("success"),
                failure     = store.countByResult("failure"),
                edgeCase    = store.edgeCaseCount(),
                untrained   = store.getUntrainedSuccesses(1000).size,
                enrichedLabels = labelStore.countEnriched(),
                totalLabels    = labelStore.count(),
            )
        }
    }

    // ─── Fix 3: Gallery fallback for Object Labeler ───────────────────────────

    /**
     * Load a gallery image URI as the labeler capture.
     * Copies the content:// URI to cache, runs OCR, and populates labelerCapture.
     * Called when screen capture is denied and user picks from gallery instead.
     */
    fun loadImageFromGallery(uri: android.net.Uri) {
        if (_labelerCapturing.value) return
        _labelerCapturing.value = true
        _labelerError.value = null
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val file = java.io.File(context.cacheDir, "labeler_gallery_${System.currentTimeMillis()}.jpg")
                context.contentResolver.openInputStream(uri)?.use { input ->
                    java.io.FileOutputStream(file).use { out -> input.copyTo(out) }
                }
                if (!file.exists() || file.length() == 0L) {
                    _labelerError.value = "Failed to read image from gallery."
                    return@launch
                }
                val bitmap = android.graphics.BitmapFactory.decodeFile(file.absolutePath)
                val ocrText = if (bitmap != null) OcrEngine.run(bitmap) else ""
                val a11yTree = AgentAccessibilityService.getSemanticTree()
                val captureUi = ScreenCaptureUi(
                    imagePath  = file.absolutePath,
                    appPackage = "gallery",
                    screenHash = file.name,
                    ocrText    = ocrText.take(2000),
                    a11yTree   = a11yTree.take(2000),
                )
                _labelerCapture.value = captureUi
                _labelerLabels.value  = emptyList()
            } catch (e: Exception) {
                _labelerError.value = e.message ?: "Gallery import failed"
            } finally {
                _labelerCapturing.value = false
            }
        }
    }

    // ─── Phase 15: task queue ─────────────────────────────────────────────────

    fun refreshTaskQueue() {
        viewModelScope.launch(Dispatchers.IO) {
            val tasks = TaskQueueManager.getAll(context)
            _taskQueue.value = tasks.map { t ->
                QueuedTaskItem(
                    id         = t.id,
                    goal       = t.goal,
                    appPackage = t.appPackage,
                    priority   = t.priority,
                    enqueuedAt = t.enqueuedAt,
                )
            }
        }
    }

    fun enqueueTask(goal: String, appPackage: String = "", priority: Int = 0) {
        viewModelScope.launch(Dispatchers.IO) {
            TaskQueueManager.enqueue(context, goal, appPackage, priority)
            refreshTaskQueue()
        }
    }

    fun removeQueuedTask(taskId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            TaskQueueManager.remove(context, taskId)
            _taskQueue.update { prev -> prev.filter { it.id != taskId } }
        }
    }

    fun clearTaskQueue() {
        viewModelScope.launch(Dispatchers.IO) {
            TaskQueueManager.clear(context)
            _taskQueue.value = emptyList()
        }
    }

    // ─── Phase 15: app skills ─────────────────────────────────────────────────

    fun refreshAppSkills() {
        viewModelScope.launch(Dispatchers.IO) {
            val skills = AppSkillRegistry.getInstance(context).getAll()
            _appSkills.value = skills.map { s ->
                AppSkillItem(
                    appPackage      = s.appPackage,
                    appName         = s.appName.ifBlank { s.appPackage.substringAfterLast('.') },
                    taskSuccess     = s.taskSuccess,
                    taskFailure     = s.taskFailure,
                    totalSteps      = s.totalSteps,
                    successRate     = s.successRate,
                    avgSteps        = s.avgStepsPerTask,
                    learnedElements = s.learnedElements.take(5),
                    promptHint      = s.promptHint,
                    lastSeen        = s.lastSeen,
                )
            }.sortedByDescending { it.lastSeen }
        }
    }

    fun clearAppSkills() {
        viewModelScope.launch(Dispatchers.IO) {
            AppSkillRegistry.getInstance(context).clear()
            _appSkills.value = emptyList()
        }
    }

    /** Delete the skill record for a single app — called from GoalsScreen Skills tab. */
    fun deleteAppSkill(appPackage: String) {
        viewModelScope.launch(Dispatchers.IO) {
            AppSkillRegistry.getInstance(context).deleteForPackage(appPackage)
            _appSkills.update { prev -> prev.filter { it.appPackage != appPackage } }
        }
    }

    /** Dismiss the chained task notification banner. */
    fun dismissChainNotification() {
        _chainedTask.value = null
    }

    // ─── T002: Session Replay ─────────────────────────────────────────────────

    /** Load recent sessions from SessionReplayStore for the ActivityScreen Replay tab. */
    fun refreshReplaySessions() {
        viewModelScope.launch(Dispatchers.IO) {
            val store = SessionReplayStore.getInstance(context)
            _replaySessions.value = store.getRecentSessions(20).map { s ->
                ReplaySessionItem(
                    sessionId = s.sessionId,
                    goal      = s.goal,
                    stepCount = s.stepCount,
                    succeeded = s.succeeded,
                    failed    = s.failed,
                    startTime = s.startTime,
                    endTime   = s.endTime,
                )
            }
        }
    }

    /** Load the step-by-step replay for a specific session (used by Replay timeline). */
    fun loadReplaySteps(sessionId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val store = SessionReplayStore.getInstance(context)
            _replaySteps.value = store.getSteps(sessionId).map { e ->
                ReplayStepItem(
                    stepIdx    = e.stepIdx,
                    screenHash = e.screenHash,
                    actionJson = e.actionJson,
                    reason     = e.reason,
                    result     = e.result,
                    appPackage = e.appPackage,
                    timestamp  = e.timestamp,
                )
            }
        }
    }

    // ─── T005: Proactive Goal Surfacing ───────────────────────────────────────

    /** Load pending automation suggestions from SuggestionStore for DashboardScreen banner. */
    fun refreshPendingSuggestions() {
        viewModelScope.launch(Dispatchers.IO) {
            val store = SuggestionStore.getInstance(context)
            _pendingSuggestions.value = store.getPending().map { s ->
                SuggestionBannerItem(
                    id             = s.id,
                    appPackage     = s.appPackage,
                    goalText       = s.goalText,
                    suggestionText = s.suggestionText,
                    repeatCount    = s.repeatCount,
                )
            }
        }
    }

    /** Accept a suggestion — mark accepted in DB, remove from UI, and launch the agent task. */
    fun acceptSuggestion(item: SuggestionBannerItem) {
        viewModelScope.launch(Dispatchers.IO) {
            SuggestionStore.getInstance(context).accept(item.id)
            _pendingSuggestions.update { list -> list.filter { it.id != item.id } }
            kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Main) {
                startAgent(item.goalText, item.appPackage)
            }
        }
    }

    /** Dismiss (snooze) a suggestion — mark dismissed in DB, remove from banner UI. */
    fun dismissSuggestion(id: Long) {
        viewModelScope.launch(Dispatchers.IO) {
            SuggestionStore.getInstance(context).dismiss(id)
            _pendingSuggestions.update { list -> list.filter { it.id != id } }
        }
    }

    // ─── Migration Phase 3: memory entries ───────────────────────────────────

    /** Load recent ExperienceStore tuples for the ActivityScreen Memory tab. */
    fun refreshMemoryEntries() {
        viewModelScope.launch(Dispatchers.IO) {
            val store = ExperienceStore.getInstance(context)
            _memoryEntries.value = store.getRecent(200).map { t ->
                MemoryEntry(
                    id         = t.id,
                    summary    = t.screenSummary.ifBlank { t.taskType },
                    app        = t.appPackage.substringAfterLast('.'),
                    taskType   = t.taskType,
                    result     = t.result,
                    reward     = t.reward,
                    isEdgeCase = t.isEdgeCase,
                    timestamp  = t.timestamp,
                )
            }
        }
    }

    // ─── Migration Phase 2/3: danger zone actions (added beyond RN) ──────────

    /**
     * Clear all experience store entries and embeddings.
     * Triggered by "Clear Memory" button in SettingsScreen and ActivityScreen.
     */
    fun clearMemory() {
        viewModelScope.launch(Dispatchers.IO) {
            ExperienceStore.getInstance(context).clearAll()
            _memoryEntries.value = emptyList()
            refreshModuleState()
        }
    }

    /**
     * Full agent reset — clears experience store, progress persistence, task queue,
     * and app skill registry. LoRA adapter files on disk are preserved.
     * Triggered by "Reset Agent" button in SettingsScreen.
     */
    fun resetAgent() {
        viewModelScope.launch(Dispatchers.IO) {
            ExperienceStore.getInstance(context).clearAll()
            ProgressPersistence.clear(context)
            AppSkillRegistry.getInstance(context).clear()
            TaskQueueManager.clear(context)
            _memoryEntries.value = emptyList()
            _appSkills.value     = emptyList()
            _taskQueue.value     = emptyList()
            refreshModuleState()
        }
    }

    // ─── Agent control ────────────────────────────────────────────────────────

    /**
     * Load the LLM engine into memory using the current stored config.
     * Matches `loadModel()` in control.tsx.  Safe to call when already loaded.
     */
    fun loadModel() {
        viewModelScope.launch(Dispatchers.IO) {
            val cfg = ConfigStore.getBlocking(context)
            runCatching {
                // Determine mmproj path and whether unified mode is appropriate.
                //
                // Custom GGUF (user-supplied):
                //   The user explicitly classified the model when they added it.
                //   If they said "Multimodal VLM" and also provided an mmproj path,
                //   load in unified mode.  Otherwise text-only load().
                //
                // Catalog model (downloaded via ARIA):
                //   Multimodal entries (mmprojFilename != null) load in unified mode when
                //   the mmproj file is fully on disk. Text-only entries (mmprojFilename == null)
                //   use LlamaEngine.load() — screen reading is handled by SmolVLM helper or
                //   falls back to accessibility tree + OCR.

                val isCustomPath = ModelManager.customModelPath(context) != null
                val mmProjPath: String? = if (isCustomPath) {
                    val customType = ModelManager.customModelType(context)
                    if (customType == ModelManager.CustomModelType.MULTIMODAL_VLM)
                        ModelManager.customMmProjPath(context)
                    else null
                } else {
                    val activeEntry = ModelManager.activeEntry(context)
                    activeEntry?.mmprojFilename?.let { fname ->
                        java.io.File(ModelManager.modelDir(context), fname)
                            .takeIf { it.exists() && it.length() > 0 }
                            ?.absolutePath
                    }
                }

                if (mmProjPath != null &&
                    java.io.File(mmProjPath).exists() &&
                    java.io.File(mmProjPath).length() > 0
                ) {
                    LlamaEngine.loadUnified(
                        modelPath   = cfg.modelPath,
                        mmProjPath  = mmProjPath,
                        contextSize = cfg.contextWindow,
                        nGpuLayers  = cfg.nGpuLayers,
                        gpuBackend  = cfg.gpuBackend,
                    )
                } else {
                    LlamaEngine.load(
                        path        = cfg.modelPath,
                        contextSize = cfg.contextWindow,
                        nGpuLayers  = cfg.nGpuLayers,
                        gpuBackend  = cfg.gpuBackend,
                    )
                }
            }
            refreshModuleState()
        }
    }

    fun startAgent(goal: String, appPackage: String = "") {
        startFloatingChat()
        viewModelScope.launch(Dispatchers.IO) {
            AgentLoop.start(context, goal, appPackage)
        }
    }

    /**
     * Start the agent in learn-only mode: observe + reason, but dispatch NO gestures.
     * Matches `startLearnOnly()` in control.tsx / AgentForegroundService.
     */
    fun startLearnOnly(goal: String, appPackage: String = "") {
        startFloatingChat()
        viewModelScope.launch(Dispatchers.IO) {
            AgentForegroundService.startLearnOnly(context, goal, appPackage)
        }
    }

    fun stopAgent() {
        AgentLoop.stop()
        _streamBuffer.value = ""
        _stepState.value = StepUiState()
        stopFloatingChat()
    }

    /** Start the floating chat overlay if SYSTEM_ALERT_WINDOW permission is granted. */
    private fun startFloatingChat() {
        if (android.provider.Settings.canDrawOverlays(context)) {
            runCatching {
                context.startService(Intent(context, FloatingChatService::class.java))
            }
        }
    }

    /** Stop the floating chat overlay service. */
    private fun stopFloatingChat() {
        runCatching {
            context.startService(
                Intent(context, FloatingChatService::class.java).apply {
                    action = FloatingChatService.ACTION_STOP
                }
            )
        }
    }

    fun pauseAgent() {
        AgentLoop.pause()
    }

    fun saveConfig(newConfig: AriaConfig) {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { ConfigStore.save(context, newConfig) }
            AgentEventBus.emit("config_updated", mapOf(
                "modelPath"        to newConfig.modelPath,
                "quantization"     to newConfig.quantization,
                "contextWindow"    to newConfig.contextWindow,
                "maxTokensPerTurn" to newConfig.maxTokensPerTurn,
                "temperatureX100"  to newConfig.temperatureX100,
                "nGpuLayers"       to newConfig.nGpuLayers,
                "rlEnabled"        to newConfig.rlEnabled,
                "loraAdapterPath"  to (newConfig.loraAdapterPath ?: ""),
                "agentCombo"       to newConfig.agentCombo.name,
                "irlEnabled"       to newConfig.irlEnabled,
                "rlVisionMode"     to newConfig.rlVisionMode.name,
                "irlVisionMode"    to newConfig.irlVisionMode.name,
            ))
        }
    }

    // ── Module combo & vision mode setters ────────────────────────────────────
    // Each setter reads the current config and writes only the changed field,
    // preserving all other settings. Uses saveConfig() so the event bus and
    // DataStore are updated atomically.

    /** Select which modules participate in agent sessions. Applies to the next started task. */
    fun setAgentCombo(combo: AgentCombo) {
        saveConfig(config.value.copy(agentCombo = combo))
    }

    /** Enable or disable IRL (processing of screen recordings). Does not affect live RL. */
    fun setIrlEnabled(enabled: Boolean) {
        saveConfig(config.value.copy(irlEnabled = enabled))
    }

    /**
     * Select which model(s) LlmRewardEnricher uses to score RL experience tuples.
     * Takes effect on the next LearningScheduler training cycle.
     */
    fun setRlVisionMode(mode: ModuleVisionMode) {
        saveConfig(config.value.copy(rlVisionMode = mode))
    }

    /**
     * Select which model(s) IrlModule uses for per-keyframe perception & action inference.
     * Takes effect on the next IRL video processing call.
     */
    fun setIrlVisionMode(mode: ModuleVisionMode) {
        saveConfig(config.value.copy(irlVisionMode = mode))
    }

    /**
     * Set the same vision mode for BOTH RL reward scoring and IRL video processing.
     * Used by the unified Vision Mode control in the UI.
     */
    fun setBothVisionModes(mode: ModuleVisionMode) {
        saveConfig(config.value.copy(rlVisionMode = mode, irlVisionMode = mode))
    }

    /**
     * Unload the current LlamaEngine instance and reload it using the current saved config.
     * Called after the user confirms the engine-reload dialog in SettingsScreen when
     * engine-affecting params (nGpuLayers, contextWindow, quantization) were changed.
     * Engine is always stopped before reload so the new params take effect immediately.
     */
    fun reloadLlamaEngine() {
        viewModelScope.launch(Dispatchers.IO) {
            if (LlamaEngine.isLoaded()) {
                runCatching { LlamaEngine.unload() }
            }
            loadModel()
        }
    }

    // ─── Migration Phase 5: Chat ──────────────────────────────────────────────

    /**
     * Send a user message through the on-device LLM.
     * Builds full context via ChatContextBuilder, then calls LlamaEngine.infer().
     * No bridge. No JS. Pure Kotlin.
     */
    fun sendChatMessage(text: String) {
        if (_chatThinking.value) return
        if (!LlamaEngine.isLoaded()) {
            _chatMessages.update { prev ->
                prev + ChatMessageItem(
                    id   = "sys-${System.currentTimeMillis()}",
                    role = "system",
                    text = "No model loaded. Go to Modules, select a model, download it, then tap Load.",
                )
            }
            return
        }

        val userMsg = ChatMessageItem(
            id   = "u-${System.currentTimeMillis()}",
            role = "user",
            text = text,
        )
        _chatMessages.update { prev -> prev + userMsg }
        _chatThinking.value = true

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val history = _chatMessages.value
                    .filter { it.role != "system" }
                    .takeLast(6)
                val historyJson = history.joinToString(",", "[", "]") {
                    // Proper JSON escaping: backslash first (prevents double-escaping),
                    // then control chars, then double-quote.
                    val escaped = it.text
                        .replace("\\", "\\\\")
                        .replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                        .replace("\"", "\\\"")
                    """{"role":"${it.role}","text":"$escaped"}"""
                }
                val systemCtx = ChatContextBuilder.build(context, text, historyJson)
                val historyBlock = if (history.isNotEmpty()) {
                    "\n[CONVERSATION]\n" + history.joinToString("\n") {
                        "${if (it.role == "user") "User" else "ARIA"}: ${it.text}"
                    } + "\n"
                } else ""
                val prompt = "$systemCtx${historyBlock}\nUser: $text\nARIA:"

                val t0  = System.currentTimeMillis()
                val raw = LlamaEngine.infer(prompt, maxTokens = 512)
                val elapsed = (System.currentTimeMillis() - t0) / 1000.0
                val wordCount = raw.trim().split(Regex("\\s+")).size
                val tps = if (elapsed > 0) wordCount / elapsed else 0.0

                val response = raw
                    .replace(Regex("^ARIA:\\s*", RegexOption.IGNORE_CASE), "")
                    .replace(Regex("^Assistant:\\s*", RegexOption.IGNORE_CASE), "")
                    .trim()
                    .ifBlank { "(no response — try rephrasing)" }

                _chatMessages.update { prev ->
                    prev + ChatMessageItem(
                        id   = "a-${System.currentTimeMillis()}",
                        role = "aria",
                        text = response,
                        tps  = tps,
                    )
                }
            } catch (e: Exception) {
                _chatMessages.update { prev ->
                    prev + ChatMessageItem(
                        id   = "err-${System.currentTimeMillis()}",
                        role = "system",
                        text = "Inference error: ${e.message ?: "unknown"}",
                    )
                }
            } finally {
                _chatThinking.value = false
            }
        }
    }

    fun clearChat() {
        _chatMessages.value = listOf(welcomeChatMsg)
        _chatThinking.value = false
    }

    // ─── Migration Phase 6: Train ─────────────────────────────────────────────

    fun refreshLearningStatus() {
        viewModelScope.launch(Dispatchers.IO) {
            val ver       = LoraTrainer.currentVersion(context)
            val path      = LoraTrainer.latestAdapterPath(context)
            val store     = ExperienceStore.getInstance(context)
            _learningStatusUi.value = LearningStatusUi(
                loraVersion        = ver,
                latestAdapterPath  = path ?: "",
                adapterExists      = path != null,
                untrainedSamples   = store.getUntrainedSuccesses(1000).size,
                policyReady        = PolicyNetwork.adamStepCount > 0,
                adamStep           = PolicyNetwork.adamStepCount,
                lastPolicyLoss     = PolicyNetwork.lastPolicyLoss,
                lastTrainedAt      = if (path != null) java.io.File(path).lastModified() else 0L,
            )
        }
    }

    fun runRlCycle() {
        if (_rlRunning.value) return
        _rlRunning.value = true
        _rlResult.value  = null
        viewModelScope.launch(Dispatchers.Default) {
            try {
                reportLoraTrainingProgress(10, 0.0, 0, "loading_data")
                val store  = ExperienceStore.getInstance(context)
                val cfg    = ConfigStore.getBlocking(context)
                reportLoraTrainingProgress(25, 0.0, 0, "training")
                val labelStore = ObjectLabelStore.getInstance(context)
                val result = LoraTrainer.train(context, store, cfg.modelPath, labelStore = labelStore)
                reportLoraTrainingProgress(
                    if (result.success) 100 else 0,
                    0.0,
                    result.samplesUsed,
                    if (result.success) "complete" else "failed"
                )
                _rlResult.value = RlResultUi(
                    success      = result.success,
                    samplesUsed  = result.samplesUsed,
                    adapterPath  = result.adapterPath,
                    loraVersion  = result.loraVersion,
                    errorMessage = result.errorMessage,
                )
                refreshLearningStatus()
                refreshModuleState()
            } catch (e: Exception) {
                reportLoraTrainingProgress(0, 0.0, 0, "failed")
                _rlResult.value = RlResultUi(false, 0, "", 0, e.message ?: "unknown")
            } finally {
                _rlRunning.value = false
            }
        }
    }

    fun processIrlVideo(
        videoUri: String,
        goal: String,
        appPackage: String,
        frameAnnotations: Map<Int, String> = emptyMap()
    ) {
        if (_irlRunning.value) return
        _irlRunning.value = true
        _irlResult.value  = null
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val resolvedPath = resolveContentUri(videoUri)
                val store  = ExperienceStore.getInstance(context)
                val irlVisionMode = config.value.irlVisionMode
                val result = IrlModule.processVideo(
                    context          = context,
                    videoPath        = resolvedPath,
                    taskGoal         = goal,
                    appPackage       = appPackage,
                    store            = store,
                    frameAnnotations = frameAnnotations,
                    visionMode       = irlVisionMode,
                )
                _irlResult.value = IrlResultUi(
                    framesProcessed  = result.framesProcessed,
                    tuplesExtracted  = result.tuplesExtracted,
                    llmAssistedCount = result.llmAssistedCount,
                    errorMessage     = result.errorMessage,
                )
                refreshLearningStatus()
            } catch (e: Exception) {
                _irlResult.value = IrlResultUi(0, 0, 0, e.message ?: "unknown")
            } finally {
                _irlRunning.value = false
            }
        }
    }

    fun extractIrlFrames(videoUri: String) {
        if (_irlExtracting.value || _irlRunning.value) return
        _irlExtracting.value = true
        _irlFramePaths.value = emptyList()
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val resolvedPath = resolveContentUri(videoUri)
                _irlFramePaths.value = IrlModule.extractKeyFramePaths(context, resolvedPath)
            } catch (e: Exception) {
                _irlFramePaths.value = emptyList()
            } finally {
                _irlExtracting.value = false
            }
        }
    }

    fun clearIrlFrames() {
        _irlFramePaths.value.forEach { path -> runCatching { java.io.File(path).delete() } }
        _irlFramePaths.value = emptyList()
    }

    fun setAutoScheduleRl(enabled: Boolean) {
        _autoScheduleRl.value = enabled
        if (enabled) {
            viewModelScope.launch(Dispatchers.IO) {
                val store = ExperienceStore.getInstance(context)
                val untrained = store.getUntrainedSuccesses(1000).size
                if (untrained > 50 && !_rlRunning.value) {
                    runRlCycle()
                }
            }
        }
    }

    // ─── Migration Phase 7: Labeler ───────────────────────────────────────────

    fun captureScreenForLabeling() {
        if (_labelerCapturing.value) return
        _labelerCapturing.value = true
        _labelerError.value     = null
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val screenshot = ScreenCaptureService.captureLatest()
                    ?: run {
                        _labelerError.value = "Screen capture service not active — grant Media Projection permission first."
                        return@launch
                    }
                val snapshot = ScreenObserver.capture()
                val ocrText  = OcrEngine.run(screenshot)
                val a11yTree = AgentAccessibilityService.getSemanticTree()

                val file = java.io.File(context.cacheDir, "labeler_capture_${System.currentTimeMillis()}.jpg")
                java.io.FileOutputStream(file).use { out ->
                    screenshot.compress(android.graphics.Bitmap.CompressFormat.JPEG, 88, out)
                }

                val captureUi = ScreenCaptureUi(
                    imagePath  = file.absolutePath,
                    appPackage = snapshot.appPackage,
                    screenHash = snapshot.screenHash(),
                    ocrText    = ocrText.take(2000),
                    a11yTree   = a11yTree.take(2000),
                )
                _labelerCapture.value = captureUi

                val store  = ObjectLabelStore.getInstance(context)
                val existing = store.getByScreen(captureUi.appPackage, captureUi.screenHash)
                _labelerLabels.value = existing
            } catch (e: Exception) {
                _labelerError.value = e.message ?: "Capture failed"
            } finally {
                _labelerCapturing.value = false
            }
        }
    }

    fun addLabelerPin(x: Float, y: Float) {
        val cap = _labelerCapture.value ?: return
        val newLabel = ObjectLabelStore.ObjectLabel(
            id          = "${System.currentTimeMillis()}_${(Math.random() * 1000).toInt()}",
            appPackage  = cap.appPackage,
            screenHash  = cap.screenHash,
            x           = x,
            y           = y,
            name        = "",
            context     = "",
            elementType = ObjectLabelStore.ElementType.BUTTON,
        )
        _labelerLabels.update { prev -> prev + newLabel }
    }

    fun updateLabelerLabel(updated: ObjectLabelStore.ObjectLabel) {
        _labelerLabels.update { prev ->
            prev.map { if (it.id == updated.id) updated else it }
        }
    }

    fun deleteLabelerLabel(id: String) {
        _labelerLabels.update { prev -> prev.filter { it.id != id } }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { ObjectLabelStore.getInstance(context).delete(id) }
        }
    }

    fun autoDetectLabelerPins() {
        val cap = _labelerCapture.value ?: return
        if (_labelerDetecting.value) return
        _labelerDetecting.value = true
        _labelerError.value     = null
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val newLabels = mutableListOf<ObjectLabelStore.ObjectLabel>()

                val detections = runCatching {
                    ObjectDetectorEngine.detectFromPath(context, cap.imagePath)
                }.getOrDefault(emptyList())
                detections.forEach { det ->
                    newLabels.add(ObjectLabelStore.ObjectLabel(
                        id             = "${System.currentTimeMillis()}_${(Math.random() * 1000).toInt()}",
                        appPackage     = cap.appPackage,
                        screenHash     = cap.screenHash,
                        x              = det.normX,
                        y              = det.normY,
                        name           = det.label.replace("_", " ")
                            .split(" ").joinToString(" ") { w -> w.replaceFirstChar { it.uppercaseChar() } },
                        context        = "EfficientDet detection",
                        elementType    = inferElementType(det.label),
                        importanceScore = maxOf(1, minOf(10, (det.confidence * 10).toInt())),
                    ))
                }

                val bitmap = runCatching {
                    android.graphics.BitmapFactory.decodeFile(cap.imagePath)
                }.getOrNull()
                if (bitmap != null) {
                    if (Sam2Engine.isLoaded()) {
                        val samRegions = runCatching {
                            Sam2Engine.segment(context, bitmap, topK = 8)
                        }.getOrDefault(emptyList())
                        val existingCoords = newLabels.map { it.x to it.y }
                        samRegions.forEach { region ->
                            val isDuplicate = existingCoords.any { (ex, ey) ->
                                Math.abs(ex - region.normX) < 0.05f && Math.abs(ey - region.normY) < 0.05f
                            }
                            if (!isDuplicate) {
                                val importance = maxOf(1, minOf(10, (region.score / 100f * 10f).toInt()))
                                newLabels.add(ObjectLabelStore.ObjectLabel(
                                    id             = "${System.currentTimeMillis()}_sam_${(Math.random() * 1000).toInt()}",
                                    appPackage     = cap.appPackage,
                                    screenHash     = cap.screenHash,
                                    x              = region.normX,
                                    y              = region.normY,
                                    name           = "",
                                    context        = "MobileSAM salient region",
                                    elementType    = ObjectLabelStore.ElementType.UNKNOWN,
                                    importanceScore = importance,
                                ))
                            }
                        }
                    }
                    bitmap.recycle()
                }

                if (newLabels.isEmpty()) {
                    _labelerError.value = "No objects detected. Try adding pins manually, or load SAM2/detector models."
                    return@launch
                }
                _labelerLabels.update { prev -> prev + newLabels }
            } catch (e: Exception) {
                _labelerError.value = e.message ?: "Auto-detect failed"
            } finally {
                _labelerDetecting.value = false
            }
        }
    }

    fun enrichAllLabelerPins() {
        val cap = _labelerCapture.value ?: return
        if (!LlamaEngine.isLoaded()) {
            _labelerError.value = "Load the LLM engine first (Control → Load LLM Engine)."
            return
        }
        if (_labelerEnriching.value) return
        _labelerEnriching.value = true
        _labelerError.value     = null
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val visionDesc = runCatching {
                    val bmp = android.graphics.BitmapFactory.decodeFile(cap.imagePath)
                    if (bmp != null && VisionEngine.isVisionModelReady(context)) {
                        VisionEngine.describe(
                            context    = context,
                            bitmap     = bmp,
                            goal       = "Describe this Android UI for element annotation",
                            screenHash = cap.screenHash,
                            maxTokens  = 128
                        ).also { bmp.recycle() }
                    } else ""
                }.getOrDefault("")

                val screenContext = buildString {
                    append(cap.ocrText.take(400))
                    if (cap.a11yTree.isNotBlank()) append("\n${cap.a11yTree.take(300)}")
                    if (visionDesc.isNotBlank()) append("\n[VISION] $visionDesc")
                }

                val enriched = _labelerLabels.value.map { label ->
                    val prompt = buildLabelEnrichPrompt(label, screenContext)
                    val raw    = LlamaEngine.infer(prompt, maxTokens = 150)
                    parseLabelEnrichOutput(raw, label)
                }
                _labelerLabels.value = enriched
            } catch (e: Exception) {
                _labelerError.value = e.message ?: "Enrichment failed"
            } finally {
                _labelerEnriching.value = false
            }
        }
    }

    fun saveLabelerLabels(onSuccess: () -> Unit, onError: (String) -> Unit) {
        val cap = _labelerCapture.value ?: return
        if (_labelerSaving.value) return
        _labelerSaving.value = true
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val store = ObjectLabelStore.getInstance(context)
                store.saveAll(_labelerLabels.value)
                _labelerSaveSuccess.value = true
                refreshModuleState()
                kotlinx.coroutines.withContext(Dispatchers.Main) { onSuccess() }
            } catch (e: Exception) {
                val msg = e.message ?: "Save failed"
                kotlinx.coroutines.withContext(Dispatchers.Main) { onError(msg) }
            } finally {
                _labelerSaving.value = false
            }
        }
    }

    fun clearLabelerCapture() {
        _labelerCapture.value    = null
        _labelerLabels.value     = emptyList()
        _labelerError.value      = null
        _labelerSaveSuccess.value = false
    }

    fun dismissLabelerError() {
        _labelerError.value = null
    }

    private fun inferElementType(cocoLabel: String): ObjectLabelStore.ElementType {
        val l = cocoLabel.lowercase()
        return when {
            l.contains("cell phone") || l.contains("remote") ||
            l.contains("keyboard")   || l.contains("mouse")  -> ObjectLabelStore.ElementType.BUTTON
            l.contains("book") || l.contains("laptop") ||
            l.contains("tv")   || l.contains("monitor") -> ObjectLabelStore.ElementType.IMAGE
            l.contains("person") || l.contains("face")   -> ObjectLabelStore.ElementType.ICON
            else -> ObjectLabelStore.ElementType.UNKNOWN
        }
    }

    private fun buildLabelEnrichPrompt(
        label: ObjectLabelStore.ObjectLabel,
        screenContext: String,
    ): String = buildString {
        append("<|begin_of_text|>")
        append("<|start_header_id|>system<|end_header_id|>\n")
        append("You are an Android UI analyst. Analyze the given UI element and output JSON metadata for an autonomous agent.\n")
        append("Respond ONLY in JSON:\n")
        append("""{"meaning":"...","interactionHint":"...","reasoningContext":"...","importanceScore":7}""")
        append("\nRules: meaning≤20 words, interactionHint≤20 words, reasoningContext≤25 words, importanceScore 0-10.\n")
        append("<|eot_id|>\n")
        append("<|start_header_id|>user<|end_header_id|>\n")
        append("Element: \"${label.name}\" (${label.elementType.name.lowercase()})\n")
        append("User description: ${label.context}\n")
        if (label.ocrText.isNotBlank()) append("OCR text: ${label.ocrText.take(100)}\n")
        append("Screen context: ${screenContext.take(400)}\n")
        append("<|eot_id|>\n")
        append("<|start_header_id|>assistant<|end_header_id|>\n")
    }

    /** Copies a content:// URI to a temp file and returns its absolute path. */
    private fun resolveContentUri(uriString: String): String {
        if (!uriString.startsWith("content://")) return uriString
        val uri     = android.net.Uri.parse(uriString)
        val tmp     = java.io.File(context.cacheDir, "irl_video_${System.currentTimeMillis()}.mp4")
        context.contentResolver.openInputStream(uri)?.use { input ->
            java.io.FileOutputStream(tmp).use { out -> input.copyTo(out) }
        }
        return tmp.absolutePath
    }

    private fun parseLabelEnrichOutput(
        rawOutput: String,
        original: ObjectLabelStore.ObjectLabel,
    ): ObjectLabelStore.ObjectLabel {
        return try {
            val start = rawOutput.indexOfFirst { it == '{' }
            val end   = rawOutput.lastIndexOf('}')
            if (start == -1 || end <= start) return original
            val json = JSONObject(rawOutput.substring(start, end + 1))
            original.copy(
                meaning         = json.optString("meaning",         original.meaning).take(200),
                interactionHint = json.optString("interactionHint", original.interactionHint).take(200),
                reasoningContext = json.optString("reasoningContext", original.reasoningContext).take(250),
                importanceScore = json.optInt("importanceScore",    original.importanceScore).coerceIn(0, 10),
                isEnriched      = true,
                updatedAt       = System.currentTimeMillis(),
            )
        } catch (_: Exception) {
            original
        }
    }

    // ─── Onboarding ───────────────────────────────────────────────────────────

    fun checkOnboardingComplete() {
        val prefs = context.getSharedPreferences("aria_prefs", android.content.Context.MODE_PRIVATE)
        _onboardingComplete.value = prefs.getBoolean("onboarding_complete", false)
    }

    fun setOnboardingComplete() {
        context.getSharedPreferences("aria_prefs", android.content.Context.MODE_PRIVATE)
            .edit().putBoolean("onboarding_complete", true).apply()
        _onboardingComplete.value = true
    }

    // ─── Safety Config ────────────────────────────────────────────────────────

    fun loadSafetyConfig() {
        viewModelScope.launch(Dispatchers.IO) {
            _safetyConfig.value = SafetyConfigStore.load(context)
        }
    }

    private fun persistSafety(config: SafetyConfig) {
        _safetyConfig.value = config
        viewModelScope.launch(Dispatchers.IO) { SafetyConfigStore.save(context, config) }
    }

    fun toggleGlobalKill() =
        persistSafety(_safetyConfig.value.copy(globalKillActive = !_safetyConfig.value.globalKillActive))

    fun setConfirmMode(enabled: Boolean) =
        persistSafety(_safetyConfig.value.copy(confirmMode = enabled))

    fun setAllowlistMode(enabled: Boolean) =
        persistSafety(_safetyConfig.value.copy(allowlistMode = enabled))

    fun addBlockedPackage(pkg: String) {
        if (pkg.isBlank()) return
        persistSafety(_safetyConfig.value.copy(blockedPackages = _safetyConfig.value.blockedPackages + pkg.trim()))
    }

    fun removeBlockedPackage(pkg: String) =
        persistSafety(_safetyConfig.value.copy(blockedPackages = _safetyConfig.value.blockedPackages - pkg))

    fun addAllowedPackage(pkg: String) {
        if (pkg.isBlank()) return
        persistSafety(_safetyConfig.value.copy(allowedPackages = _safetyConfig.value.allowedPackages + pkg.trim()))
    }

    fun removeAllowedPackage(pkg: String) =
        persistSafety(_safetyConfig.value.copy(allowedPackages = _safetyConfig.value.allowedPackages - pkg))

    // ─── Triggers ─────────────────────────────────────────────────────────────

    fun loadTriggers() {
        viewModelScope.launch(Dispatchers.IO) {
            _triggers.value = TriggerStore.load(context)
        }
    }

    fun addTrigger(trigger: TriggerItem) {
        viewModelScope.launch(Dispatchers.IO) {
            TriggerStore.add(context, trigger)
            _triggers.value = TriggerStore.load(context)
        }
    }

    fun deleteTrigger(id: String) {
        viewModelScope.launch(Dispatchers.IO) {
            TriggerStore.delete(context, id)
            _triggers.value = TriggerStore.load(context)
        }
    }

    fun toggleTrigger(id: String) {
        viewModelScope.launch(Dispatchers.IO) {
            TriggerStore.toggle(context, id)
            _triggers.value = TriggerStore.load(context)
        }
    }

    // ─── LoRA Training History ────────────────────────────────────────────────

    fun refreshLoraHistory() {
        viewModelScope.launch(Dispatchers.IO) {
            // Resolve the base lora directory the same way LoraTrainer does.
            // Adapters now live in model subdirectories: lora/<modelId>/adapter_vN.gguf
            // We must recurse into subdirs — listFiles() on the base dir only returns
            // directory entries, not the adapter files inside them.
            val loraBase = java.io.File(context.filesDir, "lora")
                .let { i -> if (i.canWrite()) i else (context.getExternalFilesDir("lora") ?: i) }
            if (!loraBase.exists()) { _loraHistory.value = emptyList(); return@launch }

            val globalLatest = LoraTrainer.latestAdapterPath(context)
            val adapterFiles = mutableListOf<java.io.File>()

            // ── Collect from per-model subdirectories (current structure) ─────
            loraBase.listFiles()
                ?.filter { it.isDirectory }
                ?.forEach { modelDir ->
                    modelDir.listFiles()
                        ?.filter { it.isFile && (it.extension == "gguf" || it.extension == "bin") }
                        ?.let { adapterFiles.addAll(it) }
                }

            // ── Collect legacy flat adapters (pre-model-aware builds) ─────────
            loraBase.listFiles()
                ?.filter { it.isFile && (it.extension == "gguf" || it.extension == "bin") }
                ?.let { adapterFiles.addAll(it) }

            val checkpoints = adapterFiles
                .mapNotNull { f ->
                    val ver = Regex("adapter_v(\\d+)").find(f.name)
                        ?.groupValues?.get(1)?.toIntOrNull() ?: return@mapNotNull null
                    LoraCheckpointItem(
                        version     = ver,
                        adapterPath = f.absolutePath,
                        sizeKb      = f.length() / 1024L,
                        createdAt   = f.lastModified(),
                        // isLatest = this file IS the globally newest adapter
                        isLatest    = f.absolutePath == globalLatest,
                    )
                }
                .sortedByDescending { it.version }
            _loraHistory.value = checkpoints
        }
    }

    /** Publish live LoRA training progress. Called by LoRA training wrapper in TrainScreen. */
    fun reportLoraTrainingProgress(pct: Int, loss: Double, samples: Int, phase: String) {
        _loraTrainingProgress.value = LoraTrainingProgress(
            percentComplete = pct,
            currentLoss     = loss,
            samplesUsed     = samples,
            phase           = phase,
        )
    }

    fun clearLoraTrainingProgress() {
        _loraTrainingProgress.value = null
        refreshLoraHistory()
    }
}
