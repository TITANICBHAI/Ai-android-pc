package com.ariaagent.mobile.core.events

import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

/**
 * AgentEventBus — the internal event backbone for ARIA's Kotlin brain.
 *
 * AgentViewModel (via StateFlow → Compose UI) subscribes to this bus.
 * The React Native bridge has been removed — this is the sole subscriber path.
 *
 * Pattern: AgentLoop/GameLoop/LearningScheduler/ThermalGuard emit here.
 *          Subscribers react independently — no coupling between them.
 *
 * SharedFlow config:
 *   - replay = 0:       No stale events replayed to late subscribers
 *   - extraBuffer = 128: Room for 128 events before any subscriber falls behind
 *   - DROP_OLDEST:      Under backpressure, newest events win (UI freshness matters)
 *
 * Thread safety: SharedFlow is coroutine-safe. tryEmit() is non-suspending and
 *   can be called from any thread (Dispatchers.Default, Main, or IO).
 *
 * Event catalogue (name → payload keys):
 *   agent_status_changed    status, currentTask, currentApp, stepCount, lastAction, lastError, gameMode
 *   token_generated         token, tokensPerSecond
 *   action_performed        tool, nodeId, success, reward, stepCount
 *   step_started            stepNumber, activity ("observe"|"reason"|"act"|"store")
 *   learning_cycle_complete loraVersion, policyVersion
 *   thermal_status_changed  level, inferenceSafe, trainingSafe, emergency
 *   game_loop_status        isActive, gameType, episodeCount, stepCount, currentScore, highScore, totalReward, lastAction, isGameOver
 *   model_download_progress percent, downloadedMb, totalMb, speedMbps
 *   model_download_complete path
 *   model_download_error    error
 *   config_updated          (same keys as getConfig return)
 *
 * Phase: 10+ (Migration complete) — sole event channel for Compose ViewModel.
 */
object AgentEventBus {

    private val _flow = MutableSharedFlow<Pair<String, Map<String, Any>>>(
        replay = 0,
        extraBufferCapacity = 128,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    /** Subscribe to all agent events as a SharedFlow. */
    val flow: SharedFlow<Pair<String, Map<String, Any>>> = _flow.asSharedFlow()

    /**
     * Emit an event. Non-suspending, safe to call from any thread.
     * Returns false only if the buffer is full and DROP_OLDEST didn't make space
     * (practically never happens with a 128-element buffer).
     */
    fun emit(name: String, data: Map<String, Any> = emptyMap()): Boolean =
        _flow.tryEmit(name to data)
}
