package com.ariaagent.mobile.core.agent

import android.content.Context
import android.util.Log
import com.ariaagent.mobile.core.memory.EmbeddingEngine
import com.ariaagent.mobile.core.memory.ExperienceStore
import com.ariaagent.mobile.core.perception.ScreenObserver
import com.ariaagent.mobile.core.rl.PolicyNetwork
import com.ariaagent.mobile.core.system.ThermalGuard
import com.ariaagent.mobile.system.actions.GestureEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * GameLoop — Fast policy-network-driven loop for game playing.
 *
 * Games are too fast for LLM inference (10–15 tok/s on Exynos 9611).
 * Instead, the PolicyNetwork (tiny MLP, <1ms per forward pass) selects
 * actions from OCR-text embeddings — no language model involved.
 *
 * Pipeline (per game step):
 *   1. Observe: ScreenObserver.capture() → snapshot (OCR + a11y)
 *   2. Thermal check: skip step if device is too hot
 *   3. Embed: EmbeddingEngine.embed(ocrText) → 384-dim → truncated to 128
 *              EmbeddingEngine.embed(goal)    → 384-dim → truncated to 128
 *   4. Act: PolicyNetwork.selectAction(screenEmb, goalEmb) → action index
 *   5. Execute: GestureEngine.executeGameAction(action, snapshot)
 *   6. Score detection: OCR diff for score/points counters
 *   7. Game-over detection: OCR for terminal state keywords
 *   8. Reward: +score_delta, -1.0 game over, +5.0 new high score
 *   9. Store: ExperienceTuple(task_type="game") → ExperienceStore
 *  10. Emit: game_loop_status event → AgentEventBus → AgentViewModel → Compose UI
 *
 * IRL bootstrapping: experience tuples from game videos (via IrlModule)
 * are stored as task_type="irl_expert" and used to pre-train the policy
 * network before the agent plays a single round itself. See Phase 6.2.
 *
 * Thermal safety:
 *   - ThermalGuard.isInferenceSafe() → false at SEVERE/CRITICAL → skip step
 *   - If game is running during training, training already paused by LearningScheduler
 *
 * Phase: 6 (Game Playing)
 */
object GameLoop {

    private const val TAG = "GameLoop"

    // Action constants matching PolicyNetwork.actionNames
    private const val ACTION_TAP        = 0
    private const val ACTION_SWIPE_UP   = 1
    private const val ACTION_SWIPE_DOWN = 2
    private const val ACTION_SWIPE_RIGHT = 3
    private const val ACTION_SWIPE_LEFT  = 4
    private const val ACTION_TYPE       = 5
    private const val ACTION_BACK       = 6

    // Timing: faster than the standard agent loop (no LLM wait)
    private const val STEP_DELAY_MS     = 400L   // ~2.5 actions/sec max
    private const val SETTLE_MS         = 200L   // screen settle after action
    private const val THERMAL_PAUSE_MS  = 8_000L // wait if overheating

    private const val MAX_STEPS_PER_EPISODE = 200

    data class GameState(
        val isActive: Boolean = false,
        val gameType: GameDetector.GameType = GameDetector.GameType.NONE,
        val episodeCount: Int = 0,
        val stepCount: Int = 0,
        val currentScore: Int = 0,
        val highScore: Int = 0,
        val totalReward: Double = 0.0,
        val lastAction: String = "",
        val isGameOver: Boolean = false
    )

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var gameJob: Job? = null

    @Volatile
    var state = GameState()
        private set

    var onEvent: ((name: String, data: Map<String, Any>) -> Unit)? = null

    // ─── Score detection ──────────────────────────────────────────────────────
    // These patterns match the raw OCR output of most Android games.
    // Scores are typically displayed as: "Score: 1234", "1,234", "SCORE 9999"

    private val SCORE_PATTERN = Regex(
        """(?i)(?:score|points?|pts)[:\s]*([0-9,]+)"""
    )

    private val GENERIC_NUMBER_PATTERN = Regex(
        """(?<!\w)(\d{3,7})(?!\w)"""   // standalone 3–7 digit number (score-like)
    )

    // ─── Game-over detection ──────────────────────────────────────────────────

    private val GAME_OVER_PATTERN = Regex(
        """(?i)(game\s*over|try\s*again|play\s*again|you\s*(?:died|lose|lost|failed)|level\s*failed|mission\s*failed|retry)""",
        RegexOption.IGNORE_CASE
    )

    // ─── Public API ───────────────────────────────────────────────────────────

    /**
     * Start the game loop for a detected game.
     * Called by AgentLoop when GameDetector signals game mode.
     *
     * @param context   Android context
     * @param goal      The user's task description (e.g. "get the highest score in Subway Surfers")
     * @param gameType  Detected game type (from GameDetector.detect())
     * @param store     ExperienceStore to persist experience tuples
     */
    fun start(
        context: Context,
        goal: String,
        gameType: GameDetector.GameType,
        store: ExperienceStore
    ) {
        if (gameJob?.isActive == true) return  // already running
        val episode = state.episodeCount + 1

        state = GameState(
            isActive    = true,
            gameType    = gameType,
            episodeCount = episode
        )
        emitStatus()

        gameJob = scope.launch {
            Log.i(TAG, "Game loop started: $gameType, episode $episode, goal=$goal")

            val goalEmbedding = EmbeddingEngine.embed(context, goal)
            var prevOcr = ""
            var highScore = state.highScore

            while (isActive && state.stepCount < MAX_STEPS_PER_EPISODE) {

                // ── 1. THERMAL CHECK ──────────────────────────────────────────
                if (!ThermalGuard.isInferenceSafe()) {
                    Log.w(TAG, "Thermal pause in GameLoop")
                    delay(THERMAL_PAUSE_MS)
                    if (!ThermalGuard.isInferenceSafe()) {
                        Log.w(TAG, "Still overheating — aborting game loop")
                        break
                    }
                }

                // ── 2. OBSERVE ────────────────────────────────────────────────
                val snapshot = ScreenObserver.capture()
                if (snapshot.isEmpty()) {
                    delay(SETTLE_MS)
                    continue
                }
                val ocrText = snapshot.ocrText

                // ── 3. GAME-OVER CHECK ────────────────────────────────────────
                val isGameOver = GAME_OVER_PATTERN.containsMatchIn(ocrText)
                if (isGameOver) {
                    Log.i(TAG, "Game over detected at step ${state.stepCount}")
                    handleGameOver(context, store, ocrText, prevOcr, goal, gameType)
                    break
                }

                // ── 4. SCORE EXTRACTION ───────────────────────────────────────
                val currentScore = extractScore(ocrText)
                val prevScore    = extractScore(prevOcr)
                val scoreDelta   = if (currentScore > 0 && prevScore >= 0) currentScore - prevScore else 0

                // ── 5. EMBED + ACTION SELECTION ───────────────────────────────
                val screenEmbedding = EmbeddingEngine.embed(context, ocrText.take(200))
                val (actionIdx, confidence) = PolicyNetwork.selectAction(screenEmbedding, goalEmbedding)
                val actionName = PolicyNetwork.actionNames.getOrElse(actionIdx) { "tap" }

                // ── 6. BUILD ACTION JSON (compatible with GestureEngine) ───────
                val actionJson = buildGameActionJson(actionIdx, snapshot)

                // ── 7. EXECUTE ────────────────────────────────────────────────
                val success = GestureEngine.executeFromJson(actionJson)
                delay(SETTLE_MS)

                // ── 8. REWARD ─────────────────────────────────────────────────
                val isNewHighScore = currentScore > highScore && currentScore > 0
                if (isNewHighScore) {
                    highScore = currentScore
                    Log.i(TAG, "New high score: $highScore")
                }
                val reward = when {
                    isNewHighScore          -> 5.0 + scoreDelta * 0.001
                    scoreDelta > 0          -> scoreDelta * 0.001            // +small per point
                    !success                -> -0.1                          // failed gesture
                    else                    -> 0.0
                }

                // ── 9. STORE EXPERIENCE ───────────────────────────────────────
                store.save(ExperienceStore.ExperienceTuple(
                    appPackage    = snapshot.appPackage,
                    taskType      = "game",
                    screenSummary = buildString {
                        append("[Game:$gameType step=${state.stepCount}] ")
                        if (currentScore > 0) append("Score:$currentScore ")
                        append(ocrText.take(300))
                    },
                    actionJson    = actionJson,
                    result        = if (success) "success" else "failure",
                    reward        = reward,
                    isEdgeCase    = false
                ))

                // ── 10. UPDATE STATE + EMIT ───────────────────────────────────
                val newStep = state.stepCount + 1
                state = state.copy(
                    stepCount    = newStep,
                    currentScore = currentScore,
                    highScore    = highScore,
                    totalReward  = state.totalReward + reward,
                    lastAction   = "$actionName (confidence=${confidence.format()})"
                )
                emitStatus()

                prevOcr = ocrText
                delay(STEP_DELAY_MS)
            }

            if (state.stepCount >= MAX_STEPS_PER_EPISODE) {
                Log.i(TAG, "Game episode capped at $MAX_STEPS_PER_EPISODE steps")
                state = state.copy(isActive = false)
                emitStatus()
            }
        }
    }

    fun stop() {
        gameJob?.cancel()
        state = state.copy(isActive = false)
        emitStatus()
    }

    fun isActive(): Boolean = state.isActive && gameJob?.isActive == true

    // ─── Game-over handling ───────────────────────────────────────────────────

    private fun handleGameOver(
        context: Context,
        store: ExperienceStore,
        ocrText: String,
        prevOcr: String,
        goal: String,
        gameType: GameDetector.GameType
    ) {
        val finalScore = extractScore(ocrText).takeIf { it > 0 } ?: extractScore(prevOcr)
        Log.i(TAG, "Game over — final score: $finalScore, total steps: ${state.stepCount}")

        store.save(ExperienceStore.ExperienceTuple(
            appPackage    = "",
            taskType      = "game",
            screenSummary = "[GAME_OVER:$gameType] Score:$finalScore ${ocrText.take(200)}",
            actionJson    = """{"tool":"Wait","reason":"game_over"}""",
            result        = "failure",
            reward        = -1.0,
            isEdgeCase    = true,
            edgeCaseNotes = "Game over detected at step ${state.stepCount}. Final score: $finalScore"
        ))

        state = state.copy(
            isActive     = false,
            isGameOver   = true,
            currentScore = finalScore,
            highScore    = maxOf(state.highScore, finalScore)
        )
        emitStatus()
    }

    // ─── Score extraction ─────────────────────────────────────────────────────

    /**
     * Extract the current score from OCR text.
     * Tries labelled patterns first ("Score: 1234"), then standalone large numbers.
     * Returns 0 if no score found.
     */
    private fun extractScore(ocrText: String): Int {
        if (ocrText.isBlank()) return 0

        // Try "Score: 1234" style first
        val labelled = SCORE_PATTERN.find(ocrText)?.groupValues?.get(1)
            ?.replace(",", "")?.toIntOrNull()
        if (labelled != null) return labelled

        // Fall back to the largest standalone number (scores are typically the biggest visible number)
        return GENERIC_NUMBER_PATTERN.findAll(ocrText)
            .mapNotNull { it.groupValues[1].toIntOrNull() }
            .maxOrNull() ?: 0
    }

    // ─── Action construction ──────────────────────────────────────────────────

    /**
     * Convert a policy network action index into a GestureEngine-compatible JSON.
     * For games we don't have reliable node IDs — use center-screen taps and directional swipes.
     * GestureEngine resolves "#center" to the middle of the screen.
     */
    private fun buildGameActionJson(actionIdx: Int, snapshot: ScreenObserver.ScreenSnapshot): String {
        // Pick a node from the accessibility tree for targeted taps,
        // or fall back to center-screen for canvas-based games with no a11y nodes.
        val hasA11yNodes = snapshot.a11yTree.contains("[#1]")
        val targetNode = if (hasA11yNodes) "#1" else "#center"

        return when (actionIdx) {
            ACTION_TAP         -> """{"tool":"Click","node_id":"$targetNode","reason":"game_policy"}"""
            ACTION_SWIPE_UP    -> """{"tool":"Swipe","direction":"up","node_id":"$targetNode","reason":"game_policy"}"""
            ACTION_SWIPE_DOWN  -> """{"tool":"Swipe","direction":"down","node_id":"$targetNode","reason":"game_policy"}"""
            ACTION_SWIPE_RIGHT -> """{"tool":"Swipe","direction":"right","node_id":"$targetNode","reason":"game_policy"}"""
            ACTION_SWIPE_LEFT  -> """{"tool":"Swipe","direction":"left","node_id":"$targetNode","reason":"game_policy"}"""
            ACTION_TYPE        -> """{"tool":"Click","node_id":"$targetNode","reason":"game_policy_type_as_tap"}"""
            ACTION_BACK        -> """{"tool":"Back","reason":"game_policy"}"""
            else               -> """{"tool":"Click","node_id":"$targetNode","reason":"game_policy_fallback"}"""
        }
    }

    // ─── Event emission ───────────────────────────────────────────────────────

    private fun emitStatus() {
        onEvent?.invoke("game_loop_status", mapOf(
            "isActive"      to state.isActive,
            "gameType"      to state.gameType.name.lowercase(),
            "episodeCount"  to state.episodeCount,
            "stepCount"     to state.stepCount,
            "currentScore"  to state.currentScore,
            "highScore"     to state.highScore,
            "totalReward"   to state.totalReward,
            "lastAction"    to state.lastAction,
            "isGameOver"    to state.isGameOver
        ))
    }

    private fun Float.format(): String = "%.2f".format(this)
}
