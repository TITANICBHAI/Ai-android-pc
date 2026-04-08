package com.ariaagent.mobile.core.agent

import com.ariaagent.mobile.core.perception.ScreenObserver
import com.ariaagent.mobile.system.accessibility.AgentAccessibilityService

/**
 * GameDetector — Determines whether the current screen is a game.
 *
 * The agent has two operating modes:
 *   - LLM-guided: for app navigation, settings, messaging, etc. (slow but precise)
 *   - Policy-network-guided: for games (fast, no language reasoning needed)
 *
 * This module decides which mode to use by analysing the current screen via
 * three independent signals — any one signal is sufficient to trigger game mode:
 *
 *   Signal 1 — Package name: known game package prefixes (e.g. com.kiloo.subwaysurf)
 *   Signal 2 — OCR patterns: "Score", "Level", "Lives", "Coins", "HP", numeric counters
 *   Signal 3 — A11y tree: absence of standard Android widgets + custom canvas view class
 *
 * Game type is coarser than genre — enough to tune reward and IRL strategy:
 *   ARCADE    → fast reflexes, score increments, game-over common
 *   PUZZLE    → slower, deliberate taps, level completion as reward
 *   STRATEGY  → long sessions, resource counters (troops, gold), no game-over
 *   NONE      → not a game; use LLM-guided agent loop
 *
 * Phase: 6 (Game Playing)
 */
object GameDetector {

    enum class GameType { NONE, ARCADE, PUZZLE, STRATEGY }

    data class GameSignal(
        val gameType: GameType,
        val confidence: Float,       // 0.0–1.0
        val triggerReason: String    // for logging/debug
    )

    // ─── Known game package prefixes ─────────────────────────────────────────
    // Exact matches are most reliable — keeps false-positive rate low.
    // Prefix matching handles variant APKs (com.supercell.clashofclans vs .clashofclans.xx).

    private val ARCADE_PACKAGES = setOf(
        "com.kiloo.subwaysurfers", "com.imangi.templerun", "com.imangi.templerun2",
        "com.frenzoo.zombie.highschool", "com.halfbrick.fruitninja",
        "com.deemedya.chickeninvaders", "com.zeptolabs.gunbros",
        "com.rovio.angrybirds", "com.rovio.anger", "com.angry.birds",
        "com.activision.callofduty", "com.ea.games.pvzfree_row",
        "com.disney.disneycrossyroad", "com.yodo1.crossyroad",
        "com.noodlecake.altosadventure", "com.snowman.altosoddyssey"
    )

    private val PUZZLE_PACKAGES = setOf(
        "com.king.candycrushsaga", "com.king.candycrush4",
        "com.ea.gp.pegglem", "com.zynga.words2", "com.zynga.scramble",
        "com.bethsoft.fallout.shelter", "com.nianticlabs.ingress",
        "com.gameloft.android.ANMP.GloftM9HM", "com.gameloft.braintrainer",
        "com.loadcomplete.2048", "com.ketchapp.stack", "com.ketchapp.ballz"
    )

    private val STRATEGY_PACKAGES = setOf(
        "com.supercell.clashofclans", "com.supercell.clashroyale",
        "com.supercell.brawlstars", "com.gram.games.townshipfarming",
        "net.iGindis.FarmVille2", "com.zynga.farmvillecountry",
        "com.ea.game.pvz2_row", "com.kiloo.skateboardparty",
        "com.reddit.frontpage"      // not a game, just testing exclusion
    )

    // ─── OCR keyword patterns ─────────────────────────────────────────────────
    // These appear in the OCR text of most game screens but rarely in normal app UIs.

    private val ARCADE_OCR = Regex(
        "(?i)(score[:\\s]*\\d|lives[:\\s]*\\d|high.?score|game.?over|tap.?to.?start|coins[:\\s]*\\d)",
        RegexOption.IGNORE_CASE
    )

    private val PUZZLE_OCR = Regex(
        "(?i)(level\\s*\\d+|moves\\s*left|stars\\s*earned|solve|puzzle|complete|next\\s+level)",
        RegexOption.IGNORE_CASE
    )

    private val STRATEGY_OCR = Regex(
        "(?i)(gold[:\\s]*\\d|troops|resources|upgrade|build|attack|defend|alliance|clan|guild|gems[:\\s]*\\d)",
        RegexOption.IGNORE_CASE
    )

    // Generic game signal (any of the above)
    private val ANY_GAME_OCR = Regex(
        "(?i)(score|level\\s*\\d|hp\\s*:\\s*\\d|mana|stamina|coins\\s*:\\s*\\d|lives\\s*:\\s*\\d|xp\\s*:\\s*\\d|power\\s*:\\s*\\d)",
        RegexOption.IGNORE_CASE
    )

    // ─── A11y tree heuristics ─────────────────────────────────────────────────
    // Games typically use a single SurfaceView or custom GLSurfaceView.
    // Standard apps have many View, Button, TextView, RecyclerView nodes.

    private val GAME_VIEW_CLASSES = setOf(
        "SurfaceView", "GLSurfaceView", "TextureView",
        "android.opengl.GLSurfaceView", "android.view.SurfaceView"
    )

    private val STANDARD_WIDGET_CLASSES = setOf(
        "RecyclerView", "ListView", "ViewPager", "BottomNavigationView",
        "Toolbar", "AppBarLayout", "CoordinatorLayout", "ConstraintLayout",
        "EditText", "CheckBox", "RadioButton", "Switch"
    )

    // ─── Detection ────────────────────────────────────────────────────────────

    /**
     * Analyse the current screen snapshot and determine if it's a game.
     * Returns a GameSignal with GameType.NONE if not a game.
     *
     * Called by AgentLoop at the start of each OBSERVE step.
     * Runs synchronously — no IO, no inference — fast enough to call every loop tick.
     */
    fun detect(snapshot: ScreenObserver.ScreenSnapshot): GameSignal {
        val pkg = snapshot.appPackage

        // 1. Package name check (highest confidence)
        when {
            ARCADE_PACKAGES.any { pkg == it || pkg.startsWith(it) } ->
                return GameSignal(GameType.ARCADE, 0.98f, "package_match_arcade")
            PUZZLE_PACKAGES.any { pkg == it || pkg.startsWith(it) } ->
                return GameSignal(GameType.PUZZLE, 0.98f, "package_match_puzzle")
            STRATEGY_PACKAGES.any { pkg == it || pkg.startsWith(it) } ->
                return GameSignal(GameType.STRATEGY, 0.98f, "package_match_strategy")
        }

        val ocr = snapshot.ocrText
        val a11y = snapshot.a11yTree

        // 2. OCR keyword match (medium-high confidence)
        val ocrType = when {
            ARCADE_OCR.containsMatchIn(ocr)   -> GameType.ARCADE
            PUZZLE_OCR.containsMatchIn(ocr)   -> GameType.PUZZLE
            STRATEGY_OCR.containsMatchIn(ocr) -> GameType.STRATEGY
            ANY_GAME_OCR.containsMatchIn(ocr) -> GameType.ARCADE  // generic → assume arcade
            else                              -> null
        }

        // 3. A11y tree structural check
        val hasGameView    = GAME_VIEW_CLASSES.any { a11y.contains(it) }
        val hasStdWidgets  = STANDARD_WIDGET_CLASSES.count { a11y.contains(it) }
        val a11yIsGameLike = hasGameView && hasStdWidgets <= 1

        // ── Combine signals ──────────────────────────────────────────────────
        return when {
            ocrType != null && a11yIsGameLike ->
                GameSignal(ocrType, 0.90f, "ocr_match + a11y_game_view")
            ocrType != null ->
                GameSignal(ocrType, 0.70f, "ocr_match_only")
            a11yIsGameLike ->
                GameSignal(GameType.ARCADE, 0.55f, "a11y_game_view_only")
            else ->
                GameSignal(GameType.NONE, 1.0f, "no_game_signals")
        }
    }

    /**
     * Minimum confidence threshold to switch to game mode.
     * Below this, stay in LLM-guided mode to avoid false positives.
     */
    const val MIN_CONFIDENCE = 0.60f
}
