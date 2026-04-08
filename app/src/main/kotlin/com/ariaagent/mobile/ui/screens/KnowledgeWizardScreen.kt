package com.ariaagent.mobile.ui.screens

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.automirrored.filled.CallSplit
import androidx.compose.material.icons.automirrored.filled.Label
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
import com.ariaagent.mobile.ui.theme.ARIAColors

/**
 * KnowledgeWizardScreen — a self-contained "encyclopedia" of how ARIA works.
 *
 * This is NOT a how-to wizard — it explains the underlying concepts behind the app:
 * what the AI engine actually does, how on-device learning works, what's real vs stub,
 * and what each component contributes.
 *
 * Accessible from Settings → "Knowledge Base" nav card.
 * Full-screen, no bottom nav bar.
 *
 * Pages:
 *   0  What is ARIA?
 *   1  The AI Engine (Llama 3.2-1B)
 *   2  Observe → Think → Act
 *   3  Vision (SmolVLM-256M)
 *   4  Memory & Experience
 *   5  On-Device Learning (RL + LoRA)
 *   6  Your Privacy
 *   7  What Needs the NDK Build
 *   8  The Permissions & Why
 *   9  Tips for Best Results
 *
 *  ── TRAINING MANUAL ──
 *  10  Training ARIA From Zero (overview)
 *  11  Mode 1: REINFORCE (Auto-Learning)
 *  12  Mode 2: IRL (Imitation from Video)
 *  13  Mode 3: Manual Labeling
 *  14  Mode 4: LoRA Fine-Tuning
 *  15  Training Recipes (combined examples)
 *
 *  ── DEEP KNOWLEDGE ──
 *  16  Why On-Device Over Cloud
 *  17  What Makes ARIA Agentic (not just a bot)
 *  18  How All Three Models See Your Screen
 *  19  Annotating Frames — Teaching Like a Sensei
 *  20  Model Size Tradeoffs on Mobile Hardware
 *  21  Model Types: Text-Only, Vision & Multimodal
 *  22  Which Model for Samsung M31? (Device Guide)
 *  23  Text-Only + SmolVLM: The Hidden Superpower
 *  24  When ARIA Fails and How to Fix It
 */

private data class KnowledgePage(
    val icon: ImageVector,
    val iconTint: Color,
    val title: String,
    val body: String,
    val bullets: List<Pair<ImageVector, String>> = emptyList(),
    val callout: String? = null,
    val calloutColor: Color = ARIAColors.Primary,
)

@Composable
fun KnowledgeWizardScreen(onBack: () -> Unit) {

    val pages: List<KnowledgePage> = listOf(
        KnowledgePage(
            icon      = Icons.Default.SmartToy,
            iconTint  = ARIAColors.Primary,
            title     = "What Is ARIA?",
            body      = "ARIA stands for AI Runtime Interface for Android. It is an autonomous agent that runs entirely on your device — no cloud servers, no subscriptions, no accounts.\n\nUnlike a chatbot, ARIA can open apps, tap buttons, type text, scroll, and complete multi-step tasks on your behalf — all decided by a local large language model (LLM) that runs in RAM.",
            bullets   = listOf(
                Icons.Default.PhoneAndroid to "100 % on-device — nothing leaves your phone",
                Icons.Default.AutoAwesome  to "Autonomous — decides what to do next, not just what to say",
                Icons.Default.Lock         to "Private by design — no account, no API key required",
                Icons.Default.FitnessCenter to "Self-improving — gets smarter with every task it runs",
            ),
            callout   = "ARIA is not a chatbot. It acts on your phone just like a human would — seeing the screen, reading text, and tapping the right things.",
        ),

        KnowledgePage(
            icon      = Icons.Default.Memory,
            iconTint  = ARIAColors.Accent,
            title     = "The AI Engine: Llama 3.2-1B",
            body      = "ARIA's decisions come from Llama 3.2-1B-Instruct — a 1-billion-parameter language model from Meta, quantised to Q4_K_M format (~700 MB) so it fits on a mid-range Android phone.\n\nInference is handled by llama.cpp, a highly-optimised C++ runtime. On a Samsung Galaxy M31 (Exynos 9611), ARIA targets 8–15 tokens per second using 4 CPU cores plus partial GPU offload via OpenCL (Mali-G72).\n\nThe model receives a structured prompt describing the screen, the goal, and past actions — and outputs a JSON action command.",
            bullets   = listOf(
                Icons.Default.Memory       to "Q4_K_M quantisation: 870 MB disk, ~1700 MB RAM with mmap",
                Icons.Default.Speed        to "8–15 tok/s on Exynos 9611 + OpenCL/Mali-G72 (target, not yet measured)",
                Icons.Default.Code         to "llama.cpp C++ runtime — same engine used by leading open-source tools",
                Icons.Default.Warning      to "Stub mode active until the NDK build compiles libllama-jni.so",
            ),
            callout   = "Until the native library is compiled, ARIA runs in stub mode — all decisions come from a hardcoded fallback, not the real model.",
            calloutColor = ARIAColors.Warning,
        ),

        KnowledgePage(
            icon      = Icons.Default.Loop,
            iconTint  = ARIAColors.Primary,
            title     = "Observe → Think → Act",
            body      = "Every step of a task follows the same three-phase cycle:\n\n1. OBSERVE: ARIA reads the Accessibility Tree (a text description of every visible UI element) and runs OCR on the screen. If vision is enabled, SmolVLM-256M also produces a visual description.\n\n2. THINK: The full observation — plus past actions, goal, and retrieved memories — is assembled into a structured prompt and sent to Llama 3.2-1B. The model outputs a JSON command such as {\"tool\":\"Click\",\"node_id\":\"#4\"}.\n\n3. ACT: GestureEngine dispatches a real programmatic touch via the Android Accessibility Service — the same mechanism used by screen readers.",
            bullets   = listOf(
                Icons.Default.Visibility   to "Observe: a11y tree + OCR + optional vision every step",
                Icons.Default.Psychology   to "Think: full prompt → LLM → JSON action command",
                Icons.Default.TouchApp     to "Act: real gesture dispatched via AccessibilityService",
                Icons.Default.Save         to "Store: result saved to SQLite as an experience tuple",
            ),
            callout   = "The loop runs up to 50 steps per task. Stuck detection aborts automatically if the screen stops changing.",
        ),

        KnowledgePage(
            icon      = Icons.Default.RemoveRedEye,
            iconTint  = ARIAColors.Primary,
            title     = "Vision: SmolVLM-256M",
            body      = "ARIA can optionally load SmolVLM-256M — a compact multimodal model that describes what it sees on screen using visual understanding, not just text.\n\nThis helps in three cases:\n  • Games and Flutter apps without an accessibility tree\n  • UI elements that are purely graphical (icons, images, buttons with no text)\n  • Detecting whether an action visually succeeded (pixel diff verification)\n\nSmolVLM uses the CLIP vision encoder to convert a JPEG screenshot into image embeddings, then Llama.cpp generates a goal-aware description. It adds ~200 MB to disk and runs in a separate model context so it never interferes with the main text model.",
            bullets   = listOf(
                Icons.Default.PhotoCamera  to "Screenshot → CLIP encode → text description every step",
                Icons.Default.Cached       to "Frame caching: ~0 ms when the screen hasn't changed",
                Icons.Default.SdCard       to "~200 MB download (base GGUF + mmproj file)",
                Icons.AutoMirrored.Filled.CallSplit     to "Separate handle — does not share context with the text LLM",
            ),
            callout   = "Vision is optional. ARIA works without it using the accessibility tree alone.",
        ),

        KnowledgePage(
            icon      = Icons.Default.Storage,
            iconTint  = ARIAColors.Accent,
            title     = "Memory & Experience",
            body      = "Every action ARIA takes is recorded in an SQLite database called ExperienceStore. Each row is an experience tuple:\n\n  (screen_hash, action_json, result, reward, timestamp)\n\nBefore each step, ARIA retrieves the 5 most relevant past experiences using MiniLM-L6 embeddings (a 23 MB ONNX model). This lets ARIA avoid repeating past mistakes and reuse successful patterns.\n\nThe Object Label Store is a separate database of UI element annotations created by you in the Labeler screen. ARIA uses these annotations at 3× weight during training — your corrections have more influence than passive experience.",
            bullets   = listOf(
                Icons.Default.TableRows    to "ExperienceStore: SQLite, success/failure/edge-case episodes",
                Icons.Default.Search       to "MiniLM embeddings: semantic similarity search over past steps",
                Icons.AutoMirrored.Filled.Label        to "ObjectLabelStore: human-annotated UI elements (3× weight)",
                Icons.Default.Memory       to "Embedding model: ~23 MB ONNX, runs on-device with ONNX Runtime",
            ),
        ),

        KnowledgePage(
            icon      = Icons.Default.School,
            iconTint  = ARIAColors.Success,
            title     = "On-Device Learning: RL + LoRA",
            body      = "ARIA improves itself through two complementary mechanisms:\n\nREINFORCE Policy Network: A 3-layer neural network (pure Kotlin, no native library required) that learns which action type to prefer given a situation. Updated after every task using the REINFORCE policy gradient algorithm with discounted returns and Adam optimisation.\n\nLoRA Fine-Tuning: The LLM itself can be fine-tuned using Low-Rank Adapters (LoRA). ARIA batches successful experience tuples into a JSONL dataset and calls llama.cpp's AdamW optimizer via JNI. This directly improves LLM decision quality — but requires the NDK build.",
            bullets   = listOf(
                Icons.Default.Psychology   to "REINFORCE: works immediately, no NDK — starts random, improves with experience",
                Icons.Default.AutoAwesome  to "LoRA: improves the LLM directly — requires NDK + compiled library",
                Icons.Default.VideoLibrary to "IRL Learning: teach by recording a video of yourself doing the task",
                Icons.Default.BarChart     to "Adam step count and policy loss visible in the Train screen",
            ),
            callout   = "The policy network starts with random weights. It needs at least 20–50 completed tasks before its suggestions become meaningful.",
            calloutColor = ARIAColors.Warning,
        ),

        KnowledgePage(
            icon      = Icons.Default.Lock,
            iconTint  = ARIAColors.Success,
            title     = "Your Privacy",
            body      = "ARIA was designed privacy-first. Here is exactly what stays on your device and what does not:\n\nStays on device: Everything. The LLM model weights, every screenshot taken, every decision made, every experience stored, your goals, your task history, your labels.\n\nLeaves your device: Only the model download (one-time, from a public source). Nothing else.\n\nThere are no analytics, no crash reporters, no telemetry, no remote logging. The app does not need an internet connection after the initial model download.",
            bullets   = listOf(
                Icons.Default.PhoneAndroid to "All inference runs locally — no API calls",
                Icons.Default.LockOpen     to "No accounts, no login, no email required",
                Icons.Default.WifiOff      to "Works fully offline after model download",
                Icons.Default.VisibilityOff to "Screenshots never transmitted — used only for on-device OCR",
            ),
            callout   = "The only network requests are the one-time model downloads (Llama, SmolVLM, MiniLM, EfficientDet).",
        ),

        KnowledgePage(
            icon      = Icons.Default.Build,
            iconTint  = ARIAColors.Warning,
            title     = "What Needs the NDK Build",
            body      = "Several features require the Android NDK to compile a native library (libllama-jni.so). Until that build runs, ARIA falls back to safe stubs and logs a clear warning.\n\nThe NDK build compiles:\n  • llama.cpp — the LLM C++ runtime\n  • NEON SIMD acceleration for the policy network\n  • LoRA AdamW optimizer\n  • CLIP vision encoder (mtmd library)\n\nTo trigger the build: clone the repo, install Android Studio with NDK r27+, run ./gradlew assembleDebug from the android/ directory. The build takes 10–20 minutes the first time.",
            bullets   = listOf(
                Icons.Default.PriorityHigh to "LLM inference (Llama 3.2-1B) — stub until built",
                Icons.Default.PriorityHigh to "Vision inference (SmolVLM) — stub until built",
                Icons.Default.Info          to "LoRA fine-tuning — writes metadata file until built",
                Icons.Default.CheckCircle   to "Policy network (REINFORCE) — works without NDK in Kotlin",
            ),
            callout   = "Stub mode is clearly labelled in logcat. Real mode activates automatically once System.loadLibrary(\"llama-jni\") succeeds at startup.",
            calloutColor = ARIAColors.Warning,
        ),

        KnowledgePage(
            icon      = Icons.Default.Accessibility,
            iconTint  = ARIAColors.Success,
            title     = "The Permissions & Why",
            body      = "ARIA requests three system permissions. Here is exactly what each one does:\n\nAccessibility Service: Required to read the UI node tree of any open app and dispatch real touch gestures. Without it, ARIA cannot see or interact with other apps. Enable once in Android Settings → Accessibility → ARIA Agent.\n\nScreen Capture (MediaProjection): Required to take screenshots for OCR and vision inference. Only active during a running task. You approve it each session.\n\nWake Lock: Keeps the CPU from sleeping during a training cycle. Released immediately when training finishes.",
            bullets   = listOf(
                Icons.Default.Accessibility to "Accessibility: reads UI + dispatches gestures (required)",
                Icons.Default.Screenshot    to "Screen Capture: screenshots for OCR + vision (per-session)",
                Icons.Default.BatteryFull   to "Wake Lock: held only during RL training, auto-released",
                Icons.Default.Notifications to "Notifications: foreground service notification while agent runs",
            ),
        ),

        KnowledgePage(
            icon      = Icons.Default.Lightbulb,
            iconTint  = ARIAColors.Accent,
            title     = "Tips for Best Results",
            body      = "ARIA performs best when you give it clear, specific goals and let it learn over time. Here are the most impactful things you can do:",
            bullets   = listOf(
                Icons.Default.Edit          to "Be specific: \"Open Chrome, search for X, tap the first result\" beats \"find something about X\"",
                Icons.AutoMirrored.Filled.Label         to "Label screens: the more UI elements you annotate in the Labeler, the better ARIA understands new apps",
                Icons.Default.VideoLibrary  to "Record yourself: IRL video training is the fastest way to teach ARIA a new workflow",
                Icons.Default.School         to "Run learn-only: lets ARIA observe without touching — safe for exploring unfamiliar apps",
                Icons.Default.BarChart      to "Check the Dashboard: thermal state, LoRA version, policy loss tell you how training is progressing",
                Icons.Default.Queue         to "Queue tasks: line up multiple tasks and ARIA chains them automatically when one completes",
            ),
            callout   = "Tip: after training on IRL video, run an RL cycle immediately. The two learning signals combine for faster convergence.",
            calloutColor = ARIAColors.Success,
        ),

        // ── TRAINING MANUAL ───────────────────────────────────────────────────

        KnowledgePage(
            icon      = Icons.Default.School,
            iconTint  = ARIAColors.Primary,
            title     = "Training ARIA From Zero",
            body      = "Training ARIA is not like training a traditional app. You are teaching a live AI agent that already knows how to reason — you just need to show it your phone, your apps, and your goals.\n\nThere are four distinct training modes. Each one teaches ARIA something different, and they all compound over time:\n\n  1. REINFORCE — learns automatically from every task it runs\n  2. IRL (Imitation) — you record a video, ARIA copies your actions\n  3. Manual Labeling — you annotate UI elements by name\n  4. LoRA Fine-Tuning — directly improves the LLM brain (advanced)\n\nYou can use all four, but start with REINFORCE — it requires zero effort and starts working immediately.",
            bullets   = listOf(
                Icons.Default.AutoMode      to "Day 1–3: just run tasks. REINFORCE learns automatically",
                Icons.Default.VideoLibrary  to "Day 4+: record IRL videos for complex workflows",
                Icons.AutoMirrored.Filled.Label         to "Week 2: annotate stubborn UI elements in the Labeler",
                Icons.Default.AutoAwesome   to "Advanced: LoRA fine-tuning once NDK is compiled",
            ),
            callout   = "No GPU, no cloud, no data labels required to start. Just run a task, and ARIA starts learning.",
            calloutColor = ARIAColors.Primary,
        ),

        KnowledgePage(
            icon      = Icons.Default.AutoMode,
            iconTint  = ARIAColors.Success,
            title     = "Mode 1: REINFORCE (Auto-Learning)",
            body      = "REINFORCE is always on. Every time ARIA completes a step — success or failure — it stores the experience and updates its policy network automatically.\n\nHow it works:\n  • ARIA runs a step and records (screen, action, result, reward)\n  • After the task ends, it calculates discounted returns: steps that led to success get high reward, failed paths get penalised\n  • The 3-layer policy network is updated with Adam gradient descent\n  • Over 20–50 tasks, ARIA learns which action types work in which context\n\nEXAMPLE — Teaching ARIA to open WhatsApp:\n  Task 1: ARIA tries random things, opens the wrong app (reward = −1)\n  Task 5: ARIA scrolls then taps WhatsApp icon (reward = +0.5)\n  Task 20: ARIA finds and opens WhatsApp in 2 steps every time (reward = +1)\n\nYou do nothing except run the task and give feedback (thumbs up / thumbs down).",
            bullets   = listOf(
                Icons.Default.ThumbUp       to "Thumbs up = +1 reward → that action path is reinforced",
                Icons.Default.ThumbDown     to "Thumbs down = −1 reward → that path is penalised",
                Icons.Default.AutoAwesome   to "No feedback = neutral reward from task completion signal",
                Icons.Default.BarChart      to "Watch policy loss in the Train screen — falling = learning",
            ),
            callout   = "The policy network starts completely random. After 20 tasks it becomes 2–3× faster. After 100 tasks it becomes highly reliable for familiar apps.",
            calloutColor = ARIAColors.Success,
        ),

        KnowledgePage(
            icon      = Icons.Default.VideoLibrary,
            iconTint  = ARIAColors.Accent,
            title     = "Mode 2: IRL (Imitation from Video)",
            body      = "IRL (Inverse Reinforcement Learning) is the fastest way to teach ARIA a specific workflow. You record a screen recording of yourself doing the task — ARIA watches it frame by frame and extracts what you tapped, where, and in what order.\n\nHow to record a teaching video:\n  1. Go to the Train screen → tap \"Record IRL Session\"\n  2. Do the task naturally (it records your screen + taps)\n  3. Stop recording — ARIA processes the video automatically\n  4. Each frame is matched to an action: tap, scroll, type, swipe\n  5. These become high-confidence training examples (3× normal weight)\n\nEXAMPLE — Teaching ARIA to recharge Paytm:\n  You record: open Paytm → tap Recharge → tap Mobile → enter number → tap Proceed → select UPI → tap Pay\n  ARIA extracts 7 action steps with coordinates and labels them\n  After just 1 video, ARIA can complete the same flow with ~80% accuracy\n\nEXAMPLE — Teaching ARIA to book a cab on Ola:\n  You record the booking flow once (takes ~90 seconds)\n  ARIA learns: home → tap Bike → enter destination → confirm pickup → tap Book\n  It generalises: if the destination changes, it still follows the same structure",
            bullets   = listOf(
                Icons.Default.Videocam      to "Record once → ARIA trains for ~2 minutes on-device",
                Icons.Default.Repeat        to "Record the same flow 3 times → accuracy jumps to 95%+",
                Icons.Default.SlowMotionVideo to "Record at normal speed — ARIA handles frame extraction",
                Icons.Default.Warning       to "Avoid recording personal info — process runs locally but store is on-device",
            ),
            callout   = "IRL + REINFORCE together: record the task once (IRL), then run it 10 times (REINFORCE). This is the fastest training recipe in ARIA.",
            calloutColor = ARIAColors.Accent,
        ),

        KnowledgePage(
            icon      = Icons.AutoMirrored.Filled.Label,
            iconTint  = ARIAColors.Warning,
            title     = "Mode 3: Manual Labeling",
            body      = "Some UI elements have no text — icons, image buttons, custom-drawn views in games, Flutter apps. ARIA cannot name these from the accessibility tree alone. Manual labeling teaches it what they mean.\n\nHow to label:\n  1. Go to the Train screen → tap \"Open Labeler\"\n  2. Take a screenshot of the app you want to teach\n  3. Draw a box around a UI element\n  4. Type its name and role: e.g. \"Send button\" or \"Hamburger menu\"\n  5. Save — ARIA immediately uses this label in all future tasks\n\nEXAMPLE — Labeling a game UI:\n  You open PUBG Mobile. The fire button has no accessibility label.\n  You draw a box around it and label it: \"Fire button\"\n  The jump button: \"Jump button\"\n  Now ARIA can play the game using these names in its reasoning\n\nEXAMPLE — Labeling a custom icon bar:\n  A banking app has 5 icon tabs with no text.\n  You label each one: \"Home\", \"Pay\", \"Cards\", \"History\", \"Profile\"\n  Next task: ARIA says \"tap Pay tab\" and finds it instantly\n\nLabeled elements get 3× reward weight in training — your annotations override ARIA's guesses.",
            bullets   = listOf(
                Icons.Default.TouchApp      to "Draw box → type name → save. Takes 5 seconds per element",
                Icons.Default.Games         to "Essential for games — accessibility tree is empty in Unity/Unreal",
                Icons.Default.Apps          to "Essential for Flutter apps — they render their own widgets",
                Icons.Default.Bolt          to "Immediate effect — ARIA uses the label from the very next task",
            ),
            callout   = "Start with the 3–5 elements ARIA gets wrong most often. Labeling those specific elements gives the biggest accuracy jump.",
            calloutColor = ARIAColors.Warning,
        ),

        KnowledgePage(
            icon      = Icons.Default.AutoAwesome,
            iconTint  = ARIAColors.Primary,
            title     = "Mode 4: LoRA Fine-Tuning",
            body      = "LoRA (Low-Rank Adaptation) is the most powerful training mode. It directly modifies the weights of Llama 3.2-1B — the AI brain itself — making it permanently smarter at your specific use cases.\n\nRequires: NDK compiled (libllama-jni.so must be built). See the \"NDK Build\" page.\n\nHow LoRA works:\n  • ARIA gathers your best experiences (high-reward episodes)\n  • Converts them to a JSONL fine-tuning dataset in llama.cpp format\n  • Trains a small set of adapter matrices (LoRA rank 8–16)\n  • These adapters are loaded on top of the base model — no re-download needed\n  • LoRA version counter increments in the Train screen\n\nEXAMPLE — Before vs After LoRA for Swiggy orders:\n  Before LoRA (Day 1): takes 18 steps, sometimes clicks wrong restaurant\n  After 1 LoRA cycle on 30 episodes: takes 9 steps, >90% correct first tap\n  After 3 LoRA cycles: completes Swiggy order in 6 steps reliably\n\nWhen to trigger LoRA:\n  • After collecting 50+ successful episodes (Train → tap \"Train LoRA\")\n  • After IRL sessions (combine IRL data with RL data)\n  • Scheduled: enable auto-training in Settings → Train runs overnight\n\nLoRA runs on CPU (4 cores, ~15 min/cycle). Battery drain is real — plug in first.",
            bullets   = listOf(
                Icons.Default.Memory        to "LoRA adapters are ~10–30 MB — tiny compared to the 800 MB base model",
                Icons.Default.Battery5Bar   to "Always plug in before LoRA training — it takes 10–20 minutes",
                Icons.Default.History       to "Old adapters are kept — revert anytime from the Train screen",
                Icons.Default.PriorityHigh  to "Requires NDK build — check Modules screen → LLM must show ACTIVE",
            ),
            callout   = "LoRA is optional. ARIA improves meaningfully through REINFORCE + IRL alone. LoRA is for users who want maximum performance.",
            calloutColor = ARIAColors.Primary,
        ),

        KnowledgePage(
            icon      = Icons.Default.RocketLaunch,
            iconTint  = ARIAColors.Success,
            title     = "Training Recipes",
            body      = "The fastest ways to train ARIA for real-world tasks, combining the four modes:\n\nRECIPE A — Learn a new app in one evening:\n  1. Run the task 5 times (REINFORCE collects data)\n  2. Record 2 IRL videos of yourself doing it\n  3. Label any icon-only buttons ARIA missed\n  4. Run LoRA fine-tuning overnight (optional)\n  → Next morning: ARIA handles that app reliably\n\nRECIPE B — Fix a recurring mistake:\n  ARIA keeps tapping the wrong button → record an IRL video of the correct tap → that action gets 3× weight → mistake disappears within 3 tasks\n\nRECIPE C — Train for gaming:\n  1. Label all game UI elements (fire, jump, aim, map)\n  2. Enable MobileSAM (Modules screen) for pixel-level detection\n  3. Switch to Game Mode in the Control screen\n  4. Run a 10-minute training session — ARIA observes your gameplay\n  5. Let REINFORCE run for 20 game sessions\n  → ARIA learns game-specific tap timing and sequences\n\nRECIPE D — Total from-scratch setup (first week):\n  Day 1: download all modules, run 5 tasks (any app)\n  Day 2–3: run 10 tasks, give thumbs up/down honestly\n  Day 4: record 3 IRL videos of your most-used workflows\n  Day 5: open Labeler, label 10–15 stubborn icons\n  Day 7: trigger first LoRA cycle (if NDK is built)\n  → End of week 1: ARIA is 3–5× faster and more accurate than Day 1",
            bullets   = listOf(
                Icons.Default.Speed         to "Fastest gain: IRL video (1 recording = 50+ labelled examples)",
                Icons.Default.Stairs        to "Steady gain: REINFORCE runs automatically — just keep using ARIA",
                Icons.Default.Tune          to "Targeted fix: labeling specific missed elements fixes them immediately",
                Icons.Default.NightsStay    to "Overnight boost: LoRA scheduled training while you sleep",
            ),
            callout   = "You do not need to do all of this. Even just running REINFORCE for two weeks of normal use makes ARIA significantly smarter at your daily apps.",
            calloutColor = ARIAColors.Success,
        ),

        // ── DEEP KNOWLEDGE ────────────────────────────────────────────────────

        KnowledgePage(
            icon      = Icons.Default.WifiOff,
            iconTint  = ARIAColors.Primary,
            title     = "Why On-Device Over Cloud",
            body      = "Cloud AI (ChatGPT, Gemini, Claude) is powerful but fundamentally incompatible with an autonomous phone agent. Here is why ARIA must run on your device:\n\nPRIVACY — A cloud agent would send every screenshot of your phone to a remote server. That includes your banking apps, your messages, your contacts, your passwords. No legitimate agent can do this.\n\nLATENCY — A cloud round-trip takes 300–1500 ms per step. ARIA runs 3–8 steps per second locally. At 1 step/second for a 20-step task, cloud adds 5–30 minutes of wait time versus 20 seconds on-device.\n\nCOST — GPT-4 Vision at ~2000 tokens/step × 50 steps/task × 10 tasks/day = $3–15/day per user. On-device: $0 forever.\n\nCONTROL — Cloud models are updated without notice. On-device, the model you trust today is the model running tomorrow. No prompt injection from model updates.\n\nThe real tradeoff: on-device models are smaller (1B–7B params vs 70B–1T) and less capable at reasoning. ARIA compensates with structured prompting, memory retrieval, and continuous on-device fine-tuning.",
            bullets   = listOf(
                Icons.Default.Lock          to "Privacy: zero screenshots leave the device — ever",
                Icons.Default.Speed         to "Latency: <100 ms/step local vs 300–1500 ms cloud round-trip",
                Icons.Default.AttachMoney   to "Cost: $0 forever vs $3–15/day for GPT-4 Vision",
                Icons.Default.PhoneAndroid  to "Control: your model, your data, your rules — no silent updates",
            ),
            callout   = "Cloud AI is not better for this use case — it is architecturally wrong. An agent that sends your banking screen to a server is a privacy violation, not a feature.",
            calloutColor = ARIAColors.Warning,
        ),

        KnowledgePage(
            icon      = Icons.Default.AutoAwesome,
            iconTint  = ARIAColors.Accent,
            title     = "What Makes ARIA Agentic",
            body      = "A chatbot answers questions. A bot follows a fixed script. An agent reasons about its environment and decides what to do next — even when the situation is new.\n\nFour things make ARIA agentic instead of just a bot:\n\n1. PERCEPTION — ARIA reads live screen state (accessibility tree + OCR + vision) at every step. It sees the actual current state, not a pre-programmed snapshot.\n\n2. REASONING — The LLM reasons about what it sees and produces a justified action (the 'reason' field in the JSON). It can handle unexpected screens by reasoning from first principles.\n\n3. MEMORY — Past experiences are retrieved and injected into every prompt. ARIA knows it has tried this before, and what happened.\n\n4. ADAPTATION — If stuck, ARIA changes strategy (tries Back, then a different path, then scrolls). It does not loop on the same failed action.\n\nEXAMPLE of agentic vs bot behaviour:\n  Bot: \"If screen = home screen, tap Gmail\"\n  ARIA: \"I see a home screen. My goal is to send email. Gmail is visible as icon #7. I will tap it. Reason: shortest path to email compose.\"\n  If Gmail is not there: Bot fails. ARIA searches the app drawer instead.",
            bullets   = listOf(
                Icons.Default.Visibility    to "Perceives real live screen state every step — not hardcoded",
                Icons.Default.Psychology    to "Reasons with justified JSON — not pattern-matched rules",
                Icons.Default.History       to "Retrieves relevant memory — learns from its own past",
                Icons.Default.AltRoute      to "Adapts when stuck — tries alternative paths autonomously",
            ),
            callout   = "The 'reason' field in every action JSON is ARIA's live reasoning trace. Reading it in the Activity screen shows exactly what the agent is thinking step by step.",
            calloutColor = ARIAColors.Accent,
        ),

        KnowledgePage(
            icon      = Icons.Default.RemoveRedEye,
            iconTint  = ARIAColors.Primary,
            title     = "How All Three Models See Your Screen",
            body      = "When ARIA processes a frame, three completely different models each contribute a different kind of understanding. They run in sequence and their outputs are fused into a single prompt.\n\nMODEL 1 — Llama 3.2-1B (The Thinker)\n  Input: structured text (accessibility tree + OCR + vision summary + memory)\n  Output: one JSON action command\n  Strength: multi-step reasoning, goal interpretation, handling novel situations\n  Weakness: cannot see raw pixels — depends on the other two for visual input\n\nMODEL 2 — SmolVLM-256M (The Eyes)\n  Input: a JPEG screenshot letterboxed to 384×384\n  Output: a 2–3 sentence natural language description of what it sees\n  Strength: describes icon-only UI, game graphics, visual states (loading, error, empty)\n  Weakness: small model — sometimes misidentifies fine text or small icons\n\nMODEL 3 — MobileSAM ViT-Tiny (The Finger Pointer)\n  Input: same JPEG screenshot at 1024×1024\n  Output: top-8 normalised (x, y) coordinates of salient regions\n  Strength: finds tappable regions even with no accessibility tree at all\n  Weakness: saliency-based — finds visually prominent regions, not logically important ones\n\nFusion: SmolVLM's description + SAM's coordinates + OCR text → all injected into Llama's prompt → Llama decides which region to tap and why.",
            bullets   = listOf(
                Icons.Default.Memory        to "Llama: reasons with text — the decision-maker",
                Icons.Default.PhotoCamera   to "SmolVLM: reads pixels — describes what human eyes see",
                Icons.Default.TouchApp      to "MobileSAM: finds tap targets — works with zero accessibility data",
                Icons.Default.MergeType     to "Fusion: all three outputs combined → richer than any alone",
            ),
            callout   = "On a Flutter app or game with no accessibility tree: SmolVLM describes the screen and SAM provides tap coordinates. Llama picks the right one. All without a single accessibility node.",
            calloutColor = ARIAColors.Primary,
        ),

        KnowledgePage(
            icon      = Icons.Default.EditNote,
            iconTint  = ARIAColors.Success,
            title     = "Annotating Frames: Teaching Like a Sensei",
            body      = "When you annotate a video frame in the IRL trainer — typing \"here I tapped the Share button at the top right\" — you are doing something much more powerful than just labeling an action.\n\nYour annotation flows into ALL THREE models simultaneously:\n\nTO SmolVLM: Your text is injected into the vision prompt. Instead of \"describe this screen\", SmolVLM is now asked: \"The user says they tapped the Share button here — confirm what you see and what the screen state shows.\" SmolVLM then produces a vision description that is grounded in your logic, not just pixel saliency.\n\nTO MobileSAM: If your annotation mentions a screen region (\"top button\", \"bottom nav\", \"left panel\"), SAM's tap candidates are re-ranked toward that area. So the coordinates ARIA receives are already biased toward where you said the action happened.\n\nTO Llama: Your annotation appears as \"Expert note\" in the action inference prompt. The LLM sees your reasoning directly and uses it to produce the correct action JSON — even if OCR or accessibility data would have suggested something different.\n\nThis is why annotating even a few frames of a complex workflow teaches ARIA exponentially faster than running the task blindly. You are teaching it your reasoning, not just your actions.",
            bullets   = listOf(
                Icons.Default.RemoveRedEye  to "SmolVLM: annotation becomes part of the vision prompt",
                Icons.Default.TouchApp      to "MobileSAM: spatial words re-rank which tap targets rank first",
                Icons.Default.Psychology    to "Llama: sees your reasoning as 'Expert note' in the prompt",
                Icons.Default.Bolt          to "Effect: one annotated video > ten unannotated task runs",
            ),
            callout   = "Best annotation style: explain your reasoning, not just what you did. 'I tapped Settings because Bluetooth toggle is only accessible from there' teaches more than 'tapped Settings'.",
            calloutColor = ARIAColors.Success,
        ),

        KnowledgePage(
            icon      = Icons.Default.Memory,
            iconTint  = ARIAColors.Warning,
            title     = "Model Size Tradeoffs on Mobile",
            body      = "ARIA uses Llama 3.2-1B by design. Here is why — and what you gain and lose versus larger models:\n\nLlama 3.2-1B Q4_K_M (~870 MB, ~10–15 tok/s on M31)\n  PRO: Fits in 6 GB RAM alongside the OS, SmolVLM, and SAM. Runs at real-time speed.\n  CON: Weaker multi-step reasoning than 3B+. May struggle with very complex tasks.\n\nLlama 3.2-3B Q4_K_M (~2.0 GB, ~4–6 tok/s on M31)\n  PRO: Noticeably better reasoning, fewer mistakes on ambiguous screens.\n  CON: Marginal fit — leaves only ~3.5 GB for OS + other apps. May trigger OOM kills.\n  STATUS: Supported in code, gated by device RAM check. Enable in Settings → Model.\n\nLlama 3.2-8B Q4_K_M (~5.0 GB, not feasible on M31)\n  Would need 8–10 GB RAM total. Would not load on 6 GB devices.\n  Feasible on flagship 12 GB devices (Galaxy S24 Ultra, Pixel 9 Pro).\n\nSmolVLM 256M vs 500M:\n  256M fits in ~200 MB. 500M doubles disk + RAM for ~15% better descriptions.\n  Currently locked to 256M for M31 safety. Future: configurable per device.\n\nRule of thumb: use the largest model that loads without OOM on your specific device. ARIA will tell you if a model is too large during the download check.",
            bullets   = listOf(
                Icons.Default.Memory        to "1B: fits any 6 GB device — current default, real-time speed",
                Icons.Default.Memory        to "3B: better reasoning — borderline on 6 GB, safe on 8 GB+",
                Icons.Default.Warning       to "8B+: requires 12 GB+ RAM — not supported on M31",
                Icons.Default.Tune          to "Q4_K_M quantisation: best quality/size tradeoff for 4-bit",
            ),
            callout   = "If you have a newer high-RAM device (8 GB+), try switching to 3B. The reasoning improvement is significant for complex multi-app workflows.",
            calloutColor = ARIAColors.Warning,
        ),

        // ── Page 21 ───────────────────────────────────────────────────────────

        KnowledgePage(
            icon      = Icons.Default.Category,
            iconTint  = ARIAColors.Primary,
            title     = "Model Types: Text, Vision & Multimodal",
            body      = "ARIA now supports three model categories. Each has a different way of understanding your screen:\n\nMULTIMODAL (Vision + Text — built-in)\n  One model does everything: it sees the screenshot AND reasons about what to do. The vision encoder (mmproj) is baked in. Examples: SmolVLM 256M/500M, Qwen2.5-VL 3B, MiniCPM-V 2.6.\n  Best for: apps with poor accessibility trees (games, Flutter, WebViews).\n  RAM cost: base model + mmproj loaded together.\n\nTEXT-ONLY (Reasoning only)\n  The model reasons and plans actions but cannot see the screen directly. It relies on the accessibility tree, OCR text, and — if downloaded — the SmolVLM helper for visual understanding. Examples: Llama 3.2 1B/3B, Gemma 3 1B/4B, Qwen2.5 1.5B.\n  Best for: standard Android apps with full accessibility trees.\n  RAM cost: model only. SmolVLM helper adds ~200 MB if enabled.\n\nAll types share the same training pipeline. Switching from a multimodal model to a text-only one — or back — does NOT discard accumulated experience. ARIA reuses every recorded episode regardless of which model collected it.",
            bullets   = listOf(
                Icons.Default.RemoveRedEye  to "Multimodal: sees screenshot natively via CLIP vision encoder",
                Icons.Default.TextFields    to "Text-only: reads accessibility tree + OCR; optionally SmolVLM",
                Icons.Default.SyncAlt       to "Training data is shared across all model types — zero loss on switch",
                Icons.Default.Memory        to "Each model keeps its own LoRA adapter — no cross-contamination",
            ),
            callout   = "ARIA automatically detects whether an active model is multimodal or text-only and loads the correct engine mode — you don't need to configure this manually.",
            calloutColor = ARIAColors.Primary,
        ),

        // ── Page 22 ───────────────────────────────────────────────────────────

        KnowledgePage(
            icon      = Icons.Default.PhoneAndroid,
            iconTint  = ARIAColors.Accent,
            title     = "Which Model for Samsung M31?",
            body      = "The Samsung Galaxy M31 has 6 GB RAM and an Exynos 9611 (8 cores, no dedicated NPU). Here is how every catalog model maps to that hardware:\n\nSAFE — fits comfortably, full speed:\n  SmolVLM 256M   │ <1 GB   │ >20 tok/s  │ Vision built-in\n  SmolVLM 500M   │ ~1.2 GB │ ~15 tok/s  │ Better vision quality\n  Llama 3.2 1B   │ ~1.2 GB │ 10–15 tok/s│ Best text reasoning for size\n  Gemma 3 1B     │ ~1.1 GB │ 12–16 tok/s│ Best JSON output\n  Qwen2.5 1.5B   │ ~1.5 GB │ 8–12 tok/s │ Strong tool-use\n  Moondream2     │ ~2 GB   │ ~8 tok/s   │ Compact VLM\n\nFITS — tight, watch for OOM during concurrent app use:\n  Qwen2.5-VL 3B  │ ~3 GB   │ ~5 tok/s   │ Best vision+text balance\n  Llama 3.2 3B   │ ~2.5 GB │ ~5 tok/s   │ Best text quality\n  Gemma 3 4B     │ ~3 GB   │ ~4 tok/s   │ Dense, high quality\n\nRISKY — 5–6 GB, may crash if other apps are open:\n  MiniCPM-V 2.6  │ ~5.5 GB │ ~2 tok/s   │ GPT-4V quality vision\n\nNOT RECOMMENDED for M31:\n  Llama 3.2 V 11B│ 8 GB+   │ <1 tok/s   │ Exceeds M31 RAM limit",
            bullets   = listOf(
                Icons.Default.CheckCircle   to "Best overall pick: SmolVLM 256M (safe, fast, vision built-in)",
                Icons.Default.AutoAwesome   to "Best text reasoning: Llama 3.2 1B — pairs with SmolVLM helper",
                Icons.Default.Videocam      to "Best vision quality safe on M31: Qwen2.5-VL 3B (tight but works)",
                Icons.Default.Warning       to "MiniCPM-V 2.6 and 11B models risk OOM on M31 — use with caution",
            ),
            callout   = "Rule of thumb for 6 GB phones: stay at or below 3 GB model RAM. Add SmolVLM helper (~200 MB) if you need vision with a text-only model.",
            calloutColor = ARIAColors.Accent,
        ),

        // ── Page 23 ───────────────────────────────────────────────────────────

        KnowledgePage(
            icon      = Icons.AutoMirrored.Filled.CallSplit,
            iconTint  = ARIAColors.Success,
            title     = "Text-Only + SmolVLM: The Power Pair",
            body      = "When you activate a text-only model (e.g. Llama 3.2 1B), ARIA doesn't lose its ability to understand the screen — it automatically delegates vision to the SmolVLM 256M helper if it's downloaded.\n\nHow the pairing works:\n  1. AgentLoop captures a screenshot each step\n  2. If SmolVLM is loaded, it produces a text description of the screen\n  3. That description is fed into Llama's prompt — Llama uses it to reason and decide\n  4. The split means each model can be the best at its specialty\n\nWhy you might prefer this over a pure VLM:\n  • Better text reasoning: Llama 3.2 1B is a pure language model trained for instruction following — it often reasons better than a 1B VLM that had to split capacity between seeing and thinking\n  • More flexibility: swap the reasoning model without losing vision; update SmolVLM independently\n  • RAM efficiency: SmolVLM 256M uses only ~200 MB extra — leaving more RAM for the reasoning model\n\nFall-back chain (in order):\n  1. SmolVLM helper (if downloaded) — visual description\n  2. OCR (Tesseract) — raw text from screenshot\n  3. Accessibility tree — UI node names and labels\n\nARIA always uses all three sources and combines them into a single structured observation.",
            bullets   = listOf(
                Icons.Default.Psychology    to "Llama reasons; SmolVLM sees — each at full capacity",
                Icons.Default.Memory        to "Total RAM: ~1.2 GB (Llama 1B) + ~200 MB (SmolVLM) = ~1.4 GB",
                Icons.Default.AutoMode      to "Pairing is automatic — no settings to configure manually",
                Icons.Default.SdCard        to "Download SmolVLM once; it pairs with any future text model",
            ),
            callout   = "This is ARIA's most RAM-efficient high-capability setup on M31: Llama 3.2 1B + SmolVLM 256M helper. Only ~1.4 GB combined, full vision, best-in-class text reasoning.",
            calloutColor = ARIAColors.Success,
        ),

        KnowledgePage(
            icon      = Icons.Default.BugReport,
            iconTint  = ARIAColors.Warning,
            title     = "When ARIA Fails and Why",
            body      = "Understanding why ARIA fails helps you fix it faster. Almost all failures fall into five categories:\n\n1. ACCESSIBILITY TREE EMPTY\n  Cause: game, Unity, Flutter, or WebView app with no a11y nodes.\n  Symptom: ARIA types Wait repeatedly, never taps anything useful.\n  Fix: enable SmolVLM + MobileSAM in Modules. Label the key UI elements in the Labeler.\n\n2. WRONG ELEMENT TAPPED\n  Cause: two elements look similar in OCR text (two buttons both labelled \"OK\").\n  Symptom: ARIA keeps tapping the wrong OK.\n  Fix: label the specific element in the Labeler with a unique name.\n\n3. STUCK LOOP (same screen for 5+ steps)\n  Cause: the action isn't registering, or the screen needs a different gesture.\n  Symptom: ARIA keeps clicking the same node, screen doesn't change.\n  Fix: ARIA auto-detects stuck at step 5 (forces Back) and step 8 (aborts). If it keeps happening, record an IRL video of the correct action.\n\n4. LLM PRODUCES INVALID JSON\n  Cause: context too long, model confused by ambiguous prompt.\n  Symptom: action shows Wait with reason 'no action parsed'.\n  Fix: shorten the goal description. Very long goals confuse the 1B model.\n\n5. MODEL NOT LOADED (stub mode)\n  Cause: NDK library not compiled yet.\n  Symptom: every action is a fixed stub output from LlamaEngine.\n  Fix: build the NDK library. See the 'What Needs the NDK Build' page.",
            bullets   = listOf(
                Icons.Default.Visibility    to "Empty a11y tree → enable SmolVLM + SAM + label key elements",
                Icons.Default.AdsClick      to "Wrong tap → label the specific confused element in Labeler",
                Icons.Default.Loop          to "Stuck loop → ARIA auto-breaks at step 5/8; record IRL to fix pattern",
                Icons.Default.Code          to "Invalid JSON → shorten goal text; 1B model has 4096 token limit",
            ),
            callout   = "The Activity screen shows every step's action JSON and reason in real time. Reading the reason field tells you exactly what the model was thinking when it failed.",
            calloutColor = ARIAColors.Warning,
        ),
    )

    var pageIdx by remember { mutableIntStateOf(0) }
    val page = pages[pageIdx]
    val isFirst = pageIdx == 0
    val isLast  = pageIdx == pages.lastIndex

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ARIAColors.Background)
    ) {
        Column(
            modifier            = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {

            // ── Top bar ───────────────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 8.dp),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                IconButton(onClick = onBack) {
                    Icon(
                        Icons.Default.Close,
                        contentDescription = "Close",
                        tint = ARIAColors.Muted
                    )
                }
                Text(
                    "KNOWLEDGE BASE",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color         = ARIAColors.Muted,
                        fontWeight    = FontWeight.Bold,
                        letterSpacing = 1.5.sp,
                        fontFamily    = FontFamily.Monospace,
                    )
                )
                Text(
                    "${pageIdx + 1} / ${pages.size}",
                    style = MaterialTheme.typography.labelSmall.copy(
                        color      = ARIAColors.Muted,
                        fontFamily = FontFamily.Monospace,
                    ),
                    modifier = Modifier.padding(end = 16.dp)
                )
            }

            // ── Progress dots ─────────────────────────────────────────────────
            Row(
                modifier              = Modifier.padding(horizontal = 28.dp),
                horizontalArrangement = Arrangement.spacedBy(5.dp),
                verticalAlignment     = Alignment.CenterVertically,
            ) {
                pages.forEachIndexed { i, _ ->
                    val active = i == pageIdx
                    val done   = i < pageIdx
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(3.dp)
                            .clip(RoundedCornerShape(2.dp))
                            .background(
                                when {
                                    active -> ARIAColors.Primary
                                    done   -> ARIAColors.Primary.copy(alpha = 0.4f)
                                    else   -> ARIAColors.Divider
                                }
                            )
                    )
                }
            }

            Spacer(Modifier.height(20.dp))

            // ── Page content ─────────────────────────────────────────────────
            AnimatedContent(
                targetState = pageIdx,
                transitionSpec = {
                    val forward = targetState > initialState
                    val enter   = fadeIn() + slideInHorizontally { w -> if (forward) w / 4 else -w / 4 }
                    val exit    = fadeOut() + slideOutHorizontally { w -> if (forward) -w / 4 else w / 4 }
                    enter togetherWith exit
                },
                label = "knowledge_page",
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
            ) { idx ->
                val p = pages[idx]
                KnowledgePageContent(p)
            }

            // ── Navigation ────────────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment     = Alignment.CenterVertically,
            ) {
                if (!isFirst) {
                    OutlinedButton(
                        onClick = { pageIdx-- },
                        modifier = Modifier.weight(1f),
                        shape    = RoundedCornerShape(10.dp),
                        colors   = ButtonDefaults.outlinedButtonColors(contentColor = ARIAColors.Muted),
                        border   = androidx.compose.foundation.BorderStroke(1.dp, ARIAColors.Divider),
                    ) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("Back")
                    }
                }

                Button(
                    onClick  = {
                        if (isLast) onBack() else pageIdx++
                    },
                    modifier = Modifier.weight(if (isFirst) 2f else 1f),
                    shape    = RoundedCornerShape(10.dp),
                    colors   = ButtonDefaults.buttonColors(
                        containerColor = if (isLast) ARIAColors.Success else ARIAColors.Primary
                    ),
                ) {
                    Text(
                        if (isLast) "Done" else "Next",
                        fontWeight = FontWeight.Bold,
                        color      = Color.White,
                    )
                    Spacer(Modifier.width(4.dp))
                    Icon(
                        if (isLast) Icons.Default.CheckCircle else Icons.AutoMirrored.Filled.ArrowForward,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint     = Color.White,
                    )
                }
            }
        }
    }
}

// ─── Page body composable ─────────────────────────────────────────────────────

@Composable
private fun KnowledgePageContent(page: KnowledgePage) {
    val scrollState = rememberScrollState()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(horizontal = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {

        // Icon
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(RoundedCornerShape(20.dp))
                .background(page.iconTint.copy(alpha = 0.12f))
                .align(Alignment.CenterHorizontally),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                page.icon,
                contentDescription = null,
                tint     = page.iconTint,
                modifier = Modifier.size(36.dp),
            )
        }

        // Title
        Text(
            page.title,
            style = MaterialTheme.typography.titleLarge.copy(
                color      = ARIAColors.OnSurface,
                fontWeight = FontWeight.Bold,
                textAlign  = TextAlign.Center,
                lineHeight = 30.sp,
            ),
            textAlign = TextAlign.Center,
            modifier  = Modifier.fillMaxWidth(),
        )

        // Body paragraphs
        Text(
            page.body,
            style = MaterialTheme.typography.bodyMedium.copy(
                color      = ARIAColors.Muted,
                lineHeight = 23.sp,
            ),
        )

        // Bullets
        if (page.bullets.isNotEmpty()) {
            Card(
                modifier  = Modifier.fillMaxWidth(),
                shape     = RoundedCornerShape(12.dp),
                colors    = CardDefaults.cardColors(containerColor = ARIAColors.Surface),
                elevation = CardDefaults.cardElevation(0.dp),
            ) {
                Column(
                    modifier            = Modifier.padding(vertical = 12.dp, horizontal = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    page.bullets.forEachIndexed { i, (icon, text) ->
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment     = Alignment.Top,
                        ) {
                            Icon(
                                icon,
                                contentDescription = null,
                                tint     = page.iconTint,
                                modifier = Modifier
                                    .size(16.dp)
                                    .padding(top = 2.dp),
                            )
                            Text(
                                text,
                                style = MaterialTheme.typography.bodySmall.copy(
                                    color      = ARIAColors.OnSurface,
                                    lineHeight = 19.sp,
                                ),
                                modifier = Modifier.weight(1f),
                            )
                        }
                        if (i < page.bullets.lastIndex) {
                            HorizontalDivider(color = ARIAColors.Divider, thickness = 0.5.dp)
                        }
                    }
                }
            }
        }

        // Callout box
        if (page.callout != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(page.calloutColor.copy(alpha = 0.08f))
                    .border(
                        width = 1.dp,
                        color = page.calloutColor.copy(alpha = 0.25f),
                        shape = RoundedCornerShape(10.dp),
                    )
                    .padding(horizontal = 14.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment     = Alignment.Top,
            ) {
                Icon(
                    Icons.Default.Info,
                    contentDescription = null,
                    tint     = page.calloutColor,
                    modifier = Modifier.size(16.dp).padding(top = 1.dp),
                )
                Text(
                    page.callout,
                    style = MaterialTheme.typography.bodySmall.copy(
                        color      = page.calloutColor,
                        lineHeight = 19.sp,
                        fontWeight = FontWeight.Medium,
                    ),
                    modifier = Modifier.weight(1f),
                )
            }
        }

        Spacer(Modifier.height(8.dp))
    }
}
