package com.ariaagent.mobile.core.rl

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import com.ariaagent.mobile.core.ai.LlamaEngine
import com.ariaagent.mobile.core.ai.Sam2Engine
import com.ariaagent.mobile.core.ai.VisionEngine
import com.ariaagent.mobile.core.config.ModuleVisionMode
import com.ariaagent.mobile.core.memory.ExperienceStore
import com.ariaagent.mobile.core.memory.ObjectLabelStore
import com.ariaagent.mobile.core.ocr.OcrEngine
import java.io.File
import java.io.FileOutputStream
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * IrlModule — Inverse Reinforcement Learning from recorded screen videos.
 *
 * Concept: watch what a human expert does on screen, then learn to do it.
 *
 * Pipeline (3-stage fusion):
 *   1. Frame extraction: 1 frame/second via MediaMetadataRetriever
 *   2. Per-frame perception fusion (mirrors AgentLoop's observe step exactly):
 *      a. ML Kit OCR         → raw text visible on screen right now
 *      b. screenHash         → stable hash of OCR text (matches ObjectLabelStore keying)
 *      c. ObjectLabelStore   → labels for THIS specific screen state first
 *                              (fallback: fuzzy-match labels whose ocrText appears in frame)
 *                              Each matched label contributes a toPromptLine() hint.
 *   3. Action inference — two-level, best quality first:
 *      LEVEL 1 (LlamaEngine loaded): LLM gets frame-specific OCR + matched labels
 *        Prompt: "Given screen A [OCR + its labels] changed to screen B [OCR + its labels],
 *                 what single UI action caused this? Reply with JSON only."
 *        Output: {"tool":"Click","node_id":"#1","reason":"..."}
 *      LEVEL 2 (LLM not available): heuristic Jaccard word-diff
 *        Full-screen text change → tap; content shrinkage → back; minor change → scroll
 *   4. Store as ExperienceTuple (result="success", reward=1.0)
 *      Expert demonstrations are treated as correct by definition.
 *      screenSummary includes the per-frame matched labels for traceability.
 *      These tuples feed into LoraTrainer on the next training cycle.
 *
 * Primary use case: screen recordings showing how to use an Android app.
 *   User records their screen, taps "Process video", ARIA learns from the expert.
 *   No manual labeling required — OCR + per-frame labels + LLM do all the interpretation.
 *
 * Limitation: tap coordinates cannot be recovered from OCR alone.
 *   Actions are high-level (tap / swipe / type), not pixel coordinates.
 *   Coordinate binding requires accessibility events (Phase 5.2+).
 *
 * Phase: 5.2 (IRL Processing)
 */
object IrlModule {

    private const val TAG = "IrlModule"
    private const val FRAMES_PER_SECOND = 1
    private const val MIN_OCR_CHANGE_THRESHOLD = 0.15  // 15% word-Jaccard change = new state
    private const val MAX_FRAMES = 300                  // cap at 5-minute videos

    // Max tokens for LLM action inference — keep short to stay fast on Exynos 9611
    private const val LLM_MAX_TOKENS = 96

    // Max label lines injected into the LLM prompt per frame
    private const val MAX_LABELS_PER_PROMPT = 6

    // ── Inverse Dynamics / Chunking constants ─────────────────────────────────
    // Scene-change detection: pixel-diff threshold above which a frame is a "key frame"
    // Expressed as normalised diff (0.0–1.0) averaged over a downsampled 32×32 comparison.
    // 0.08 = 8% average pixel change → scene cut or significant UI transition.
    private const val SCENE_CHANGE_PIXEL_THRESHOLD = 0.08

    // Even without a scene change, force a key frame every N seconds to catch slow transitions.
    private const val FORCE_KEYFRAME_INTERVAL_FRAMES = 5

    // Downscale video frames to this size before pixel-diff comparison — saves CPU on M31.
    private const val DIFF_THUMB_SIZE = 32

    data class IrlResult(
        val videoPath: String,
        val framesProcessed: Int,
        val tuplesExtracted: Int,
        val llmAssistedCount: Int = 0,
        val errorMessage: String = ""
    )

    /**
     * Extract key frames from a video and save them as JPEG files in cacheDir.
     * Used by the annotation UI to show each frame before full IRL processing.
     *
     * @param context   Android context
     * @param videoPath Absolute path to the .mp4 file
     * @return Ordered list of absolute JPEG file paths (indices match frameAnnotations keys).
     *         Returns empty list on failure.
     */
    suspend fun extractKeyFramePaths(context: Context, videoPath: String): List<String> =
        withContext(Dispatchers.IO) {
            val videoFile = File(videoPath)
            if (!videoFile.exists()) return@withContext emptyList()
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(videoPath)
                val durationMs = retriever
                    .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                    ?.toLongOrNull() ?: return@withContext emptyList()

                val frames = extractFrames(retriever, durationMs)
                val paths  = mutableListOf<String>()
                frames.forEachIndexed { idx, bitmap ->
                    val file = File(context.cacheDir, "irl_frame_$idx.jpg")
                    runCatching {
                        FileOutputStream(file).use { out ->
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
                        }
                        paths.add(file.absolutePath)
                    }
                    bitmap.recycle()
                }
                paths
            } catch (e: Exception) {
                Log.e(TAG, "extractKeyFramePaths failed: ${e.message}")
                emptyList()
            } finally {
                retriever.release()
            }
        }

    private data class FrameData(
        val ocrText: String,
        val visionDesc: String,
        val samRegions: List<String>
    )

    /**
     * Process a screen recording and extract expert experience tuples.
     *
     * @param context    Android context
     * @param videoPath  Path to the .mp4 recording file
     * @param taskGoal   Human description of what the video demonstrates
     * @param appPackage Package name of the app being demonstrated
     * @param visionMode   Which models to use for per-frame perception & action inference.
     *                     AUTO: use VisionEngine + Sam2 if loaded, LlamaEngine if loaded.
     *                     LLM_ONLY: skip VisionEngine & Sam2; use only OCR + LlamaEngine.
     *                     VISION_ONLY: use VisionEngine & Sam2; skip LlamaEngine action inference.
     *                     BOTH: force use of both VisionEngine and LlamaEngine (best accuracy).
     */
    suspend fun processVideo(
        context: Context,
        videoPath: String,
        taskGoal: String,
        appPackage: String,
        store: ExperienceStore,
        frameAnnotations: Map<Int, String> = emptyMap(),
        visionMode: ModuleVisionMode = ModuleVisionMode.AUTO,
    ): IrlResult {
        val videoFile = File(videoPath)
        if (!videoFile.exists()) {
            return IrlResult(videoPath, 0, 0, errorMessage = "video_file_not_found")
        }

        Log.i(TAG, "Processing IRL video: $videoPath (goal: $taskGoal, app: $appPackage)")

        val labelStore = ObjectLabelStore.getInstance(context)

        // Pre-load all enriched labels for this app once — used for fuzzy fallback only.
        // Per-frame exact lookup uses getByScreen(); these are the fallback pool.
        val allAppLabels = labelStore.getEnrichedByApp(appPackage)

        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(videoPath)

            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: return IrlResult(videoPath, 0, 0, errorMessage = "cannot_read_duration")

            val frames = extractFrames(retriever, durationMs)
            Log.i(TAG, "Extracted ${frames.size} frames from ${durationMs / 1000}s video")

            val visionReady = VisionEngine.isVisionModelReady(context)
            val sam2Ready   = Sam2Engine.isLoaded()
            val llmReady    = LlamaEngine.isLoaded()

            // ── Resolve effective model usage from visionMode + availability ──────
            // BOTH:         use vision + LLM if both loaded; graceful fallback otherwise
            // VISION_ONLY:  run VisionEngine/Sam2; skip LlamaEngine for action inference
            // LLM_ONLY:     skip VisionEngine/Sam2; use only OCR + LlamaEngine
            // AUTO:         use whatever is available (current default behavior)
            val useVision: Boolean
            val useLlmForAction: Boolean
            when (visionMode) {
                ModuleVisionMode.BOTH -> {
                    useVision       = visionReady
                    useLlmForAction = llmReady
                }
                ModuleVisionMode.VISION_ONLY -> {
                    useVision       = visionReady
                    useLlmForAction = false
                }
                ModuleVisionMode.LLM_ONLY -> {
                    useVision       = false
                    useLlmForAction = llmReady
                }
                ModuleVisionMode.AUTO -> {
                    useVision       = visionReady
                    useLlmForAction = llmReady
                }
            }
            Log.i(TAG, "IRL vision mode=$visionMode → useVision=$useVision useLlm=$useLlmForAction")

            // Stage 2: Per-frame perception fusion — OCR + VisionEngine + Sam2Engine
            // Use a for loop so suspend functions (OcrEngine, VisionEngine) can be awaited.
            //
            // Annotation enrichment: if the user has annotated frame [idx], that annotation
            // is passed to BOTH VisionEngine and Sam2Engine so all three models reason with
            // the user's logic — not just pixel patterns.
            //   • VisionEngine: annotation injected into SmolVLM prompt → annotation-aware description
            //   • Sam2Engine:   annotation used to spatially re-rank SAM peaks toward described area
            val frameDataList = mutableListOf<FrameData>()
            for ((idx, bitmap) in frames.withIndex()) {
                val ocr = runCatching { OcrEngine.run(bitmap) }.getOrDefault("")

                // Per-frame annotation from IRL labeling UI (empty string if user skipped this frame)
                val frameAnnotation = frameAnnotations[idx]?.trim() ?: ""

                val visionDesc = if (useVision) {
                    runCatching {
                        VisionEngine.describe(
                            context     = context,
                            bitmap      = bitmap,
                            goal        = taskGoal,
                            screenHash  = "",
                            annotation  = frameAnnotation,
                            maxTokens   = 80
                        )
                    }.getOrDefault("")
                } else ""

                val samRegions = if (useVision && sam2Ready) {
                    runCatching {
                        Sam2Engine.segment(
                            context        = context,
                            bitmap         = bitmap,
                            topK           = 6,
                            annotationHint = frameAnnotation
                        ).mapIndexed { i, r ->
                            "[tap-${i + 1}] x=%.2f y=%.2f score=%.2f".format(r.normX, r.normY, r.score)
                        }
                    }.getOrDefault(emptyList())
                } else emptyList()

                if (frameAnnotation.isNotEmpty()) Log.d(TAG, "Frame $idx annotation: ${frameAnnotation.take(60)}")
                if (visionDesc.isNotBlank()) Log.d(TAG, "Frame $idx vision: ${visionDesc.take(80)}")
                if (samRegions.isNotEmpty()) Log.d(TAG, "Frame $idx SAM: ${samRegions.size} regions")

                bitmap.recycle()
                frameDataList.add(FrameData(ocrText = ocr, visionDesc = visionDesc, samRegions = samRegions))
            }

            // Stage 3: Build experience tuples from consecutive frame pairs
            val tuples = buildExperienceTuples(
                context              = context,
                frameDataList        = frameDataList,
                labelStore           = labelStore,
                allAppLabels         = allAppLabels,
                taskGoal             = taskGoal,
                appPackage           = appPackage,
                frameAnnotations     = frameAnnotations,
                llmAvailableForAction= useLlmForAction
            )

            tuples.forEach { store.save(it.first) }
            val llmCount = tuples.count { it.second }

            Log.i(TAG, "Saved ${tuples.size} IRL tuples (${llmCount} LLM-inferred, ${tuples.size - llmCount} heuristic)")

            return IrlResult(
                videoPath        = videoPath,
                framesProcessed  = frames.size,
                tuplesExtracted  = tuples.size,
                llmAssistedCount = llmCount
            )

        } catch (e: Exception) {
            Log.e(TAG, "Video processing failed: ${e.message}")
            return IrlResult(videoPath, 0, 0, errorMessage = e.message ?: "unknown_error")
        } finally {
            retriever.release()
        }
    }

    // ─── Frame extraction — Inverse Dynamics Chunking ────────────────────────
    //
    // Strategy: instead of naively extracting 1 frame/sec (which passes every identical
    // "no action" frame to the expensive LLM pipeline), use scene-change detection to
    // select only "key frames" — moments where something visually significant happened
    // on screen (a tap was registered, a new page loaded, a menu appeared).
    //
    // This implements the Inverse Dynamics objective described in Phase 14.2:
    //   "Seeing a menu open implies a tap event occurred at the previous frame's icon location."
    //   The agent analyses consecutive KEY states (not every frame) to infer the action.
    //
    // Two-pass approach:
    //   Pass 1: Extract candidate frames at 2× rate (every 0.5s) — into 32×32 thumbnails only.
    //   Pass 2: For each pair, compute pixel diff. Keep frame[i+1] if:
    //             a) pixel diff ≥ SCENE_CHANGE_PIXEL_THRESHOLD (visible UI change), OR
    //             b) it is a forced keyframe (every FORCE_KEYFRAME_INTERVAL_FRAMES frames)
    //   Pass 3: For kept key frames, load the full 512×512 bitmap for OCR.
    //
    // Effect on Exynos 9611:
    //   A 3-minute video at 2fps = 360 candidates.
    //   ~60–80% of consecutive frames in a YouTube tutorial are static (no UI change).
    //   Key frame selection typically reduces the OCR+LLM load to 60–100 frames — a 4–5× speedup.

    private fun extractFrames(retriever: MediaMetadataRetriever, durationMs: Long): List<Bitmap> {
        val candidateStepUs = 500_000L  // 0.5s between candidates (2× normal rate)
        val durationUs = durationMs * 1000L

        // ── Pass 1: collect thumbnails for diff comparison ─────────────────────
        data class Candidate(val timeUs: Long, val thumb: Bitmap)

        val candidates = mutableListOf<Candidate>()
        var timeUs = 0L
        while (timeUs < durationUs && candidates.size < MAX_FRAMES * 2) {
            val frame = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            if (frame != null) {
                val thumb = Bitmap.createScaledBitmap(frame, DIFF_THUMB_SIZE, DIFF_THUMB_SIZE, false)
                candidates.add(Candidate(timeUs, thumb))
                if (thumb !== frame) frame.recycle()
            }
            timeUs += candidateStepUs
        }

        Log.d(TAG, "Pass 1: ${candidates.size} candidate frames from ${durationMs / 1000}s video")

        // ── Pass 2: select key frames via pixel diff ───────────────────────────
        val keyFrameTimestamps = mutableListOf<Long>()
        if (candidates.isNotEmpty()) {
            keyFrameTimestamps.add(candidates.first().timeUs)  // always include first frame
        }

        for (i in 1 until candidates.size) {
            val prev = candidates[i - 1].thumb
            val curr = candidates[i].thumb
            val diff = computeThumbDiff(prev, curr)
            val isForced = (i % FORCE_KEYFRAME_INTERVAL_FRAMES == 0)

            if (diff >= SCENE_CHANGE_PIXEL_THRESHOLD || isForced) {
                keyFrameTimestamps.add(candidates[i].timeUs)
            }
        }

        // Release thumbnail memory
        candidates.forEach { runCatching { it.thumb.recycle() } }

        Log.i(TAG, "Pass 2: ${keyFrameTimestamps.size} key frames selected " +
            "(${((1.0 - keyFrameTimestamps.size.toDouble() / candidates.size.coerceAtLeast(1)) * 100).toInt()}% skipped)")

        // ── Pass 3: load full 512×512 bitmaps for selected key frames ─────────
        val frames = mutableListOf<Bitmap>()
        for (tsUs in keyFrameTimestamps.take(MAX_FRAMES)) {
            val frame = retriever.getFrameAtTime(tsUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            if (frame != null) {
                val scaled = Bitmap.createScaledBitmap(frame, 512, 512, false)
                frames.add(scaled)
                if (scaled !== frame) frame.recycle()
            }
        }

        return frames
    }

    /**
     * Compute normalised pixel diff between two 32×32 thumbnails.
     * Fast enough to run on every candidate frame pair on Exynos 9611.
     */
    private fun computeThumbDiff(a: Bitmap, b: Bitmap): Double {
        val w = minOf(a.width,  b.width)
        val h = minOf(a.height, b.height)
        if (w <= 0 || h <= 0) return 0.0

        val pA = IntArray(w * h)
        val pB = IntArray(w * h)
        a.getPixels(pA, 0, w, 0, 0, w, h)
        b.getPixels(pB, 0, w, 0, 0, w, h)

        var total = 0L
        for (i in pA.indices) {
            total += Math.abs(((pA[i] shr 16) and 0xFF) - ((pB[i] shr 16) and 0xFF)).toLong()
            total += Math.abs(((pA[i] shr  8) and 0xFF) - ((pB[i] shr  8) and 0xFF)).toLong()
            total += Math.abs(( pA[i]         and 0xFF) - ( pB[i]         and 0xFF)).toLong()
        }
        return total.toDouble() / (3.0 * w * h * 255.0)
    }

    // ─── Screen hash ──────────────────────────────────────────────────────────

    /**
     * Stable hash for ObjectLabelStore lookup — matches ScreenObserver.screenHash() exactly:
     *
     *   combined = "$appPackage|${a11yTree.take(200)}|${ocrText.take(100)}"
     *   hash     = combined.hashCode().toString(16)
     *
     * In IRL we have no accessibility tree (video-only), so a11yTree is passed as "".
     * This means exact screen hash hits only occur for labels that were also annotated
     * without an a11y tree (e.g. pure-OCR sessions or manually created labels).
     * Labels from live agent runs (which have a11y data) are recovered via the
     * fuzzy ocrText substring fallback in resolveFrameLabels().
     */
    fun computeScreenHash(appPackage: String, ocrText: String, a11yTree: String = ""): String {
        val combined = "$appPackage|${a11yTree.take(200)}|${ocrText.take(100)}"
        return combined.hashCode().toString(16)
    }

    // ─── Per-frame label resolution ──────────────────────────────────────────

    /**
     * Find the most relevant ObjectLabels for a given frame's OCR text.
     *
     * Strategy (priority order):
     *   1. Exact screen match: getByScreen(screenHash) — labels that were annotated
     *      on a screen with this exact content signature.
     *   2. OCR fuzzy match: from allAppLabels, keep labels whose stored ocrText
     *      appears (case-insensitive substring) anywhere in the current frame OCR.
     *   3. Top importance fallback: if both above are empty, take the top-N highest
     *      importance labels for the app as weak context hints.
     *
     * Returns at most MAX_LABELS_PER_PROMPT labels, sorted by importance DESC.
     */
    private fun resolveFrameLabels(
        labelStore: ObjectLabelStore,
        allAppLabels: List<ObjectLabelStore.ObjectLabel>,
        appPackage: String,
        ocrText: String
    ): List<ObjectLabelStore.ObjectLabel> {
        val screenHash = computeScreenHash(appPackage, ocrText)

        // 1. Exact screen match
        val exact = labelStore.getByScreen(appPackage, screenHash)
        if (exact.isNotEmpty()) {
            return exact.take(MAX_LABELS_PER_PROMPT)
        }

        // 2. Fuzzy: label's own ocrText appears in this frame's OCR
        val ocrLower = ocrText.lowercase()
        val fuzzy = allAppLabels
            .filter { label ->
                label.ocrText.isNotBlank() && ocrLower.contains(label.ocrText.lowercase().take(30))
            }
            .sortedByDescending { it.importanceScore }
        if (fuzzy.isNotEmpty()) {
            return fuzzy.take(MAX_LABELS_PER_PROMPT)
        }

        // 3. Importance fallback (weak context — no screen match, no ocrText match)
        return allAppLabels.take(MAX_LABELS_PER_PROMPT)
    }

    // ─── Tuple building ───────────────────────────────────────────────────────

    /**
     * Compare consecutive frame data (OCR + vision + SAM regions).
     * For each significant change, resolve per-frame labels, then:
     *   1. Try LLM inference with frame-specific context (OCR + vision + labels + SAM regions)
     *   2. Fall back to heuristic word-diff
     *
     * Returns pairs of (ExperienceTuple, wasLlmInferred).
     */
    private suspend fun buildExperienceTuples(
        context: Context,
        frameDataList: List<FrameData>,
        labelStore: ObjectLabelStore,
        allAppLabels: List<ObjectLabelStore.ObjectLabel>,
        taskGoal: String,
        appPackage: String,
        frameAnnotations: Map<Int, String> = emptyMap(),
        llmAvailableForAction: Boolean = LlamaEngine.isLoaded(),
    ): List<Pair<ExperienceStore.ExperienceTuple, Boolean>> {

        val tuples = mutableListOf<Pair<ExperienceStore.ExperienceTuple, Boolean>>()
        val sessionId = UUID.randomUUID().toString()
        val llmAvailable = llmAvailableForAction

        for (i in 0 until frameDataList.size - 1) {
            val prev = frameDataList[i]
            val next = frameDataList[i + 1]
            val changeRatio = textChangeRatio(prev.ocrText, next.ocrText)

            if (changeRatio < MIN_OCR_CHANGE_THRESHOLD) continue

            // Per-frame label resolution — what elements are visible on frame i?
            val prevLabels = resolveFrameLabels(labelStore, allAppLabels, appPackage, prev.ocrText)
            val nextLabels = resolveFrameLabels(labelStore, allAppLabels, appPackage, next.ocrText)

            val prevLabelBlock = buildLabelBlock(prevLabels)
            val nextLabelBlock = buildLabelBlock(nextLabels)

            // User annotation for this frame (empty string if none provided)
            val annotation = frameAnnotations[i]?.trim() ?: ""

            val (actionJson, usedLlm) = if (llmAvailable) {
                inferActionWithLlm(
                    prevOcr      = prev.ocrText,
                    nextOcr      = next.ocrText,
                    prevLabels   = prevLabelBlock,
                    nextLabels   = nextLabelBlock,
                    prevVision   = prev.visionDesc,
                    prevSam      = prev.samRegions,
                    taskGoal     = taskGoal,
                    appPackage   = appPackage,
                    frameIdx     = i,
                    annotation   = annotation
                )
            } else if (annotation.isNotEmpty()) {
                // Heuristic mode: user has explained this frame — use their logic directly
                // instead of a blind word-diff. Parse the annotation for action keywords
                // so the user's frame description drives the action, not just text delta.
                Pair(inferActionFromAnnotation(annotation, i), false)
            } else {
                Pair(inferActionHeuristic(prev.ocrText, next.ocrText, i), false)
            }

            // screenSummary includes labels, vision, SAM regions and user annotation for traceability
            val screenSummary = buildString {
                append("[IRL frame $i] ")
                if (annotation.isNotEmpty()) append("UserNote: $annotation | ")
                if (prevLabelBlock.isNotEmpty()) append("Labels: $prevLabelBlock | ")
                if (prev.visionDesc.isNotBlank()) append("Vision: ${prev.visionDesc.take(120)} | ")
                if (prev.samRegions.isNotEmpty()) append("SAM: ${prev.samRegions.take(3).joinToString(",")} | ")
                append("OCR: ${prev.ocrText.take(150)}")
            }

            tuples.add(Pair(
                ExperienceStore.ExperienceTuple(
                    appPackage    = appPackage,
                    taskType      = taskGoal.take(60),
                    screenSummary = screenSummary,
                    actionJson    = actionJson,
                    result        = "success",
                    reward        = 1.0,
                    isEdgeCase    = false,
                    sessionId     = sessionId
                ),
                usedLlm
            ))
        }

        return tuples
    }

    /**
     * Format a list of ObjectLabels into a compact prompt block.
     * Uses each label's toPromptLine() which includes name, type, importance,
     * interaction hint, and any LLM-generated reasoning context.
     */
    private fun buildLabelBlock(labels: List<ObjectLabelStore.ObjectLabel>): String {
        if (labels.isEmpty()) return ""
        return labels.joinToString(" | ") { it.toPromptLine() }
    }

    // ─── LLM-assisted action inference ───────────────────────────────────────

    /**
     * Use the on-device LLM to reason about what changed between two frames.
     *
     * Now includes per-frame SmolVLM description and MobileSAM tap-candidate regions
     * alongside OCR text and matched object labels, giving the LLM richer context.
     *
     * Prompt budget: ~700 chars to stay fast on Exynos 9611.
     * If output cannot be parsed as valid JSON, falls back to heuristic.
     *
     * @return (actionJson, usedLlm=true)
     */
    private suspend fun inferActionWithLlm(
        prevOcr: String,
        nextOcr: String,
        prevLabels: String,
        nextLabels: String,
        prevVision: String,
        prevSam: List<String>,
        taskGoal: String,
        appPackage: String,
        frameIdx: Int,
        annotation: String = ""
    ): Pair<String, Boolean> {
        val prevLabelLine    = if (prevLabels.isNotEmpty())  "Elements A: $prevLabels\n" else ""
        val nextLabelLine    = if (nextLabels.isNotEmpty())  "Elements B: $nextLabels\n" else ""
        val visionLine       = if (prevVision.isNotBlank())  "Vision A: ${prevVision.take(120)}\n" else ""
        val samLine          = if (prevSam.isNotEmpty())     "Tap candidates A: ${prevSam.take(4).joinToString(", ")}\n" else ""
        val annotationLine   = if (annotation.isNotBlank())  "Expert note: ${annotation.take(200)}\n" else ""

        val prompt = """<|system|>
You are an Android UI action classifier. Reply with a single JSON object only.
JSON format: {"tool":"Click|Swipe|Type|Back|Scroll","node_id":"#N","reason":"brief"}
If a tap candidate coordinate best explains the action, use TapXY: {"tool":"TapXY","x":0.55,"y":0.32,"reason":"brief"}
<|user|>
App: $appPackage  Goal: $taskGoal
${visionLine}${annotationLine}${samLine}${prevLabelLine}OCR A: ${prevOcr.take(200)}
${nextLabelLine}OCR B: ${nextOcr.take(200)}
What single UI action caused screen A → screen B?
<|assistant|>
""".trimIndent()

        return try {
            val raw = LlamaEngine.infer(prompt, maxTokens = LLM_MAX_TOKENS, onToken = null)
            val json = extractJson(raw) ?: inferActionHeuristic(prevOcr, nextOcr, frameIdx)
            Pair(json, true)
        } catch (e: Exception) {
            Log.w(TAG, "LLM action inference failed at frame $frameIdx: ${e.message}")
            Pair(inferActionHeuristic(prevOcr, nextOcr, frameIdx), false)
        }
    }

    /**
     * Extract the first {...} JSON object from raw LLM output.
     * Returns null if output is not parseable.
     */
    private fun extractJson(raw: String): String? {
        val start = raw.indexOf('{')
        val end   = raw.lastIndexOf('}')
        if (start < 0 || end <= start) return null
        val candidate = raw.substring(start, end + 1).trim()
        return if (candidate.contains("\"tool\"")) candidate else null
    }

    // ─── Annotation-driven action inference (heuristic mode + user context) ──

    /**
     * Produce a JSON action from a user-provided frame annotation.
     *
     * This is used when the LLM is not available but the user HAS annotated the frame.
     * Instead of a blind word-diff, we parse the annotation text for action keywords
     * and emit the best-matching action JSON. Falls back to a "UserAction" custom
     * reason if no keyword is recognised, so at least the annotation is recorded.
     *
     * Keyword mapping (case-insensitive):
     *   tap / click / press / button / icon / link    → Click
     *   type / typed / entered / wrote / input / text → Type
     *   swipe / swiped / scroll / scrolled            → Swipe (up default)
     *   back / went back / returned / navigate back   → Back
     *   long press / hold / held                      → LongPress
     *   otherwise                                     → Click (safest default)
     */
    private fun inferActionFromAnnotation(annotation: String, frameIdx: Int): String {
        val a = annotation.lowercase()
        val reason = annotation.take(120).replace("\"", "'")
        return when {
            a.containsAny("type", "typed", "entered", "wrote", "input", "keyboard", "text field") ->
                """{"tool":"Type","node_id":"#1","text":"","reason":"IRL annotation frame $frameIdx: $reason"}"""
            a.containsAny("swipe left", "swiped left") ->
                """{"tool":"Swipe","direction":"left","reason":"IRL annotation frame $frameIdx: $reason"}"""
            a.containsAny("swipe right", "swiped right") ->
                """{"tool":"Swipe","direction":"right","reason":"IRL annotation frame $frameIdx: $reason"}"""
            a.containsAny("swipe up", "swiped up", "scroll up", "scrolled up") ->
                """{"tool":"Swipe","direction":"up","reason":"IRL annotation frame $frameIdx: $reason"}"""
            a.containsAny("swipe down", "swiped down", "scroll down", "scrolled down") ->
                """{"tool":"Swipe","direction":"down","reason":"IRL annotation frame $frameIdx: $reason"}"""
            a.containsAny("swipe", "scroll") ->
                """{"tool":"Swipe","direction":"up","reason":"IRL annotation frame $frameIdx: $reason"}"""
            a.containsAny("back", "went back", "returned", "navigate back", "press back") ->
                """{"tool":"Back","reason":"IRL annotation frame $frameIdx: $reason"}"""
            a.containsAny("long press", "long-press", "hold", "held", "press and hold") ->
                """{"tool":"LongPress","node_id":"#1","reason":"IRL annotation frame $frameIdx: $reason"}"""
            else ->
                """{"tool":"Click","node_id":"#1","reason":"IRL annotation frame $frameIdx: $reason"}"""
        }
    }

    private fun String.containsAny(vararg keywords: String): Boolean =
        keywords.any { this.contains(it) }

    // ─── Heuristic action inference ───────────────────────────────────────────

    /**
     * Word-level Jaccard diff heuristic. Used when LLM is not loaded AND no annotation.
     *
     *   Both new AND lost words large → full screen transition → tap
     *   More new words than lost      → content appeared → tap
     *   More lost words than new      → content removed → back
     *   Similar count, different words → scroll
     */
    private fun inferActionHeuristic(prevOcr: String, nextOcr: String, frameIdx: Int): String {
        val prevWords = prevOcr.split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()
        val nextWords = nextOcr.split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()
        val newWords  = nextWords - prevWords
        val lostWords = prevWords - nextWords

        return when {
            newWords.size > 5 && lostWords.size > 5 ->
                """{"tool":"Click","node_id":"#1","reason":"IRL heuristic frame $frameIdx: full screen transition"}"""
            newWords.size > lostWords.size ->
                """{"tool":"Click","node_id":"#1","reason":"IRL heuristic frame $frameIdx: new content appeared"}"""
            lostWords.size > newWords.size ->
                """{"tool":"Back","reason":"IRL heuristic frame $frameIdx: content removed, likely back navigation"}"""
            else ->
                """{"tool":"Swipe","direction":"up","reason":"IRL heuristic frame $frameIdx: minimal text change, likely scroll"}"""
        }
    }

    // ─── Jaccard text-change metric ───────────────────────────────────────────

    private fun textChangeRatio(a: String, b: String): Double {
        val wordsA = a.split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()
        val wordsB = b.split(Regex("\\s+")).filter { it.isNotEmpty() }.toSet()
        if (wordsA.isEmpty() && wordsB.isEmpty()) return 0.0
        val intersection = (wordsA intersect wordsB).size
        val union = (wordsA union wordsB).size
        return if (union == 0) 0.0 else 1.0 - (intersection.toDouble() / union)
    }
}
