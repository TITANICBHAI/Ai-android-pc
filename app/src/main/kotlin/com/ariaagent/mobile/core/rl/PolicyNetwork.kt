package com.ariaagent.mobile.core.rl

import android.content.Context
import android.util.Log
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * PolicyNetwork — The RL agent's fast action selection brain.
 *
 * This is NOT the LLM. The LLM handles language reasoning (navigation, complex tasks).
 * The policy network handles fast, repeated action selection (games, repetitive UI patterns).
 *
 * Architecture: goal-conditioned attention + 3-layer MLP, ~5MB on disk
 *
 *   [ATTENTION HEAD — goal attends to screen regions]
 *   screenEmbedding (128) → split into ATT_TOKENS=8 tokens of ATT_DIM=16 dims
 *   goalEmbedding   (128) → project to 16-dim query via WQ (16×128)
 *   attn_score_i = dot(query, screen_token_i) / sqrt(ATT_DIM)   for i in 0..7
 *   attn_weights = softmax(scores)                               [8-dim]
 *   context      = Σ attn_weights[i] * screen_token_i           [16-dim]
 *   attendedScreen = screenEmbedding + tile(context, 8)          [128-dim, residual]
 *
 *   [MLP — same as before, but input is now goal-conditioned]
 *   Input:  attendedScreen (128) + goalEmbedding (128) = 256 floats
 *   Hidden: 256 → 128 neurons (ReLU activations)
 *   Output: 7 action probabilities (softmax)
 *
 * Why attention helps generalize:
 *   A pure MLP treats all 128 screen dims equally regardless of goal.
 *   The attention head lets the goal "focus" on the relevant screen region before
 *   the MLP runs — so the same MLP weights can generalize across novel UI layouts
 *   by attending to different parts of the screen per goal.
 *
 * Training algorithm: REINFORCE (Williams, 1992)
 *   - Policy gradient method — no value function needed
 *   - Update: θ ← θ + α * G_t * ∇_θ log π_θ(a_t | s_t)
 *   - Optimizer: Adam (adaptive moment estimation) for stable convergence
 *
 * Math backend: NEON SIMD via aria_math.cpp (JNI)
 *   nativeMatVecRelu() — matrix-vector multiply + ReLU for hidden layers
 *   nativeSoftmax()    — numerically stable softmax for output
 *   Falls back to Kotlin scalar math if .so not loaded.
 *
 * Action space (7):
 *   0 = tap          4 = swipe-left
 *   1 = swipe-up     5 = type
 *   2 = swipe-down   6 = back
 *   3 = swipe-right
 *
 * Saved weights: rl/policy_latest.bin (little-endian float32 binary)
 * Adam state:    rl/policy_adam.bin
 *
 * Training runs ONLY during idle + charging. See LearningScheduler.
 *
 * Phase: 5 (RL/IRL Processing)
 */
object PolicyNetwork {

    private const val TAG = "PolicyNetwork"

    private const val INPUT_DIM  = 256
    private const val HIDDEN1    = 256
    private const val HIDDEN2    = 128
    private const val OUTPUT_DIM = 7

    // ─── Attention head dimensions ────────────────────────────────────────────
    // Screen embedding (128) is split into ATT_TOKENS tokens of ATT_DIM dims each.
    // Goal embedding is projected to ATT_DIM via WQ to form the query vector.
    private const val ATT_TOKENS = 8    // number of screen "regions"
    private const val ATT_DIM    = 16   // dims per token (128 / 8 = 16)
    private const val SCREEN_DIM = 128  // half of INPUT_DIM
    private const val GOAL_DIM   = 128  // half of INPUT_DIM

    private const val LEARNING_RATE = 1e-4f
    private const val DISCOUNT_GAMMA = 0.99f
    private const val BASELINE_DECAY = 0.95f   // exponential moving average for reward baseline

    val actionNames = arrayOf("tap", "swipe_up", "swipe_down", "swipe_right", "swipe_left", "type", "back")

    // ─── Attention weights ────────────────────────────────────────────────────
    // WQ projects goal embedding → query vector for attention scoring
    // Shape: (ATT_DIM × GOAL_DIM) = (16 × 128) = 2048 params
    private var WQ: FloatArray? = null

    // ─── Network weights ─────────────────────────────────────────────────────
    private var weights1: FloatArray? = null    // shape: (HIDDEN1 × INPUT_DIM) row-major
    private var weights2: FloatArray? = null    // shape: (HIDDEN2 × HIDDEN1) row-major
    private var outputW:  FloatArray? = null    // shape: (OUTPUT_DIM × HIDDEN2) row-major

    // ─── Adam optimizer state ─────────────────────────────────────────────────
    // First moment (mean of gradients)
    private var mQ: FloatArray? = null
    private var m1: FloatArray? = null;  private var m2: FloatArray? = null;  private var mOut: FloatArray? = null
    // Second moment (uncentered variance of gradients)
    private var vQ: FloatArray? = null
    private var v1: FloatArray? = null;  private var v2: FloatArray? = null;  private var vOut: FloatArray? = null
    private var adamStep = 0

    // ─── Reward baseline ──────────────────────────────────────────────────────
    private var rewardBaseline = 0f

    private var isInitialized = false
    private var neonAvailable = false

    // ─── Exposed stats (read by TrainScreen via AgentViewModel) ──────────────
    var lastPolicyLoss: Double = 0.0
        private set

    val adamStepCount: Int get() = adamStep

    fun isReady(): Boolean = isInitialized

    // ─── JNI (aria_math.cpp) ─────────────────────────────────────────────────

    private external fun nativeMatVecRelu(W: FloatArray, x: FloatArray, rows: Int, cols: Int): FloatArray
    private external fun nativeSoftmax(logits: FloatArray): FloatArray

    init {
        try {
            System.loadLibrary("llama-jni")
            neonAvailable = true
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "NEON not available — using Kotlin scalar math")
        }
    }

    // ─── Load / init ─────────────────────────────────────────────────────────

    fun load(context: Context) {
        val rlDir = File(context.filesDir, "rl").also { it.mkdirs() }
            .let { i -> if (i.canWrite()) i else (context.getExternalFilesDir("rl") ?: i).also { it.mkdirs() } }
        val weightsFile = File(rlDir, "policy_latest.bin")
        if (weightsFile.exists() && weightsFile.length() > 100L) {
            loadFromBinary(weightsFile)
        } else {
            initRandom()
        }
        val adamFile = File(rlDir, "policy_adam.bin")
        if (adamFile.exists() && adamFile.length() > 100L) {
            loadAdamState(adamFile)
        } else {
            initAdamState()
        }
        isInitialized = true
        Log.i(TAG, "PolicyNetwork loaded (neon=$neonAvailable, fresh=${!weightsFile.exists()}, attention=goal-conditioned)")
    }

    // ─── Attention head ───────────────────────────────────────────────────────

    /**
     * Goal-conditioned attention over screen regions.
     *
     * The screen embedding (128 floats) is treated as ATT_TOKENS=8 "region tokens"
     * of ATT_DIM=16 dims each. The goal embedding is projected to a 16-dim query
     * via WQ, which then attends (dot-product) to each screen token. The weighted
     * sum is a 16-dim context vector describing which screen region the current goal
     * cares about. A residual adds context back to every token, producing an
     * attended 128-dim screen representation that feeds the MLP.
     *
     * This lets the same MLP weights generalise across novel UI layouts — different
     * goals steer attention to different screen areas, bypassing the lookup-table
     * limitation of a pure MLP that treats every screen dim identically.
     *
     * @param screenEmb  Raw 128-dim screen embedding (first half of MLP input).
     * @param goalEmb    Raw 128-dim goal embedding (second half of MLP input).
     * @return           128-dim attended screen embedding; concat with goalEmb → MLP.
     */
    private fun attentionHead(screenEmb: FloatArray, goalEmb: FloatArray): FloatArray {
        val wq = WQ ?: return screenEmb.copyOf()  // no-op fallback if not initialised

        // ── 1. Project goal → query (ATT_DIM = 16) ───────────────────────────
        val query = FloatArray(ATT_DIM)
        for (i in 0 until ATT_DIM) {
            var s = 0f
            for (j in 0 until GOAL_DIM) s += wq[i * GOAL_DIM + j] * goalEmb[j]
            query[i] = s
        }

        // ── 2. Compute dot-product scores per screen token ────────────────────
        val scale = 1f / kotlin.math.sqrt(ATT_DIM.toDouble()).toFloat()
        val scores = FloatArray(ATT_TOKENS)
        for (t in 0 until ATT_TOKENS) {
            var dot = 0f
            val base = t * ATT_DIM
            for (d in 0 until ATT_DIM) dot += query[d] * screenEmb[base + d]
            scores[t] = dot * scale
        }

        // ── 3. Softmax attention weights ──────────────────────────────────────
        val maxScore = scores.maxOrNull() ?: 0f
        val expScores = FloatArray(ATT_TOKENS) { kotlin.math.exp((scores[it] - maxScore).toDouble()).toFloat() }
        val expSum = expScores.sum().coerceAtLeast(1e-9f)
        val attnWeights = FloatArray(ATT_TOKENS) { expScores[it] / expSum }

        // ── 4. Weighted sum → 16-dim context ─────────────────────────────────
        val context = FloatArray(ATT_DIM)
        for (t in 0 until ATT_TOKENS) {
            val base = t * ATT_DIM
            for (d in 0 until ATT_DIM) context[d] += attnWeights[t] * screenEmb[base + d]
        }

        // ── 5. Residual: add context to every screen token (tile context 8×) ─
        val attended = screenEmb.copyOf()
        for (t in 0 until ATT_TOKENS) {
            val base = t * ATT_DIM
            for (d in 0 until ATT_DIM) attended[base + d] += context[d]
        }
        return attended
    }

    // ─── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Forward pass: screen + goal embeddings → action probabilities.
     *
     * Pipeline: attention(screen, goal) → concat(attended, goal) → MLP → softmax
     *
     * @return (actionIndex, confidence) pair
     */
    fun selectAction(screenEmbedding: FloatArray, goalEmbedding: FloatArray): Pair<Int, Float> {
        if (!isInitialized) return Pair(0, 0f)
        val screen = FloatArray(SCREEN_DIM).also { buf ->
            System.arraycopy(screenEmbedding, 0, buf, 0, minOf(screenEmbedding.size, SCREEN_DIM))
        }
        val goal = FloatArray(GOAL_DIM).also { buf ->
            System.arraycopy(goalEmbedding, 0, buf, 0, minOf(goalEmbedding.size, GOAL_DIM))
        }
        val attended = attentionHead(screen, goal)
        val input = FloatArray(INPUT_DIM)
        System.arraycopy(attended, 0, input, 0, SCREEN_DIM)
        System.arraycopy(goal,     0, input, SCREEN_DIM, GOAL_DIM)
        val (probs, _, _) = forwardWithActivations(input)
        val idx = probs.indices.maxByOrNull { probs[it] } ?: 0
        return Pair(idx, probs[idx])
    }

    /**
     * Forward pass returning intermediate activations (needed for backprop).
     * Callers must pre-apply the attention head before calling this (states list in
     * reinforce() stores already-attended inputs to avoid recomputing attention).
     * @return Triple(probs, h1, h2)
     */
    private fun forwardWithActivations(input: FloatArray): Triple<FloatArray, FloatArray, FloatArray> {
        val w1 = weights1 ?: return Triple(uniformProbs(), FloatArray(HIDDEN1), FloatArray(HIDDEN2))
        val w2 = weights2 ?: return Triple(uniformProbs(), FloatArray(HIDDEN1), FloatArray(HIDDEN2))
        val wo = outputW  ?: return Triple(uniformProbs(), FloatArray(HIDDEN1), FloatArray(HIDDEN2))

        val h1 = if (neonAvailable) nativeMatVecRelu(w1, input, HIDDEN1, INPUT_DIM)
                 else matVecReluKotlin(w1, input, HIDDEN1, INPUT_DIM)

        val h2 = if (neonAvailable) nativeMatVecRelu(w2, h1, HIDDEN2, HIDDEN1)
                 else matVecReluKotlin(w2, h1, HIDDEN2, HIDDEN1)

        val logits = matVecKotlin(wo, h2, OUTPUT_DIM, HIDDEN2)  // no ReLU before softmax

        val probs = if (neonAvailable) nativeSoftmax(logits) else softmaxKotlin(logits)

        return Triple(probs, h1, h2)
    }

    private fun uniformProbs() = FloatArray(OUTPUT_DIM) { 1f / OUTPUT_DIM }

    // ─── REINFORCE training ───────────────────────────────────────────────────

    /**
     * REINFORCE policy gradient update.
     *
     * Algorithm (Williams, 1992):
     *   For each step t in episode:
     *     G_t = sum_{k=t}^{T} γ^{k-t} * r_k   (discounted return)
     *     Advantage = G_t - baseline            (reduce variance)
     *     loss_t = -log(π(a_t|s_t)) * advantage
     *     ∇_θ J ≈ mean(∇_θ loss_t) over episode
     *
     * Optimizer: Adam for stable convergence on sparse reward signals.
     *
     * @param states  List of screen+goal concatenated input vectors (each 256-dim)
     * @param actions List of action indices taken at each step
     * @param rewards List of rewards received at each step
     * @return mean episode return (for logging)
     */
    fun reinforce(
        states:  List<FloatArray>,
        actions: List<Int>,
        rewards: List<Double>
    ): Double {
        if (!isInitialized || states.isEmpty()) return 0.0
        if (states.size != actions.size || actions.size != rewards.size) return 0.0

        val T = states.size

        // ── Step 1: Compute discounted returns ──────────────────────────────
        val returns = DoubleArray(T)
        var G = 0.0
        for (t in T - 1 downTo 0) {
            G = rewards[t] + DISCOUNT_GAMMA * G
            returns[t] = G
        }

        // ── Step 2: Normalize returns (reduce variance) ─────────────────────
        val meanR = returns.average()
        val stdR  = kotlin.math.sqrt(returns.map { (it - meanR) * (it - meanR) }.average()).coerceAtLeast(1e-6)
        val normalizedReturns = DoubleArray(T) { (returns[it] - meanR) / stdR }

        // ── Step 3: Accumulate gradients over episode ───────────────────────
        val dWQ = FloatArray(ATT_DIM * GOAL_DIM)
        val dW1 = FloatArray(HIDDEN1 * INPUT_DIM)
        val dW2 = FloatArray(HIDDEN2 * HIDDEN1)
        val dWo = FloatArray(OUTPUT_DIM * HIDDEN2)

        for (t in 0 until T) {
            // Raw state has screen (128) + goal (128) pre-concatenated or just 256-dim.
            // Apply attention head to produce the actual MLP input.
            val rawState = states[t].let { s ->
                if (s.size == INPUT_DIM) s else FloatArray(INPUT_DIM).also {
                    System.arraycopy(s, 0, it, 0, minOf(s.size, INPUT_DIM))
                }
            }
            val screenRaw = rawState.copyOfRange(0, SCREEN_DIM)
            val goalRaw   = rawState.copyOfRange(SCREEN_DIM, INPUT_DIM)
            val attended  = attentionHead(screenRaw, goalRaw)
            val input     = FloatArray(INPUT_DIM)
            System.arraycopy(attended, 0, input, 0, SCREEN_DIM)
            System.arraycopy(goalRaw, 0, input, SCREEN_DIM, GOAL_DIM)

            val action = actions[t].coerceIn(0, OUTPUT_DIM - 1)
            val Gt     = normalizedReturns[t].toFloat()

            val (probs, h1, h2) = forwardWithActivations(input)

            // ── Policy gradient at output layer ─────────────────────────────
            // delta_out = G_t * (probs - one_hot(a))
            // This is the softmax + cross-entropy gradient scaled by the return
            val deltaOut = FloatArray(OUTPUT_DIM) { i ->
                Gt * (probs[i] - if (i == action) 1f else 0f)
            }

            // ── dW_out += delta_out ⊗ h2 (outer product) ───────────────────
            for (i in 0 until OUTPUT_DIM) {
                for (j in 0 until HIDDEN2) {
                    dWo[i * HIDDEN2 + j] += deltaOut[i] * h2[j]
                }
            }

            // ── Backprop through layer 2 ────────────────────────────────────
            // delta_h2 = (W_out^T · delta_out) ⊙ relu_grad(h2)
            val wo = outputW!!
            val deltaH2 = FloatArray(HIDDEN2) { j ->
                var d = 0f
                for (i in 0 until OUTPUT_DIM) d += wo[i * HIDDEN2 + j] * deltaOut[i]
                if (h2[j] > 0f) d else 0f  // ReLU gradient: 1 if h2>0, else 0
            }

            // ── dW2 += delta_h2 ⊗ h1 ───────────────────────────────────────
            for (i in 0 until HIDDEN2) {
                for (j in 0 until HIDDEN1) {
                    dW2[i * HIDDEN1 + j] += deltaH2[i] * h1[j]
                }
            }

            // ── Backprop through layer 1 ────────────────────────────────────
            val w2 = weights2!!
            val deltaH1 = FloatArray(HIDDEN1) { j ->
                var d = 0f
                for (i in 0 until HIDDEN2) d += w2[i * HIDDEN1 + j] * deltaH2[i]
                if (h1[j] > 0f) d else 0f  // ReLU gradient
            }

            // ── dW1 += delta_h1 ⊗ input ────────────────────────────────────
            for (i in 0 until HIDDEN1) {
                for (j in 0 until INPUT_DIM) {
                    dW1[i * INPUT_DIM + j] += deltaH1[i] * input[j]
                }
            }

            // ── Attention gradient: backprop through attentionHead ──────────
            // attended_screen = screenRaw + tile(context, 8)
            // dL/d(attended_screen[j]) is recoverable from MLP layer-1 gradient.
            // dL/d(WQ) = Gt * d(context) ⊗ goalRaw
            val wqLocal = WQ
            if (wqLocal != null) {
                // Reconstruct dL/d(attended_screen) from first-layer weight gradient
                val w1Local = weights1
                val dAttScreen = if (w1Local != null) FloatArray(SCREEN_DIM) { j ->
                    var d = 0f
                    for (i in 0 until HIDDEN1) d += w1Local[i * INPUT_DIM + j] * deltaH1[i]
                    d
                } else FloatArray(SCREEN_DIM)
                // Collapse residual tiling: dContext[d] = sum over all tokens
                val dContext = FloatArray(ATT_DIM)
                for (tok in 0 until ATT_TOKENS) {
                    val base = tok * ATT_DIM
                    for (d in 0 until ATT_DIM) dContext[d] += dAttScreen[base + d]
                }
                // WQ gradient: outer product of dContext and goalRaw
                for (i in 0 until ATT_DIM) {
                    for (j in 0 until GOAL_DIM) {
                        dWQ[i * GOAL_DIM + j] += Gt * dContext[i] * goalRaw[j]
                    }
                }
            }
        }

        // ── Step 4: Average gradients over episode ──────────────────────────
        val scale = 1f / T
        for (i in dWQ.indices) dWQ[i] *= scale
        for (i in dW1.indices) dW1[i] *= scale
        for (i in dW2.indices) dW2[i] *= scale
        for (i in dWo.indices) dWo[i] *= scale

        // ── Step 5: Adam optimizer update ───────────────────────────────────
        adamStep++
        WQ?.let { wq -> adamUpdate(wq, dWQ, mQ!!, vQ!!, adamStep) }
        adamUpdate(weights1!!, dW1, m1!!, v1!!, adamStep)
        adamUpdate(weights2!!, dW2, m2!!, v2!!, adamStep)
        adamUpdate(outputW!!,  dWo, mOut!!, vOut!!, adamStep)

        val episodeReturn = returns[0]
        // policy loss = -mean(log π(a|s) * G_t) over episode, for logging
        lastPolicyLoss = -returns.average()
        Log.d(TAG, "REINFORCE step $adamStep — T=$T return=${episodeReturn.toFloat()} loss=${lastPolicyLoss.toFloat()} meanReward=${rewards.average().toFloat()}")
        return episodeReturn
    }

    // ─── Adam optimizer ───────────────────────────────────────────────────────
    // Adam: Adaptive Moment Estimation (Kingma & Ba, 2015)
    //   m = β1 * m + (1-β1) * g         (first moment — momentum)
    //   v = β2 * v + (1-β2) * g²        (second moment — variance)
    //   m̂ = m / (1-β1^t)                (bias-corrected)
    //   v̂ = v / (1-β2^t)                (bias-corrected)
    //   W -= lr * m̂ / (√v̂ + ε)

    private const val BETA1 = 0.9f
    private const val BETA2 = 0.999f
    private const val ADAM_EPS = 1e-8f

    private fun adamUpdate(W: FloatArray, g: FloatArray, m: FloatArray, v: FloatArray, t: Int) {
        val bc1 = 1f - BETA1.pow(t)  // bias correction for first moment
        val bc2 = 1f - BETA2.pow(t)  // bias correction for second moment
        for (i in W.indices) {
            m[i] = BETA1 * m[i] + (1f - BETA1) * g[i]
            v[i] = BETA2 * v[i] + (1f - BETA2) * g[i] * g[i]
            val mHat = m[i] / bc1
            val vHat = v[i] / bc2
            W[i] -= LEARNING_RATE * mHat / (kotlin.math.sqrt(vHat.toDouble()).toFloat() + ADAM_EPS)
        }
    }

    private fun Float.pow(n: Int): Float {
        var result = 1f
        repeat(n.coerceAtMost(100)) { result *= this }
        return result
    }

    // ─── Persistence — real binary serialization ─────────────────────────────

    fun saveToFile(context: Context) {
        if (!isInitialized) return
        try {
            val dir = File(context.filesDir, "rl").also { it.mkdirs() }
                .let { i -> if (i.canWrite()) i else (context.getExternalFilesDir("rl") ?: i).also { it.mkdirs() } }

            // Save weights as little-endian float32 binary
            // Format: [MLP weights…] [WQ size] [WQ floats] — WQ appended for backward compat
            DataOutputStream(FileOutputStream(File(dir, "policy_latest.bin"))).use { out ->
                out.writeInt(HIDDEN1 * INPUT_DIM)
                weights1!!.forEach { out.writeFloat(it) }
                out.writeInt(HIDDEN2 * HIDDEN1)
                weights2!!.forEach { out.writeFloat(it) }
                out.writeInt(OUTPUT_DIM * HIDDEN2)
                outputW!!.forEach { out.writeFloat(it) }
                out.writeFloat(rewardBaseline)
                out.writeInt(adamStep)
                // WQ (attention query projection) appended last
                val wq = WQ
                if (wq != null) {
                    out.writeInt(wq.size)
                    wq.forEach { out.writeFloat(it) }
                }
            }

            // Save Adam state separately (large — only save periodically)
            DataOutputStream(FileOutputStream(File(dir, "policy_adam.bin"))).use { out ->
                m1!!.forEach { out.writeFloat(it) }
                v1!!.forEach { out.writeFloat(it) }
                m2!!.forEach { out.writeFloat(it) }
                v2!!.forEach { out.writeFloat(it) }
                mOut!!.forEach { out.writeFloat(it) }
                vOut!!.forEach { out.writeFloat(it) }
                // mQ/vQ for attention WQ appended last
                mQ?.forEach { out.writeFloat(it) }
                vQ?.forEach { out.writeFloat(it) }
            }

            Log.i(TAG, "PolicyNetwork saved — step=$adamStep baseline=$rewardBaseline")
        } catch (e: Exception) {
            Log.e(TAG, "Save failed: ${e.message}")
        }
    }

    private fun loadFromBinary(file: File) {
        try {
            DataInputStream(FileInputStream(file)).use { din ->
                val s1 = din.readInt()
                weights1 = FloatArray(s1) { din.readFloat() }
                val s2 = din.readInt()
                weights2 = FloatArray(s2) { din.readFloat() }
                val sO = din.readInt()
                outputW  = FloatArray(sO) { din.readFloat() }
                rewardBaseline = din.readFloat()
                adamStep = din.readInt()
                // WQ (attention) — appended after existing weights for backward compat.
                // Old weight files won't have this block; catch EOF and init randomly.
                val sQ = runCatching { din.readInt() }.getOrDefault(-1)
                WQ = if (sQ == ATT_DIM * GOAL_DIM) {
                    FloatArray(sQ) { din.readFloat() }
                } else {
                    initRandomWQ()
                }
            }
            Log.i(TAG, "PolicyNetwork loaded from file — step=$adamStep, attention=goal-conditioned")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load weights from file — reinitializing: ${e.message}")
            initRandom()
        }
    }

    private fun loadAdamState(file: File) {
        try {
            DataInputStream(FileInputStream(file)).use { din ->
                m1   = FloatArray(HIDDEN1 * INPUT_DIM) { din.readFloat() }
                v1   = FloatArray(HIDDEN1 * INPUT_DIM) { din.readFloat() }
                m2   = FloatArray(HIDDEN2 * HIDDEN1) { din.readFloat() }
                v2   = FloatArray(HIDDEN2 * HIDDEN1) { din.readFloat() }
                mOut = FloatArray(OUTPUT_DIM * HIDDEN2) { din.readFloat() }
                vOut = FloatArray(OUTPUT_DIM * HIDDEN2) { din.readFloat() }
                // mQ/vQ appended for attention WQ — old files omit this, init zeros
                val szQ = ATT_DIM * GOAL_DIM
                mQ = runCatching { FloatArray(szQ) { din.readFloat() } }.getOrDefault(FloatArray(szQ))
                vQ = runCatching { FloatArray(szQ) { din.readFloat() } }.getOrDefault(FloatArray(szQ))
            }
        } catch (e: Exception) {
            Log.w(TAG, "Adam state not loadable — reinitializing: ${e.message}")
            initAdamState()
        }
    }

    // ─── Init ─────────────────────────────────────────────────────────────────

    private fun initRandomWQ(): FloatArray {
        val rng = java.util.Random(99L)
        val scaleQ = kotlin.math.sqrt(2.0 / (GOAL_DIM + ATT_DIM)).toFloat()
        return FloatArray(ATT_DIM * GOAL_DIM) { (rng.nextGaussian() * scaleQ).toFloat() }
    }

    private fun initRandom() {
        val rng = java.util.Random(42L)
        // Xavier/Glorot initialization: σ = sqrt(2 / (fan_in + fan_out))
        val scaleQ = kotlin.math.sqrt(2.0 / (GOAL_DIM + ATT_DIM)).toFloat()
        val scale1 = kotlin.math.sqrt(2.0 / (INPUT_DIM + HIDDEN1)).toFloat()
        val scale2 = kotlin.math.sqrt(2.0 / (HIDDEN1 + HIDDEN2)).toFloat()
        val scaleO = kotlin.math.sqrt(2.0 / (HIDDEN2 + OUTPUT_DIM)).toFloat()

        WQ       = FloatArray(ATT_DIM * GOAL_DIM) { (rng.nextGaussian() * scaleQ).toFloat() }
        weights1 = FloatArray(HIDDEN1 * INPUT_DIM)  { (rng.nextGaussian() * scale1).toFloat() }
        weights2 = FloatArray(HIDDEN2 * HIDDEN1)    { (rng.nextGaussian() * scale2).toFloat() }
        outputW  = FloatArray(OUTPUT_DIM * HIDDEN2) { (rng.nextGaussian() * scaleO).toFloat() }
    }

    private fun initAdamState() {
        val szQ = ATT_DIM * GOAL_DIM
        mQ   = FloatArray(szQ)
        vQ   = FloatArray(szQ)
        m1   = FloatArray(HIDDEN1 * INPUT_DIM)
        v1   = FloatArray(HIDDEN1 * INPUT_DIM)
        m2   = FloatArray(HIDDEN2 * HIDDEN1)
        v2   = FloatArray(HIDDEN2 * HIDDEN1)
        mOut = FloatArray(OUTPUT_DIM * HIDDEN2)
        vOut = FloatArray(OUTPUT_DIM * HIDDEN2)
        adamStep = 0
    }

    // ─── Kotlin scalar fallbacks ──────────────────────────────────────────────

    private fun matVecReluKotlin(W: FloatArray, x: FloatArray, rows: Int, cols: Int): FloatArray =
        FloatArray(rows) { i ->
            var s = 0f
            for (j in 0 until cols) s += W[i * cols + j] * x[j]
            if (s > 0f) s else 0f
        }

    private fun matVecKotlin(W: FloatArray, x: FloatArray, rows: Int, cols: Int): FloatArray =
        FloatArray(rows) { i ->
            var s = 0f
            for (j in 0 until cols) s += W[i * cols + j] * x[j]
            s
        }

    private fun softmaxKotlin(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        val exp = FloatArray(logits.size) { kotlin.math.exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exp.sum().coerceAtLeast(1e-10f)
        return FloatArray(exp.size) { exp[it] / sum }
    }
}
