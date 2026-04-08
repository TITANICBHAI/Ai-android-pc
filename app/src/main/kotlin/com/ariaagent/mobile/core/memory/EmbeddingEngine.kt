package com.ariaagent.mobile.core.memory

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * EmbeddingEngine — On-device text embedding for experience retrieval.
 *
 * Model: MiniLM-L6-v2 (all-MiniLM-L6-v2) ONNX export from sentence-transformers
 *   - Disk size: ~23MB
 *   - Output: 384-dimensional sentence embedding (mean-pooled, L2-normalised)
 *   - ONNX Runtime Android: ~50-80ms per inference on Exynos 9611
 *   - Model file: internal storage → models/minilm-l6-v2.onnx
 *
 * Input format (BERT-style):
 *   input_ids      — int64[1, 128]  (whitespace-split word hashes, clamped to vocab 0-30521)
 *   attention_mask — int64[1, 128]  (1 for real tokens, 0 for padding)
 *   token_type_ids — int64[1, 128]  (all zeros — single-sentence input)
 *
 * Output format:
 *   last_hidden_state — float32[1, 128, 384]  (per-token embeddings)
 *   Mean-pooled over non-padding positions → 384-dim sentence vector → L2-normalised.
 *
 * Math backend: NEON SIMD via aria_math.cpp (JNI)
 *   - nativeCosineSimilarity() — 384-dim dot product + L2 norms, ~4× faster than Kotlin loop
 *   - nativeL2Normalize()      — in-place unit vector normalization
 *   - Falls back to Kotlin scalar math if .so not loaded (same result, ~4× slower)
 *
 * Fallback when MiniLM not downloaded yet:
 *   deterministic hash embedding — same text always produces same vector,
 *   cosine similarities are semantically meaningful within session.
 *
 * Phase: 4 (Data Collection)
 */
object EmbeddingEngine {

    private const val TAG = "EmbeddingEngine"
    private const val EMBEDDING_DIM = 384
    private const val SEQ_LEN = 128
    private const val MODEL_FILENAME = "minilm-l6-v2.onnx"

    // ONNX Runtime session — loaded lazily on first embed() call when model file is present.
    // Null while model not yet downloaded or if ORT is unavailable.
    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null
    private var neonAvailable = false

    private fun modelsDir(context: Context): File {
        val internal = File(context.filesDir, "models").also { it.mkdirs() }
        if (internal.canWrite()) return internal
        return (context.getExternalFilesDir("models") ?: internal).also { it.mkdirs() }
    }

    fun isModelAvailable(context: Context): Boolean =
        File(modelsDir(context), MODEL_FILENAME).exists()

    // ─── JNI (aria_math.cpp) ─────────────────────────────────────────────────

    private external fun nativeCosineSimilarity(a: FloatArray, b: FloatArray): Float
    private external fun nativeL2Normalize(v: FloatArray): FloatArray
    private external fun nativeDotProduct(a: FloatArray, b: FloatArray): Float

    init {
        try {
            System.loadLibrary("llama-jni")
            neonAvailable = true
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "NEON math not available — using Kotlin fallback")
        }
    }

    // ─── Public API ───────────────────────────────────────────────────────────

    /**
     * Embed text into a 384-dim L2-normalised unit vector.
     * Uses MiniLM ONNX (via ONNX Runtime Android) if available, falls back to hash embedding.
     */
    fun embed(context: Context, text: String): FloatArray {
        val modelFile = File(modelsDir(context), MODEL_FILENAME)
        if (modelFile.exists() && ortSession == null) initOrtSession(modelFile)
        // Load WordPiece vocab whenever it is present — independent of the ONNX model
        // so the tokenizer quality improves as soon as vocab.txt is downloaded.
        if (wordPieceVocab == null) loadVocab(context)

        val raw = if (ortSession != null) runOnnxInference(text) else hashEmbedFallback(text)

        return if (neonAvailable) {
            nativeL2Normalize(raw)
        } else {
            normalizeL2Kotlin(raw)
        }
    }

    /**
     * Retrieve top-K most similar past experiences for LLM prompt injection.
     *
     * Hybrid scoring = 0.7 × cosine(query_embedding, doc_embedding)
     *                + 0.3 × BM25_keyword(query_terms, doc_text)
     *
     * Pure embedding retrieval misses exact keyword matches (e.g. a rare app name
     * or package ID in the goal). BM25 keyword scoring adds a sparse signal that
     * boosts documents containing the same words as the query, improving precision
     * for short or highly specific goals ("open com.whatsapp and send 'Hi'").
     *
     * BM25 parameters: k1=1.5, b=0.75 (standard defaults). The corpus here is
     * small (≤50 docs), so per-query TF/IDF is recomputed inline without overhead.
     */
    fun retrieve(
        context: Context,
        query: String,
        store: ExperienceStore,
        topK: Int = 3
    ): List<String> {
        val tuples = store.getUntrainedSuccesses(limit = 50)
        if (tuples.isEmpty()) return emptyList()

        val queryVec   = embed(context, query)
        val queryTerms = tokeniseForBm25(query)
        val docs       = tuples.map { it.screenSummary + " " + it.taskType + " " + it.appPackage }
        val avgDocLen  = docs.sumOf { it.length }.toFloat() / docs.size

        return tuples
            .mapIndexed { idx, tuple ->
                val docVec   = embed(context, tuple.screenSummary)
                val cosScore = cosineSimilarity(queryVec, docVec)
                val bm25     = bm25Score(queryTerms, docs[idx], avgDocLen)
                val hybrid   = 0.7f * cosScore + 0.3f * bm25
                Pair(tuple, hybrid)
            }
            .sortedByDescending { it.second }
            .take(topK)
            .map { (tuple, score) ->
                val simPct = (score * 100).toInt()
                "[$simPct% match] App=${tuple.appPackage} Task=${tuple.taskType} " +
                "Action=${tuple.actionJson.take(60)} Result=${tuple.result}"
            }
    }

    // ─── BM25 keyword scoring ─────────────────────────────────────────────────

    /**
     * Tokenise text for BM25: lowercase, split on non-alphanumeric,
     * filter very short tokens (< 2 chars) and a tiny stop-list.
     */
    private fun tokeniseForBm25(text: String): List<String> {
        val stops = setOf("the", "a", "an", "is", "in", "to", "on", "at", "and", "or", "of", "for")
        return text.lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { it.length >= 2 && it !in stops }
    }

    /**
     * BM25 relevance score for [queryTerms] against a single [doc].
     * Returns a value in [0, ~3] (unnormalized; used only for relative ranking).
     * k1 = 1.5, b = 0.75 (standard BM25 defaults).
     */
    private fun bm25Score(queryTerms: List<String>, doc: String, avgDocLen: Float): Float {
        if (queryTerms.isEmpty()) return 0f
        val k1 = 1.5f
        val b  = 0.75f
        val docTokens = tokeniseForBm25(doc)
        val docLen    = docTokens.size.toFloat().coerceAtLeast(1f)
        val tf        = docTokens.groupingBy { it }.eachCount()

        return queryTerms.sumOf { term ->
            val termFreq = tf[term]?.toFloat() ?: 0f
            if (termFreq == 0f) return@sumOf 0.0
            // Simplified BM25 without IDF (corpus too small for reliable IDF)
            val numerator   = termFreq * (k1 + 1f)
            val denominator = termFreq + k1 * (1f - b + b * docLen / avgDocLen)
            (numerator / denominator).toDouble()
        }.toFloat()
    }

    /**
     * Retrieve top-K most relevant object labels for a given screen query.
     * Used by AgentLoop + PromptBuilder to find relevant human annotations.
     */
    fun retrieveLabels(
        context: Context,
        query: String,
        labelStore: ObjectLabelStore,
        appPackage: String,
        topK: Int = 5
    ): List<ObjectLabelStore.ObjectLabel> {
        val allLabels = labelStore.getEnrichedByApp(appPackage)
        if (allLabels.isEmpty()) return emptyList()

        val queryVec = embed(context, query)

        return allLabels
            .map { label ->
                val labelText = "${label.name} ${label.context} ${label.ocrText}"
                val docVec = embed(context, labelText)
                val score = cosineSimilarity(queryVec, docVec)
                Pair(label, score)
            }
            .sortedByDescending { it.second }
            .take(topK)
            .map { it.first }
    }

    // ─── Similarity math ──────────────────────────────────────────────────────

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        return if (neonAvailable) {
            nativeCosineSimilarity(a, b)
        } else {
            cosineSimilarityKotlin(a, b)
        }
    }

    // ─── ONNX Runtime inference ───────────────────────────────────────────────

    private fun initOrtSession(modelFile: File) {
        try {
            val env = OrtEnvironment.getEnvironment()
            ortEnvironment = env
            ortSession = env.createSession(modelFile.absolutePath)
            Log.i(TAG, "ONNX Runtime session created for ${modelFile.name}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ORT session: ${e.message}")
            ortSession = null
        }
    }

    /**
     * Run BERT-style ONNX inference for the given text.
     * Input:  input_ids, attention_mask, token_type_ids — int64[1, 128]
     * Output: last_hidden_state — float32[1, 128, 384] → mean-pool → 384 floats
     */
    private fun runOnnxInference(text: String): FloatArray {
        val session = ortSession ?: return hashEmbedFallback(text)
        val env     = ortEnvironment ?: return hashEmbedFallback(text)

        try {
            val tokens        = tokenize(text)
            val inputIds      = LongArray(SEQ_LEN) { if (it < tokens.size) tokens[it] else 0L }
            val attentionMask = LongArray(SEQ_LEN) { if (it < tokens.size) 1L else 0L }
            val tokenTypeIds  = LongArray(SEQ_LEN) { 0L }
            val shape         = longArrayOf(1L, SEQ_LEN.toLong())

            val inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds),      shape)
            val attMaskTensor  = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape)
            val tokTypeTensor  = OnnxTensor.createTensor(env, LongBuffer.wrap(tokenTypeIds),  shape)

            val inputs = LinkedHashMap<String, OnnxTensor>()
            inputs["input_ids"]      = inputIdsTensor
            inputs["attention_mask"] = attMaskTensor
            inputs["token_type_ids"] = tokTypeTensor

            session.run(inputs).use { result ->
                val outputName = session.outputNames.first()
                val rawValue   = result.get(outputName).get().getValue()

                // rawValue is float[][][]: [batch=1][seq_len=128][hidden=384]
                @Suppress("UNCHECKED_CAST")
                val hiddenStates = rawValue as Array<Array<FloatArray>>

                // Mean-pool over non-padding positions
                val seqEmb    = hiddenStates[0]  // shape [128][384]
                val seqLength = attentionMask.count { it == 1L }.coerceAtLeast(1)
                val pooled    = FloatArray(EMBEDDING_DIM)
                for (t in 0 until seqLength) {
                    for (d in 0 until EMBEDDING_DIM) {
                        pooled[d] += seqEmb[t][d]
                    }
                }
                for (d in 0 until EMBEDDING_DIM) pooled[d] /= seqLength.toFloat()
                return pooled
            }
        } catch (e: Exception) {
            Log.w(TAG, "ORT inference failed — using hash fallback: ${e.message}")
            return hashEmbedFallback(text)
        }
    }

    // ─── WordPiece tokenizer ──────────────────────────────────────────────────

    // Loaded lazily when the ONNX model is first used.
    // Maps token string → BERT vocab id (0–30521).
    // Null while vocab.txt has not been downloaded yet.
    private var wordPieceVocab: Map<String, Int>? = null

    /**
     * Load the BERT vocab.txt file into memory.
     * Each line is one vocabulary entry; its 0-based line index is the token id.
     * Called once lazily from runOnnxInference() when the vocab file is present.
     */
    fun loadVocab(context: Context) {
        if (wordPieceVocab != null) return
        val vocabFile = File(modelsDir(context), "bert-vocab.txt")
        if (!vocabFile.exists()) return
        val vocab = HashMap<String, Int>(32_000)
        vocabFile.bufferedReader().useLines { lines ->
            lines.forEachIndexed { index, token -> vocab[token.trim()] = index }
        }
        wordPieceVocab = vocab
        Log.i(TAG, "WordPiece vocab loaded: ${vocab.size} tokens")
    }

    /**
     * Real BERT WordPiece tokenizer.
     *
     * Algorithm for each whitespace-split word:
     *   1. Try the full lower-cased word — if in vocab, emit its id.
     *   2. Otherwise greedily split into the longest known sub-words, prefixing
     *      continuations with "##" (e.g. "playing" → "play", "##ing").
     *   3. If any sub-word is completely unknown, emit [UNK]=100 for the whole word.
     *
     * Special tokens:  [CLS]=101  [SEP]=102  [UNK]=100  [PAD]=0
     *
     * Falls back to the hash tokenizer if vocab.txt is not yet downloaded.
     */
    private fun tokenize(text: String): List<Long> {
        val vocab = wordPieceVocab ?: return tokenizeHash(text)

        val ids = mutableListOf<Long>()
        ids.add(101L)  // [CLS]

        val words = text.lowercase()
            .split(Regex("\\s+|(?=[^a-z0-9])|(?<=[^a-z0-9])"))
            .filter { it.isNotEmpty() }

        for (word in words) {
            if (ids.size >= SEQ_LEN - 1) break   // reserve space for [SEP]

            val wordPieces = mutableListOf<Int>()
            var start = 0
            var failed = false

            while (start < word.length) {
                var end = word.length
                var found = false

                while (end > start) {
                    val sub = if (start == 0) word.substring(start, end)
                              else "##" + word.substring(start, end)
                    val id = vocab[sub]
                    if (id != null) {
                        wordPieces.add(id)
                        start = end
                        found = true
                        break
                    }
                    end--
                }

                if (!found) {
                    failed = true
                    break
                }
            }

            if (failed || wordPieces.isEmpty()) {
                ids.add(100L)  // [UNK]
            } else {
                wordPieces.forEach { ids.add(it.toLong()) }
            }
        }

        ids.add(102L)  // [SEP]
        return ids
    }

    /**
     * Hash-based fallback tokenizer — used when vocab.txt is not yet downloaded.
     * Maps each whitespace-split word to a stable id in [1, 30521] via hashCode().
     * Semantically less accurate than WordPiece but always available.
     */
    private fun tokenizeHash(text: String): List<Long> {
        val words = text.lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { it.isNotEmpty() }
            .take(SEQ_LEN - 2)
        val ids = mutableListOf<Long>()
        ids.add(101L)  // [CLS]
        for (word in words) {
            val raw = word.hashCode().toLong() and 0x7FFFFFFF
            ids.add(1L + (raw % 30520L))
        }
        ids.add(102L)  // [SEP]
        return ids
    }

    // ─── Fallback embedding ───────────────────────────────────────────────────

    private fun hashEmbedFallback(text: String): FloatArray {
        val vec = FloatArray(EMBEDDING_DIM)
        val words = text.lowercase().split(Regex("[^a-z0-9]+")).filter { it.isNotEmpty() }
        for (word in words) {
            val hash = word.hashCode()
            val idx1 = ((hash ushr 0)  and 0xFF) % EMBEDDING_DIM
            val idx2 = ((hash ushr 8)  and 0xFF) % EMBEDDING_DIM
            val idx3 = ((hash ushr 16) and 0xFF) % EMBEDDING_DIM
            val idx4 = ((hash ushr 24) and 0xFF) % EMBEDDING_DIM
            vec[idx1] += 1.0f
            vec[idx2] += 0.5f
            vec[idx3] += 0.25f
            vec[idx4] += 0.125f
        }
        return vec
    }

    // ─── Kotlin scalar fallbacks (used when NEON .so not loaded) ─────────────

    private fun cosineSimilarityKotlin(a: FloatArray, b: FloatArray): Float {
        var dot = 0.0f; var normA = 0.0f; var normB = 0.0f
        for (i in a.indices) { dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i] }
        val denom = kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB)
        return if (denom > 0f) dot / denom else 0f
    }

    private fun normalizeL2Kotlin(vec: FloatArray): FloatArray {
        val norm = kotlin.math.sqrt(vec.fold(0f) { acc, v -> acc + v * v })
        return if (norm > 0f) FloatArray(vec.size) { vec[it] / norm } else vec
    }
}
