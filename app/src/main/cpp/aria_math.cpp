/**
 * aria_math.cpp — NEON SIMD accelerated math for ARIA's on-device ML.
 *
 * Why C++ here?
 * The Cortex-A73 (Exynos 9611 big cores) has ARMv8 NEON SIMD:
 *   - 128-bit SIMD registers
 *   - Processes 4 float32 values per instruction (vfmaq_f32, vmulq_f32, etc.)
 *   - For 384-dim embeddings: 96 SIMD iterations vs 384 scalar iterations
 *   - Real speedup: ~3-4× over Kotlin FloatArray loops for dot products
 *
 * Operations exposed to Kotlin via JNI:
 *   nativeCosineSimilarity(a, b): Float   → EmbeddingEngine.retrieve()
 *   nativeL2Normalize(v): FloatArray      → EmbeddingEngine.embed()
 *   nativeMatVecRelu(W, x, rows, cols)    → PolicyNetwork.forward()
 *   nativeSoftmax(logits): FloatArray     → PolicyNetwork.selectAction()
 *   nativeDotProduct(a, b): Float         → general purpose
 *
 * These are the ONLY places C++ is used in ARIA outside of llama.cpp,
 * because they are the only other compute-bound operations (≥384 float ops
 * that run inside the agent loop multiple times per second).
 */

#include <jni.h>
#include <arm_neon.h>     // ARMv8 NEON intrinsics — guaranteed on Exynos 9611
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <algorithm>

#define TAG "AriaMath"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)

// ─── Helper: pin float array from JVM → native pointer ───────────────────────
// Must call env->ReleaseFloatArrayElements() after use to unpin.

// ─── Dot product with NEON ───────────────────────────────────────────────────
// Processes 4 floats per cycle using vfmaq_f32 (fused multiply-add).
// Handles tail (n % 4 != 0) with scalar fallback.

static float dot_neon(const float* __restrict__ a, const float* __restrict__ b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;

    // NEON: 4 floats per iteration
    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb);  // acc += va * vb (fused multiply-add)
    }

    // Horizontal add of 4 accumulators
    float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    float result = vget_lane_f32(vpadd_f32(sum2, sum2), 0);

    // Scalar tail
    for (; i < n; i++) result += a[i] * b[i];

    return result;
}

// ─── L2 norm with NEON ───────────────────────────────────────────────────────

static float l2_norm_neon(const float* v, int n) {
    return sqrtf(dot_neon(v, v, n));
}

extern "C" {

// ─── nativeCosineSimilarity ───────────────────────────────────────────────────
// EmbeddingEngine calls this to compare query embedding vs stored experience embeddings.
// dim=384 for MiniLM-L3-v2. Returns [-1, 1] (higher = more similar).
JNIEXPORT jfloat JNICALL
Java_com_ariaagent_mobile_core_memory_EmbeddingEngine_nativeCosineSimilarity(
    JNIEnv* env, jobject /* thiz */,
    jfloatArray a_arr, jfloatArray b_arr
) {
    jint n = env->GetArrayLength(a_arr);
    jfloat* a = env->GetFloatArrayElements(a_arr, nullptr);
    jfloat* b = env->GetFloatArrayElements(b_arr, nullptr);

    float dot  = dot_neon(a, b, n);
    float normA = l2_norm_neon(a, n);
    float normB = l2_norm_neon(b, n);

    env->ReleaseFloatArrayElements(a_arr, a, JNI_ABORT);
    env->ReleaseFloatArrayElements(b_arr, b, JNI_ABORT);

    float denom = normA * normB;
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

// ─── nativeL2Normalize ────────────────────────────────────────────────────────
// In-place L2 normalization — used after embedding to produce unit vectors.
// Returns a new normalized FloatArray (does not modify input).
JNIEXPORT jfloatArray JNICALL
Java_com_ariaagent_mobile_core_memory_EmbeddingEngine_nativeL2Normalize(
    JNIEnv* env, jobject /* thiz */,
    jfloatArray v_arr
) {
    jint n = env->GetArrayLength(v_arr);
    jfloat* v = env->GetFloatArrayElements(v_arr, nullptr);

    float norm = l2_norm_neon(v, n);

    jfloatArray result = env->NewFloatArray(n);
    jfloat* out = env->GetFloatArrayElements(result, nullptr);

    if (norm > 1e-8f) {
        float32x4_t vNorm = vdupq_n_f32(1.0f / norm);
        int i = 0;
        for (; i <= n - 4; i += 4) {
            float32x4_t vv = vld1q_f32(v + i);
            vst1q_f32(out + i, vmulq_f32(vv, vNorm));
        }
        for (; i < n; i++) out[i] = v[i] / norm;
    } else {
        memcpy(out, v, n * sizeof(float));
    }

    env->ReleaseFloatArrayElements(v_arr, v, JNI_ABORT);
    env->ReleaseFloatArrayElements(result, out, 0);
    return result;
}

// ─── nativeDotProduct ─────────────────────────────────────────────────────────
// General purpose dot product. Used in PolicyNetwork forward pass.
JNIEXPORT jfloat JNICALL
Java_com_ariaagent_mobile_core_memory_EmbeddingEngine_nativeDotProduct(
    JNIEnv* env, jobject /* thiz */,
    jfloatArray a_arr, jfloatArray b_arr
) {
    jint n = env->GetArrayLength(a_arr);
    jfloat* a = env->GetFloatArrayElements(a_arr, nullptr);
    jfloat* b = env->GetFloatArrayElements(b_arr, nullptr);
    float result = dot_neon(a, b, n);
    env->ReleaseFloatArrayElements(a_arr, a, JNI_ABORT);
    env->ReleaseFloatArrayElements(b_arr, b, JNI_ABORT);
    return result;
}

// ─── nativeMatVecRelu ─────────────────────────────────────────────────────────
// Matrix-vector multiply + ReLU for PolicyNetwork hidden layers.
// W: (rows × cols) row-major, x: (cols,), out: (rows,)
// out[i] = max(0, W[i,:] · x)
// Used in PolicyNetwork.forward() — replaces Kotlin scalar loop.
JNIEXPORT jfloatArray JNICALL
Java_com_ariaagent_mobile_core_rl_PolicyNetwork_nativeMatVecRelu(
    JNIEnv* env, jobject /* thiz */,
    jfloatArray W_arr, jfloatArray x_arr,
    jint rows, jint cols
) {
    jfloat* W = env->GetFloatArrayElements(W_arr, nullptr);
    jfloat* x = env->GetFloatArrayElements(x_arr, nullptr);

    jfloatArray out_arr = env->NewFloatArray(rows);
    jfloat* out = env->GetFloatArrayElements(out_arr, nullptr);

    for (int i = 0; i < rows; i++) {
        float val = dot_neon(W + i * cols, x, cols);
        out[i] = val > 0.0f ? val : 0.0f;  // ReLU
    }

    env->ReleaseFloatArrayElements(W_arr, W, JNI_ABORT);
    env->ReleaseFloatArrayElements(x_arr, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(out_arr, out, 0);
    return out_arr;
}

// ─── nativeSoftmax ───────────────────────────────────────────────────────────
// Numerically stable softmax for PolicyNetwork output layer.
// exp(x - max(x)) / sum(exp(x - max(x)))
JNIEXPORT jfloatArray JNICALL
Java_com_ariaagent_mobile_core_rl_PolicyNetwork_nativeSoftmax(
    JNIEnv* env, jobject /* thiz */,
    jfloatArray logits_arr
) {
    jint n = env->GetArrayLength(logits_arr);
    if (n == 0) return env->NewFloatArray(0);   // guard: empty input → empty output

    jfloat* logits = env->GetFloatArrayElements(logits_arr, nullptr);

    jfloatArray out_arr = env->NewFloatArray(n);
    jfloat* out = env->GetFloatArrayElements(out_arr, nullptr);

    // Find max for numerical stability
    float maxVal = logits[0];
    for (int i = 1; i < n; i++) maxVal = std::max(maxVal, logits[i]);

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(logits[i] - maxVal);
        sum += out[i];
    }

    // Normalize
    float invSum = (sum > 1e-8f) ? (1.0f / sum) : (1.0f / n);
    for (int i = 0; i < n; i++) out[i] *= invSum;

    env->ReleaseFloatArrayElements(logits_arr, logits, JNI_ABORT);
    env->ReleaseFloatArrayElements(out_arr, out, 0);
    return out_arr;
}

} // extern "C"
