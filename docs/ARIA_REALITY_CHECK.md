# ARIA Agent — Reality Check

> A brutally honest audit of what the code actually does vs. what it claims.
> Based on reading every core implementation file: `AgentLoop.kt`, `LlamaEngine.kt`,
> `llama_jni.cpp`, `PolicyNetwork.kt`, `IrlModule.kt`, `OcrEngine.kt`,
> `ObjectDetectorEngine.kt`, `AgentAccessibilityService.kt`, and `LoraTrainer.kt`.

---

## Q1: Is the on-device LLM (Llama 3.2-1B) inference real?

**Partially — and here's exactly why.**

The JNI bridge (`llama_jni.cpp`, 823 lines) is completely real. It calls the actual
llama.cpp C API: `llama_model_load_from_file`, `llama_tokenize`, `llama_decode`,
`llama_sampler_sample`, `llama_token_to_piece` — the full inference loop. Streaming
tokens back to Kotlin via a `TokenCallback` JNI callback is also properly wired.

The llama.cpp library itself is **physically present** in the repo
(`android/app/src/main/cpp/llama.cpp/`) with its full CMakeLists.txt.

**The catch:** `LlamaEngine.kt` has a `jniAvailable` flag that starts `false`.
It only flips to `true` if `System.loadLibrary("llama-jni")` succeeds at runtime —
which requires the C++ code to have been compiled into a `.so` via
`./gradlew assembleDebug` with the Android NDK on a proper build machine.
Until that build runs, every `infer()` call returns this hardcoded stub:

```json
{"tool":"Click","node_id":"#1","reason":"stub inference — llama.cpp not compiled"}
```

**Verdict:** The architecture and C++ code are real and correct. Real inference
requires: (1) a successful NDK build, (2) the 870 MB GGUF model downloaded to the
device. Without those two things, the agent loop runs on stub output.

---

## Q2: Does the agent actually control the Android device?

**Yes — when Accessibility permission is granted.**

`AgentAccessibilityService.kt` is a proper `AccessibilityService` subclass. It:
- Traverses the live UI node tree (`rootInActiveWindow`) on every screen change
- Builds LLM-friendly semantic text: `[#1] Button: "Play" (center, clickable)`
- Stores `AccessibilityNodeInfo` copies in a node registry keyed by `#1`, `#2`, etc.
- Dispatches real touch gestures via `GestureDescription.Builder` (Android's official
  programmatic gesture API — the same one screen readers use)
- Handles tap, swipe, and global Back

`GestureEngine.kt` parses the LLM's JSON output (`{"tool":"Click","node_id":"#3"}`)
and calls the service's gesture dispatch.

**The friction:** The user must manually enable ARIA in Android Accessibility Settings.
This is a high-friction step that many users won't complete, and without it the agent
loop sees `"(accessibility service not active)"` for every observation.

**Verdict:** Gesture control is real and correctly implemented. The blocker is a
user permission that must be manually granted.

---

## Q3: Is the RL training (LoRA fine-tuning) real or fake?

**The pipeline is real; the JNI execution has a stub fallback.**

`LoraTrainer.kt` correctly:
1. Pulls successful `(screen, action)` pairs from SQLite (`ExperienceStore`)
2. Pulls human-annotated labels from `ObjectLabelStore` (3× weight)
3. Writes a real JSONL training dataset with proper `{"input":…,"output":…,"weight":…}` format
4. Calls `nativeTrainLora(modelPath, datasetPath, outputPath, rank=4, epochs=1)` via JNI

`llama_jni.cpp` has `Java_…_LoraTrainer_nativeTrainLora` implemented using
`llama_opt_init` / `llama_opt_epoch` from llama.cpp's optimizer API (the same
AdamW path used by the official llama.cpp `finetune` example).

**The catch (same as Q1):** If `libllama-jni.so` hasn't been compiled, the JNI call
throws `UnsatisfiedLinkError`, which `LoraTrainer` catches and handles by writing a
metadata stub file instead of a real adapter:

```kotlin
// stubTrainLora():
File(adapterPath).writeText("""{"lora_stub":true,"rank":4,"version":N}""")
```

The UI will show "LoRA v1" trained, but the "adapter" is just a text file.

**Verdict:** The full on-device LoRA pipeline is architected and coded correctly. It
works end-to-end only after the NDK build compiles the native library.

---

## Q4: Is the "REINFORCE policy network" a real ML model?

**Yes — this one works out of the box, no NDK required.**

`PolicyNetwork.kt` (430 lines) implements from scratch in pure Kotlin:
- A 3-layer MLP: 256 input → 256 hidden (ReLU) → 128 hidden (ReLU) → 7 output (softmax)
- The REINFORCE policy gradient algorithm (Williams, 1992) with discounted returns
- Adam optimizer (β1=0.9, β2=0.999) with bias correction
- Xavier/Glorot weight initialization
- Correct full backpropagation through all three layers
- Binary float32 serialization to `rl/policy_latest.bin` + `rl/policy_adam.bin`

NEON SIMD acceleration (`nativeMatVecRelu`, `nativeSoftmax` via JNI) is used when
the native library is available, otherwise it falls back to the Kotlin scalar math.

**The honest limitation:** This MLP only handles 7 actions (tap, swipe in 4 directions,
type, back) and starts with random weights. It needs many episodes of agent experience
to learn anything meaningful. Fresh install = random action selection.

**Verdict:** Genuinely implemented. The math is correct and it persists state across
sessions. It just starts untrained.

---

## Q5: Does OCR (reading screen text) work?

**Yes — fully, out of the box.**

`OcrEngine.kt` uses Google ML Kit `TextRecognition` (bundled, no API key needed).
It wraps the async Task callback in a Kotlin `suspendCoroutine` correctly.
~50–150ms latency, ~100 MB RAM. Used by the agent loop on every observation step.

**Verdict:** Real, works immediately after install. No caveats.

---

## Q6: Does the object detector (visual perception beyond text) work?

**Yes — after a 4.4 MB download.**

`ObjectDetectorEngine.kt` uses MediaPipe `EfficientDet-Lite0 INT8`, downloaded from
the Google CDN on first use. It builds a real `ObjectDetector`, processes a
`BitmapImageBuilder` image, and returns bounding boxes with COCO category labels.

Detected objects are formatted as `det-1: keyboard (87%, center 50%×80%)` and
injected into the LLM prompt's `[VISUAL DETECTIONS]` block.

**Limitation:** EfficientDet-Lite0 is trained on 80 COCO categories (person, car,
phone, etc.), not on Android UI elements. It catches game sprites and images well,
but not buttons, text fields, or custom views — that's the accessibility tree's job.

**Verdict:** Real and works after the small model download.

---

## Q7: Is the IRL (learning from video recordings) feature real?

**Yes — it's one of the most complete modules.**

`IrlModule.kt` (507 lines) does genuine work:
- Uses `MediaMetadataRetriever` to extract video frames
- Implements scene-change detection via 32×32 pixel diff to skip static frames (4–5×
  speedup — not a gimmick, a real optimisation)
- Runs ML Kit OCR on each key frame
- Resolves per-frame UI labels from `ObjectLabelStore`
- Calls the on-device LLM to infer what action caused screen A → screen B
- Falls back to a word-Jaccard heuristic if LLM isn't loaded
- Saves results to `ExperienceStore` as expert demonstrations (reward=1.0)

**Limitation:** Without LLM inference (stub mode), action inference falls back to the
heuristic, which just classifies "big text change = tap / content shrinks = back /
scroll otherwise." Coordinates are never recovered — only action types.

**Verdict:** Pipeline is real and well-engineered. Quality of the action inference
degrades significantly without the compiled LLM.

---

## Q8: Is the "8–15 tok/s on Exynos 9611" performance claim realistic?

**It's a documented design target, not a measured result from this codebase.**

The claim is plausible — Q4_K_M quantization of a 1B-parameter model with partial
Vulkan GPU offload on a Mali-G72 GPU is in the right ballpark based on public
llama.cpp benchmarks on similar hardware. The specific `n_gpu_layers=32` and
`n_threads=4` settings in `llama_jni.cpp` match what the community reports for Exynos 9611.

However, the `.so` has not been compiled in this repo, so no benchmark has been run.
The 1700 MB RSS estimate is hardcoded, not measured:

```cpp
// llama_jni.cpp line 67:
g_memory_mb.store(1700.0);  // hardcoded estimate, not measured
```

**Verdict:** Plausible and research-backed, but not verified from this build.

---

## Q9: Is the multimodal (vision) inference real?

**The C++ code is there and the Kotlin API is now fully exposed. Active once the NDK `.so` compiles.**

`llama_jni.cpp` has a complete `nativeRunVisionInference` implementation (150+ lines)
using the `mtmd` (multimodal) library from llama.cpp. It properly encodes JPEG/PNG
bytes through the CLIP vision encoder and prepends image embeddings to the text prompt.

`LlamaEngine.kt` (lines 138–268) now exposes the full vision API:
- `loadVision(visionModelPath, mmProjPath, contextSize, nGpuLayers): Boolean` — loads SmolVLM base + CLIP mmproj into independent handles that coexist with the text model
- `isVisionLoaded(): Boolean` — readiness check
- `inferWithVision(imageBytes, prompt, maxTokens, onToken): String` — CLIP encode + text generation; falls back to a stub description in stub mode
- `unloadVision()` — frees the vision handles without touching the main text model

`VisionEngine.kt` wraps these into `VisionEngine.describe(imageBytes, goal)` and is
called by `AgentLoop` every step when the vision model is loaded.

The onboarding wizard (step 2) lets the user download SmolVLM-256M + mmproj (~200 MB)
right during setup.

**Verdict:** C++ implementation and Kotlin API are both real and complete. Vision
inference activates automatically once `libllama-jni.so` is compiled via the NDK build.
Until then, `inferWithVision()` returns a clearly-labelled stub string so callers know
the mode they're in.

---

## Q10: How much of the claimed "autonomous agent" actually runs without manual setup?

**Summary of what works immediately vs. what needs extra setup:**

| Feature | Works immediately? | Requires extra setup |
|---|---|---|
| All 11 UI screens, buttons, navigation | ✅ Yes | — |
| OCR (screen text reading) | ✅ Yes | — |
| Object detection (visual) | ✅ After 4.4 MB download | — |
| REINFORCE policy network | ✅ Yes (starts untrained) | — |
| IRL video processing (heuristic) | ✅ Yes | — |
| Experience storage (SQLite) | ✅ Yes | — |
| LLM inference (Llama 3.2-1B) | ❌ Stub mode | NDK build + 870 MB model download |
| LoRA fine-tuning | ❌ Stub mode | Same NDK build |
| IRL video (LLM-assisted) | ❌ Heuristic only | Same NDK build |
| Device gesture control | ❌ Until permission granted | User enables Accessibility Service |
| Screen capture (visual input) | ❌ Until permission granted | User grants MediaProjection |
| "8–15 tok/s" performance | ❌ Not measurable | NDK build + real device |

**Overall verdict:** ARIA is a genuinely ambitious, technically serious project with
correct implementations across all major subsystems. It is **not vaporware** — the
code is real, the algorithms are correct, and the architecture is sound.

The honest gap is that the most important feature — LLM inference — is in stub mode
until someone runs a full NDK build (`./gradlew assembleDebug` with Android SDK + NDK
installed) and downloads the 870 MB model to the device. Without that build step, the
"agent" makes decisions by returning the same hardcoded JSON string every step.

The UI, the permissions flow, the training screens, the labeler, and the memory system
all work as shown. What the agent *decides to do* is what depends on the compiled LLM.
