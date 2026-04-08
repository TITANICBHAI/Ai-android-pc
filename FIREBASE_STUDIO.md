# ARIA — Firebase Studio: Import → Build → Emulate

> **Important before you read anything else:**
>
> "Firebase Studio" is the name of Google's cloud IDE (formerly Project IDX).
> It has nothing to do with Firebase Auth, Firestore, Realtime Database,
> Storage, or Functions — ARIA uses **none** of those services.
> ARIA is 100% on-device. No cloud backend. No emulator SDK. No `firebase.json`.
> Any advice about `Firebase.auth.useEmulator()`, port 9099, `10.0.2.2`,
> or `firebase emulators:start` is for different apps and does not apply here.

---

## What Firebase Studio gives you for this project

| What | How |
|---|---|
| JDK 17 | Installed via `.idx/dev.nix` automatically |
| NDK 27.1.12297006 | Installed via `.idx/dev.nix` automatically |
| Android SDK 35 + Build-tools 35.0.0 | Installed via `.idx/dev.nix` automatically |
| `local.properties` with correct `sdk.dir` / `ndk.dir` | Written by `.idx/dev.nix` on every workspace start |
| arm64-v8a emulator (Pixel 7, API 35) | Provisioned via `.idx/dev.nix` automatically |
| Gradle caching | Enabled in `gradle.properties` |
| Kotlin + Gradle extensions | Installed into the embedded VS Code |

You do not configure any of the above manually. Open the project, wait for setup, build.

---

## Step 1 — Open the project

In Firebase Studio: **Import from GitHub** (or open from your workspace if already cloned).

Open the **`android/`** subdirectory as the project root, not the monorepo root:

```
File → Open Folder → select …/android/
```

Firebase Studio will detect `.idx/dev.nix` and begin provisioning automatically.
This takes 3–8 minutes on a cold workspace (SDK + NDK download).

---

## Step 2 — Verify the workspace provisioned correctly

Open the terminal and run:

```bash
echo "SDK: $ANDROID_HOME"
cat android/local.properties
```

Expected output:

```
SDK: /home/user/Android/sdk
sdk.dir=/home/user/Android/sdk
ndk.dir=/home/user/Android/sdk/ndk/27.1.12297006
```

If `local.properties` is missing or wrong, re-run the onStart hook manually:

```bash
printf 'sdk.dir=%s\nndk.dir=%s/ndk/27.1.12297006\n' \
  "$ANDROID_HOME" "$ANDROID_HOME" > android/local.properties
cat android/local.properties
```

---

## Step 3 — Gradle sync

Click **Sync Now** in the yellow notification bar, or:

```
View → Command Palette → Gradle: Refresh Gradle Project
```

In the Build output watch for this line — it confirms native-only mode is active:

```
[ARIA] node_modules not found — NATIVE-ONLY mode (Firebase Studio / Android Studio).
[ARIA] RN/Expo sub-projects and plugins are skipped.
```

If you see this, the sync is correct. All React Native and Expo sub-projects are
automatically excluded — no `node_modules` is needed and none should be present.

**Sync takes 2–5 minutes** on first run (Gradle downloads all Jetpack, ML Kit,
ONNX, MediaPipe, and OkHttp artifacts). Subsequent syncs use the Gradle cache
and take seconds.

---

## Step 4 — Build

```bash
cd android
./gradlew assembleDebug
```

Expected final line:

```
BUILD SUCCESSFUL in Xs
```

APK location:

```
android/app/build/outputs/apk/debug/app-debug.apk
```

### What the NDK build compiles

The NDK step builds `llama-jni.so` from two C++ source files:

| File | What it is |
|---|---|
| `app/src/main/cpp/llama_jni.cpp` | JNI bridge to llama.cpp (LLM inference) |
| `app/src/main/cpp/aria_math.cpp` | NEON SIMD math for ONNX embeddings + RL policy |
| `app/src/main/cpp/llama.cpp/` | llama.cpp git submodule (already committed) |

NDK compile is the slowest part (~4 min cold, ~30 s incremental with caching on).

---

## Step 5 — Run on the emulator

### Start the emulator

In the Firebase Studio sidebar: **Android panel → Start emulator**

Or from terminal:

```bash
$ANDROID_HOME/emulator/emulator \
  -avd Pixel_7_API_35 \
  -no-snapshot-load \
  &
```

Wait until the emulator home screen appears (~60–90 seconds on Axion ARM64).

### Install and launch

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.ariaagent.mobile/.ui.ComposeMainActivity
```

Or press the **▶ Run** button in the Firebase Studio Android panel after selecting
`app` as the run configuration and the emulator as the target device.

The app opens directly to the Jetpack Compose UI — Dashboard tab, 7-tab nav bar.
There is no React Native splash, no Metro bundler, no JS bridge initialisation.

---

## Step 6 — Enable Accessibility Service (required for agent to function)

The agent cannot read screen content or dispatch gestures without this.

1. On the emulator: **Settings → Accessibility → ARIA Agent → Enable**
2. Confirm the permission dialog
3. Back in the app: **Settings tab** — the Accessibility row should show a green
   granted indicator

This persists across app restarts but resets if the APK is uninstalled.

---

## Step 7 — Load the LLM model

ARIA uses an on-device GGUF model (~870 MB). It is not bundled in the APK
(too large). On first launch:

1. Go to **Settings tab** → set the model path
   (default: `/sdcard/Download/Llama-3.2-1B-Instruct-Q4_K_M.gguf`)
2. Push the model file to the emulator:
   ```bash
   adb push /path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
     /sdcard/Download/Llama-3.2-1B-Instruct-Q4_K_M.gguf
   ```
3. Go to **Control tab** → tap the **Load Model** card that appears when no model
   is loaded

The ONNX embedding model (`all-MiniLM-L6-v2`) and the MediaPipe object detection
model download automatically on first run via `ModelBootstrap` (small files,
no action needed).

---

## Known issues / things to watch

### 1. CMake version resolution
`app/build.gradle` specifies `cmake { version "3.22.1" }`. Firebase Studio
installs CMake 3.30.x via Nix. AGP 8.8 will use the system CMake from PATH
when the exact version is not found in the SDK CMake directory — this works,
but the Build output may show:

```
No version of CMake 3.22.1 found in PATH or by cmake.dir property.
Using CMake from PATH: /nix/store/…/cmake-3.30.x/bin/cmake
```

This is harmless. If it becomes an error, install the exact version:

```bash
$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager "cmake;3.22.1"
```

### 2. arm64-v8a emulator only
The NDK ABI filter is `arm64-v8a`. **Do not create an x86_64 emulator** — it
will install the APK but crash immediately when `System.loadLibrary("llama-jni")`
is called because the `.so` was compiled for ARM only.

Firebase Studio Axion instances are ARM64, so the arm64-v8a emulator gets
hardware acceleration. If you are on a local Intel/AMD machine, use a physical
ARM device instead (or accept slow emulation via QEMU).

### 3. First Gradle sync is slow
Gradle downloads ~800 MB of dependencies (Compose BOM, ML Kit, ONNX, MediaPipe,
OkHttp, etc.). Gradle caching is enabled (`org.gradle.caching=true`). After the
first successful sync the cache is warm and syncs are fast.

### 4. ExpoModulesPackageList.kt stub
`android/app/src/main/kotlin/expo/modules/ExpoModulesPackageList.kt` exists but
is intentionally empty (a comment-only file). It exists to keep the package
declaration valid in native-only mode. It will be deleted in Phase 8 of
`migration.md` once all Compose screens are emulator-verified.

### 5. MainActivity.kt stub
`MainActivity.kt` is a no-op `AppCompatActivity` stub. It has no launcher
intent-filter. It exists only to keep `AndroidManifest.xml` valid during
migration. Delete it in Phase 8.

### 6. Heap size
`gradle.properties` allocates `Xmx4096m` (4 GB) for the Gradle JVM. Firebase
Studio Axion instances have 8–16 GB RAM — this is safe. Do not reduce it below
2 GB or the NDK (llama.cpp) + Compose compiler passes will OOM.

### 7. Missing mipmap PNG densities
Only `mipmap-anydpi-v26/` exists (adaptive icon). There are no density-bucketed
PNG files (`mipmap-hdpi/`, `mipmap-xxhdpi/`, etc.). The emulator is API 35 so
adaptive icons are used — no visual issue. For a production release targeting
API < 26 devices, add PNG fallback icons.

---

## Adjustments already made for native-only mode

These were updated as part of the migration and are already in the codebase:

| File | What changed |
|---|---|
| `android/app/proguard-rules.pro` | Removed all React Native, Hermes, TurboModule, gesture-handler, screens, async-storage, and Expo keep rules — they don't belong in a native release build |
| `.idx/dev.nix` | Removed `nodejs_20` package and `pnpm install` onCreate step — no JS runtime needed |
| `android/gradle.properties` | Added `org.gradle.caching=true` and `org.gradle.daemon=true` |
| All 16 Kotlin core files | Removed stale `AgentCoreModule → JS` comments — replaced with Compose/AgentViewModel references |
| `android/app/src/main/kotlin/com/ariaagent/mobile/ui/ComposeMainActivity.kt` | Fixed comment that still said ReactActivity was the launcher |

---

## Build modes reference

| Mode | Trigger | What compiles |
|---|---|---|
| **Native-only** (Firebase Studio) | No `node_modules` in tree | Pure Kotlin + NDK. All RN/Expo sub-projects and plugins skipped. |
| **Hybrid** (pnpm monorepo) | `node_modules` with `react-native` present | Adds RN bridge + Expo modules. Used for EAS builds during Phases 1-7. |

Mode is detected **automatically** by `android/settings.gradle`. No flag to set.

---

## Quick reference — useful adb commands

```bash
# Install debug APK
adb install -r android/app/build/outputs/apk/debug/app-debug.apk

# Launch ARIA directly
adb shell am start -n com.ariaagent.mobile/.ui.ComposeMainActivity

# Push GGUF model to emulator
adb push Llama-3.2-1B-Instruct-Q4_K_M.gguf /sdcard/Download/

# Tail logcat (filter to ARIA tags)
adb logcat -s ARIA AgentLoop LlamaEngine OcrEngine AgentViewModel

# Open Accessibility settings directly
adb shell am start -a android.settings.ACCESSIBILITY_SETTINGS

# Take screenshot
adb exec-out screencap -p > screen.png

# Clean build (if Gradle cache is corrupt)
cd android && ./gradlew clean assembleDebug
```

---

## Migration phase reference

See `migration.md` at the monorepo root for the full plan.

| Phase | Status | Description |
|---|---|---|
| 1 — Compose launcher | Done | ComposeMainActivity is the launcher |
| 2 — SettingsScreen | Done | Feature-complete, emulator verify pending |
| 3 — ActivityScreen | Done | Feature-complete, emulator verify pending |
| 4 — ControlScreen | Done | Feature-complete, emulator verify pending |
| 5 — ChatScreen | Written | Emulator verify pending |
| 6 — TrainScreen | Written | Emulator verify pending |
| 7 — LabelerScreen | Written | Emulator verify pending |
| 8 — Delete RN layer | Blocked | Waiting for all screens to be `[x]` verified |
| 9 — Strip build system | Blocked | Waiting for Phase 8 |
