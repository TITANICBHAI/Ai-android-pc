# ARIA Agent — UI ↔ Backend Connectivity Audit

Generated: 2026-04-05
Scope: all interactive controls across 8 Compose screens AND all backend services.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully wired end-to-end |
| ⚠️  | Partially wired — backend exists and runs, but feedback to UI is broken or missing |
| ❌ | Gap — one side exists, the other does not |

---

# Part A — UI exists, tracing backend connection

## Control Screen

| Control | ViewModel call | Real backend | Status | Notes |
|---------|---------------|--------------|--------|-------|
| Goal text field | feeds `startAgent` / `startLearnOnly` | `AgentLoop` | ✅ | |
| Target app field | feeds `startAgent` | `AgentLoop` | ✅ | |
| Learn-only toggle | switches call path | `AgentForegroundService` | ✅ | |
| START AGENT button | `vm.startAgent()` | `AgentLoop.start()` | ✅ | |
| START LEARNING button | `vm.startLearnOnly()` | `AgentForegroundService.startLearnOnly()` | ✅ | |
| PAUSE / RESUME button | `vm.pauseAgent()` / `vm.startAgent()` | `AgentLoop.pause()` | ✅ | |
| STOP button | `vm.stopAgent()` | `AgentLoop.stop()` | ✅ | |
| Load LLM Engine button | `vm.loadModel()` | `LlamaEngine.load()` | ✅ | |
| Task queue goal field | feeds `enqueueTask` | `TaskQueueManager` | ✅ | |
| Task queue app field | feeds `enqueueTask` | `TaskQueueManager` | ✅ | |
| Add to Queue button | `vm.enqueueTask()` | `TaskQueueManager.enqueue()` | ✅ | |
| Clear Queue button | `vm.clearTaskQueue()` | `TaskQueueManager.clear()` | ✅ | |
| Per-task remove button | `vm.removeQueuedTask()` | `TaskQueueManager.remove()` | ✅ | |
| Dismiss chain notification | `vm.dismissChainNotification()` | state only | ✅ | |

## Chat Screen

| Control | ViewModel call | Real backend | Status | Notes |
|---------|---------------|--------------|--------|-------|
| Message text field | feeds `sendChatMessage` | `LlamaEngine.infer()` | ✅ | |
| Send button | `vm.sendChatMessage()` | `LlamaEngine.infer()` | ✅ | Guards against unloaded model |
| Preset prompt chips | `vm.sendChatMessage(preset)` | `LlamaEngine.infer()` | ✅ | |
| Clear chat icon | `vm.clearChat()` | state clear | ✅ | |

## Modules Screen

| Control | ViewModel call | Real backend | Status | Notes |
|---------|---------------|--------------|--------|-------|
| Download — Llama 3.2-1B | `vm.downloadLlmModel()` | `ModelDownloadService` | ⚠️ | Service runs, emits `model_download_progress` events every 2 MB, but ViewModel resets `_llmDownloading` after a fake 2-second delay instead of listening to events. No progress % shown in UI. |
| Download — EfficientDet | `vm.downloadDetectorModel()` | `ObjectDetectorEngine.ensureModel()` | ✅ | Small file, flag reset on coroutine completion |
| GRANT — Screen Capture | `vm.requestScreenCapturePermission()` | `MediaProjectionManager` chooser | ✅ | Fixed this session |
| GRANT — Accessibility | ❌ no button | `Settings.ACTION_ACCESSIBILITY_SETTINGS` deep-link | ❌ | Row shows DENIED but no action |
| OCR status chip | `refreshModuleState()` | — | ⚠️ | `ocrReady` hardcoded `true` — never checks ML Kit init |

## Train Screen

| Control | ViewModel call | Real backend | Status | Notes |
|---------|---------------|--------------|--------|-------|
| Run RL Cycle button | `vm.runRlCycle()` | `LoraTrainer.train()` | ✅ | Real on-device LoRA fine-tuning |
| Auto-schedule RL toggle | `vm.setAutoScheduleRl()` | `ExperienceStore` + `runRlCycle()` | ✅ | Triggers if > 50 untrained successes |
| IRL video picker | → `vm.processIrlVideo()` | `IrlModule.processVideo()` | ✅ | |
| IRL goal / app fields | feed `processIrlVideo` | `IrlModule` | ✅ | |
| Run IRL button | `vm.processIrlVideo()` | `IrlModule.processVideo()` | ✅ | |
| Refresh header button | `vm.refreshLearningStatus()` | `LoraTrainer` + `ExperienceStore` | ✅ | |
| Navigate to Labeler | nav callback | — | ✅ | |

## Labeler Screen

| Control | ViewModel call | Real backend | Status | Notes |
|---------|---------------|--------------|--------|-------|
| Capture Screen button | `vm.captureScreenForLabeling()` | `ScreenCaptureService` + `OcrEngine` | ✅ | |
| Grant Screen Capture button | `onRequestScreenCapture()` | `MediaProjectionManager` | ✅ | Fixed this session |
| Import from Gallery button | gallery launcher → `vm.loadImageFromGallery()` | `OcrEngine.run()` | ✅ | Fixed this session |
| Tap to place pin | `vm.addLabelerPin()` | state | ✅ | |
| Drag pin | `vm.updateLabelerLabel()` | state | ✅ | |
| Auto-detect button | `vm.autoDetectLabelerPins()` | `ObjectDetectorEngine.detectFromPath()` | ✅ | |
| Enrich button | `vm.enrichAllLabelerPins()` | `LlamaEngine.infer()` per pin | ✅ | |
| Save button | `vm.saveLabelerLabels()` | `ObjectLabelStore` | ✅ | |
| Delete label | `vm.deleteLabelerLabel()` | state | ✅ | |
| Dismiss error | `vm.dismissLabelerError()` | state | ✅ | Auto-dismissed via LaunchedEffect |

## Activity Screen

| Control | ViewModel call | Real backend | Status | Notes |
|---------|---------------|--------------|--------|-------|
| Clear Memory button | `vm.clearMemory()` | `ExperienceStore.clearAll()` | ✅ | Confirmation dialog |
| (memory list auto-loads) | `vm.refreshMemoryEntries()` | `ExperienceStore.getRecent(200)` | ✅ | LaunchedEffect on enter |

## Settings Screen

| Control | ViewModel call | Real backend | Status | Notes |
|---------|---------------|--------------|--------|-------|
| Model path field | → `saveConfig` | `ConfigStore` | ✅ | |
| Quantization selector | → `saveConfig` | `ConfigStore` | ✅ | |
| Context window selector | → `saveConfig` | `ConfigStore` | ✅ | |
| GPU layers selector | → `saveConfig` | `ConfigStore` | ✅ | |
| Temperature slider | → `saveConfig` | `ConfigStore` | ✅ | |
| RL enabled switch | → `saveConfig` | `ConfigStore` | ✅ | |
| LoRA adapter path field | → `saveConfig` | `ConfigStore` | ✅ | |
| Save Configuration button | `vm.saveConfig()` | `ConfigStore.save()` + `AgentEventBus` | ✅ | |
| Clear Memory button | `vm.clearMemory()` | `ExperienceStore.clearAll()` | ✅ | |
| Reset Agent button | `vm.resetAgent()` | full state wipe | ✅ | |

## Dashboard Screen

| Control | ViewModel call | Real backend | Status | Notes |
|---------|---------------|--------------|--------|-------|
| Dismiss chain notification | `vm.dismissChainNotification()` | state | ✅ | |
| (all other content is read-only display) | — | — | — | |

---

# Part B — Backend ↔ UI Wiring (Historical gaps — all closed in Part C)

These services exist in Kotlin and work correctly. Each originally had no UI entry point. All gaps listed below were subsequently closed; descriptions updated to reflect current state.

## 1. Unload LLM from RAM ✅ Fixed (see Part C, gap #5)

| Item | File | What it does |
|------|------|--------------|
| `LlamaEngine.unload()` | `core/ai/LlamaEngine.kt:103` | Frees the JNI model handle and ~800 MB of RAM |
| `LlamaEngine.loadLora()` | `core/ai/LlamaEngine.kt:156` | Hot-swaps a LoRA adapter into a loaded model |

**Status:** `ModulesScreen` "Free RAM (Unload)" button calls `AgentViewModel.unloadLlmModel()` → `LlamaEngine.unload()`. The LoRA adapter path from config is passed automatically by `AgentLoop` when a task starts.

---

## 2. Embedding Model (MiniLM) — status + download button ✅ Fixed (see Part C, gap #4)

| Item | File | What it does |
|------|------|--------------|
| `EmbeddingModelManager.isModelReady()` | `core/memory/EmbeddingModelManager.kt:63` | Checks if the ~23 MB ONNX MiniLM model is on disk |
| `EmbeddingModelManager.isVocabReady()` | `core/memory/EmbeddingModelManager.kt:68` | Checks if the tokenizer vocab file is present |
| `EmbeddingModelManager.download()` | `core/memory/EmbeddingModelManager.kt:88` | Downloads model + vocab from HuggingFace |
| `EmbeddingModelManager.cancelDownload()` | `core/memory/EmbeddingModelManager.kt:205` | Cancels an in-progress download |
| `EmbeddingEngine.isModelAvailable()` | `core/memory/EmbeddingEngine.kt:51` | Combined readiness check |

**Status:** `ModulesScreen` MiniLM card shows readiness status and a download button. `AgentViewModel.downloadEmbeddingModel()` wired.

---

## 3. Local Web Server — toggle in SettingsScreen ✅ Fixed (see Part C, gap #6)

| Item | File | What it does |
|------|------|--------------|
| `LocalDeviceServer.start(port)` | `core/monitoring/LocalDeviceServer.kt:56` | Starts an HTTP server on the device (default port 8080) |
| `LocalDeviceServer.stop()` | `core/monitoring/LocalDeviceServer.kt:81` | Stops the server |
| `LocalDeviceServer.serverUrl()` | `core/monitoring/LocalDeviceServer.kt:110` | Returns `http://<device-ip>:8080` |
| `MonitoringPusher.start()` | `core/monitoring/MonitoringPusher.kt:65` | Pushes live agent state to the local server |
| `MonitoringPusher.stop()` | `core/monitoring/MonitoringPusher.kt:84` | Stops the push loop |

**Status:** `SettingsScreen` "Web Dashboard" section (lines 436–503) has a start/stop toggle, live URL display, and clipboard copy. `MonitoringPusher` started/stopped alongside the server.

---

## 4. Learning Scheduler — runs hidden, no status ✅ Partially addressed

| Item | File | What it does |
|------|------|--------------|
| `LearningScheduler.start()` | `core/rl/LearningScheduler.kt:57` | Starts an auto-training background loop |
| `LearningScheduler.stop()` | `core/rl/LearningScheduler.kt:64` | Stops it |
| `LearningScheduler.isTrainingRunning()` | `core/rl/LearningScheduler.kt:187` | Whether a training pass is currently running |

**Status:** `TrainScreen` "Auto-schedule RL" toggle wired to `vm.setAutoScheduleRl()`, which starts/stops `LearningScheduler`. The "Run RL Cycle" button shows a spinner while any training is running. A dedicated running-indicator widget is a future nice-to-have.

---

## 5. Object Label browse and stats ✅ Fixed (see Part C, gap #7)

| Item | File | What it does |
|------|------|--------------|
| `ObjectLabelStore.getAllEnriched()` | `core/memory/ObjectLabelStore.kt:212` | Returns all enriched UI element labels |
| `ObjectLabelStore.countEnriched()` | `core/memory/ObjectLabelStore.kt:228` | How many labels have been LLM-enriched |
| `ObjectLabelStore.getByScreen()` | `core/memory/ObjectLabelStore.kt:182` | Labels for a specific screen hash |
| `ObjectLabelStore.clearAll()` | `core/memory/ObjectLabelStore.kt:171` | Wipes all stored UI element labels |
| `ObjectLabelStore.getEnrichedByApp()` | `core/memory/ObjectLabelStore.kt:197` | Labels for a specific app |

**Status:** `ActivityScreen` Labels tab shows a full browseable list of stored labels per-app, with a clear-all dialog. `AgentViewModel.loadLabelEntries()` and `clearAllLabels()` wired.

---

## 6. ExperienceStore breakdown ✅ Fixed (see Part C, gap #8)

| Item | File | What it does |
|------|------|--------------|
| `ExperienceStore.edgeCaseCount()` | `core/memory/ExperienceStore.kt:112` | Count of experiences flagged as edge cases |
| `ExperienceStore.countByResult("failure")` | `core/memory/ExperienceStore.kt:107` | Failure episode count |

**Status:** `ActivityScreen` Memory tab stats bar shows Total / Success / Fail / Edge-case / Untrained breakdown. All four counts wired from `ExperienceStore`.

---

## 7. LLM download progress ✅ Fixed (see Part C, gap #1)

| Item | File | What it does |
|------|------|--------------|
| `AgentEventBus "model_download_progress"` | `ModelDownloadService.kt:137` | Emits `{percent, downloadedMb, totalMb, speedMbps}` every 2 MB |
| `AgentEventBus "model_download_error"` | `ModelDownloadService.kt:150` | Emits `{error}` on failure |

**Status:** `AgentViewModel.handleLlmDownloadProgress()` handles `model_download_progress` events and updates `_llmDownloadPercent`. `ModulesScreen` shows a real progress bar with percentage and MB/s speed. Error events surface as a dismissible error card.

---

# Part C — Gaps (all closed)

| # | Gap | Status | Where fixed |
|---|-----|--------|-------------|
| 1 | LLM download: real progress bar + error handling | ✅ Done | `AgentViewModel.handleLlmDownloadProgress()` + `ModulesScreen` progress bar |
| 2 | Accessibility Service: GRANT button | ✅ Done | `ModulesScreen` `onGrantAccessibility` + `ARIAComposeApp` wires `ACTION_ACCESSIBILITY_SETTINGS` |
| 3 | OCR: real readiness check | ✅ Done | `AgentViewModel.refreshModuleState()` calls `OcrEngine.isAvailable()` |
| 4 | Embedding model: status card + download button | ✅ Done | `ModulesScreen` MiniLM card + `AgentViewModel.downloadEmbeddingModel()` |
| 5 | Unload LLM button | ✅ Done | `ModulesScreen` "Free RAM (Unload)" button + `AgentViewModel.unloadLlmModel()` |
| 6 | Local web server toggle + URL display | ✅ Done | `SettingsScreen` "Web Dashboard" section + toggle + copy-URL button |
| 7 | Object Label browser + clear-all | ✅ Done | `ActivityScreen` Labels tab + `LabelsList` + `LabelRow` + clear dialog |
| 8 | ExperienceStore breakdown (success/fail/edge) | ✅ Done | `ActivityScreen` Memory tab stats bar (Total / Success / Fail / Edge / Untrained) |
