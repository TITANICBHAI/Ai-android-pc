package com.ariaagent.mobile.core.rl

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import android.util.Log
import com.ariaagent.mobile.core.agent.AgentLoop
import com.ariaagent.mobile.core.ai.LlamaEngine
import com.ariaagent.mobile.core.ai.ModelManager
import com.ariaagent.mobile.core.rl.DreamEngine
import com.ariaagent.mobile.core.config.ConfigStore
import com.ariaagent.mobile.core.config.ModuleVisionMode
import com.ariaagent.mobile.core.events.AgentEventBus
import com.ariaagent.mobile.core.memory.ExperienceStore
import com.ariaagent.mobile.core.memory.ObjectLabelStore
import com.ariaagent.mobile.core.system.ThermalGuard
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * LearningScheduler — decides when training runs and when it pauses.
 *
 * Training rules (from technical documents):
 *   ✓ Run ONLY when: device is charging + screen is off (idle)
 *   ✓ Pause if: battery temp > 40°C
 *   ✓ Pause if: user unlocks screen
 *   ✓ Pause if: available RAM < 1GB
 *   ✓ Stop if: unplugged from charger
 *
 * Training sequence when triggered:
 *   1. PolicyNetwork REINFORCE update (game/app experience tuples)
 *   2. IRL processing (if video frames captured)
 *   3. LoRA training (successful LLM traces, if ≥50 new tuples)
 *   4. Notify JS: learning_cycle_complete event
 *
 * Phase: 7 (Continuous Learning Scheduler)
 */
class LearningScheduler(private val context: Context) {

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var isRunning = false
    private var wakeLock: PowerManager.WakeLock? = null

    private val chargingReceiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            // Sync battery temperature to ThermalGuard on every battery update
            val tempTenths = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)
            ThermalGuard.updateFromBatteryTemp(tempTenths)

            val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL
            if (isCharging) maybeStartTraining() else cancelTraining()
        }
    }

    fun start() {
        context.registerReceiver(
            chargingReceiver,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )
    }

    fun stop() {
        try { context.unregisterReceiver(chargingReceiver) } catch (_: Exception) {}
        cancelTraining()
    }

    private fun maybeStartTraining() {
        if (isRunning) return
        // Never start a LoRA/REINFORCE cycle while the agent loop is actively running.
        // Both compete for LlamaEngine, RAM, and CPU — concurrent execution risks OOM
        // kills and corrupts in-flight inference. Training resumes on the next charging
        // event once the agent session ends.
        if (AgentLoop.isRunning()) {
            Log.i(TAG, "Training deferred — agent loop is active")
            return
        }
        if (!ThermalGuard.isTrainingSafe()) {
            Log.i(TAG, "Training skipped — thermal level: ${ThermalGuard.currentLevel}")
            return
        }
        if (!isDeviceIdle()) return
        runTrainingCycle()
    }

    private fun cancelTraining() {
        isRunning = false
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    // Callback emitted to AgentEventBus on cycle completion (read by TrainScreen via AgentViewModel)
    var onLearningCycleComplete: ((loraVersion: Int, policyVersion: Int) -> Unit)? = null

    private fun runTrainingCycle() {
        if (isRunning) return
        isRunning = true

        // Keep the CPU alive during training. The scheduler runs while the screen
        // is off and the device is charging — prime Doze mode territory. Without a
        // partial wake lock the OS can suspend the process mid-epoch, silently
        // killing the training coroutine and leaving isRunning stuck at true.
        // Timeout: 2 h safety cap so the lock can never be permanently orphaned
        // (e.g. if the finally block is skipped by a hard process kill).
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "AriaAgent:LearningScheduler"
        ).also { it.acquire(2 * 60 * 60 * 1000L) }
        Log.d(TAG, "WakeLock acquired for training cycle")

        scope.launch {
            try {
                AgentEventBus.emit("scheduler_training_started", mapOf("source" to "LearningScheduler"))

                val store = ExperienceStore.getInstance(context)

                // ── Step 1: Policy network REINFORCE update ────────────────────
                // Runs on successful experience tuples for repeated UI tasks
                PolicyNetwork.load(context)
                Log.i(TAG, "PolicyNetwork loaded for REINFORCE update")

                // ── Step 1.5: LLM reward enrichment ───────────────────────────
                // Re-score experience tuples using the loaded LlamaEngine so LoRA
                // training gets continuous quality weights instead of binary ±1.
                // Skipped automatically when LlamaEngine is not loaded (RAM conflict).
                //
                // IMPORTANT: resolve modelPath and modelId BEFORE enrichment so the
                // enrichment query reads from the SAME per-model experience set that
                // LoraTrainer.train() will later consume. Using the global flag here
                // would score a different (possibly disjoint) set of experiences,
                // making enriched quality weights useless for the actual training batch.
                val modelPath  = ModelManager.modelPath(context).absolutePath
                val enrichModelId = LoraTrainer.modelId(modelPath)

                // Read rlVisionMode from config — determines which model(s) score the rewards.
                val rlVisionMode: ModuleVisionMode = runCatching {
                    ConfigStore.getBlocking(context).rlVisionMode
                }.getOrElse { ModuleVisionMode.AUTO }

                val enrichedRewards = try {
                    LlmRewardEnricher.enrich(
                        tuples     = store.getUntrainedSuccessesFor(enrichModelId, limit = 20),
                        visionMode = rlVisionMode,
                    )
                } catch (e: Exception) {
                    Log.w(TAG, "LLM reward enrichment failed: ${e.message}")
                    emptyMap<String, Double>()
                }
                if (enrichedRewards.isNotEmpty()) {
                    Log.i(TAG, "LLM enriched ${enrichedRewards.size} rewards before LoRA training")
                }

                // ── Step 2: LoRA fine-tuning ───────────────────────────────────
                // Requires the base GGUF model path.
                // Both ExperienceStore (agent-generated) and ObjectLabelStore
                // (human-annotated, 3× weight) are passed so the trainer uses
                // all available signal — not just agent experiences.
                val labelStore = ObjectLabelStore.getInstance(context)
                val loraResult = LoraTrainer.train(context, store, modelPath,
                    labelStore      = labelStore,
                    enrichedRewards = enrichedRewards)

                if (loraResult.success && loraResult.adapterPath.isNotEmpty()) {
                    Log.i(TAG, "LoRA training complete: v${loraResult.loraVersion}, ${loraResult.samplesUsed} samples")

                    // ── Step 2a: Persist new adapter path to ConfigStore (DataStore) ──
                    // Phase 10: ConfigStore is the authoritative config store. Writing here
                    // ensures loadModel() and getConfig() always see the latest adapter path
                    // without any manual user action.
                    runCatching {
                        val current = ConfigStore.getBlocking(context)
                        ConfigStore.save(context, current.copy(loraAdapterPath = loraResult.adapterPath))
                        Log.i(TAG, "Persisted new loraAdapterPath → ${loraResult.adapterPath}")
                    }.onFailure { e ->
                        Log.w(TAG, "Failed to persist adapter path to ConfigStore: ${e.message}")
                    }

                    // ── Step 2b: Hot-reload adapter into running LlamaEngine ───
                    // If the model is currently loaded, swap in the new adapter
                    // without a full reload. The base weights remain in memory;
                    // only the LoRA delta matrices are swapped.
                    if (LlamaEngine.isLoaded()) {
                        val hotReloaded = LlamaEngine.loadLora(loraResult.adapterPath, scale = 0.8f)
                        if (hotReloaded) {
                            Log.i(TAG, "LoRA adapter v${loraResult.loraVersion} hot-reloaded into running LlamaEngine")
                        } else {
                            Log.w(TAG, "Hot-reload failed — adapter will be picked up on next loadModel()")
                        }
                    } else {
                        Log.i(TAG, "LlamaEngine not loaded — adapter saved; will load on next session start")
                    }
                } else {
                    Log.i(TAG, "LoRA skipped: ${loraResult.errorMessage}")
                }

                // ── Step 3: Save policy weights ────────────────────────────────
                PolicyNetwork.saveToFile(context)

                // ── Step 4: Passive Dream Mode ─────────────────────────────────
                // After real training, use the LLM to generate synthetic counterfactual
                // experience pairs from existing memories. Skipped automatically if
                // LlamaEngine is not loaded (avoids RAM competition with LoRA).
                val dreamCount = try {
                    DreamEngine.dream(context)
                } catch (e: Exception) {
                    Log.w(TAG, "Dream mode failed: ${e.message}")
                    0
                }
                if (dreamCount > 0) {
                    Log.i(TAG, "Dream mode: generated $dreamCount synthetic experiences")
                    AgentEventBus.emit("dream_cycle_complete", mapOf("count" to dreamCount))
                }

                // ── Step 4: Notify JS ──────────────────────────────────────────
                // policyVersion = cumulative Adam optimizer steps across all training cycles.
                // This strictly increases with each REINFORCE gradient update, giving JS
                // a monotonic counter it can display as "policy generation N".
                val loraVersion   = loraResult.loraVersion
                val policyVersion = PolicyNetwork.adamStepCount
                onLearningCycleComplete?.invoke(loraVersion, policyVersion)

            } catch (e: Exception) {
                Log.e(TAG, "Training cycle failed: ${e.message}")
            } finally {
                isRunning = false
                wakeLock?.let { if (it.isHeld) it.release() }
                wakeLock = null
                Log.d(TAG, "WakeLock released after training cycle")
                AgentEventBus.emit("scheduler_training_stopped", mapOf("source" to "LearningScheduler"))
            }
        }
    }

    companion object {
        private const val TAG = "LearningScheduler"
    }

    private fun syncBatteryTempToThermalGuard() {
        // Feed battery temperature to ThermalGuard for API < 29 fallback
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val tempTenths = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        ThermalGuard.updateFromBatteryTemp(tempTenths)
    }

    private fun isDeviceIdle(): Boolean {
        val pm = context.getSystemService(PowerManager::class.java)
        return !pm.isInteractive // screen is off
    }

    fun isTrainingRunning(): Boolean = isRunning
}
