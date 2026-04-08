package com.ariaagent.mobile

import android.app.Application

/**
 * MainApplication — ARIA application entry point.
 *
 * React Native, Expo, and SoLoader have been fully removed.
 * The NDK llama.cpp JNI library (llama-jni) is loaded lazily by LlamaEngine
 * via System.loadLibrary("llama-jni") in its companion object init block.
 * No explicit load is needed here.
 *
 * All UI is handled by ComposeMainActivity (Jetpack Compose).
 * All native services (LLM, OCR, RL, screen capture) are started by
 * AgentViewModel through AgentForegroundService.
 *
 * Migration phase: 8 — SoLoader / RN host fully removed.
 */
class MainApplication : Application() {

    override fun onCreate() {
        super.onCreate()
    }
}
