# ─── ARIA Agent — ProGuard / R8 rules (native-only build) ───────────────────
#
# This is the NATIVE-ONLY proguard configuration.
# React Native / Expo / Hermes / bridge rules have been removed — those classes
# are not on the classpath in a pure Kotlin build and R8 would error on them.
#
# Crash symptom to watch for in release builds: immediate crash after launch
# with no readable stack trace → add -keepattributes below and rebuild.
# ─────────────────────────────────────────────────────────────────────────────

# Preserve line numbers in crash stack traces (critical for debugging)
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# ─── ARIA Agent — all classes ─────────────────────────────────────────────────
-keep class com.ariaagent.mobile.** { *; }

# ─── JNI methods (llama.cpp, aria_math.cpp) ───────────────────────────────────
# R8 cannot see that native code calls these by name — keep all native methods.
-keepclasseswithmembernames class * {
    native <methods>;
}

# ─── ONNX Runtime (MiniLM sentence embeddings + MobileSAM encoder) ───────────
# The artifact is com.microsoft.onnxruntime:onnxruntime-android but the Java
# package inside is ai.onnxruntime — both must be kept so R8 does not strip
# classes that EmbeddingEngine and Sam2Engine reference by name.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
-keep class com.microsoft.onnxruntime.** { *; }
-dontwarn com.microsoft.onnxruntime.**

# ─── ML Kit OCR ───────────────────────────────────────────────────────────────
-keep class com.google.mlkit.** { *; }
-dontwarn com.google.mlkit.**

# ─── MediaPipe (object detection) ────────────────────────────────────────────
-keep class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# ─── OkHttp (model download with Range header resume) ────────────────────────
-keep class okhttp3.** { *; }
-dontwarn okhttp3.**
-dontwarn okio.**

# ─── Kotlin coroutines ────────────────────────────────────────────────────────
-keep class kotlinx.coroutines.** { *; }
-dontwarn kotlinx.coroutines.**

# ─── Jetpack / AndroidX ───────────────────────────────────────────────────────
-keep class androidx.datastore.** { *; }
-keep class androidx.compose.** { *; }

# ─── Suppress annotation-processor-only warnings ─────────────────────────────
-dontwarn javax.lang.model.**
-dontwarn autovalue.shaded.**
-dontwarn com.google.auto.value.**
