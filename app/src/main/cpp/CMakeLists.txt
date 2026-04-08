cmake_minimum_required(VERSION 3.22.1)

project(aria-llama-jni VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Custom CMake module path — must come first so our FindOpenCL.cmake is ───
# found before CMake's built-in one.  This makes find_package(OpenCL REQUIRED)
# (called inside ggml-opencl/CMakeLists.txt) resolve to the inline stub target.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ─── llama.cpp submodule path ────────────────────────────────────────────────
# To add llama.cpp: from artifacts/mobile/android/app/src/main/cpp/
#   git submodule add https://github.com/ggerganov/llama.cpp llama.cpp
#   git submodule update --init --recursive
set(LLAMA_DIR ${CMAKE_SOURCE_DIR}/llama.cpp)

# ─── Hardware flags for Exynos 9611 (Samsung Galaxy M31) ─────────────────────
# ARM Cortex-A73 big cores → use 4 threads (arm64-v8a, neon enabled by default)
#
# Vulkan is disabled: Mali-G72 MP3 ships Vulkan 1.1 but llama.cpp's ggml-vulkan
# requires Vulkan 1.2, so it would never be used at runtime. Disabling it here
# avoids requiring vulkan.hpp (the C++ wrapper header) which the Android NDK
# does not ship.
set(GGML_VULKAN OFF CACHE BOOL "Vulkan disabled — Mali-G72 is Vulkan 1.1 only; llama.cpp requires 1.2" FORCE)
message(STATUS "Vulkan: disabled for Android build (Mali-G72 Vulkan 1.1 < required 1.2)")
set(GGML_METAL OFF CACHE BOOL "Metal is iOS only" FORCE)
set(GGML_CUDA  OFF CACHE BOOL "No CUDA on Android" FORCE)
set(GGML_BLAS  OFF CACHE BOOL "No external BLAS needed for Q4_K_M" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Cross-compiling for arm64-v8a" FORCE)

# ─── OpenCL (Mali-G72 OpenCL 2.0) ────────────────────────────────────────────
# Strategy (three-tier, evaluated in order):
#
#   Tier 1 — Inline build:  stubs/opencl_stub.c exists AND CL headers found.
#     Compiles a weak-symbol ARM64 libOpenCL.so from source using the NDK
#     toolchain (already set up for cross-compilation). The real Mali vendor
#     libOpenCL.so overrides the weak symbols at runtime via Bionic's DT_NEEDED
#     resolution. CL headers are fetched from Khronos via FetchContent if not
#     already in stubs/include/CL/.
#
#   Tier 2 — Pre-built stub:  stubs/libOpenCL.so exists (legacy CI artifact).
#     Headers still required in stubs/include/CL/ or fetched via FetchContent.
#
#   Tier 3 — CPU fallback:  Neither stub nor source found.
#     GGML_OPENCL=OFF; inference runs on CPU (slower but correct).
#
# Our cmake/FindOpenCL.cmake overrides the built-in Find module so that
# find_package(OpenCL REQUIRED) in ggml-opencl/CMakeLists.txt picks up the
# aria_opencl_stub target (Tier 1/2) rather than searching the host system.

set(_opencl_stub_src  "${CMAKE_CURRENT_SOURCE_DIR}/stubs/opencl_stub.c")
set(_opencl_prebuilt  "${CMAKE_CURRENT_SOURCE_DIR}/stubs/libOpenCL.so")
set(_opencl_inc_local "${CMAKE_CURRENT_SOURCE_DIR}/stubs/include")

# ── Locate or download OpenCL headers ────────────────────────────────────────
if(EXISTS "${_opencl_inc_local}/CL/cl.h")
    set(ARIA_OPENCL_INCLUDE_DIR "${_opencl_inc_local}")
    message(STATUS "OpenCL headers: found locally at stubs/include/CL/cl.h")
else()
    message(STATUS "OpenCL headers: stubs/include/CL/cl.h not found — fetching from Khronos")
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_Declare(
        opencl_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
        GIT_TAG        v2024.10.24
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(opencl_headers)
    if(opencl_headers_POPULATED)
        set(ARIA_OPENCL_INCLUDE_DIR "${opencl_headers_SOURCE_DIR}")
        message(STATUS "OpenCL headers: fetched → ${ARIA_OPENCL_INCLUDE_DIR}")
    else()
        message(WARNING "OpenCL headers: FetchContent failed — OpenCL will be disabled")
        set(ARIA_OPENCL_INCLUDE_DIR "")
    endif()
endif()

# ── Build or locate the stub library ─────────────────────────────────────────
if(ARIA_OPENCL_INCLUDE_DIR AND EXISTS "${_opencl_stub_src}")
    # Tier 1: build ARM64 stub inline using the NDK toolchain
    add_library(aria_opencl_stub SHARED "${_opencl_stub_src}")
    target_compile_definitions(aria_opencl_stub PRIVATE
        CL_TARGET_OPENCL_VERSION=200
        CL_USE_DEPRECATED_OPENCL_1_2_APIS)
    set_target_properties(aria_opencl_stub PROPERTIES
        OUTPUT_NAME "OpenCL"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/opencl_stub")
    set(GGML_OPENCL                  ON    CACHE BOOL   "OpenCL enabled — inline stub built from source" FORCE)
    set(GGML_OPENCL_EMBED_KERNELS    ON    CACHE BOOL   "Embed .cl kernels into binary"                  FORCE)
    set(GGML_OPENCL_TARGET_VERSION   "300" CACHE STRING "Target OpenCL 3.0 — Mali-G72 r0p1 Full Profile" FORCE)
    set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL  "Disable Adreno kernels — device is Mali not Adreno" FORCE)
    message(STATUS "OpenCL: Tier 1 — building inline ARM64 stub from ${_opencl_stub_src}")

elseif(ARIA_OPENCL_INCLUDE_DIR AND EXISTS "${_opencl_prebuilt}")
    # Tier 2: use a pre-built stub from stubs/libOpenCL.so (legacy CI artifact)
    add_library(aria_opencl_stub SHARED IMPORTED GLOBAL)
    set_target_properties(aria_opencl_stub PROPERTIES
        IMPORTED_LOCATION "${_opencl_prebuilt}")
    set(GGML_OPENCL                  ON    CACHE BOOL   "OpenCL enabled — pre-built stub"               FORCE)
    set(GGML_OPENCL_EMBED_KERNELS    ON    CACHE BOOL   "Embed .cl kernels into binary"                  FORCE)
    set(GGML_OPENCL_TARGET_VERSION   "300" CACHE STRING "Target OpenCL 3.0 — Mali-G72 r0p1 Full Profile" FORCE)
    set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL  "Disable Adreno kernels — device is Mali not Adreno" FORCE)
    message(STATUS "OpenCL: Tier 2 — using pre-built stub at ${_opencl_prebuilt}")

else()
    # Tier 3: CPU fallback
    set(GGML_OPENCL OFF CACHE BOOL "OpenCL disabled — no stub source or pre-built lib found" FORCE)
    if(NOT ARIA_OPENCL_INCLUDE_DIR)
        message(STATUS "OpenCL: Tier 3 (CPU fallback) — headers unavailable")
    else()
        message(STATUS "OpenCL: Tier 3 (CPU fallback) — opencl_stub.c missing from stubs/")
    endif()
endif()
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
# LLAMA_BUILD_COMMON defaults to LLAMA_STANDALONE, which is OFF when llama.cpp
# is consumed via add_subdirectory (not a top-level build). Force it ON so the
# 'common' CMake target is always produced and can be linked by llama-jni.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build llama.cpp common utilities" FORCE)

# ─── Add llama.cpp as a static library ───────────────────────────────────────
add_subdirectory(${LLAMA_DIR} llama_build)

# ─── mtmd: multimodal (vision) library ───────────────────────────────────────
# mtmd is under tools/mtmd/ and is NOT built by add_subdirectory(llama_dir)
# unless LLAMA_BUILD_TOOLS=ON (which requires LLAMA_STANDALONE). We add it
# directly so the 'mtmd' CMake target is available for linking.
# mtmd exposes its own PUBLIC include path (tools/mtmd/) so we do NOT need to
# manually add it to target_include_directories below.
#
# LLAMA_INSTALL_VERSION is set inside llama.cpp's own root CMakeLists.txt scope
# and does NOT propagate back to this parent scope (CMake subdirectory scoping).
# tools/mtmd/CMakeLists.txt:40 calls set_target_properties(... VERSION
# ${LLAMA_INSTALL_VERSION} ...) which silently expands to an empty string here,
# causing CMake to see malformed arguments and abort with "incorrect number of
# arguments". Define a fallback so the property call is always well-formed.
if(NOT DEFINED LLAMA_INSTALL_VERSION)
    set(LLAMA_INSTALL_VERSION "0.0.0")
endif()
add_subdirectory(${LLAMA_DIR}/tools/mtmd mtmd_build)

# ─── Android log library ─────────────────────────────────────────────────────
find_library(LOG_LIB log)

# ─── Vulkan library (for pick_backend_device() vkEnumerateInstanceVersion check)
# libvulkan.so is a system library on every Android device (API 24+). The NDK
# provides the headers; the loader is present at runtime even on devices with no
# Vulkan GPU.  Only link it when Vulkan was compiled in so the
# #include <vulkan/vulkan.h> guard and this link target stay in sync.
if(GGML_VULKAN)
    find_library(VULKAN_LIB vulkan)
endif()

# ─── Our JNI bridge shared library ───────────────────────────────────────────
# Two source files:
#   llama_jni.cpp  — wraps llama.cpp for LLM inference
#   aria_math.cpp  — NEON SIMD math for EmbeddingEngine + PolicyNetwork
add_library(
    llama-jni
    SHARED
    llama_jni.cpp
    aria_math.cpp
)

target_include_directories(llama-jni PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/ggml/include
    ${LLAMA_DIR}/ggml/src
)

# NOTE: 'common' target is produced by llama.cpp's own CMake when
# LLAMA_BUILD_COMMON=ON (set above). The include path for common/ headers is
# already listed in target_include_directories above.
target_link_libraries(llama-jni
    llama           # llama.cpp core
    common          # llama.cpp common utilities
    mtmd            # multimodal: CLIP image encoder + projection layer
    ${LOG_LIB}      # __android_log_print
)
# Link libvulkan.so when Vulkan is compiled in so that vkEnumerateInstanceVersion
# (called in pick_backend_device) resolves at link time.
if(GGML_VULKAN AND VULKAN_LIB)
    target_link_libraries(llama-jni ${VULKAN_LIB})
endif()
# Ensure the OpenCL stub shared library is packaged in the APK alongside llama-jni.
# ggml-opencl links to it transitively via llama, but we explicitly add it here so
# the Android Gradle plugin sees it as a direct dependency of llama-jni and includes
# libOpenCL.so in the jni/arm64-v8a/ directory of the APK.
if(GGML_OPENCL AND TARGET aria_opencl_stub)
    target_link_libraries(llama-jni aria_opencl_stub)
endif()

# ─── Compiler flags ──────────────────────────────────────────────────────────
# -march=armv8-a      : Exynos 9611 CPU arch (A73 = ARMv8.0-A, A53 = ARMv8.0-A).
#                       NEON is always enabled on arm64-v8a — no -mfpu flag needed.
# -mcpu=cortex-a73    : Tune instruction scheduling for the A73 big cores.
#                       Uses ARMv8.0-A instructions only (no ARMv8.2 extensions),
#                       so binaries run correctly on both A73 (big) and A53 (LITTLE).
# -ffast-math         : Aggressive float optimisation for inference throughput.
# -fno-finite-math-only: ggml/vec.h rejects __FINITE_MATH_ONLY__ (set by -ffast-math).
target_compile_options(llama-jni PRIVATE
    -O3
    -march=armv8-a
    -mcpu=cortex-a73
    -ffast-math
    -fno-finite-math-only
    -funroll-loops
)

# ─── On-device LoRA training via llama_opt_* API ─────────────────────────────
# ENABLED: LLAMA_HAS_TRAINING is defined. The JNI function nativeTrainLora()
# in llama_jni.cpp is compiled with the real llama_opt_* / ggml_opt_* code path.
#
# The build uses the vendored llama.cpp submodule at android/app/src/main/cpp/llama.cpp/.
# The llama_jni.cpp training block is guarded by #if defined(LLAMA_HAS_TRAINING)
# so removing this line is safe and causes a graceful fallback:
#   → nativeTrainLora() returns JNI_FALSE
#   → LoraTrainer.kt calls stubTrainLora() — writes metadata-only .bin
#   → No inference regression
#
# If the vendored llama.cpp version does not export these symbols, the linker
# will fail at build time (not at runtime). Remove the definition below to
# disable training and get a guaranteed clean build, then re-enable once the
# API surface is verified.
add_compile_definitions(LLAMA_HAS_TRAINING)
