/*
 * opencl_stub.c — Link-time OpenCL 2.0 stub for Android/Mali builds.
 *
 * Purpose: satisfies link-time symbol resolution for ggml-opencl without
 * requiring the Mali vendor libOpenCL.so to be present at build time.
 * At runtime, Android's Bionic linker resolves DT_NEEDED "libOpenCL.so"
 * to the real Mali vendor library, which overrides these weak stubs.
 *
 * On devices without OpenCL (emulators, non-Mali), the weak functions
 * remain active and return CL_OUT_OF_HOST_MEMORY (-6), causing ggml-opencl
 * to fail initialisation gracefully and fall back to CPU inference.
 *
 * All types are defined manually so this file compiles without <CL/cl.h>,
 * breaking the circular dependency (headers needed to compile the stub
 * that provides the headers' linkage).
 */

#include <stddef.h>
#include <stdint.h>

typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef int64_t  cl_long;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef cl_uint  cl_bool;

typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_queue_properties;
typedef cl_ulong    cl_mem_properties;
typedef cl_uint     cl_buffer_create_type;
typedef cl_uint     cl_addressing_mode;
typedef cl_uint     cl_filter_mode;

typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;
typedef void* cl_event;
typedef void* cl_sampler;

typedef size_t cl_context_properties;

/* All stubs are weak symbols so the real Mali libOpenCL.so overrides them */
#define CL_STUB __attribute__((visibility("default"))) __attribute__((weak))

/* Error code for "no OpenCL platform found" */
#define CL_STUB_ERR (-6)

/* Platform / Device */
CL_STUB cl_int clGetPlatformIDs(cl_uint n, cl_platform_id* p, cl_uint* np)
    { if (np) *np = 0; return CL_STUB_ERR; }
CL_STUB cl_int clGetPlatformInfo(cl_platform_id p, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }
CL_STUB cl_int clGetDeviceIDs(cl_platform_id p, cl_device_type t, cl_uint n, cl_device_id* d, cl_uint* nd)
    { if (nd) *nd = 0; return CL_STUB_ERR; }
CL_STUB cl_int clGetDeviceInfo(cl_device_id d, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }

/* Context */
CL_STUB cl_context clCreateContext(
    const cl_context_properties* p, cl_uint n, const cl_device_id* d,
    void (*f)(const char*, const void*, size_t, void*), void* u, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_context clCreateContextFromType(
    const cl_context_properties* p, cl_device_type t,
    void (*f)(const char*, const void*, size_t, void*), void* u, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_int clReleaseContext(cl_context c)  { return CL_STUB_ERR; }
CL_STUB cl_int clRetainContext(cl_context c)   { return CL_STUB_ERR; }
CL_STUB cl_int clGetContextInfo(cl_context c, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }

/* Command Queue */
CL_STUB cl_command_queue clCreateCommandQueueWithProperties(
    cl_context c, cl_device_id d, const cl_queue_properties* p, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_command_queue clCreateCommandQueue(
    cl_context c, cl_device_id d, cl_bitfield p, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_int clReleaseCommandQueue(cl_command_queue q) { return CL_STUB_ERR; }
CL_STUB cl_int clRetainCommandQueue(cl_command_queue q)  { return CL_STUB_ERR; }
CL_STUB cl_int clGetCommandQueueInfo(cl_command_queue q, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }
CL_STUB cl_int clFlush(cl_command_queue q)  { return CL_STUB_ERR; }
CL_STUB cl_int clFinish(cl_command_queue q) { return CL_STUB_ERR; }

/* Memory */
CL_STUB cl_mem clCreateBuffer(cl_context c, cl_mem_flags f, size_t s, void* p, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_mem clCreateSubBuffer(cl_mem b, cl_mem_flags f,
    cl_buffer_create_type t, const void* i, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_mem clCreateBufferWithProperties(cl_context c,
    const cl_mem_properties* props, cl_mem_flags f, size_t s, void* p, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_mem clCreateImage(cl_context c, cl_mem_flags f,
    const void* fmt, const void* desc, void* p, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_int clReleaseMemObject(cl_mem m) { return CL_STUB_ERR; }
CL_STUB cl_int clRetainMemObject(cl_mem m)  { return CL_STUB_ERR; }
CL_STUB cl_int clGetMemObjectInfo(cl_mem m, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }

/* Enqueue read/write/copy/fill/map */
CL_STUB cl_int clEnqueueReadBuffer(cl_command_queue q, cl_mem b, cl_bool bl,
    size_t o, size_t s, void* p, cl_uint n, const cl_event* el, cl_event* e)
    { return CL_STUB_ERR; }
CL_STUB cl_int clEnqueueWriteBuffer(cl_command_queue q, cl_mem b, cl_bool bl,
    size_t o, size_t s, const void* p, cl_uint n, const cl_event* el, cl_event* e)
    { return CL_STUB_ERR; }
CL_STUB cl_int clEnqueueCopyBuffer(cl_command_queue q, cl_mem src, cl_mem dst,
    size_t so, size_t doff, size_t sz, cl_uint n, const cl_event* el, cl_event* e)
    { return CL_STUB_ERR; }
CL_STUB cl_int clEnqueueFillBuffer(cl_command_queue q, cl_mem b, const void* pat,
    size_t ps, size_t o, size_t s, cl_uint n, const cl_event* el, cl_event* e)
    { return CL_STUB_ERR; }
CL_STUB void* clEnqueueMapBuffer(cl_command_queue q, cl_mem b, cl_bool bl,
    cl_mem_flags f, size_t o, size_t s, cl_uint n, const cl_event* el, cl_event* e, cl_int* r)
    { if (r) *r = CL_STUB_ERR; return 0; }
CL_STUB cl_int clEnqueueUnmapMemObject(cl_command_queue q, cl_mem m, void* p,
    cl_uint n, const cl_event* el, cl_event* e)
    { return CL_STUB_ERR; }

/* Program */
CL_STUB cl_program clCreateProgramWithSource(cl_context c, cl_uint n,
    const char** s, const size_t* l, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_program clCreateProgramWithBinary(cl_context c, cl_uint n,
    const cl_device_id* d, const size_t* l, const unsigned char** b,
    cl_int* bs, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_int clBuildProgram(cl_program p, cl_uint n, const cl_device_id* d,
    const char* o, void (*f)(cl_program, void*), void* u)
    { return CL_STUB_ERR; }
CL_STUB cl_int clReleaseProgram(cl_program p)  { return CL_STUB_ERR; }
CL_STUB cl_int clRetainProgram(cl_program p)   { return CL_STUB_ERR; }
CL_STUB cl_int clGetProgramInfo(cl_program p, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }
CL_STUB cl_int clGetProgramBuildInfo(cl_program p, cl_device_id d,
    cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }

/* Kernel */
CL_STUB cl_kernel clCreateKernel(cl_program p, const char* n, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_int clReleaseKernel(cl_kernel k)  { return CL_STUB_ERR; }
CL_STUB cl_int clRetainKernel(cl_kernel k)   { return CL_STUB_ERR; }
CL_STUB cl_int clSetKernelArg(cl_kernel k, cl_uint i, size_t s, const void* v)
    { return CL_STUB_ERR; }
CL_STUB cl_int clGetKernelInfo(cl_kernel k, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }
CL_STUB cl_int clGetKernelWorkGroupInfo(cl_kernel k, cl_device_id d,
    cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }
CL_STUB cl_int clGetKernelSubGroupInfo(cl_kernel k, cl_device_id d,
    cl_uint n, size_t is, const void* iv, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }
CL_STUB cl_int clEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k,
    cl_uint wd, const size_t* go, const size_t* gs, const size_t* ls,
    cl_uint n, const cl_event* el, cl_event* e)
    { return CL_STUB_ERR; }

/* Event */
CL_STUB cl_int clReleaseEvent(cl_event e)         { return CL_STUB_ERR; }
CL_STUB cl_int clRetainEvent(cl_event e)           { return CL_STUB_ERR; }
CL_STUB cl_int clWaitForEvents(cl_uint n, const cl_event* el) { return CL_STUB_ERR; }
CL_STUB cl_int clGetEventInfo(cl_event e, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }
CL_STUB cl_int clGetEventProfilingInfo(cl_event e, cl_uint n, size_t s, void* v, size_t* sr)
    { return CL_STUB_ERR; }
CL_STUB cl_int clSetEventCallback(cl_event e, cl_int t,
    void (*f)(cl_event, cl_int, void*), void* u)
    { return CL_STUB_ERR; }
CL_STUB cl_int clEnqueueMarkerWithWaitList(cl_command_queue q, cl_uint n,
    const cl_event* el, cl_event* e)
    { return CL_STUB_ERR; }
CL_STUB cl_int clEnqueueBarrierWithWaitList(cl_command_queue q, cl_uint n,
    const cl_event* el, cl_event* e)
    { return CL_STUB_ERR; }
CL_STUB cl_int clEnqueueMarker(cl_command_queue q, cl_event* e)  { return CL_STUB_ERR; }
CL_STUB cl_int clEnqueueBarrier(cl_command_queue q)               { return CL_STUB_ERR; }

/* Sampler */
CL_STUB cl_sampler clCreateSampler(cl_context c, cl_bool n,
    cl_addressing_mode m, cl_filter_mode f, cl_int* e)
    { if (e) *e = CL_STUB_ERR; return 0; }
CL_STUB cl_int clReleaseSampler(cl_sampler s) { return CL_STUB_ERR; }
CL_STUB cl_int clGetSamplerInfo(cl_sampler s, cl_uint n, size_t sz, void* v, size_t* sr)
    { return CL_STUB_ERR; }

/* Extension function queries */
CL_STUB void* clGetExtensionFunctionAddress(const char* n)                       { return 0; }
CL_STUB void* clGetExtensionFunctionAddressForPlatform(cl_platform_id p, const char* n) { return 0; }

/* Unload compiler */
CL_STUB cl_int clUnloadPlatformCompiler(cl_platform_id p) { return CL_STUB_ERR; }
