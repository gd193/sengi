#include "optix_mesh.h"
#include <optix_device.h>
#include <float.h>

extern "C" {
__constant__ RayBatch batch;
}
enum Mode { First, Any, Count, All, Classify };

template <Mode mode> __device__ void run() {
    unsigned i = optixGetLaunchIndex().x;
    float3 origin = mode == Classify ? batch.points[i] : batch.origins[i];
    float3 direction = mode == Classify ? make_float3(1, 0, 0) : batch.directions[i];
    float cap = mode == Classify ? FLT_MAX : fminf(batch.max_t[i], FLT_MAX);
    float start = mode == Classify ? 0.f : batch.min_t[i];
    unsigned value = 0, face = ~0u, distance = __float_as_uint(cap);
    const unsigned flags =
        mode == First ? OPTIX_RAY_FLAG_DISABLE_ANYHIT
        : mode == Any ? (OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT)
                      : OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    if constexpr (mode == All)
        for (unsigned j = 0; j < batch.max_hits; ++j) {
            batch.face[i * batch.max_hits + j] = -1;
            batch.dist[i * batch.max_hits + j] = __int_as_float(0x7f800000);
        }
    optixTrace(batch.tree, origin, direction, start, cap, 0.f, 255, flags, 0, 1, 0, value, face,
               distance);
    if constexpr (mode == First) {
        batch.face[i] = int(face);
        batch.dist[i] = __uint_as_float(distance);
    } else
        batch.value[i] = mode == Classify ? ((value & 1) ? -1 : 1) : int(value);
}
extern "C" __global__ void __raygen__first_hit() { run<First>(); }
extern "C" __global__ void __raygen__any_hit() { run<Any>(); }
extern "C" __global__ void __raygen__count_hits() { run<Count>(); }
extern "C" __global__ void __raygen__all_hits() { run<All>(); }
extern "C" __global__ void __raygen__classify_point() { run<Classify>(); }
extern "C" __global__ void __closesthit__first() {
    optixSetPayload_0(1);
    optixSetPayload_1(optixGetPrimitiveIndex());
    optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
}
extern "C" __global__ void __closesthit__any() { optixSetPayload_0(1); }
extern "C" __global__ void __anyhit__count() {
    optixSetPayload_0(optixGetPayload_0() + 1);
    optixIgnoreIntersection();
}
extern "C" __global__ void __anyhit__all() {
    unsigned count = optixGetPayload_0(), i = optixGetLaunchIndex().x;
    if (count < batch.max_hits) {
        batch.face[batch.max_hits * i + count] = optixGetPrimitiveIndex();
        batch.dist[batch.max_hits * i + count] = optixGetRayTmax();
    }
    optixSetPayload_0(count + 1);
    optixIgnoreIntersection();
}
extern "C" __global__ void __miss__empty() { optixSetPayload_0(optixGetPayload_0()); }
