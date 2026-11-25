#pragma once
#include <optix.h>
#include <vector_types.h>
struct RayBatch {
    OptixTraversableHandle tree;
    const float3 *points, *origins, *directions;
    const float *min_t, *max_t;
    int *value, *face;
    unsigned max_hits;
    float *dist;
};
