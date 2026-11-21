#pragma once
#include "triangle_distance.h"
#include "box_distance.h"

namespace bvh {

CUDA_CALLABLE inline wp::vec4 triangle_nearest_candidate(const wp::vec3 &query, const wp::vec3 &a,
                                                          const wp::vec3 &b, const wp::vec3 &c,
                                                          float limit) {
    const wp::vec3 lower = wp::min(a, wp::min(b, c)), upper = wp::max(a, wp::max(b, c));
    if (conservative_box_score<true>(query, lower, upper) > limit)
        return wp::vec4(0.f, 0.f, 0.f, -1.f);
    const wp::vec3 point = wp::min(wp::max(triangle_closest_fp32(query, a, b, c), lower), upper);
    const wp::vec3 delta = query - point;
    // Match the box bound's accumulation order. Explicit FMA also gives
    // CPU and CUDA the same FP32 score when a candidate lies at the cap.
    const float xy = fmaf(delta[0], delta[0], delta[1] * delta[1]);
    const float score = fmaf(delta[2], delta[2], xy);
    return wp::vec4(point[0], point[1], point[2], score);
}

template <typename Data>
CUDA_CALLABLE inline wp::vec4 triangle_nearest_at(const wp::vec3 &query, const Data &data,
                                                   int slot, float limit) {
    const wp::vec3 a = data.a.data[slot], b = data.b.data[slot], c = data.c.data[slot];
    return triangle_nearest_candidate(query, a, b, c, limit);
}

} // namespace bvh
