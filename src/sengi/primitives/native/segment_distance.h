#pragma once
#include "builtin.h"

namespace bvh {

template <typename Vec>
CUDA_CALLABLE inline Vec closest_segment(const Vec &p, const Vec &a, const Vec &b) {
    const auto ab = b - a;
    const float denom = wp::dot(ab, ab);
    float t = 0.0f;
    if (denom > 0.0f)
        t = wp::clamp(wp::dot(p - a, ab) / denom, 0.0f, 1.0f);
    return a + t * ab;
}

} // namespace bvh
