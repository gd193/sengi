#pragma once
#include "builtin.h"

namespace bvh {

template <typename Vec>
CUDA_CALLABLE inline Vec closest_segment(const Vec &p, const Vec &a, const Vec &b) {
    const auto ab = b - a;
    const float from_a = wp::dot(p - a, ab);
    const float from_b = wp::dot(b - p, ab);
    const float length2 = wp::dot(ab, ab);

    // Use one division for either anchor. Measuring the offset from the
    // nearer endpoint preserves short displacements along long segments.
    const float offset = wp::clamp(wp::min(from_a, from_b) / (length2 > 0.f ? length2 : 1.f), 0.f, 0.5f);
    return from_a <= from_b ? a + offset * ab : b - offset * ab;
}

} // namespace bvh
