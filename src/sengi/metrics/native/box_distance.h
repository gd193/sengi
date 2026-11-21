#pragma once
#include "builtin.h"
#include "rounding.h"

namespace bvh {

CUDA_CALLABLE inline float box_distance_sq(const wp::vec3 &p, const wp::vec3 &lower,
                                           const wp::vec3 &upper) {
    const float dx = wp::max(wp::max(lower[0] - p[0], 0.0f), p[0] - upper[0]);
    const float dy = wp::max(wp::max(lower[1] - p[1], 0.0f), p[1] - upper[1]);
    const float dz = wp::max(wp::max(lower[2] - p[2], 0.0f), p[2] - upper[2]);
    return dx * dx + dy * dy + dz * dz;
}

CUDA_CALLABLE inline float box_round_toward_zero(float value) {
    const uint32_t bits = triangle_vertex_bits(value);
    const uint32_t magnitude = bits & 0x7fffffffu;
    return magnitude != 0 && magnitude <= 0x7f800000u ? triangle_vertex_float(bits - 1) : value;
}

template <bool FloatOnly = false>
CUDA_CALLABLE inline float conservative_box_score(const wp::vec3 &query, const wp::vec3 &lower,
                                                  const wp::vec3 &upper) {
    const wp::vec3 point = wp::min(wp::max(query, lower), upper);
#if defined(__CUDA_ARCH__)
    const float x = __fsub_rz(query[0], point[0]);
    const float y = __fsub_rz(query[1], point[1]);
    const float z = __fsub_rz(query[2], point[2]);
    if constexpr (FloatOnly)
        return __fmaf_rd(z, z, __fmaf_rd(x, x, __fmul_rd(y, y)));
    return __fmaf_rd(z, z, __fmaf_rd(y, y, __fmul_rd(x, x)));
#else
    if constexpr (!FloatOnly) {
        const double x = double(query[0]) - double(point[0]);
        const double y = double(query[1]) - double(point[1]);
        const double z = double(query[2]) - double(point[2]);
        const double score = x*x + y*y + z*z;
        float rounded = float(score);
        // Positive float bit patterns are ordered by value. Step down when
        // rounding to nearest crossed above the double-precision box score.
        if (double(rounded) > score) {
            uint32_t bits;
            memcpy(&bits, &rounded, sizeof(bits));
            --bits;
            memcpy(&rounded, &bits, sizeof(rounded));
        }
        return rounded;
    }
    // Step toward zero after each host operation so the result is a lower
    // bound under round-to-nearest without changing the rounding environment.
    const float x = box_round_toward_zero(query[0] - point[0]);
    const float y = box_round_toward_zero(query[1] - point[1]);
    const float z = box_round_toward_zero(query[2] - point[2]);
    const float yy = box_round_toward_zero(y * y);
    const float xy = box_round_toward_zero(fmaf(x, x, yy));
    return box_round_toward_zero(fmaf(z, z, xy));
#endif
}

} // namespace bvh
