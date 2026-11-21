#pragma once
#include "builtin.h"

namespace bvh {

CUDA_CALLABLE inline uint32_t triangle_vertex_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

CUDA_CALLABLE inline float triangle_vertex_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// Each float is a signed 24-bit significand times 2^(shift - 149).
// Accumulate the nine terms of sum(p² + v² - 2pv) in units of 2^-298.
// 576 bits cover even the sum of three squared differences of opposite
// maximum finite coordinates. Unsigned carries and borrows are modulo 2^576.

} // namespace bvh
