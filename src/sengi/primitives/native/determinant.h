#pragma once
#include "builtin.h"

namespace bvh {

CUDA_CALLABLE inline double cross2(double ax, double ay, double bx, double by) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(__dmul_rn(ax, by), -__dmul_rn(ay, bx));
#else
    volatile double a = ax * by;
    volatile double b = ay * bx;
    return a - b;
#endif
}

} // namespace bvh
