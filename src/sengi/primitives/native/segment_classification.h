#pragma once
#include "determinant.h"

namespace bvh {
template <class Meta>
CUDA_CALLABLE inline int classify_segment(const wp::vec2 &point, const Meta &meta, int slot) {
    const wp::vec3 a = meta.vertices.data[meta.indices.data[2 * slot]];
    const wp::vec3 b = meta.vertices.data[meta.indices.data[2 * slot + 1]];
    const double ax = double(a[0]) - double(point[0]), ay = double(a[1]) - double(point[1]);
    const double bx = double(b[0]) - double(point[0]), by = double(b[1]) - double(point[1]);
    const double cross = cross2(ax, ay, bx, by);
    if (cross == 0.0 && wp::min(ax, bx) <= 0 && wp::max(ax, bx) >= 0 && wp::min(ay, by) <= 0 &&
        wp::max(ay, by) >= 0)
        return 2;
    if ((ay <= 0 && by > 0) || (by <= 0 && ay > 0))
        return cross / (by - ay) > 0 ? 1 : 0;
    return 0;
}
} // namespace bvh
