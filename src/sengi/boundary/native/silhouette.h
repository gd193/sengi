#pragma once
#include "../../primitives/native/segment_distance.h"

namespace bvh {

template <typename Lower, typename Upper, typename Angles, typename Axes, typename Radii>
CUDA_CALLABLE inline bool
cone_may_have_silhouette(const wp::vec3 &p, int node, const Lower &node_lower,
                         const Upper &node_upper, const Angles &node_cone_sin_cos,
                         const Axes &node_cone_axis, const Radii &node_cone_radius) {
    const wp::vec2 angles = node_cone_sin_cos.data[node];
    const float sin_alpha = angles[0];
    const float cos_alpha = angles[1];
    if (sin_alpha < 0.0f)
        return false;
    if (cos_alpha <= 0.0f)
        return true;
    const wp::vec3 lower = node_lower.data[node];
    const wp::vec3 upper = node_upper.data[node];
    const wp::vec3 view = 0.5f * (lower + upper) - p;
    const wp::vec3 extent = 0.5f * (upper - lower);
    const wp::vec3 axis = node_cone_axis.data[node];
    const float projection = wp::dot(axis, view);
    // Box support along the cone axis bounds every edge endpoint.
    const float support =
        wp::abs(axis[0]) * extent[0] + wp::abs(axis[1]) * extent[1] + wp::abs(axis[2]) * extent[2];
    const float gap = wp::abs(projection) - support;
    if (gap <= 0.0f)
        return true;
    const wp::vec3 perpendicular = view - projection * axis;
    const float radius = node_cone_radius.data[node];
    const float limit = sin_alpha * (wp::sqrt(wp::dot(perpendicular, perpendicular)) + radius);
    const float slack = 2.0e-6f * (wp::sqrt(wp::dot(view, view)) + radius);
    return gap * cos_alpha <= limit + slack;
}

} // namespace bvh
