#pragma once
#include "builtin.h"

namespace bvh {

template <typename Vec>
CUDA_CALLABLE inline bool
box_interval(const Vec &o, const Vec &d, const Vec &inv_dir, int dimension, bool fast_aabb,
             const wp::vec3 &lower, const wp::vec3 &upper, float tmin, float tmax, float &enter) {
    enter = tmin;
    float leave = tmax;
    const auto update = [&](float t0, float t1) {
        float near_t = wp::min(t0, t1), far_t = wp::max(t0, t1);
        enter = wp::max(enter, near_t);
        leave = wp::min(leave, far_t);
    };
    if (fast_aabb) {
        for (int axis = 0; axis < dimension; ++axis) {
            const float t0 = (lower[axis] - o[axis]) * inv_dir[axis];
            const float t1 = (upper[axis] - o[axis]) * inv_dir[axis];
            update(t0, t1);
        }
    } else {
        for (int axis = 0; axis < dimension; ++axis) {
            const float component = d[axis];
            const float origin_component = o[axis];
            if (component == 0.0f) {
                if (origin_component < lower[axis] || origin_component > upper[axis])
                    return false;
            } else {
                const float t0 = (lower[axis] - origin_component) / component;
                const float t1 = (upper[axis] - origin_component) / component;
                update(t0, t1);
            }
        }
    }
    // Expand the reduced interval; narrow slabs can invert through rounding.
    const float padding = 9.5367431640625e-7f * wp::max(wp::abs(enter), wp::abs(leave)) + 1.e-37f;
    if (wp::isfinite(padding)) {
        enter -= padding;
        return enter <= leave + padding;
    }
    enter = tmin;
    return true;
}

template <class Vec>
CUDA_CALLABLE inline bool valid_ray(const Vec &origin, const Vec &direction, float minimum,
                                    float maximum, int excluded, int count, bool line,
                                    bool exclusion) {

    for (float v : origin.c)
        if (!(v == v && wp::abs(v) < 3.4028234e38f))
            return false;
    bool nonzero = false;
    for (float v : direction.c) {
        if (!(v == v && wp::abs(v) < 3.4028234e38f))
            return false;
        nonzero |= v != 0.f;
    }
    if (!(minimum <= maximum) || minimum == INFINITY || maximum == -INFINITY)
        return false;
    if (exclusion && (excluded < -1 || excluded >= count))
        return false;
    return line ? nonzero : minimum >= 0.f && wp::dot(direction, direction) > 1.e-30f;
}

template <class Ray>
CUDA_CALLABLE inline wp::vec2 ray_box(const Ray &ray, const wp::vec3 &lower, const wp::vec3 &upper,
                                      float minimum, float maximum, int dimension) {
    float enter;
    const bool hit = box_interval(ray.origin, ray.direction, ray.inverse, dimension, ray.fast,
                                  lower, upper, minimum, maximum, enter);
    return wp::vec2(enter, hit ? 1.f : 0.f);
}

template <class Ray>
CUDA_CALLABLE inline wp::vec3 first_ray_inverse(const Ray &ray) {
    if (ray.fast)
        return ray.inverse;
    wp::vec3 inverse;
    for (int axis = 0; axis < 3; ++axis) {
        const float d = ray.direction[axis];
        if (d != 0.f && wp::abs(d) < 1.e-20f)
            return wp::vec3(NAN); // Use the division-based slab test for tiny nonzero directions.
        inverse[axis] = d == 0.f ? INFINITY : 1.f / d;
    }
    return inverse;
}

template <class Ray>
CUDA_CALLABLE inline wp::vec2 first_ray_box(const Ray &ray, const wp::vec3 &inverse,
                                            const wp::vec3 &lower, const wp::vec3 &upper,
                                            float minimum, float maximum) {
    float enter = minimum, leave = maximum;
    for (int axis = 0; axis < 3; ++axis) {
        const float t0 = (lower[axis] - ray.origin[axis]) * inverse[axis];
        const float t1 = (upper[axis] - ray.origin[axis]) * inverse[axis];
        const bool positive = inverse[axis] > 0.f;
        // Float min/max ignore 0 * infinity on contact with a parallel slab.
        enter = wp::max(enter, positive ? t0 : t1);
        leave = wp::min(leave, positive ? t1 : t0);
    }
    if (wp::isfinite(enter) && leave > -INFINITY) {
        // First-hit rays have enter >= 0. For a near miss, enter >= leave;
        // bound both slab endpoints' rounding by 18 float epsilons on enter.
        // The unpadded value is used only to order children.
        constexpr float scale = 1.f - 18.f * 1.1920928955078125e-7f;
        return wp::vec2(enter, scale * enter - 2.1e-37f <= leave ? 1.f : 0.f);
    }
    for (int axis = 0; axis < 3; ++axis)
        if (ray.direction[axis] == 0.f &&
            (ray.origin[axis] < lower[axis] || ray.origin[axis] > upper[axis]))
            return wp::vec2(minimum, 0.f);
    return wp::vec2(minimum, 1.f);
}

} // namespace bvh
