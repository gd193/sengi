#pragma once
#include "determinant.h"

namespace bvh {

CUDA_CALLABLE inline bool owns_edge(double ax, double ay, double bx, double by,
                                    double orientation) {
    const double dx = orientation * (bx - ax), dy = orientation * (by - ay);
    return dy > 0 || (dy == 0 && dx < 0);
}

// Project each vertex in the same cyclic coordinate order as the prepared ray.
CUDA_CALLABLE inline wp::vec3 ray_frame_vertex(const wp::vec3 &v, int dominant_axis) {
    if (dominant_axis == 0)
        return wp::vec3(v[1], v[2], v[0]);
    if (dominant_axis == 1)
        return wp::vec3(v[2], v[0], v[1]);
    return v;
}

template <class Ray>
CUDA_CALLABLE inline wp::vec2 triangle_crossing(const Ray &ray, float min_t, float max_t,
                                                const wp::vec3 &a0, const wp::vec3 &b0,
                                                const wp::vec3 &c0, bool half_open, bool store_times) {
    const wp::vec3 o = ray.origin, d = ray.direction;
    const float lo_t = min_t, hi_t = max_t;

    const auto triangle_t = [&](float &result) {
        // Reuse one predicate at both precisions. -1 asks for the double path.
        const auto solve = [&](auto zero) {
            using S = decltype(zero);
            S x[3], y[3], z[3], uncertainty[3];
            const S sx = sizeof(S) == sizeof(float) ? static_cast<S>(ray.shear[0])
                                                    : static_cast<S>(ray.frame_direction[0]) /
                                                          static_cast<S>(ray.frame_direction[2]);
            const S sy = sizeof(S) == sizeof(float) ? static_cast<S>(ray.shear[1])
                                                    : static_cast<S>(ray.frame_direction[1]) /
                                                          static_cast<S>(ray.frame_direction[2]);
            constexpr float rounding = 8.0f * 1.1920928955078125e-7f;
            for (int i = 0; i < 3; ++i) {
                const wp::vec3 v = i == 0 ? a0 : i == 1 ? b0 : c0;
                const wp::vec3 frame = ray_frame_vertex(v, ray.dominant_axis);
                const S a = static_cast<S>(frame[0]) - static_cast<S>(ray.frame_origin[0]);
                const S b = static_cast<S>(frame[1]) - static_cast<S>(ray.frame_origin[1]);
                z[i] = static_cast<S>(frame[2]) - static_cast<S>(ray.frame_origin[2]);
                x[i] = a - sx * z[i];
                y[i] = b - sy * z[i];
                if constexpr (sizeof(S) == sizeof(float)) {
                    uncertainty[i] = rounding * wp::max(wp::abs(a) + wp::abs(sx * z[i]),
                                                        wp::abs(b) + wp::abs(sy * z[i])) +
                                     static_cast<S>(1.e-37f);
                }
            }
            S e[3], error[3];
            bool positive = false, negative = false, uncertain = false;
            for (int i = 0; i < 3; ++i) {
                const int a = (i + 1) % 3, b = (i + 2) % 3;
                if constexpr (sizeof(S) == sizeof(float)) {
                    e[i] = x[a] * y[b] - y[a] * x[b];
                    error[i] =
                        rounding * (wp::abs(x[a] * y[b]) + wp::abs(y[a] * x[b])) +
                        uncertainty[a] * (wp::abs(x[b]) + wp::abs(y[b]) + 2 * uncertainty[b]) +
                        uncertainty[b] * (wp::abs(x[a]) + wp::abs(y[a]));
                } else {
                    e[i] = cross2(x[a], y[a], x[b], y[b]);
                    error[i] = 0;
                }
                positive |= e[i] > error[i];
                negative |= e[i] < -error[i];
                if constexpr (sizeof(S) == sizeof(float))
                    uncertain |= !(wp::abs(e[i]) > error[i]);
            }
            if (positive && negative)
                return 0;
            if constexpr (sizeof(S) == sizeof(float)) {
                if (uncertain)
                    return -1;
            }
            const S det = e[0] + e[1] + e[2];
            if (det == 0)
                return 0; // Parallel/coplanar faces have no isolated crossing.
            S t;
            if constexpr (sizeof(S) == sizeof(float)) {
                const S numerator = e[0] * z[0] + e[1] * z[1] + e[2] * z[2];
                const S denominator = det * static_cast<S>(ray.frame_direction[2]);
                t = numerator / denominator;
                const S num_error =
                    rounding *
                        (wp::abs(e[0] * z[0]) + wp::abs(e[1] * z[1]) + wp::abs(e[2] * z[2])) +
                    error[0] * wp::abs(z[0]) + error[1] * wp::abs(z[1]) + error[2] * wp::abs(z[2]);
                const S den_error =
                    wp::abs(ray.frame_direction[2]) * (error[0] + error[1] + error[2]) +
                    rounding * wp::abs(denominator);
                if (!(wp::abs(denominator) > den_error) || !wp::isfinite(t))
                    return -1;
                const S t_error =
                    (num_error + wp::abs(t) * den_error) / (wp::abs(denominator) - den_error);
                if (!wp::isfinite(t_error))
                    return -1;
                if (t + t_error < lo_t || t - t_error > hi_t)
                    return 0;
                if (t - t_error < lo_t || t + t_error > hi_t)
                    return -1;
            } else {
                // Recompute from the original plane at ambiguous interval endpoints.
                const wp::vec3 a = a0;
                const wp::vec3 b = b0;
                const wp::vec3 c = c0;
                S ab[3], ac[3], normal[3];
                for (int i = 0; i < 3; ++i) {
                    ab[i] = static_cast<S>(b[i]) - static_cast<S>(a[i]);
                    ac[i] = static_cast<S>(c[i]) - static_cast<S>(a[i]);
                }
                S plane = 0, velocity = 0;
                for (int i = 0; i < 3; ++i) {
                    const int u = (i + 1) % 3, v = (i + 2) % 3;
                    normal[i] = cross2(ab[u], ab[v], ac[u], ac[v]);
                    plane += normal[i] * (static_cast<S>(a[i]) - static_cast<S>(o[i]));
                    velocity += normal[i] * static_cast<S>(d[i]);
                }
                if (velocity == 0)
                    return 0;
                t = plane / velocity;
                if (t < lo_t || t > hi_t)
                    return 0;
                const S orientation = det > 0 ? 1 : -1;
                for (int i = 0; i < 3; ++i) {
                    if (half_open && e[i] == 0) {
                        const int a = (i + 1) % 3, b = (i + 2) % 3;
                        if (!owns_edge(x[a], y[a], x[b], y[b], orientation))
                            return 0;
                    }
                }
            }
            result = static_cast<float>(t);
            return store_times && !wp::isfinite(result) ? -2 : 1;
        };
        const int fast = solve(0.0f);
        return fast == -1 ? solve(0.0) : fast;
    };

    float t = 0;
    const int hit = triangle_t(t);
    return wp::vec2(t, hit < 0 ? -1.f : static_cast<float>(hit));
}

} // namespace bvh
