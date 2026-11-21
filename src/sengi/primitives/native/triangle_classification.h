#pragma once
#include "triangle_crossing.h"

namespace bvh {
CUDA_CALLABLE inline int triangle_event(const wp::vec3 &p, const wp::array_t<wp::vec3> &vertices,
                                        const wp::vec3i &ids) {

    const wp::vec3 a = vertices.data[ids[0]] - p;
    const wp::vec3 b = vertices.data[ids[1]] - p;
    const wp::vec3 c = vertices.data[ids[2]] - p;
    const wp::vec3 e_fast(b[1] * c[2] - b[2] * c[1], c[1] * a[2] - c[2] * a[1],
                          a[1] * b[2] - a[2] * b[1]);
    // Bounds include coordinate subtraction, products and summation.
    constexpr float rounding = 8.0f * 1.1920928955078125e-7f;
    const wp::vec3 error(
        wp::max(rounding * (wp::abs(b[1] * c[2]) + wp::abs(b[2] * c[1])), 1.e-37f),
        wp::max(rounding * (wp::abs(c[1] * a[2]) + wp::abs(c[2] * a[1])), 1.e-37f),
        wp::max(rounding * (wp::abs(a[1] * b[2]) + wp::abs(a[2] * b[1])), 1.e-37f));
    bool positive = false, negative = false, uncertain = false;
    for (int i = 0; i < 3; ++i) {
        positive |= e_fast[i] > error[i];
        negative |= e_fast[i] < -error[i];
        uncertain |= !(wp::abs(e_fast[i]) > error[i]);
    }
    if (positive && negative)
        return 0;
    if (!uncertain) {
        const wp::vec3 terms(e_fast[0] * a[0], e_fast[1] * b[0], e_fast[2] * c[0]);
        const float t = terms[0] + terms[1] + terms[2];
        const float t_error =
            rounding * (wp::abs(terms[0]) + wp::abs(terms[1]) + wp::abs(terms[2])) +
            error[0] * wp::abs(a[0]) + error[1] * wp::abs(b[0]) + error[2] * wp::abs(c[0]);
        if (wp::abs(t) > t_error)
            return (t > 0.0f) == positive ? 1 : 0;
    }
    double v[3][3];
    for (int i = 0; i < 3; ++i) {
        const wp::vec3 vertex = vertices.data[ids[i]];
        for (int j = 0; j < 3; ++j)
            v[i][j] = static_cast<double>(vertex[j]) - static_cast<double>(p[j]);
    }
    double e[3];
    for (int i = 0; i < 3; ++i)
        e[i] = cross2(v[(i + 1) % 3][1], v[(i + 1) % 3][2], v[(i + 2) % 3][1], v[(i + 2) % 3][2]);
    double det = e[0] + e[1] + e[2];
    const bool parallel = det == 0.0;
    int axis = 0;
    if (parallel) {
        // A parallel face can contain the query without crossing the ray.
        double normal[3];
        for (int j = 0; j < 3; ++j) {
            const int u = (j + 1) % 3, w = (j + 2) % 3;
            normal[j] =
                cross2(v[1][u] - v[0][u], v[1][w] - v[0][w], v[2][u] - v[0][u], v[2][w] - v[0][w]);
        }
        axis = wp::abs(normal[1]) > wp::abs(normal[2]) ? 1 : 2;
        const int u = (axis + 1) % 3, w = (axis + 2) % 3;
        for (int i = 0; i < 3; ++i)
            e[i] =
                cross2(v[(i + 1) % 3][u], v[(i + 1) % 3][w], v[(i + 2) % 3][u], v[(i + 2) % 3][w]);
        det = e[0] + e[1] + e[2];
        if (det == 0.0)
            return 0;
    }
    const double orientation = det > 0.0 ? 1.0 : -1.0;
    for (int i = 0; i < 3; ++i) {
        e[i] *= orientation;
        if (e[i] < 0.0)
            return 0;
    }
    const double t = e[0] * v[0][axis] + e[1] * v[1][axis] + e[2] * v[2][axis];
    if (t == 0.0)
        return 2;
    if (parallel || t < 0.0)
        return 0;
    // Top-left ownership counts a shared edge or vertex exactly once.
    for (int i = 0; i < 3; ++i) {
        if (e[i] == 0.0) {
            const int a = (i + 1) % 3, b = (i + 2) % 3;
            if (!owns_edge(v[a][1], v[a][2], v[b][1], v[b][2], orientation))
                return 0;
        }
    }
    return 1;
}

template <class Meta>
CUDA_CALLABLE inline int classify_triangle(const wp::vec3 &point, const Meta &meta, int slot) {
    const wp::vec3i ids(meta.indices.data[3 * slot], meta.indices.data[3 * slot + 1],
                        meta.indices.data[3 * slot + 2]);
    return triangle_event(point, meta.vertices, ids);
}
} // namespace bvh
