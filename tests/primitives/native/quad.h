#pragma once

#include "quad_helper.h"

namespace external_quad {
CUDA_CALLABLE inline wp::vec3 closest(wp::vec3 query, const wp::mat_t<4, 3, float>& q)
{
    const auto a = q.get_row(0);
    const auto normal = wp::normalize(wp::cross(q.get_row(1) - a, q.get_row(3) - a));
    bool inside = true;
    auto best = a;
    for (int i = 0; i < 4; ++i) {
        const auto start = q.get_row(i);
        const auto edge = q.get_row((i + 1) % 4) - start;
        const auto delta = query - start;
        inside = inside && wp::dot(wp::cross(edge, delta), normal) >= 0.0f;
        const auto candidate = start + edge * wp::clamp(wp::dot(delta, edge) / wp::dot(edge, edge), 0.0f, 1.0f);
        if (wp::dot(query - candidate, query - candidate) < wp::dot(query - best, query - best)) {
            best = candidate;
        }
    }
    return inside ? query - normal * wp::dot(query - a, normal) + wp::vec3(interior_offset(), 0.0f, 0.0f) : best;
}
}
