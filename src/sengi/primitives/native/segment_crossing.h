#pragma once
#include "builtin.h"

namespace bvh {

template <typename Ray, typename Data>
CUDA_CALLABLE inline wp::vec2 segment_intersection(const Ray &ray, float minimum, float maximum,
                                                   const Data &data, int slot, bool half_open,
                                                   bool store_times) {
    const auto origin = ray.origin, direction = ray.direction;
    const wp::vec2 a = data.a.data[slot], b = data.b.data[slot];
    // Filter endpoint signs; uncertain signs retain the double predicate.
    // Four rounded operations per term are bounded by 16u times the sum
    // of product magnitudes; padding also covers gradual underflow/FTZ.
    const float fax = a[0] - origin[0], fay = a[1] - origin[1];
    const float fbx = b[0] - origin[0], fby = b[1] - origin[1];
    const float ap = fax * direction[1], aq = fay * direction[0];
    const float bp = fbx * direction[1], bq = fby * direction[0];
    const float fa = ap - aq, fb = bp - bq;
    const float padding =
        4.70197740328915e-38f * (wp::abs(direction[0]) + wp::abs(direction[1]) + wp::abs(fax) +
                                 wp::abs(fay) + wp::abs(fbx) + wp::abs(fby) + 1.f);
    const float ea = 9.5367431640625e-7f * (wp::abs(ap) + wp::abs(aq)) + padding;
    const float eb = 9.5367431640625e-7f * (wp::abs(bp) + wp::abs(bq)) + padding;
    const float shared_error = ea + eb;
    if ((fa > shared_error && fb > shared_error) || (fa < -shared_error && fb < -shared_error))
        return wp::vec2(0.f, 0.f);
    const bool interior =
        (fa > shared_error && fb < -shared_error) || (fa < -shared_error && fb > shared_error);
    // Certify the original interval expressions; uncertain signs,
    // bounds and overflow retain the double predicate.
    bool certified = false;
    if (interior) {
        const float ex = b[0] - a[0], ey = b[1] - a[1];
        const float dp = direction[0] * ey, dq = direction[1] * ex;
        const float np = fax * ey, nq = fay * ex;
        const float sign = fa < 0.f ? -1.f : 1.f;
        const float scale = sign * (dp - dq), numerator = sign * (np - nq);
        const float ed = 9.5367431640625e-7f * (wp::abs(dp) + wp::abs(dq)) +
                         4.70197740328915e-38f * (wp::abs(direction[0]) + wp::abs(direction[1]) +
                                                  wp::abs(ex) + wp::abs(ey) + 1.f);
        const float en =
            9.5367431640625e-7f * (wp::abs(np) + wp::abs(nq)) +
            4.70197740328915e-38f * (wp::abs(fax) + wp::abs(fay) + wp::abs(ex) + wp::abs(ey) + 1.f);
        if (scale > ed) {
            const float low = minimum * scale, high = maximum * scale;
            const float el = en + wp::abs(minimum) * ed +
                             2.384185791015625e-7f * (wp::abs(numerator) + wp::abs(low)) +
                             4.70197740328915e-38f * (wp::abs(minimum) + scale + 1.f);
            const float eh = en + wp::abs(maximum) * ed +
                             2.384185791015625e-7f * (wp::abs(numerator) + wp::abs(high)) +
                             4.70197740328915e-38f * (wp::abs(maximum) + scale + 1.f);
            certified = (minimum == -INFINITY || numerator - low > el) &&
                        (maximum == INFINITY || high - numerator > eh);
            if (certified && !store_times)
                return wp::vec2(0.f, 1.f);
        }
    }
    if (certified && store_times) {
        const bool vertical = a[0] == b[0], horizontal = a[1] == b[1];
        if (vertical || horizontal) {
            const double edge = double(vertical ? b[1] : b[0]) - double(vertical ? a[1] : a[0]);
            // Only exact FP64 power-of-two factors cancel without rounding.
            uint64_t bits;
            memcpy(&bits, &edge, 8);
            if ((bits & 0x7fffffffffffffffull) != 0 && (bits & 0x000fffffffffffffull) == 0) {
                const double offset =
                    double(vertical ? a[0] : a[1]) - double(vertical ? origin[0] : origin[1]);
                const float component = vertical ? direction[0] : direction[1];
                const float offset32 = static_cast<float>(offset);
                const float result =
                    double(offset32) == offset && wp::abs(component) >= 1.1754943508222875e-38f
                        ? offset32 / component
                        : static_cast<float>(offset / double(component));
                if (wp::isfinite(result) && wp::abs(result) >= 1.1754943508222875e-38f)
                    return wp::vec2(result, 1.f);
            }
        }
    }
    const double ex = double(b[0]) - double(a[0]), ey = double(b[1]) - double(a[1]);
    const double dx = direction[0], dy = direction[1];
    const double ax = double(a[0]) - double(origin[0]), ay = double(a[1]) - double(origin[1]);
    const double denominator = dx * ey - dy * ex;
    double sign = 1.0, scale = denominator;
    if (!certified) {
        if (denominator == 0)
            return wp::vec2(0.f, 0.f);
        sign = denominator < 0 ? -1.0 : 1.0;
        scale = sign * denominator;
    }
    // Strict opposite signs certify interior and half-open ownership.
    if (!interior) {
        const double sa = ax * dy - ay * dx, u = sign * sa;
        if (u < 0 || u > scale)
            return wp::vec2(0.f, 0.f);
        if (half_open) {
            const double sb =
                (double(b[0]) - double(origin[0])) * dy - (double(b[1]) - double(origin[1])) * dx;
            if (!((sa <= 0 && sb > 0) || (sb <= 0 && sa > 0)))
                return wp::vec2(0.f, 0.f);
        }
    }
    double numerator = ax * ey - ay * ex;
    if (!certified) {
        numerator = sign * numerator;
        if (numerator < double(minimum) * scale || numerator > double(maximum) * scale)
            return wp::vec2(0.f, 0.f);
    }
    const float result = store_times ? static_cast<float>(numerator / scale) : 0.f;
    return wp::vec2(result, store_times && !wp::isfinite(result) ? -1.f : 1.f);
}

} // namespace bvh
