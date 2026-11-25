// Resident, single-mesh cuBQL triangle queries. Pass the exported fixture directory.
#include <cuda_runtime.h>
#include <climits>
#include <cuBQL/queries/triangleData/closestPointOnAnyTriangle.h>
#include <cuBQL/traversal/fixedBoxQuery.h>
#include <cuBQL/queries/triangleData/anyWithinRadius.h>
#include <cuBQL/queries/triangleData/boxInsideOutsideIntersects.h>
#include <cuBQL/queries/triangleData/math/rayTriangleIntersections.h>
#include "cuda_fixture.h"
#include <iostream>
#include <type_traits>
#include <vector>

using namespace cuBQL;
using Vec = vec3f;
enum Mode {
    first_hit,
    any_hit,
    count_hits,
    all_hits,
    closest_point,
    classify_point,
    aabb_overlap,
    knn,
    radius_any,
    radius_count,
    region_any,
    region_count,
    classify_box
};
constexpr const char *names[] = {"first_hit",     "any_hit",        "count_hits",   "all_hits",
                                 "closest_point", "classify_point", "aabb_overlap", "knn",
                                 "radius_any",    "radius_count",   "region_any",   "region_count",
                                 "classify_box"};

struct Inputs {
    bvh3f tree;
    Triangle *tris;
    Vec *p, *o, *d, *lower, *upper;
    float *radii, *min_t, *max_t;
    int *excluded;
    int n, max_hits;
};
struct Outputs {
    int *value, *ids;
    float *dist;
    Vec *points;
};
template <Mode mode> __global__ void query(Inputs in, Outputs out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= in.n)
        return;
    Vec p = in.p ? in.p[i] : Vec(0);
    auto triangle = [&](uint32_t id) { return in.tris[id]; };
    if constexpr (mode == closest_point) {
        triangles::CPAT hit;
        hit.runQuery(in.tris, in.tree, p, in.radii[i]);
        out.ids[i] = hit.triangleIdx;
        out.dist[i] = sqrtf(hit.sqrDist);
        out.points[i] = hit.P;
    } else if constexpr (mode == radius_any || mode == radius_count) {
        out.value[i] = triangles::numWithinRadius(in.tree, triangle, p, in.radii[i],
                                                  mode == radius_any ? 1 : INT_MAX);
    } else if constexpr (mode == region_any || mode == region_count) {
        out.value[i] = triangles::countTrianglesIntersectingQueryBox(
            in.tree, triangle, box3f(in.lower[i], in.upper[i]), mode == region_any ? 1 : INT_MAX);
    } else if constexpr (mode == aabb_overlap) {
        out.value[i] =
            triangles::countTrianglesWhoseBoundsOverlapQueryBox(in.tree, triangle, box3f(in.lower[i], in.upper[i]), 1);
    } else if constexpr (mode == classify_point) {
        out.value[i] = triangles::pointIsInsideSurface(in.tree, triangle, p) ? -1 : 1;
    } else if constexpr (mode == classify_box) {
        using namespace triangles::boxInsideOutsideIntersects;
        auto sign = queryVsActualTriangles(in.tree, triangle, box3f(in.lower[i], in.upper[i]));
        out.value[i] = sign == INSIDE ? 1 : sign == OUTSIDE ? -1 : 0;
    } else if constexpr (mode == knn) {
        float cap = in.radii[i] * in.radii[i];
        const float outside = nextafterf(cap, INFINITY);
        float score[4] = {outside, outside, outside, outside};
        int ids[4] = {-1, -1, -1, -1};
        auto visit = [&](uint32_t id) {
            float s = triangles::computeClosestPoint(triangle(id), p).sqrDist;
            int candidate = id;
            for (int j = 0; j < 4; ++j)
                if (s < score[j]) {
                    float old = score[j];
                    score[j] = s;
                    s = old;
                    int previous = ids[j];
                    ids[j] = candidate;
                    candidate = previous;
                }
            return score[3];
        };
        shrinkingRadiusQuery::forEachPrim(visit, in.tree, p, outside);
        for (int j = 0; j < 4; ++j) {
            out.dist[4 * i + j] = ids[j] < 0 ? INFINITY : sqrtf(score[j]);
            out.ids[4 * i + j] = ids[j];
        }
    } else {
        Ray ray(in.o[i], in.d[i], in.min_t[i], in.max_t[i]);
        int hits = 0, face = -1;
        if constexpr (mode == first_hit) {
            auto visit = [&](uint32_t id) {
                RayTriangleIntersection r;
                if (r.compute(ray, triangle(id)) && r.t < ray.tMax) {
                    ray.tMax = r.t;
                    face = id;
                }
                return ray.tMax;
            };
            shrinkingRayQuery::forEachPrim(visit, in.tree, ray);
            out.ids[i] = face;
            out.dist[i] = ray.tMax;
        } else {
            if constexpr (mode == all_hits)
                for (int j = 0; j < in.max_hits; ++j) {
                    out.ids[i * in.max_hits + j] = -1;
                    out.dist[i * in.max_hits + j] = INFINITY;
                }
            auto visit = [&](uint32_t id) {
                RayTriangleIntersection r;
                if ((!in.excluded || id != uint32_t(in.excluded[i])) && r.compute(ray, triangle(id))) {
                    if constexpr (mode == all_hits)
                        if (hits < in.max_hits) {
                            out.ids[i * in.max_hits + hits] = id;
                            out.dist[i * in.max_hits + hits] = r.t;
                        }
                    ++hits;
                    if constexpr (mode == any_hit)
                        return CUBQL_TERMINATE_TRAVERSAL;
                }
                return CUBQL_CONTINUE_TRAVERSAL;
            };
            fixedRayQuery::forEachPrim(visit, in.tree, ray);
            out.value[i] = hits;
        }
    }
}

template <Mode mode>
void bench(const Fixture &fixture, const std::string &path, Inputs in, Outputs out, cudaStream_t stream) {
    const int block = fixture.block;
    auto launch = [&] { query<mode><<<(in.n + block - 1) / block, block, 0, stream>>>(in, out); };
    measure(launch, fixture, stream, path);
    if constexpr (mode != first_hit && mode != closest_point && mode != knn)
        save(path + ".value", out.value, in.n);
    if constexpr (mode == first_hit || mode == closest_point || mode == knn || mode == all_hits) {
        int count = in.n * (mode == knn ? 4 : mode == all_hits ? in.max_hits : 1);
        save(path + ".ids", out.ids, count);
        save(path + ".dist", out.dist, count);
    }
    if constexpr (mode == closest_point)
        save(path + ".points", out.points, in.n);
}
int main(int argc, char **argv) try {
    if (argc != 2)
        throw std::runtime_error("usage: cubql_mesh FIXTURE");
    Fixture fixture(argv[1]);
    auto vertices = fixture.read<Vec>("vertices");
    auto faces = fixture.read<vec3i>("faces");
    std::vector<Triangle> tris;
    std::vector<box3f> bounds;
    for (auto f : faces) {
        tris.emplace_back(vertices.at(f.x), vertices.at(f.y), vertices.at(f.z));
        bounds.push_back(tris.back().bounds());
    }
    Inputs in{};
    in.tris = upload(tris);
    auto boxes = upload(bounds);
    BuildConfig config;
    config.makeLeafThreshold = fixture.leaf;
    config.enableSAH();
    DeviceMemoryResource memory;
    const auto buildStart = NativeClock::now();
    gpuBuilder(in.tree, boxes, bounds.size(), config, 0, memory);
    check(cudaDeviceSynchronize());
    const auto buildMs = elapsedMs(buildStart);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const auto &item : fixture.cases) {
        writeBuildTimes(item, buildMs);
        auto array = [&](const char *name, auto *&dest) {
            using T = std::remove_pointer_t<std::decay_t<decltype(dest)>>;
            dest = item.arrays.count(name) ? upload(item.read<T>(name)) : nullptr;
        };
        array("points", in.p);
        array("origins", in.o);
        array("directions", in.d);
        array("lower", in.lower);
        array("upper", in.upper);
        array("radii", in.radii);
        array("min_t", in.min_t);
        array("max_t", in.max_t);
        array("excluded", in.excluded);
        in.n = item.count;
        in.max_hits = item.mode == "all_hits" ? item.integer("max_hits") : 0;
        const size_t width = item.mode == "knn" ? 4 : std::max(in.max_hits, 1);
        Outputs out{};
        check(cudaMalloc(&out.value, sizeof(int) * in.n));
        check(cudaMalloc(&out.ids, sizeof(int) * width * in.n));
        check(cudaMalloc(&out.dist, sizeof(float) * width * in.n));
        check(cudaMalloc(&out.points, sizeof(Vec) * in.n));
        auto run = [&] {
#define RUN(selected) if (item.mode == names[selected]) return bench<selected>(fixture, item.prefix, in, out, stream)
            RUN(first_hit);
            RUN(any_hit);
            RUN(count_hits);
            RUN(all_hits);
            RUN(closest_point);
            RUN(classify_point);
            RUN(aabb_overlap);
            RUN(knn);
            RUN(radius_any);
            RUN(radius_count);
            RUN(region_any);
            RUN(region_count);
            RUN(classify_box);
#undef RUN
            throw std::runtime_error("unknown mode: " + item.mode);
        };
        run();
        for (void *ptr : {static_cast<void *>(in.p), static_cast<void *>(in.o),
                          static_cast<void *>(in.d), static_cast<void *>(in.lower),
                          static_cast<void *>(in.upper), static_cast<void *>(in.radii),
                          static_cast<void *>(in.min_t), static_cast<void *>(in.max_t),
                          static_cast<void *>(in.excluded),
                          static_cast<void *>(out.value), static_cast<void *>(out.ids),
                          static_cast<void *>(out.dist), static_cast<void *>(out.points)})
            check(cudaFree(ptr));
    }
    check(cudaStreamDestroy(stream));
    cuBQL::cuda::free(in.tree, 0, memory);
    check(cudaFree(in.tris));
    check(cudaFree(boxes));
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
}
