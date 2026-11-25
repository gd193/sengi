#include "cuda_fixture.h"
#include <cuBQL/queries/pointData/findClosest.h>
#include <cuBQL/queries/pointData/knn.h>

using Vec = cuBQL::vec_t<float, BENCH_DIM>;
using Box = cuBQL::box_t<float, BENCH_DIM>;
using Tree = cuBQL::BinaryBVH<float, BENCH_DIM>;

__global__ void nearest(Tree tree, const Vec *data, const Vec *queries, int count,
                        float cap, Vec *points, float *distances, int *ids) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    int id = cuBQL::points::findClosest(tree, data, queries[i], cap);
    ids[i] = id;
    points[i] = id >= 0 ? data[id] : Vec(0.f);
    distances[i] = id >= 0 ? sqrtf(cuBQL::sqrLength(data[id] - queries[i])) : INFINITY;
}

template<int K>
__global__ void knn(Tree tree, const Vec *data, const Vec *queries, int count, float cap,
                    float *scores, int *ids, int *counts, int *status) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count)
        return;
    cuBQL::knn::Candidate candidates[K];
    auto result = cuBQL::points::findKNN(candidates, K, tree, data, queries[row], nextafterf(cap, INFINITY));
    // Upstream may fill the candidate list with points from a partially overlapping leaf.
    while (result.numFound && candidates[result.numFound - 1].sqrDist > cap)
        --result.numFound;
    for (int j = 0; j < K; ++j) {
        scores[row * K + j] = j < result.numFound ? candidates[j].sqrDist : INFINITY;
        ids[row * K + j] = j < result.numFound ? candidates[j].primID : -1;
    }
    counts[row] = result.numFound;
    status[row] = result.numFound ? 1 : 0;
}

template<class Kernel>
void kernelInfo(const NativeCase &item, Kernel kernel, const Tree &tree) {
    cudaFuncAttributes properties;
    check(cudaFuncGetAttributes(&properties, kernel));
    std::ofstream(item.prefix + ".json") << "{\"registers\":" << properties.numRegs
        << ",\"local_bytes\":" << properties.localSizeBytes
        << ",\"nodes\":" << tree.numNodes << "}\n";
}

int main(int argc, char **argv) try {
    if (argc != 2)
        throw std::runtime_error("usage: cubql_points FIXTURE");
    Fixture fixture(argv[1]);
    auto host = fixture.read<Vec>("points");
    std::vector<Box> bounds;
    for (auto point : host)
        bounds.emplace_back(point, point);
    auto data = upload(host);
    auto boxes = upload(bounds);
    Tree tree;
    cuBQL::BuildConfig config;
    config.makeLeafThreshold = fixture.leaf;
    if constexpr (BENCH_DIM == 3)
        config.enableSAH();
    cuBQL::DeviceMemoryResource memory;
    const auto buildStart = NativeClock::now();
    cuBQL::gpuBuilder(tree, boxes, bounds.size(), config, 0, memory);
    check(cudaDeviceSynchronize());
    const auto buildMs = elapsedMs(buildStart);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int block = fixture.block;
    for (const auto &item : fixture.cases) {
        writeBuildTimes(item, buildMs);
        auto queries = upload(item.read<Vec>("points"));
        const int n = item.count;
        const int blocks = (n + block - 1) / block;
        const float cap = item.scalar("cap_squared");
        const float exclusiveCap = nextafterf(cap, INFINITY);
        if (item.mode == "closest_point") {
            Vec *points;
            float *distances;
            int *ids;
            check(cudaMalloc(&points, n * sizeof(Vec)));
            check(cudaMalloc(&distances, n * sizeof(float)));
            check(cudaMalloc(&ids, n * sizeof(int)));
            measure([&] {
                nearest<<<blocks, block, 0, stream>>>(tree, data, queries, n, exclusiveCap,
                                                    points, distances, ids);
            }, fixture, stream, item.prefix);
            save(item.prefix + ".points", points, n);
            save(item.prefix + ".dist", distances, n);
            save(item.prefix + ".ids", ids, n);
            kernelInfo(item, nearest, tree);
            check(cudaFree(points));
            check(cudaFree(distances));
            check(cudaFree(ids));
        } else if (item.mode == "knn") {
            dispatchK(item, [&](auto k) {
                auto kernel = knn<decltype(k)::value>;
                runKnn(fixture, item, stream, [&](float *scores, int *ids, int *counts, int *status) {
                    knn<decltype(k)::value><<<blocks, block, 0, stream>>>(
                        tree, data, queries, n, cap, scores, ids, counts, status);
                });
                kernelInfo(item, kernel, tree);
            });
        } else {
            throw std::runtime_error("unknown mode: " + item.mode);
        }
        check(cudaFree(queries));
    }
    cuBQL::cuda::free(tree, 0, memory);
    check(cudaFree(data));
    check(cudaFree(boxes));
    check(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
