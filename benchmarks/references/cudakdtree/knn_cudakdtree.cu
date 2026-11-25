#include "cuda_fixture.h"
#include <cukd/builder.h>
#include <cukd/knn.h>
#include <cfloat>

#if BENCH_DIM == 2
using Vec = float2;
#else
using Vec = float3;
#endif

struct Item {
    Vec p;
    int id;
};
struct Traits : cukd::default_data_traits<Vec> {
    using data_t = Item;
    static __host__ __device__ int get_dim(const Item&) { return 0; }
    static __host__ __device__ void set_dim(Item&, int) {}
    static __host__ __device__ const Vec& get_point(const Item& a) { return a.p; }
    static __host__ __device__ float get_coord(const Item& a, int d) {
        return cukd::get_coord(a.p, d);
    }
};

#if BENCH_HEAP
template<int K> using List = cukd::HeapCandidateList<K>;
#else
template<int K> using List = cukd::FixedCandidateList<K>;
#endif

template<int K> __global__ void query(cukd::box_t<Vec> bounds, const Item* p, int np,
                      const Vec* q, int nq, float cap, float* scores,
                      int* ids, int* counts, int* status) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nq)
        return;
    // cuKDTree squares its radius and prunes at the open boundary.
    List<K> c(nextafterf(sqrtf(fmaxf(cap, FLT_MIN)), INFINITY));
#if BENCH_STACK
    cukd::stackBased::knn<List<K>, Item, Traits>(c, q[row], p, np);
#else
    cukd::cct::knn<List<K>, Item, Traits>(c, q[row], bounds, p, np);
#endif
#if BENCH_HEAP
    sort_entries(c.entry);
#endif
    // Sorting and original-ID remapping remain inside the timed kernel.
    int found = 0;
    for (int j = 0; j < K; ++j) {
        int id = c.get_dist2(j) <= cap ? c.get_pointID(j) : -1;
        scores[row * K + j] = id >= 0 ? c.get_dist2(j) : INFINITY;
        ids[row * K + j] = id >= 0 ? p[id].id : -1;
        found += id >= 0;
    }
    counts[row] = found;
    status[row] = found ? 1 : 0;
}

int main(int argc, char** argv) try {
    if (argc != 2)
        throw std::runtime_error("usage: binary FIXTURE");
    Fixture fixture(argv[1]);
    auto hp = fixture.read<Vec>("points");
    // Keep original IDs alongside points through the builder's permutation.
    std::vector<Item> hi;
    for (int i = 0; i < (int)hp.size(); ++i)
        hi.push_back({hp[i], i});
    auto p = upload(hi);

    cukd::box_t<Vec>* db;
    check(cudaMalloc(&db, sizeof(*db)));
    const auto buildStart = NativeClock::now();
    cukd::buildTree<Item, Traits>(p, hi.size(), db);
    cukd::box_t<Vec> bounds;
    check(cudaMemcpy(&bounds, db, sizeof(bounds), cudaMemcpyDeviceToHost));
    check(cudaDeviceSynchronize());

    const auto buildMs = elapsedMs(buildStart);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (const auto &item : fixture.cases) {
        writeBuildTimes(item, buildMs);
        if (item.mode != "knn")
            throw std::runtime_error("unknown mode: " + item.mode);
        auto q = upload(item.read<Vec>("points"));
        dispatchK(item, [&](auto k) {
            const float cap = item.scalar("cap_squared");
            auto kernel = query<decltype(k)::value>;
            runKnn(fixture, item, stream, [&](float *scores, int *ids, int *counts, int *status) {
                int block = fixture.block;
                query<decltype(k)::value><<<(item.count + block - 1) / block, block, 0, stream>>>(
                    bounds, p, hi.size(), q, item.count, cap, scores, ids, counts, status);
            });
            cudaFuncAttributes properties;
            check(cudaFuncGetAttributes(&properties, kernel));
            std::ofstream(item.prefix + ".json") << "{\"registers\":" << properties.numRegs
                << ",\"local_bytes\":" << properties.localSizeBytes << "}\n";
        });
        check(cudaFree(q));
    }
    check(cudaStreamDestroy(stream));
    check(cudaFree(p));
    check(cudaFree(db));
    return 0;
} catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
