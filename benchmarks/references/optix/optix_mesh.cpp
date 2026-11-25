// Hardware triangle intersection using an indexed OptiX GAS and one ray per query.
#include "fixture.h"
#include "optix_mesh.h"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <algorithm>
#include <mutex>
#include <iostream>
#include <array>
#include <type_traits>

void check(OptixResult r) {
    if (r != OPTIX_SUCCESS)
        throw std::runtime_error(optixGetErrorName(r));
}
void check(cudaError_t r) {
    if (r != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(r));
}
CUdeviceptr storage(size_t bytes) {
    void *ptr;
    check(cudaMalloc(&ptr, bytes));
    return reinterpret_cast<CUdeviceptr>(ptr);
}
template <class T> CUdeviceptr upload(const T *p, size_t n) {
    auto dst = storage(n * sizeof(T));
    check(cudaMemcpy((void *)dst, p, n * sizeof(T), cudaMemcpyHostToDevice));
    return dst;
}
template <class T> void dump(std::string path, T *data, int n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), data, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::ofstream(path, std::ios::binary).write((char *)h.data(), n * sizeof(T));
}
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {
    char bytes[OPTIX_SBT_RECORD_HEADER_SIZE];
};
// CUPTI timestamps each native OptiX launch on the GPU without replaying its internal bindings.
std::vector<std::pair<uint64_t, uint64_t>> intervals;
std::mutex intervalsMutex;
uint32_t measuredContext, measuredStream;
void CUPTIAPI allocateBuffer(uint8_t **buffer, size_t *size, size_t *maxRecords) {
    *size = 1024 * 1024;
    *maxRecords = 0;
    *buffer = static_cast<uint8_t *>(std::malloc(*size));
}
void CUPTIAPI consumeBuffer(CUcontext, uint32_t, uint8_t *buffer, size_t, size_t valid) {
    std::lock_guard lock(intervalsMutex);
    CUpti_Activity *record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto kernel = reinterpret_cast<CUpti_ActivityKernel11 *>(record);
            if (kernel->streamId != measuredStream || kernel->contextId != measuredContext)
                continue;
            intervals.emplace_back(uint64_t(kernel->start), uint64_t(kernel->end));
        }
    }
    std::free(buffer);
}
void check(CUptiResult r) {
    if (r != CUPTI_SUCCESS) {
        const char *msg;
        cuptiGetResultString(r, &msg);
        throw std::runtime_error(msg);
    }
}

int main(int argc, char **argv) try {
    if (argc != 2)
        throw std::runtime_error("usage: optix_mesh FIXTURE");
    Fixture fixture(argv[1]);
    check(cudaFree(nullptr));
    check(optixInit());
    check(cuptiActivityRegisterCallbacks(allocateBuffer, consumeBuffer));
    OptixDeviceContext ctx;
    OptixDeviceContextOptions options{};
    if (std::getenv("OPTIX_BENCH_DEBUG")) {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        options.logCallbackLevel = 4;
        options.logCallbackFunction = [](unsigned, const char *tag, const char *msg, void *) {
            std::cerr << tag << " " << msg << "\n";
        };
    }
    check(optixDeviceContextCreate(nullptr, &options, &ctx));
    auto v = fixture.read<float3>("vertices");
    auto f = fixture.read<uint3>("faces");
    auto vb = upload(v.data(), v.size());
    const unsigned flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto &tri = input.triangleArray;
    tri.vertexBuffers = &vb;
    tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri.numVertices = v.size();
    tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tri.numIndexTriplets = f.size();
    tri.indexBuffer = upload(f.data(), f.size());
    tri.flags = &flags;
    tri.numSbtRecords = 1;
    OptixAccelBuildOptions build{};
    build.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    build.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes;
    check(optixAccelComputeMemoryUsage(ctx, &build, &input, 1, &sizes));
    auto temp = storage(sizes.tempSizeInBytes), gas = storage(sizes.outputSizeInBytes);
    RayBatch batch{};
    const auto buildStart = NativeClock::now();
    check(optixAccelBuild(ctx, nullptr, &build, &input, 1, temp, sizes.tempSizeInBytes, gas,
                          sizes.outputSizeInBytes, &batch.tree, nullptr, 0));
    check(cudaDeviceSynchronize());
    const auto buildMs = elapsedMs(buildStart);
    check(cudaFree((void *)temp));
    auto ptx = readFixture<char>(
        (std::filesystem::path(argv[0]).parent_path() / "optix_programs.ptx").string());
    OptixModule module;
    OptixModuleCompileOptions mo{};
    OptixPipelineCompileOptions po{};
    po.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    po.numPayloadValues = 3;
    po.numAttributeValues = 2;
    po.pipelineLaunchParamsVariableName = "batch";
    po.pipelineLaunchParamsSizeInBytes = sizeof(RayBatch);
    po.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    check(optixModuleCreate(ctx, &mo, &po, ptx.data(), ptx.size(), nullptr, nullptr, &module));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUcontext context;
    if (cuStreamGetCtx(stream, &context) != CUDA_SUCCESS)
        throw std::runtime_error("cannot identify the OptiX stream context");
    check(cuptiGetContextId(context, &measuredContext));
    check(cuptiGetStreamIdEx(context, stream, 0, &measuredStream));
    for (const auto &item : fixture.cases) {
        writeBuildTimes(item, buildMs);
        const auto &mode = item.mode;
        int n = item.count;
        auto batchInput = [&](const char *name, auto *&dest) {
            using T = std::remove_const_t<std::remove_pointer_t<std::decay_t<decltype(dest)>>>;
            if (item.arrays.count(name)) {
                auto data = item.read<T>(name);
                dest = reinterpret_cast<T *>(upload(data.data(), n));
            } else {
                dest = nullptr;
            }
        };
        batchInput("points", batch.points);
        batchInput("origins", batch.origins);
        batchInput("directions", batch.directions);
        batchInput("min_t", batch.min_t);
        batchInput("max_t", batch.max_t);
        batch.max_hits = mode == "all_hits" ? item.integer("max_hits") : 1;
        batch.value = (int *)storage(n * sizeof(int));
        batch.face = (int *)storage(n * batch.max_hits * sizeof(int));
        batch.dist = (float *)storage(n * batch.max_hits * sizeof(float));
        auto params = upload(&batch, 1);
        std::array<OptixProgramGroupDesc, 3> desc{};
        std::array<OptixProgramGroup, 3> groups{};
        std::string rg = "__raygen__" + mode;
        desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc[0].raygen = {module, rg.c_str()};
        desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc[1].miss = {module, "__miss__empty"};
        desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (mode == "first_hit" || mode == "any_hit") {
            desc[2].hitgroup.moduleCH = module;
            desc[2].hitgroup.entryFunctionNameCH =
                mode == "first_hit" ? "__closesthit__first" : "__closesthit__any";
        } else {
            desc[2].hitgroup.moduleAH = module;
            desc[2].hitgroup.entryFunctionNameAH =
                mode == "all_hits" ? "__anyhit__all" : "__anyhit__count";
        }
        OptixProgramGroupOptions go{};
        check(optixProgramGroupCreate(ctx, desc.data(), 3, &go, nullptr, nullptr, groups.data()));
        OptixPipelineLinkOptions lo{};
        lo.maxTraceDepth = 1;
        OptixPipeline pipeline;
        check(optixPipelineCreate(ctx, &po, &lo, groups.data(), 3, nullptr, nullptr, &pipeline));
        Record rec[3];
        for (int j = 0; j < 3; ++j)
            check(optixSbtRecordPackHeader(groups[j], &rec[j]));
        auto records = upload(rec, 3);
        OptixShaderBindingTable sbt{};
        sbt.raygenRecord = records;
        sbt.missRecordBase = records + sizeof(Record);
        sbt.hitgroupRecordBase = records + 2 * sizeof(Record);
        sbt.hitgroupRecordStrideInBytes = sbt.missRecordStrideInBytes = sizeof(Record);
        sbt.hitgroupRecordCount = sbt.missRecordCount = 1;
        check(cudaDeviceSynchronize());
        auto launch = [&] {
            check(optixLaunch(pipeline, stream, params, sizeof(batch), &sbt, n, 1, 1));
        };
        if (std::getenv("OPTIX_BENCH_DEBUG"))
            std::cerr << mode << " warmup\n";
        for (int j = 0; j < fixture.warmups; ++j)
            launch();
        check(cudaStreamSynchronize(stream));
        check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        for (int r = 0; r < fixture.repeats; ++r) {
            for (int j = 0; j < fixture.launches; ++j) launch();
            check(cudaStreamSynchronize(stream));
        }
        check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        check(cuptiActivityFlushAll(0));
        const auto &path = item.prefix;
        {
            std::lock_guard lock(intervalsMutex);
            if (intervals.size() != size_t(fixture.launches * fixture.repeats))
                throw std::runtime_error( "unexpected number of timed kernels: " +
                                         std::to_string(intervals.size()));
            std::sort(intervals.begin(), intervals.end());
            std::ofstream samples(path + ".ms");
            samples.precision(12);
            for (int r = 0; r < fixture.repeats; ++r) {
                uint64_t ns = 0;
                for (int j = 0; j < fixture.launches; ++j)
                    ns += intervals[r * fixture.launches + j].second - intervals[r * fixture.launches + j].first;
                samples << ns / (fixture.launches * 1e6) << "\n";
            }
            intervals.clear();
        }
        if (mode != "first_hit")
            dump(path + ".value", batch.value, n);
        if (mode == "first_hit" || mode == "all_hits") {
            int count = mode == "all_hits" ? n * batch.max_hits : n;
            dump(path + ".ids", batch.face, count);
            dump(path + ".dist", batch.dist, count);
        }
        check(optixPipelineDestroy(pipeline));
        for (auto pg : groups)
            check(optixProgramGroupDestroy(pg));
        for (const void *ptr : {static_cast<const void *>(batch.points),
                                static_cast<const void *>(batch.origins),
                                static_cast<const void *>(batch.directions),
                                static_cast<const void *>(batch.min_t),
                                static_cast<const void *>(batch.max_t),
                                static_cast<const void *>(batch.value),
                                static_cast<const void *>(batch.face),
                                static_cast<const void *>(batch.dist),
                                reinterpret_cast<const void *>(params),
                                reinterpret_cast<const void *>(records)})
            check(cudaFree(const_cast<void *>(ptr)));
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
}
