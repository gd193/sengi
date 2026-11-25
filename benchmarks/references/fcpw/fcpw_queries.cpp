// CUDA-event timings of resident FCPW primitive and silhouette hierarchies.
#include <fcpw/fcpw_gpu.h>
#include "fcpw_timing.h"
#include "fixture.h"
#include <iostream>

using namespace fcpw;

bool needsFeatures(const std::string &mode) {
    return mode == "closest_silhouette" || mode == "star_radius";
}

template<int D>
std::vector<fcpw::float3> coordinates(const NativeCase &item, const std::string &name) {
    auto flat = item.read<float>(name);
    std::vector<fcpw::float3> result(item.count);
    if (flat.size() != result.size() * D)
        throw std::runtime_error("unexpected coordinate shape: " + name);
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = {flat[D*i], flat[D*i + 1], D == 3 ? flat[D*i + 2] : 0.f};
    return result;
}

void adjustThreads(const std::string &source, const std::string &destination, int block) {
    auto bytes = readFixture<char>(source);
    std::string shader(bytes.begin(), bytes.end());
    const std::string token = "numthreads(256,";
    const std::string replacement = "numthreads(" + std::to_string(block) + ",";
    for (size_t at = shader.find(token); at != std::string::npos;
         at = shader.find(token, at + replacement.size()))
        shader.replace(at, token.size(), replacement);
    std::ofstream(destination) << shader;
}

template<int D>
void execute(const Fixture &fixture, const std::string &source,
             const std::string &directory, const std::string &boundaryShader) {
    auto vertices = fixture.read<Vector<D>>("vertices");
    const auto shaders = source + "/include/fcpw/gpu";
    const std::string prepared = directory + "/shaders";
    std::filesystem::create_directories(prepared);
    adjustThreads(shaders + "/bvh.cs.slang", prepared + "/bvh.cs.slang", fixture.block);
    const bool needsBoundary = std::any_of(fixture.cases.begin(), fixture.cases.end(),
        [](const NativeCase &item) {
            return item.mode == "sample_boundary" || item.mode == "star_radius";
        });
    if (needsBoundary)
        adjustThreads(boundaryShader, prepared + "/fcpw_boundary.slang", fixture.block);

    for (bool features : {false, true}) {
        if (std::none_of(fixture.cases.begin(), fixture.cases.end(), [&](const NativeCase &item) {
            return features == needsFeatures(item.mode);
        }))
            continue;
        Scene<D> scene;
        scene.setObjectCount(1);
        scene.setObjectVertices(vertices, 0);
        if constexpr (D == 3)
            scene.setObjectTriangles(fixture.read<Vector3i>("faces"), 0);
        else
            scene.setObjectLineSegments(fixture.read<Vector2i>("indices"), 0);
        double boundaryMs = -1.;
        if (features) {
            const auto boundaryStart = NativeClock::now();
            scene.computeSilhouettes();
            boundaryMs = elapsedMs(boundaryStart);
        }
        const auto buildStart = NativeClock::now();
        scene.build(AggregateType::Bvh_SurfaceArea, false);
        const auto buildMs = elapsedMs(buildStart);
        GPUContext context;
        context.searchPaths = {prepared, shaders};
        context.macros = {{"FCPW_BVH_TYPE", std::to_string(D - 1 + (features ? 2 : 0))}};
        context.initDevice(DeviceType::CUDA);
        std::vector<GPUModule> libs(1);
        libs[0].load(context, shaders + "/fcpw.slang");
        GPUModule module;
        module.load(context, prepared + "/bvh.cs.slang");
        GPUModule boundary;
        if (needsBoundary)
            boundary.load(context, prepared + "/fcpw_boundary.slang");
        GPUBvhBuffers buffers;
        buffers.allocate<D>(context, scene.getSceneData(), true, features, true, false);
        auto bind = [&](const ComputeShader &shader, const ShaderCursor &cursor) {
            auto obj = shader.createShaderObject(context, buffers.getReflectionType());
            ShaderCursor bindings(obj);
            buffers.setResources(bindings, false);
            cursor.getPath("gBvh").setObject(obj);
        };
        for (const auto &item : fixture.cases) {
            const auto &mode = item.mode;
            if (features != needsFeatures(mode))
                continue;
            writeBuildTimes(item, buildMs, boundaryMs);
            auto run = [&](const char *entry, auto &query) {
                ComputeShader shader;
                shader.loadProgram(context, mode == "star_radius" || mode == "sample_boundary"
                                                ? boundary : module, libs, entry);
                std::vector<GPUInteraction> hits;
                auto times = timeKernelOnly(context, shader, query, bind,
                    countThreadGroups(item.count, fixture.block, false),
                    fixture.launches, fixture.repeats, fixture.warmups, hits);
                std::ofstream samples(item.prefix + ".ms");
                samples.precision(12);
                for (auto ms : times.samples_ms)
                    samples << ms << "\n";

                writeArray(item.prefix + ".bin", hits);
                std::ofstream(item.prefix + ".json")
                    << "{\"registers\":" << times.regs << ",\"local_bytes\":" << times.local_bytes
                    << ",\"captured_kernels\":" << times.captured_kernels << "}\n";
            };
            if (mode == "first_hit" || mode == "any_hit") {
                auto origins = coordinates<D>(item, "origins");
                auto directions = coordinates<D>(item, "directions");
                auto caps = item.read<float>("max_t");
                std::vector<GPURay> rays;
                for (size_t i = 0; i < origins.size(); ++i)
                    rays.emplace_back(origins[i], directions[i], caps.at(i));
                GPURunRayIntersectionQuery q;
                q.allocate(context, rays);
                q.checkForOcclusion = mode == "any_hit";
                run("rayIntersection", q);
            } else {
                auto points = coordinates<D>(item, "points");
                auto radii = item.read<float>("radii");
                std::vector<GPUBoundingSphere> spheres;
                for (size_t i = 0; i < points.size(); ++i)
                    spheres.emplace_back(points[i], mode == "star_radius" ? radii.at(i) :
                        std::min(maxFloat, radii.at(i) * radii.at(i)));
                if (features) {
                    GPURunClosestSilhouettePointQuery q;
                    q.allocate(context, spheres, std::vector<uint32_t>(points.size(), 0));
                    q.precision = q.squaredMinRadius = 0;
                    run(mode == "star_radius" ? "cappedSilhouettePoint" : "closestSilhouettePoint", q);
                } else if (mode == "closest_point") {
                    GPURunClosestPointQuery q;
                    q.allocate(context, spheres);
                    run("closestPoint", q);
                } else if (mode == "sample_boundary") {
                    // The sampler expects three random coordinates for either dimension.
                    auto randoms = item.read<fcpw::float3>("randoms");
                    GPURunSphereIntersectionQuery q;
                    q.allocate(context, spheres, randoms);
                    run("sampleInBall", q);
                } else {
                    throw std::runtime_error("unknown mode: " + mode);
                }
            }
        }
    }
}

int main(int argc, char **argv) try {
    if (argc != 4)
        throw std::runtime_error("usage: fcpw_queries FIXTURE FCPW_SOURCE DIMENSION");
    Fixture fixture(argv[1]);
    const auto boundaryShader = (std::filesystem::path(argv[0]).parent_path()
                                 / "fcpw_boundary.slang").string();
    const std::string dimension = argv[3];
    if (dimension == "3")
        execute<3>(fixture, argv[2], argv[1], boundaryShader);
    else if (dimension == "2")
        execute<2>(fixture, argv[2], argv[1], boundaryShader);
    else
        throw std::runtime_error("FCPW reference needs a two- or three-dimensional fixture");
    return 0;
} catch (const std::exception &err) {
    std::cerr << err.what() << "\n";
    return 1;
}
