#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

template <typename T> std::vector<T> readFixture(const std::string &file) {
    const auto bytes = std::filesystem::file_size(file);
    std::vector<T> values(bytes / sizeof(T));
    if (bytes % sizeof(T) ||
        !std::ifstream(file, std::ios::binary).read(reinterpret_cast<char *>(values.data()), bytes))
        throw std::runtime_error("invalid fixture array: " + file);
    return values;
}

struct NativeArray {
    std::string path;
    size_t bytes;

    template<class T> std::vector<T> read() const {
        auto data = readFixture<T>(path);
        if (data.size() * sizeof(T) != bytes)
            throw std::runtime_error("array size differs from manifest: " + path);
        return data;
    }
};

struct NativeCase {
    std::string key, mode, prefix;
    int count;
    std::unordered_map<std::string, NativeArray> arrays;
    std::unordered_map<std::string, std::string> parameters;

    template<class T> std::vector<T> read(const std::string &name) const {
        return arrays.at(name).read<T>();
    }
    int integer(const std::string &name) const { return std::stoi(parameters.at(name)); }
    float scalar(const std::string &name) const { return static_cast<float>(std::stod(parameters.at(name))); }
};

struct Fixture {
    std::unordered_map<std::string, NativeArray> geometry;
    std::vector<NativeCase> cases;
    int block, warmups, launches, repeats, leaf;

    explicit Fixture(const std::string &directory) {
        std::ifstream manifest(directory + "/manifest.tsv");
        std::string version;
        if (!(manifest >> version >> block >> warmups >> launches >> repeats >> leaf) ||
            version != "sengi-native-1" || block < 1 || block > 1024 || warmups < 0 ||
            launches < 1 || repeats < 1 || leaf < 1)
            throw std::runtime_error("invalid native fixture manifest");
        std::string line;
        std::getline(manifest, line);
        while (std::getline(manifest, line)) {
            std::istringstream row(line);
            std::string record, name, path, dtype, shape;
            row >> record;
            if (record == "case") {
                NativeCase item;
                if (!(row >> item.key >> item.mode >> item.count) || item.count < 1)
                    throw std::runtime_error("invalid case in native fixture");
                item.prefix = directory + "/out/" + item.key;
                cases.push_back(std::move(item));
            } else if (record == "geometry" || record == "input") {
                if (!(row >> name >> path >> dtype >> shape))
                    throw std::runtime_error("invalid array in native fixture");
                if (record == "input" && cases.empty())
                    throw std::runtime_error("input record must follow a case");
                if (dtype != "<f4" && dtype != "<i4")
                    throw std::runtime_error("unsupported fixture array dtype: " + dtype);
                size_t bytes = 4;
                std::istringstream dimensions(shape);
                std::string dim;
                while (std::getline(dimensions, dim, ',')) {
                    const auto width = std::stoull(dim);
                    if (!width || width > std::numeric_limits<size_t>::max() / bytes)
                        throw std::runtime_error("invalid fixture array shape: " + shape);
                    bytes *= width;
                }
                auto &arrays = record == "geometry" ? geometry : cases.back().arrays;
                if (!arrays.emplace(name, NativeArray{directory + "/" + path, bytes}).second)
                    throw std::runtime_error("duplicate fixture array: " + name);
            } else if (record == "parameter") {
                if (cases.empty() || !(row >> name >> path))
                    throw std::runtime_error("invalid parameter in native fixture");
                cases.back().parameters.emplace(name, path);
            } else {
                throw std::runtime_error("unknown native fixture record: " + record);
            }
        }
        if (cases.empty())
            throw std::runtime_error("native fixture has no cases");
    }

    template<class T> std::vector<T> read(const std::string &name) const {
        return geometry.at(name).read<T>();
    }
};

template<class T> void writeArray(const std::string &path, const std::vector<T> &data) {
    if (!std::ofstream(path, std::ios::binary).write(
            reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T)))
        throw std::runtime_error("failed to write native result: " + path);
}

using NativeClock = std::chrono::steady_clock;

inline double elapsedMs(NativeClock::time_point start) {
    return std::chrono::duration<double, std::milli>(NativeClock::now() - start).count();
}

inline void writeBuildTimes(const NativeCase &item, double build, double boundary = -1.) {
    std::ofstream times(item.prefix + ".prepare.json");
    times << "{\"build_ms\":" << build;
    if (boundary >= 0.)
        times << ",\"boundary_ms\":" << boundary;
    times << "}\n";
}
