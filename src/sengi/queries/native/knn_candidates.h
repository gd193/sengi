#pragma once
#include "../../bvh/traversal/native/traversal.h"

namespace bvh {
// The host specialization selects register-list or heap storage.
// Separate standard-layout records keep the native ABI and ID-only storage.
template<int K, bool Small, bool TrackSlots = false> struct KnnCandidates;

template<int K> struct KnnCandidates<K, true, false> {
    float scores[K];
    int32_t ids[K];
    int32_t size;
    CUDA_CALLABLE KnnCandidates() : size(0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            scores[i] = INFINITY;
            ids[i] = -1;
        }
    }
};

template<int K> struct KnnCandidates<K, false, false> {
    uint64_t entries[K];
    int32_t size;
    CUDA_CALLABLE KnnCandidates() : size(0) {}
};

template<int K> struct KnnCandidates<K, true, true> {
    int32_t slots[K];
    float scores[K];
    int32_t ids[K];
    int32_t size;
    CUDA_CALLABLE KnnCandidates() : size(0) {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            slots[i] = -1;
            scores[i] = INFINITY;
            ids[i] = -1;
        }
    }
};

template<int K> struct KnnCandidates<K, false, true> {
    int32_t slots[K];
    uint64_t entries[K];
    int32_t size;
    CUDA_CALLABLE KnnCandidates() : size(0) {}
};

template<int K, bool Small, bool TrackSlots>
CUDA_CALLABLE inline void knn_insert(KnnCandidates<K, Small, TrackSlots>& c, float score, int id, int slot = -1) {
    const int n = c.size;
    if constexpr (Small) {
        // Constant indices let CUDA keep these candidate lists in registers.
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            const float previous_score = c.scores[i];
            const int previous_id = c.ids[i];
            const bool replace = previous_score > score;
            if constexpr (TrackSlots) {
                const int previous_slot = c.slots[i];
                c.slots[i] = replace ? slot : previous_slot;
                slot = replace ? previous_slot : slot;
            }
            c.scores[i] = replace ? score : previous_score;
            c.ids[i] = replace ? id : previous_id;
            score = replace ? previous_score : score;
            id = replace ? previous_id : id;
        }
        if (n < K) c.size = n+1;
    } else {
        // Keep the ID in the low word for storage. Compare only the score.
        const uint64_t value = cache_entry(uint32_t(id), score);
        if (n < K) {
            int p = n;
            c.size = n+1;
            while (p > 0) {
                const int parent = (p-1)/4;
                if (score <= entry_score(c.entries[parent])) break;
                c.entries[p] = c.entries[parent];
                if constexpr (TrackSlots) c.slots[p] = c.slots[parent];
                p = parent;
            }
            c.entries[p] = value;
            if constexpr (TrackSlots) c.slots[p] = slot;
        } else {
            if (score >= entry_score(c.entries[0])) return;
            int p = 0;
            // Four children shorten the sift. Retain the selected value to
            // avoid reloading it through a runtime-selected local index.
            while (4*p+1 < K) {
                int child = 4*p+1;
                uint64_t child_value = c.entries[child];
                #pragma unroll
                for (int offset = 1; offset < 4; ++offset) {
                    const int other = 4*p+1+offset;
                    if (other < K) {
                        const uint64_t other_value = c.entries[other];
                        if (entry_score(other_value) > entry_score(child_value)) {
                            child = other;
                            child_value = other_value;
                        }
                    }
                }
                if (entry_score(child_value) <= entry_score(value)) break;
                c.entries[p] = child_value;
                if constexpr (TrackSlots) c.slots[p] = c.slots[child];
                p = child;
            }
            c.entries[p] = value;
            if constexpr (TrackSlots) c.slots[p] = slot;
        }
    }
}

template<int K, bool Small, bool TrackSlots>
CUDA_CALLABLE inline float knn_score(const KnnCandidates<K, Small, TrackSlots>& c, int i) {
    if constexpr (Small) return c.scores[i];
    else return entry_score(c.entries[i]);
}
template<int K, bool Small, bool TrackSlots>
CUDA_CALLABLE inline int knn_id(const KnnCandidates<K, Small, TrackSlots>& c, int i) {
    if constexpr (Small) return c.ids[i];
    else return static_cast<int32_t>(uint32_t(c.entries[i]));
}
template<int K, bool Small, bool TrackSlots>
CUDA_CALLABLE inline float knn_limit(const KnnCandidates<K, Small, TrackSlots>& c, float initial) {
    return c.size < K ? initial : knn_score(c, Small ? K-1 : 0);
}
template<int K, bool Small, bool TrackSlots>
CUDA_CALLABLE inline void knn_sort(KnnCandidates<K, Small, TrackSlots>& c) {
    if constexpr (!Small) {
        for (int end = c.size-1; end > 0; --end) {
            const uint64_t value = c.entries[end];
            int slot = -1;
            if constexpr (TrackSlots) {
                slot = c.slots[end];
                c.slots[end] = c.slots[0];
            }
            c.entries[end] = c.entries[0];
            int p = 0;
            while (4*p+1 < end) {
                int child = 4*p+1;
                uint64_t child_value = c.entries[child];
                #pragma unroll
                for (int offset = 1; offset < 4; ++offset) {
                    const int other = 4*p+1+offset;
                    if (other < end) {
                        const uint64_t other_value = c.entries[other];
                        if (entry_score(other_value) > entry_score(child_value)) {
                            child = other;
                            child_value = other_value;
                        }
                    }
                }
                if (entry_score(child_value) <= entry_score(value)) break;
                c.entries[p] = child_value;
                if constexpr (TrackSlots) c.slots[p] = c.slots[child];
                p = child;
            }
            c.entries[p] = value;
            if constexpr (TrackSlots) c.slots[p] = slot;
        }
    }
}
} // namespace bvh
