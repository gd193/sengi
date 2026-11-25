// Use FCPW's silhouette traversal and materialize the capped radius on misses.
#include "bvh.cs.slang"

#if FCPW_BVH_HAS_SILHOUETTE_DATA
[shader("compute")]
[numthreads(256, 1, 1)]
void cappedSilhouettePoint(uint3 threadId : SV_DispatchThreadID,
                           uniform StructuredBuffer<BoundingSphere> boundingSpheres,
                           uniform StructuredBuffer<uint> flipNormalOrientation,
                           uniform RWStructuredBuffer<Interaction> interactions,
                           uniform float squaredMinRadius,
                           uniform float precision,
                           uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries) return;
    BoundingSphere sphere = boundingSpheres[index];
    float cap = sphere.r2; // This entry point receives the original, unsquared cap.
    sphere.r2 = min(cap * cap, 3.402823466e38f);
    Interaction hit;
    bool found = gBvh.findClosestSilhouettePoint(sphere, flipNormalOrientation[index] != 0,
                                                  squaredMinRadius, precision, hit);
    if (!found) hit.d = cap;
    interactions[index] = hit;
}
#else
[shader("compute")]
[numthreads(256, 1, 1)]
void sampleInBall(uint3 threadId : SV_DispatchThreadID,
                  uniform StructuredBuffer<BoundingSphere> boundingSpheres,
                  uniform StructuredBuffer<float3> randNums,
                  uniform RWStructuredBuffer<Interaction> interactions,
                  uniform uint nQueries)
{
    uint index = threadId.x;
    if (index >= nQueries) return;
    BoundingSphere sphere = boundingSpheres[index];
    ConstantBranchTraversalWeight weight;
    Interaction hit;
    bool found = gBvh.intersect<ConstantBranchTraversalWeight>(sphere, randNums[index], weight, hit);
    if (found) {
        float3 delta = hit.p - boundingSpheres[index].c;
        found = dot(delta, delta) <= boundingSpheres[index].r2;
    }
    if (!found) {
        hit.p = hit.n = float3(0);
        hit.index = ~0u;
        hit.d = 0;
    }
    interactions[index] = hit;
}
#endif
