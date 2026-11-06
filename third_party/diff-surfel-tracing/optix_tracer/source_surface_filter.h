#pragma once

// A geometric source-pixel candidate test, evaluated lazily at each hit.
// The same detached decision is used by forward and backward traversal.
// Layout: camera origin(3), unit incoming direction(3), launch position(3),
// unit source normal(3), depth tolerance, max hit distance, normal cosine, min alpha.
static __forceinline__ __device__ bool exclude_source_surface(
    const Params& p, unsigned int gidx, const float3& hit)
{
    if (p.source_surface == nullptr) return false;
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    const float* f = p.source_surface + 16 * (idx.x * dim.y + idx.y);
    const float3 camera = make_float3(f[0], f[1], f[2]);
    const float3 incoming = make_float3(f[3], f[4], f[5]);
    const float3 launch = make_float3(f[6], f[7], f[8]);
    const float3 source_n = make_float3(f[9], f[10], f[11]);
    if (length(hit - launch) > f[13]) return false;
    float3 R[3];
    quat_to_rotmat_transpose(p.rotations[gidx], R);
    const float3 n = R[2];
    if (fabsf(dot(n, source_n)) < f[14]) return false;
    const float denom = dot(n, incoming);
    if (fabsf(denom) <= 1e-6f) return false;
    const float t = dot(p.means3D[gidx] - camera, n) / denom;
    const float source_t = dot(launch - camera, incoming);
    if (t <= 0 || fabsf(t - source_t) > f[12]) return false;
    const float3 delta = camera + t * incoming - p.means3D[gidx];
    const float u = dot(delta, R[0]) / p.scales[gidx].x;
    const float v = dot(delta, R[1]) / p.scales[gidx].y;
    // Match the actual finite OptiX surfel support as well as Gaussian alpha.
    if (fabsf(u) > 3.0f || fabsf(v) > 3.0f) return false;
    // Classify against the original camera-visible surface even when the
    // reflected ray composites with material-attenuated opacity.
    const float opacity = p.source_opacities != nullptr
        ? p.source_opacities[gidx] : p.opacities[gidx];
    const float alpha = opacity * expf(-0.5f * (u*u + v*v));
    return alpha >= f[15];
}
