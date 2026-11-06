#pragma once

// Normalize only the SH input, never the ray used for intersections/z-depth.
// For a degenerate direction use d / eps (the existing homogeneous SH basis
// extension off the unit sphere), with the matching constant-denominator VJP.
// This branch is finite at zero; scale invariance applies above the epsilon.
struct SHDirection {
    float3 unit;
    float inverse_length;
    bool clamped;
};

static __forceinline__ __device__ SHDirection normalize_sh_direction(const float3& d)
{
    constexpr float epsilon = 1e-12f;
    // hypot avoids squared-norm overflow/underflow for finite extreme inputs.
    const float norm = hypotf(hypotf(d.x, d.y), d.z);
    const float inverse_length = 1.0f / fmaxf(norm, epsilon);
    return {d * inverse_length, inverse_length, norm <= epsilon};
}

static __forceinline__ __device__ float3 sh_direction_backward(
    const SHDirection& direction, const float3& gradient)
{
    if (direction.clamped)
        return gradient * direction.inverse_length;
    return (gradient - direction.unit * dot(direction.unit, gradient))
        * direction.inverse_length;
}
