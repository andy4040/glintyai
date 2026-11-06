"""Run the actual native SH functions, isolated from ray-intersection boundaries.

The CPU harness compiles the function bodies extracted from the production CUDA
sources; the optional CUDA harness runs those same bodies in a CUDA kernel.
No SH formula is copied into the test harness. Native colors include +0.5 and
the positive RGB clamp; the PyTorch reference evaluator does not.

The zero/sub-epsilon tests check the documented homogeneous native extension,
not equality with the Python evaluator's unit-sphere-simplified polynomials.
Set GLINT_TEST_SH_CUDA=1 to also build and run the CUDA kernel harness.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from glint.spherical_harmonics import evaluate_spherical_harmonics


_NATIVE = Path(__file__).resolve().parents[1] / "third_party/diff-surfel-tracing/optix_tracer"


def _extract_function(source: str, name: str) -> str:
    match = re.search(r"__device__\s+(?:float3|void)\s+" + name + r"\s*\(", source)
    assert match is not None, f"Native function {name} is missing"
    opening = source.index("{", match.end())
    depth = 1
    for cursor in range(opening + 1, len(source)):
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        if depth == 0:
            return source[match.start() : cursor + 1]
    raise AssertionError(f"Unterminated native function {name}")


def _harness_source(cuda: bool) -> str:
    auxiliary = (_NATIVE / "auxiliary.h").read_text()
    constants = re.search(
        r"__device__ const float SH_C0\s*=[\s\S]*?__device__ const float SH_C3\[\]\s*=\s*\{[\s\S]*?\};",
        auxiliary,
    )
    assert constants is not None
    forward = (_NATIVE / "forward.cu").read_text()
    backward = (_NATIVE / "backward.cu").read_text()
    direction = (_NATIVE / "sh_direction.h").read_text().replace("#pragma once", "")
    prefix = "#include <cuda_runtime.h>\n" if cuda else """
#define __device__
#define __forceinline__ inline
struct float3 { float x, y, z; };
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
"""
    math = r"""
#include <cmath>
__device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ inline float3 operator*(float3 a, float s) { return make_float3(a.x*s,a.y*s,a.z*s); }
__device__ inline float3 operator*(float s, float3 a) { return a*s; }
__device__ inline float3& operator+=(float3& a, float3 b) { a=a+b; return a; }
__device__ inline float3& operator+=(float3& a, float s) { a=a+make_float3(s,s,s); return a; }
__device__ inline float dot(float3 a, float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ inline float3 max(float3 a, float s) { return make_float3(fmaxf(a.x,s),fmaxf(a.y,s),fmaxf(a.z,s)); }
"""
    bodies = "\n".join(
        [constants.group(), direction,
         _extract_function(forward, "computeColorFromSH"),
         _extract_function(backward, "computeColorFromSH"),
         _extract_function(backward, "computeColorFromSHBackward")]
    )
    # Each row has 57 input floats: direction(3), SH(48), upstream(3), seed(3).
    # Output has forward(3), recomputed(3), SH gradient(48), dir gradient(3), mask(3).
    single = r"""
__device__ void evaluate_row(int degree, const float* input, float* output) {
    float3 direction=make_float3(input[0],input[1],input[2]);
    float3 sh[16], grad_sh[16];
    for(int j=0;j<16;++j) {
        sh[j]=make_float3(input[3+3*j],input[4+3*j],input[5+3*j]);
        grad_sh[j]=make_float3(0,0,0);
    }
    float3 first=computeColorFromSH(degree,sh,direction);
    float mask[3];
    float3 second=computeColorFromSH(degree,sh,direction,mask);
    float3 dir_grad=make_float3(input[54],input[55],input[56]);
    computeColorFromSHBackward(degree,sh,direction,mask,input+51,grad_sh,dir_grad);
    output[0]=first.x; output[1]=first.y; output[2]=first.z;
    output[3]=second.x; output[4]=second.y; output[5]=second.z;
    for(int j=0;j<16;++j) {
        output[6+3*j]=grad_sh[j].x;
        output[7+3*j]=grad_sh[j].y;
        output[8+3*j]=grad_sh[j].z;
    }
    output[54]=dir_grad.x; output[55]=dir_grad.y; output[56]=dir_grad.z;
    output[57]=mask[0]; output[58]=mask[1]; output[59]=mask[2];
}
"""
    if cuda:
        entry = r"""
__global__ void evaluate_kernel(int degree,int count,const float* input,float* output) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) evaluate_row(degree,input+57*i,output+60*i);
}
extern "C" int evaluate_native(int degree,int count,const float* input,float* output) {
    float *d_input=nullptr,*d_output=nullptr;
    cudaError_t status=cudaMalloc(&d_input,count*57*sizeof(float));
    if(status!=cudaSuccess) return static_cast<int>(status);
    status=cudaMalloc(&d_output,count*60*sizeof(float));
    if(status==cudaSuccess) status=cudaMemcpy(d_input,input,count*57*sizeof(float),cudaMemcpyHostToDevice);
    if(status==cudaSuccess) {
        evaluate_kernel<<<(count+63)/64,64>>>(degree,count,d_input,d_output);
        status=cudaGetLastError();
    }
    if(status==cudaSuccess) status=cudaDeviceSynchronize();
    if(status==cudaSuccess) status=cudaMemcpy(output,d_output,count*60*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_input); cudaFree(d_output);
    return static_cast<int>(status);
}
"""
    else:
        entry = r"""
extern "C" int evaluate_native(int degree,int count,const float* input,float* output) {
    for(int i=0;i<count;++i) evaluate_row(degree,input+57*i,output+60*i);
    return 0;
}
"""
    return prefix + math + bodies + single + entry


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def native_sh(request, tmp_path_factory):
    cuda = request.param == "cuda"
    if cuda and os.environ.get("GLINT_TEST_SH_CUDA") != "1":
        pytest.skip("Set GLINT_TEST_SH_CUDA=1 to enable the CUDA native harness")
    if os.name == "nt":
        pytest.skip("Native harness uses the Linux compiler/linker flags used by GLINT")
    compiler = shutil.which("nvcc" if cuda else os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip(f"{'nvcc' if cuda else 'C++ compiler'} unavailable")
    if cuda and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable; CPU native checks still run")
    build = tmp_path_factory.mktemp(f"native_sh_{request.param}")
    source = build / ("sh_harness.cu" if cuda else "sh_harness.cpp")
    library = build / "sh_harness.so"
    source.write_text(_harness_source(cuda))
    flags = ["-Xcompiler", "-fPIC"] if cuda else ["-fPIC"]
    if cuda:
        major, minor = torch.cuda.get_device_capability()
        flags += [f"-arch=sm_{major}{minor}"]
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-shared", *flags, str(source), "-o", str(library)],
        check=True, capture_output=True, text=True,
    )
    handle = ctypes.CDLL(str(library))
    call = handle.evaluate_native
    float_ptr = ctypes.POINTER(ctypes.c_float)
    call.argtypes = [ctypes.c_int, ctypes.c_int, float_ptr, float_ptr]
    call.restype = ctypes.c_int

    def evaluate(degree, directions, coeffs, upstream=None, seed=None):
        directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        count = len(directions)
        coeffs = np.asarray(coeffs, dtype=np.float32).reshape(count, 16, 3)
        upstream = np.ones((count, 3), np.float32) if upstream is None else np.asarray(upstream, np.float32)
        seed = np.zeros((count, 3), np.float32) if seed is None else np.broadcast_to(seed, (count, 3))
        values = np.ascontiguousarray(np.concatenate((directions, coeffs.reshape(count, 48), upstream, seed), axis=1), dtype=np.float32)
        output = np.zeros((count, 60), np.float32)
        status = call(degree, count, values.ctypes.data_as(float_ptr), output.ctypes.data_as(float_ptr))
        assert status == 0, f"CUDA harness error code {status}"
        return {
            "color": output[:, :3], "recomputed": output[:, 3:6],
            "coeff_grad": output[:, 6:54].reshape(count, 16, 3),
            "dir_grad": output[:, 54:57], "clamped": output[:, 57:60],
        }

    return evaluate


def _inputs():
    rng = np.random.default_rng(1234)
    directions = np.array([[0.3, -0.7, 2.1], [-1.1, 0.2, 1.7], [0.5, 1.4, -0.9]], np.float32)
    coeffs = rng.normal(0, 0.06, (3, 16, 3)).astype(np.float32)
    coeffs[:, 0] += 0.3  # Keep all ordinary test colors safely above the clamp.
    upstream = np.array([[0.7, -0.2, 0.4], [-0.3, 0.5, 0.9], [0.4, 0.2, -0.6]], np.float32)
    return directions, coeffs, upstream


def _reference(degree, directions, coeffs, upstream):
    d = torch.tensor(directions, requires_grad=True)
    c = torch.tensor(coeffs, requires_grad=True)
    color = (evaluate_spherical_harmonics(degree, d, c) + 0.5).clamp_min(0)
    dc, dd = torch.autograd.grad((color * torch.tensor(upstream)).sum(), (c, d), allow_unused=True)
    return color.detach().numpy(), dc.detach().numpy(), np.zeros_like(directions) if dd is None else dd.detach().numpy()


@pytest.mark.parametrize("degree", range(4))
def test_native_sh_matches_reference_and_preserves_existing_gradient(native_sh, degree):
    d, c, upstream = _inputs()
    seed = np.array([0.7, -0.3, 0.2], np.float32)
    actual = native_sh(degree, d, c, upstream, seed)
    color, coeff_grad, dir_grad = _reference(degree, d, c, upstream)
    np.testing.assert_allclose(actual["color"], color, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(actual["recomputed"], actual["color"], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(actual["coeff_grad"], coeff_grad, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(actual["dir_grad"], seed + dir_grad, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("degree", range(4))
def test_native_sh_direction_scale_invariance(native_sh, degree):
    d, c, upstream = _inputs()
    a = native_sh(degree, d, c, upstream)
    b = native_sh(degree, 2*d, c, upstream)
    unit = native_sh(degree, F.normalize(torch.tensor(d), dim=-1).numpy(), c, upstream)
    for result in (b, unit):
        np.testing.assert_allclose(result["color"], a["color"], atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(result["coeff_grad"], a["coeff_grad"], atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(b["dir_grad"], a["dir_grad"] / 2, atol=2e-6, rtol=3e-5)
    # Normalization removes the radial derivative for nondegenerate directions.
    np.testing.assert_allclose((a["dir_grad"] * d).sum(-1), 0, atol=2e-6)


@pytest.mark.parametrize("degree", range(4))
def test_native_sh_direction_finite_difference_away_from_clamp(native_sh, degree):
    d, c, upstream = _inputs()
    actual = native_sh(degree, d, c, upstream)
    assert actual["color"].min() > 0.2
    h = 1e-3
    finite = np.empty_like(d)
    for axis in range(3):
        plus, minus = d.copy(), d.copy()
        plus[:, axis] += h
        minus[:, axis] -= h
        p, m = native_sh(degree, plus, c, upstream), native_sh(degree, minus, c, upstream)
        assert not p["clamped"].any() and not m["clamped"].any()
        finite[:, axis] = ((p["color"]-m["color"])*upstream).sum(-1)/(2*h)
    np.testing.assert_allclose(actual["dir_grad"], finite, atol=5e-5, rtol=5e-3)


@pytest.mark.parametrize("degree", range(4))
def test_native_sh_clamped_channels_keep_original_behavior(native_sh, degree):
    d, c, upstream = _inputs()
    c[:, 0, (0, 2)] = -3.0
    actual = native_sh(degree, d, c, upstream)
    color, coeff_grad, dir_grad = _reference(degree, d, c, upstream)
    assert (actual["clamped"][:, (0, 2)] == 1).all()
    assert (actual["color"][:, (0, 2)] == 0).all()
    np.testing.assert_allclose(actual["color"], color, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(actual["coeff_grad"], coeff_grad, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(actual["dir_grad"], dir_grad, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("degree", range(4))
def test_native_sh_zero_and_tiny_directions_are_consistent_and_finite(native_sh, degree):
    _, c, upstream = _inputs()
    d = np.array([[0, 0, 0], [2e-15, -3e-15, 4e-15], [2e-10, -3e-10, 4e-10]], np.float32)
    actual = native_sh(degree, d, c, upstream)
    for value in actual.values():
        assert np.isfinite(value).all()
    np.testing.assert_allclose(actual["recomputed"], actual["color"], atol=1e-7, rtol=1e-6)
    # Finite differences strictly within the constant-denominator branch. The
    # unit-sphere Python polynomial is not an oracle for that native extension.
    h = 1e-15
    finite = np.empty((2, 3), np.float32)
    for axis in range(3):
        plus, minus = d[:2].copy(), d[:2].copy()
        plus[:, axis] += h
        minus[:, axis] -= h
        p = native_sh(degree, plus, c[:2], upstream[:2])
        m = native_sh(degree, minus, c[:2], upstream[:2])
        finite[:, axis] = ((p["color"]-m["color"])*upstream[:2]).sum(-1)/(2*h)
    np.testing.assert_allclose(actual["dir_grad"][:2], finite, atol=5e7, rtol=1e-2)
