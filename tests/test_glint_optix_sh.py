# SPDX-FileCopyrightText: Copyright (c) 2026 GLINT Authors
# SPDX-License-Identifier: MIT

"""Opt-in end-to-end checks of SH evaluation in the actual OptiX extension.

Build/install the modified extension in an isolated environment, then run:
    GLINT_TEST_OPTIX_SH=1 python -m pytest -q tests/test_glint_optix_sh.py

These tests intentionally do not build or replace any extension themselves.
Zero/tiny directions cannot hit this surfel and are covered by the SH unit tests.
"""

from __future__ import annotations

import os

import pytest
import torch

from glint.checkpoint import GlintGaussianSet
from glint.optix_backend import OptixSurfelTracer
from glint.spherical_harmonics import evaluate_spherical_harmonics
from glint.torch_tracer import TorchSurfelTracer
from glint.transport import RayBundle


pytestmark = pytest.mark.skipif(
    os.environ.get("GLINT_TEST_OPTIX_SH") != "1",
    reason="requires opt-in GLINT_TEST_OPTIX_SH=1 and an isolated OptiX build",
)


@pytest.fixture
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("actual OptiX SH integration requires a CUDA device")
    return torch.device("cuda")


def _scene(degree: int, device: torch.device) -> GlintGaussianSet:
    generator = torch.Generator().manual_seed(20261106)
    coefficients = (0.08 * torch.randn(1, 16, 3, generator=generator)).to(device)
    coefficients[:, 0] = coefficients.new_tensor([0.12, 0.20, 0.28])
    state = {
        "g.active_sh_degree": torch.tensor([degree], device=device),
        "g._xyz": torch.tensor([[0.0, 0.0, 2.0]], device=device),
        "g._features_dc": coefficients[:, :1],
        "g._features_rest": coefficients[:, 1:],
        "g._scaling": torch.tensor([[0.9, 1.2]], device=device).log(),
        "g._rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "g._opacity": torch.tensor([[0.65]], device=device).logit(),
    }
    return GlintGaussianSet(state, "g", role="reflection", trainable=True)


def _rays(directions: torch.Tensor) -> RayBundle:
    return RayBundle(
        origins=directions.new_tensor([[[0.1, -0.2, 0.0]]]),
        directions=directions,
    )


def _rgb_loss(result) -> torch.Tensor:
    return (result.rgb * result.rgb.new_tensor([0.7, -0.2, 0.4])).sum()


def _render_with_grad(tracer, scene, directions, background):
    directions = directions.detach().clone().requires_grad_(True)
    result = tracer.trace(scene, _rays(directions), background=background)
    dc_grad, rest_grad, direction_grad = torch.autograd.grad(
        _rgb_loss(result),
        (scene._features_dc, scene._features_rest, directions),
    )
    return {
        "rgb": result.rgb.detach(),
        "alpha": result.alpha.detach(),
        "depth": result.depth.detach(),
        "sh_grad": torch.cat((dc_grad, rest_grad), dim=1).detach(),
        "direction_grad": direction_grad.detach(),
    }


@pytest.mark.parametrize("degree", range(4))
def test_optix_sh_matches_torch_and_preserves_z_depth(degree, cuda_device):
    native = OptixSurfelTracer()
    reference = TorchSurfelTracer()
    scene = _scene(degree, cuda_device)
    directions = torch.tensor([[[0.32, 0.24, 1.7]]], device=cuda_device)
    # Use black background to isolate this SH regression. The existing native
    # alpha backward subtracts nonzero background twice (out_rgb already
    # includes it); correcting that independent compositing bug is out of scope.
    # The off-center hit still gives a nonzero geometric direction derivative.
    background = torch.zeros(3, device=cuda_device)

    # This ray stays well inside one triangle, away from its diagonal, the
    # Gaussian support/alpha cutoffs and RGB clamp. Finite differences therefore
    # keep the same primitive and active branches.
    color = evaluate_spherical_harmonics(
        degree, directions, scene.get_features
    ) + 0.5
    assert color.min().item() > 0.2
    assert color.max().item() < 0.9

    scales = (1.0, 2.0, 1.0 / directions.norm().item())
    native_results = []
    for scale in scales:
        actual = _render_with_grad(native, scene, directions * scale, background)
        expected = _render_with_grad(reference, scene, directions * scale, background)
        for name in ("rgb", "alpha", "depth"):
            torch.testing.assert_close(
                actual[name], expected[name], rtol=2e-4, atol=2e-5,
                msg=lambda message, name=name: f"degree={degree}, {name}: {message}",
            )
        for name in ("sh_grad", "direction_grad"):
            torch.testing.assert_close(
                actual[name], expected[name], rtol=5e-4, atol=5e-5,
                msg=lambda message, name=name: f"degree={degree}, {name}: {message}",
            )
        native_results.append(actual)

    base = native_results[0]
    assert 0.1 < base["alpha"].item() < 0.9
    # Degree zero has no SH direction derivative: its nonzero ray gradient is
    # entirely from intersection/opacity and must survive the SH backward call.
    assert base["direction_grad"].norm().item() > 1e-3
    for scale, actual in zip(scales[1:], native_results[1:]):
        for name in ("rgb", "alpha", "sh_grad"):
            torch.testing.assert_close(actual[name], base[name], rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(
            actual["depth"], base["depth"] / scale, rtol=2e-4, atol=2e-5
        )
        torch.testing.assert_close(
            actual["direction_grad"], base["direction_grad"] / scale,
            rtol=5e-4, atol=5e-5,
        )

    # Independently check the complete native RGB derivative. This includes the
    # geometric direction derivative plus the normalized SH contribution.
    step = 1e-3
    with torch.no_grad():
        finite_difference = torch.empty_like(directions)
        for axis in range(3):
            delta = torch.zeros_like(directions)
            delta[..., axis] = step
            plus = native.trace(scene, _rays(directions + delta), background=background)
            minus = native.trace(scene, _rays(directions - delta), background=background)
            finite_difference[..., axis] = (_rgb_loss(plus) - _rgb_loss(minus)) / (2 * step)
        torch.testing.assert_close(
            base["direction_grad"], finite_difference, rtol=2e-2, atol=1e-4
        )

        # Probe a nonzero basis at each degree, away from clamp boundaries.
        coefficient_index = (0, 2, 6, 12)[degree]
        parameter = scene._features_dc if coefficient_index == 0 else scene._features_rest
        local_index = max(0, coefficient_index - 1)
        original = parameter[0, local_index, 1].item()
        try:
            parameter[0, local_index, 1] = original + step
            plus = _rgb_loss(native.trace(scene, _rays(directions), background=background))
            parameter[0, local_index, 1] = original - step
            minus = _rgb_loss(native.trace(scene, _rays(directions), background=background))
        finally:
            parameter[0, local_index, 1] = original
        torch.testing.assert_close(
            base["sh_grad"][0, coefficient_index, 1], (plus - minus) / (2 * step),
            rtol=2e-2, atol=1e-4,
        )
