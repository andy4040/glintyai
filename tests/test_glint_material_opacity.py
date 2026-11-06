"""Shared-reflection visibility must respect material transparency.

The CPU reference runs unconditionally. To exercise the real extension, build
it in an isolated environment and set GLINT_TEST_OPTIX_MATERIAL=1.
"""

import math
import os

import pytest
import torch

from glint.checkpoint import GlintCheckpoint
from glint.optix_backend import OptixSurfelTracer
from glint.torch_tracer import TorchSurfelTracer
from glint.transport import RayBundle, make_source_surface_filter, trace_reflection_paths
from test_glint_transport import _gaussian_set


@pytest.fixture(params=("torch", "optix"))
def material_backend(request):
    if request.param == "optix":
        if os.environ.get("GLINT_TEST_OPTIX_MATERIAL") != "1":
            pytest.skip("requires opt-in GLINT_TEST_OPTIX_MATERIAL=1 and an isolated build")
        if not torch.cuda.is_available():
            pytest.skip("actual OptiX material integration requires CUDA")
        return OptixSurfelTracer(), torch.device("cuda")
    return TorchSurfelTracer(), torch.device("cpu")


def _two_layers(device, *, sign=1, transparency=.75, gradient_scene=False):
    scene = _gaussian_set(
        [[0, 0, sign], [0, 0, 2 * sign]], role="interface", opacity=.8,
        transparency=transparency, device=device,
    )
    colors = scene._xyz.new_tensor(
        [[.7, .2, .1], [.1, .3, .8]] if gradient_scene else [[1., 0., 0.], [0., 0., 1.]]
    )
    with torch.no_grad():
        scene._features_dc.copy_(((colors - .5) / .28209479177387814)[:, None])
        scene._transmission_coeff[1, 0] = torch.logit(
            scene._xyz.new_tensor(.15 if gradient_scene else 0.)
        )
    # Off the triangle diagonal/support boundaries, with non-unit direction.
    rays = RayBundle(
        scene._xyz.new_tensor([[[.1, -.2, 0.]]]),
        scene._xyz.new_tensor([[[0., 0., 2. * sign]]]),
    )
    return scene, rays


def _checkpoint(scene, *, normal_cosine=.9, env=None):
    zero = scene._xyz.new_zeros(3)
    return GlintCheckpoint(
        scene, env, None, zero, zero, zero, 0,
        source_surface_filter=True, source_normal_cosine=normal_cosine,
    )


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("transparency", [0., .75, 1.])
def test_transparent_foreground_reveals_target_from_both_faces(material_backend, sign, transparency):
    tracer, device = material_backend
    scene, rays = _two_layers(device, sign=sign, transparency=transparency)
    zero = scene._xyz.new_zeros(3)
    result = tracer.trace(scene, rays, background=zero, material_aware_opacity=True)
    geometric_alpha = .8 * math.exp(-.5 * (.1 ** 2 + .2 ** 2))
    front_alpha = geometric_alpha * (1. - transparency)
    target_weight = (1. - front_alpha) * geometric_alpha
    expected = scene._xyz.new_tensor([[[front_alpha, 0., target_weight]]])
    torch.testing.assert_close(result.rgb, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(
        result.alpha, scene._xyz.new_tensor([[[front_alpha + target_weight]]]),
        atol=2e-5, rtol=2e-4,
    )
    # Plain tracing retains geometric coverage regardless of material tau.
    plain = tracer.trace(scene, rays, background=zero)
    torch.testing.assert_close(
        plain.rgb,
        scene._xyz.new_tensor([[[geometric_alpha, 0., (1. - geometric_alpha) * geometric_alpha]]]),
        atol=2e-5, rtol=2e-4,
    )
    if transparency == 0.:
        torch.testing.assert_close(result.rgb, plain.rgb, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("role", ["reflection", "transmission"])
def test_sets_without_transparency_keep_geometric_opacity(material_backend, role):
    tracer, device = material_backend
    scene = _gaussian_set([[0, 0, 2]], role=role, device=device)
    rays = RayBundle(scene._xyz.new_tensor([[[.1, -.2, 0.]]]), scene._xyz.new_tensor([[[0., 0., 1.]]]))
    zero = scene._xyz.new_zeros(3)
    plain = tracer.trace(scene, rays, background=zero)
    aware = tracer.trace(scene, rays, background=zero, material_aware_opacity=True)
    torch.testing.assert_close(aware.rgb, plain.rgb)
    torch.testing.assert_close(aware.alpha, plain.alpha)


def test_shared_path_uses_material_opacity_without_mutating_direct_or_extra(material_backend):
    tracer, device = material_backend
    scene, rays = _two_layers(device)
    extra = _gaussian_set([[0, 0, 3]], role="reflection", color=(.2, .3, .4), device=device)
    checkpoint = _checkpoint(scene, env=extra)
    zero = scene._xyz.new_zeros(3)
    original_opacity = scene._opacity.detach().clone()
    direct_before = tracer.trace(scene, rays, background=zero)
    combined, shared, separate = trace_reflection_paths(
        checkpoint, tracer, rays, shared=True, separate=True,
    )
    expected = tracer.trace(scene, rays, background=zero, material_aware_opacity=True)
    expected_extra = tracer.trace(extra, rays, background=zero)
    torch.testing.assert_close(shared.rgb, expected.rgb, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(separate.rgb, expected_extra.rgb, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(combined.rgb, shared.rgb + (1. - shared.alpha) * separate.rgb)
    direct_after = tracer.trace(scene, rays, background=zero)
    torch.testing.assert_close(direct_before.rgb, direct_after.rgb)
    torch.testing.assert_close(scene._opacity, original_opacity)


def _rgb_loss(result):
    return (result.rgb * result.rgb.new_tensor([.7, -.2, .4])).sum()


def test_material_and_opacity_gradients_match_reference_and_finite_difference(material_backend):
    tracer, device = material_backend
    scene, rays = _two_layers(device, transparency=.35, gradient_scene=True)
    zero = scene._xyz.new_zeros(3)
    parameters = (scene._opacity, scene._transmission_coeff, scene._features_dc)

    def render(backend):
        return backend.trace(scene, rays, background=zero, material_aware_opacity=True)

    actual = render(tracer)
    gradients = torch.autograd.grad(_rgb_loss(actual), parameters)
    expected = render(TorchSurfelTracer())
    reference_gradients = torch.autograd.grad(_rgb_loss(expected), parameters)
    torch.testing.assert_close(actual.rgb, expected.rgb, atol=2e-5, rtol=2e-4)
    for actual_gradient, reference_gradient in zip(gradients, reference_gradients):
        assert torch.isfinite(actual_gradient).all()
        assert actual_gradient.abs().sum().item() > 1e-4
        torch.testing.assert_close(actual_gradient, reference_gradient, atol=5e-5, rtol=5e-4)

    # Perturb activated-material/opacity logits and a color coefficient, keeping
    # geometry, support, alpha thresholds, and RGB clamp branches unchanged.
    step = 1e-3
    probes = [(0, (0, 0)), (0, (1, 0)), (1, (0, 0)), (1, (1, 0)), (2, (0, 0, 1))]
    with torch.no_grad():
        for parameter_index, index in probes:
            parameter = parameters[parameter_index]
            original = parameter[index].item()
            try:
                parameter[index] = original + step
                plus = _rgb_loss(render(tracer))
                parameter[index] = original - step
                minus = _rgb_loss(render(tracer))
            finally:
                parameter[index] = original
            finite_difference = (plus - minus) / (2. * step)
            torch.testing.assert_close(
                gradients[parameter_index][index], finite_difference,
                atol=3e-5, rtol=2e-2,
            )


def test_source_membership_uses_original_geometric_opacity(material_backend):
    tracer, device = material_backend
    scene = _gaussian_set(
        [[.0025, .0001, 2.]], scale=(.001, .001), opacity=.8,
        transparency=.98, role="interface", device=device,
    )
    checkpoint = _checkpoint(scene)
    incoming = RayBundle(scene._xyz.new_zeros(1, 1, 3), scene._xyz.new_tensor([[[0., 0., 1.]]]))
    rays = RayBundle(scene._xyz.new_tensor([[[0., 0., 2.001]]]), scene._xyz.new_tensor([[[2.5, 0., -1.]]]))
    source = make_source_surface_filter(incoming, rays, scene._xyz.new_tensor([[[0., 0., -1.]]]), checkpoint)
    geometric_source_alpha = .8 * math.exp(-.5 * (2.5 ** 2 + .1 ** 2))
    assert geometric_source_alpha > 1. / 255.
    assert geometric_source_alpha * .02 < 1. / 255.
    unfiltered = tracer.trace(scene, rays, background=checkpoint.bg_color, material_aware_opacity=True)
    assert unfiltered.alpha.item() > 1. / 255.
    filtered = tracer.trace(
        scene, rays, background=checkpoint.bg_color,
        material_aware_opacity=True, source_surface=source,
    )
    torch.testing.assert_close(filtered.alpha, torch.zeros_like(filtered.alpha), atol=1e-6, rtol=0.)
    filtered.rgb.sum().backward()
    for parameter in (scene._opacity, scene._transmission_coeff, scene._features_dc):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), atol=1e-7, rtol=0.)


@pytest.mark.parametrize("normal_cosine,threshold,excluded", [(.929, .9, True), (.929, .95, False), (.89, .9, False)])
def test_normal_threshold_point_nine_excludes_tilted_source_but_keeps_other_faces(
    material_backend, normal_cosine, threshold, excluded,
):
    tracer, device = material_backend
    scene = _gaussian_set([[0., 0., 2.]], opacity=.8, transparency=.3, role="interface", device=device)
    angle = math.acos(normal_cosine)
    with torch.no_grad():
        scene._rotation[0] = scene._xyz.new_tensor([math.cos(angle / 2), 0., math.sin(angle / 2), 0.])
    checkpoint = _checkpoint(scene, normal_cosine=threshold)
    incoming = RayBundle(scene._xyz.new_zeros(1, 1, 3), scene._xyz.new_tensor([[[0., 0., 1.]]]))
    rays = RayBundle(scene._xyz.new_tensor([[[0., 0., 2.001]]]), scene._xyz.new_tensor([[[.08, -.06, -1.]]]))
    source = make_source_surface_filter(incoming, rays, scene._xyz.new_tensor([[[0., 0., -1.]]]), checkpoint)
    result = tracer.trace(
        scene, rays, background=checkpoint.bg_color,
        material_aware_opacity=True, source_surface=source,
    )
    if excluded:
        assert result.alpha.item() < 1e-6
    else:
        assert result.alpha.item() > .55

