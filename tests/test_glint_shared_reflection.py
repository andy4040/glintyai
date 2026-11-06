"""Shared-scene reflection curriculum and layered radiance regression tests."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from glint.checkpoint import GlintCheckpoint, load_glint_checkpoint
from glint.camera import GlintCamera
from glint.model import GlintStageSchedule
from glint.trainer import GlintTrainer
from glint.transport import RayBundle, TraceResult, render_glint_transport, trace_reflection_paths
from test_glint_training import _training_model
from test_glint_transport import _gaussian_set
from glint.torch_tracer import TorchSurfelTracer


def test_curriculum_and_optimizer_activation():
    s = GlintStageSchedule(transmission_start=2, shared_reflection_start=4, reflection_start=6)
    assert [s.stage(i) for i in (1, 2, 3, 4, 5, 6)] == [
        "interface", "transmission", "transmission", "shared_reflection", "shared_reflection", "full"
    ]
    stub = SimpleNamespace(schedule=s)
    assert GlintTrainer._active_sets(stub, 4) == ("interface", "transmission")
    assert GlintTrainer._active_sets(stub, 6) == ("interface", "transmission", "reflection")


@pytest.mark.parametrize("shared", [-1, 6, 7])
def test_invalid_curriculum(shared):
    with pytest.raises(ValueError):
        GlintStageSchedule(transmission_start=2, shared_reflection_start=shared, reflection_start=6)


def _shared_first_schedule():
    return GlintStageSchedule(
        transmission_start=3_000,
        shared_reflection_start=1_000,
        reflection_start=6_000,
        separate_reflection_enabled=False,
    )


@pytest.mark.parametrize(
    "step,stage,active",
    [
        (0, "interface", ("interface",)),
        (999, "interface", ("interface",)),
        (1_000, "shared_reflection_only", ("interface",)),
        (2_999, "shared_reflection_only", ("interface",)),
        (3_000, "shared_reflection", ("interface", "transmission")),
        (5_999, "shared_reflection", ("interface", "transmission")),
        (6_000, "shared_reflection", ("interface", "transmission")),
        (40_000, "shared_reflection", ("interface", "transmission")),
    ],
)
def test_shared_first_boundaries_and_disabled_reflection(step, stage, active):
    schedule = _shared_first_schedule()
    assert schedule.stage(step) == stage
    assert GlintTrainer._active_sets(SimpleNamespace(schedule=schedule), step) == active


@pytest.mark.parametrize("step", [999, 1_000, 2_999, 3_000, 6_000, 40_000])
def test_shared_first_checkpoint_roundtrip(tmp_path, step):
    model = _training_model()
    schedule = _shared_first_schedule()
    path = tmp_path / "shared-first.pt"
    torch.save(
        dict(model=model.legacy_state_dict(), step=step, epoch=step, schedule=asdict(schedule)),
        path,
    )
    loaded = load_glint_checkpoint(path)
    live = model.as_checkpoint(step, schedule=schedule)
    assert loaded.render_stage == schedule.stage(step)
    assert loaded.shared_reflection == (step >= 1_000)
    assert loaded.shared_reflection == live.shared_reflection
    assert not loaded.separate_reflection_enabled
    assert not live.separate_reflection_enabled


def test_shared_first_can_enable_separate_reflection_later():
    schedule = GlintStageSchedule(
        transmission_start=3_000, shared_reflection_start=1_000,
        reflection_start=6_000,
    )
    assert schedule.stage(1_000) == "shared_reflection_only"
    assert schedule.stage(3_000) == "shared_reflection"
    assert schedule.stage(6_000) == "full"
    assert GlintTrainer._active_sets(SimpleNamespace(schedule=schedule), 6_000) == (
        "interface", "transmission", "reflection"
    )


class ConstantTracer:
    def __init__(self, alpha):
        self.alpha = alpha
        self.shared_color = torch.tensor([[[.2, .3, .4]]], requires_grad=True)
        self.extra_color = torch.tensor([[[.8, .6, .1]]], requires_grad=True)
        self.calls = []

    def trace(self, g, rays, *, background):
        self.calls.append((g, rays, background))
        a = self.alpha if g == "shared" else .5
        color = self.shared_color if g == "shared" else self.extra_color
        return TraceResult(rgb=a * color + (1-a)*background, alpha=torch.full((1,1,1),a),
                           depth=torch.ones(1,1,1), normal=torch.ones(1,1,3),
                           backend_data={"sentinel":g})


def _checkpoint():
    return GlintCheckpoint(pcd="shared", env="extra", trans_env=None,
                           bg_color=torch.ones(3), env_bg_color=torch.zeros(3),
                           trans_env_bg_color=torch.zeros(3), epoch=0)


@pytest.mark.parametrize("alpha", [0., .4, 1.])
def test_composition_and_gradient_routing(alpha):
    t=ConstantTracer(alpha)
    rays=RayBundle(torch.zeros(1,1,3),torch.tensor([[[0.,0.,2.]]]))
    combined, shared, extra=trace_reflection_paths(_checkpoint(),t,rays,shared=True,separate=True)
    torch.testing.assert_close(combined.rgb, shared.rgb+(1-alpha)*extra.rgb)
    torch.testing.assert_close(combined.alpha,torch.full((1,1,1),alpha+(1-alpha)*.5))
    assert extra.backend_data == {"sentinel":"extra"}
    assert torch.count_nonzero(t.calls[0][2]) == 0
    torch.testing.assert_close(t.calls[0][1].origins,torch.tensor([[[0.,0.,1e-4]]]))
    combined.rgb.sum().backward()
    torch.testing.assert_close(t.shared_color.grad,torch.full((1,1,3),alpha))
    torch.testing.assert_close(t.extra_color.grad,torch.full((1,1,3),(1-alpha)*.5))


def test_warmup_does_not_query_or_train_extra():
    t=ConstantTracer(.5)
    result, shared, extra=trace_reflection_paths(_checkpoint(),t,
        RayBundle(torch.zeros(1,1,3),torch.ones(1,1,3)),shared=True,separate=False)
    result.rgb.sum().backward()
    assert len(t.calls)==1 and extra is None
    assert t.shared_color.grad is not None and t.extra_color.grad is None


@pytest.mark.parametrize("step,stage", [(1,"interface"),(2,"transmission"),(4,"shared_reflection"),(6,"full")])
def test_checkpoint_roundtrip_preserves_curriculum(tmp_path,step,stage):
    model=_training_model()
    schedule=GlintStageSchedule(transmission_start=2,shared_reflection_start=4,reflection_start=6)
    path=tmp_path/'model.pt'
    torch.save(dict(model=model.legacy_state_dict(),step=step,epoch=step,schedule=asdict(schedule)),path)
    loaded=load_glint_checkpoint(path)
    live=model.as_checkpoint(step,schedule=schedule)
    assert loaded.render_stage==stage
    assert loaded.shared_reflection==live.shared_reflection
    assert loaded.separate_reflection_enabled==live.separate_reflection_enabled


def test_shared_tracing_skips_launch_plane_and_hits_scene():
    # Launch at z=0 toward -z: the interface plane at the origin must not
    # obscure the second existing scene surfel at z=-2.
    shared=_gaussian_set([[0,0,0],[0,0,-2]],opacity=.9,role="interface")
    cp=_checkpoint();cp.pcd=shared
    rays=RayBundle(torch.zeros(1,1,3),torch.tensor([[[0.,0.,-1.]]]))
    result,_,_=trace_reflection_paths(cp,TorchSurfelTracer(),rays,shared=True,separate=False)
    torch.testing.assert_close(result.alpha,torch.full((1,1,1),.9),atol=1e-5,rtol=0)
    result.rgb.sum().backward()
    assert shared._features_dc.grad[1].abs().sum()>0
    assert shared._features_dc.grad[0].abs().sum()==0


@pytest.mark.parametrize("omit_environment_sets", [False, True])
@pytest.mark.parametrize("preserve_material", [False, True])
def test_shared_only_render_has_no_transmission_or_extra_path(monkeypatch, omit_environment_sets, preserve_material):
    """The new warm-up uses the interface for both direct and reflected color."""
    interface = _gaussian_set(
        [[0, 0, 2], [0, 0, 0]], role="interface", opacity=.9,
        transparency=.7, specularity=.2,
    )
    with torch.no_grad():
        colors = torch.tensor([[.4, .3, .2], [.1, .7, .2]])
        interface._features_dc.copy_(((colors - .5) / .28209479177387814)[:, None])
    transmission = _gaussian_set([[0, 0, 4]], role="transmission")
    extra = _gaussian_set([[0, 0, -2]])
    checkpoint = GlintCheckpoint(
        pcd=interface, trans_env=None if omit_environment_sets else transmission,
        env=None if omit_environment_sets else extra,
        bg_color=torch.zeros(3), trans_env_bg_color=torch.zeros(3),
        env_bg_color=torch.zeros(3), epoch=1_000, shared_reflection=True,
        separate_reflection_enabled=False, render_stage="shared_reflection_only",
        preserve_transparency_without_transmission=preserve_material,
    )
    raster_calls = []

    def rasterize(camera, gaussian_set, *args, **kwargs):
        raster_calls.append(gaussian_set)
        assert gaussian_set is interface, "inactive environment set was rasterized"
        return SimpleNamespace(
            render=(interface._features_dc[0, 0] * .28209479177387814 + .5).reshape(3, 1, 1),
            surf_depth=torch.full((1, 1, 1), 2.),
            rend_normal=torch.tensor([0., 0., -1.]).reshape(3, 1, 1),
            transparency=interface.get_transmission_coeff[0].reshape(1, 1, 1),
            conditional_transparency=interface.get_transmission_coeff[0].reshape(1, 1, 1),
            specular=interface.get_specular[0].reshape(1, 1, 1),
            rend_alpha=torch.ones(1, 1, 1),
        )

    monkeypatch.setattr("glint.transport.render_glint_camera", rasterize)

    class RecordingTracer:
        def __init__(self):
            self.calls = []
            self.reference = TorchSurfelTracer()

        def trace(self, gaussian_set, rays, *, background):
            self.calls.append(gaussian_set)
            assert gaussian_set is interface, "inactive environment set was traced"
            return self.reference.trace(gaussian_set, rays, background=background)

    tracer = RecordingTracer()
    camera = GlintCamera(
        K=torch.tensor([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]),
        R=torch.eye(3), T=torch.zeros(3, 1), image_width=1, image_height=1,
    )
    # Even explicitly requested direct/secondary diagnostics cannot activate a
    # transmission path in a checkpoint saved during shared-only warm-up.
    output = render_glint_transport(
        camera, checkpoint, tracer,
        render_transmission_direct=True, render_secondary_reflection=True,
    )
    assert output.stage == "shared_reflection_only"
    assert raster_calls == [interface]
    assert tracer.calls == [interface]
    assert output.transmission_trace is None
    assert output.transmission_direct is None
    assert output.secondary_reflection_trace is None
    assert output.extra_reflection_trace is None
    assert output.shared_reflection_trace is not None
    assert torch.count_nonzero(output.trans_render) == 0
    assert torch.count_nonzero(output.extra_ref_render) == 0
    torch.testing.assert_close(output.ref_render, output.shared_ref_render)
    torch.testing.assert_close(output.energy_sum, torch.ones_like(output.energy_sum))
    # Retain material diagnostics even though this warm-up reallocates the
    # unavailable transmission weight to diffuse radiance.
    torch.testing.assert_close(output.trans_map, torch.full_like(output.trans_map, .7))
    torch.testing.assert_close(output.render, output.dif_render + output.ref_render)
    diffuse_weight = (1 - (.2 + .8 * .04)) * (.3 if preserve_material else 1.)
    torch.testing.assert_close(output.dif_render.flatten(), colors[0] * diffuse_weight)
    output.render.sum().backward()
    assert interface._features_dc.grad[0].abs().sum() > 0  # direct path
    assert interface._features_dc.grad[1].abs().sum() > 0  # reflected path
    assert interface._specular.grad.abs().sum() > 0
    if preserve_material:
        assert interface._transmission_coeff.grad.abs().sum() > 0
    else:
        assert interface._transmission_coeff.grad is None
    for gaussian_set in (transmission, extra):
        assert all(parameter.grad is None for parameter in gaussian_set.parameter_map().values())
