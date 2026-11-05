"""Shared-scene reflection curriculum and layered radiance regression tests."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from glint.checkpoint import GlintCheckpoint, load_glint_checkpoint
from glint.model import GlintStageSchedule
from glint.trainer import GlintTrainer
from glint.transport import RayBundle, TraceResult, trace_reflection_paths
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


@pytest.mark.parametrize("shared", [1, 6, 7])
def test_invalid_curriculum(shared):
    with pytest.raises(ValueError):
        GlintStageSchedule(transmission_start=2, shared_reflection_start=shared, reflection_start=6)


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
