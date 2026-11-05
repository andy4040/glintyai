"""Source-local rejection must remove self-hits, not nearby real objects."""
import math
import pytest
import torch
from dataclasses import asdict

from glint.checkpoint import GlintCheckpoint, load_glint_checkpoint
from glint.camera import GlintCamera
from glint.model import GlintStageSchedule
from glint.optix_backend import OptixSurfelTracer
from glint.torch_tracer import TorchSurfelTracer
from glint.transport import RayBundle, make_source_surface_filter, trace_reflection_paths, render_glint_transport
from test_glint_transport import _gaussian_set


@pytest.fixture(params=['torch', 'optix'])
def backend(request):
    if request.param == 'optix':
        if not torch.cuda.is_available():
            pytest.skip('CUDA required')
        return OptixSurfelTracer(), torch.device('cuda')
    return TorchSurfelTracer(), torch.device('cpu')


def setup_scene(device, positions, *, direction=(0,0,-1), launch=(0,0,2.0033345), scale=(1,1)):
    g=_gaussian_set(positions, color=(1,0,0), opacity=.8, scale=scale, role='interface',device=device)
    cp=GlintCheckpoint(g,None,None,torch.zeros(3,device=device),torch.zeros(3,device=device),
                       torch.zeros(3,device=device),0,source_surface_filter=True)
    o=torch.tensor(launch,device=device,dtype=torch.float32).reshape(1,1,3)
    d=torch.tensor(direction,device=device,dtype=torch.float32).reshape(1,1,3)
    incoming=RayBundle(torch.zeros_like(o),torch.tensor([[[0.,0.,1.]]],device=device))
    rays=RayBundle(o,d)
    f=make_source_surface_filter(incoming,rays,torch.tensor([[[0.,0.,-1.]]],device=device),cp)
    return g,cp,rays,f


def test_multiple_self_surfels_removed_and_real_object_gradient_preserved(backend):
    tracer,device=backend
    g,cp,rays,f=setup_scene(device,[[0,0,2],[0,0,2.001],[0,0,-2]])
    with torch.no_grad():
        g._features_dc[-1,0]=torch.tensor([-.5,-.5,.5],device=device)/.28209479177387814
    old,_,_=trace_reflection_paths(cp,tracer,rays,shared=True,separate=False)
    result,_,_=trace_reflection_paths(cp,tracer,rays,shared=True,separate=False,source_surface=f)
    assert old.rgb[0,0,0]>.9  # reproduces source contamination
    torch.testing.assert_close(result.rgb,torch.tensor([[[0.,0.,.8]]],device=device),atol=2e-5,rtol=0)
    result.rgb.sum().backward()
    torch.testing.assert_close(g._features_dc.grad[:2],torch.zeros_like(g._features_dc.grad[:2]),atol=1e-7,rtol=0)
    assert g._features_dc.grad[-1].abs().sum()>0
    assert not f.requires_grad


def test_all_source_hits_give_background_and_no_nan(backend):
    tracer,device=backend
    g,cp,rays,f=setup_scene(device,[[0,0,2],[0,0,2.001]])
    result,_,_=trace_reflection_paths(cp,tracer,rays,shared=True,separate=False,source_surface=f)
    torch.testing.assert_close(result.alpha,torch.zeros_like(result.alpha),atol=1e-6,rtol=0)
    assert torch.isfinite(result.rgb).all()
    result.rgb.sum().backward()
    assert torch.isfinite(g._features_dc.grad).all()


def test_close_parallel_object_outside_source_depth_band_is_preserved(backend):
    tracer,device=backend
    _,cp,rays,f=setup_scene(device,[[0,0,1.99]])
    result,_,_=trace_reflection_paths(cp,tracer,rays,shared=True,separate=False,source_surface=f)
    torch.testing.assert_close(result.alpha,torch.full_like(result.alpha,.8),atol=2e-5,rtol=0)


def test_nearby_different_normal_is_preserved(backend):
    tracer,device=backend
    g,cp,rays,f=setup_scene(device,[[0,0,2]])
    with torch.no_grad():
        g._rotation[0]=torch.tensor([math.cos(math.pi/8),0,math.sin(math.pi/8),0],device=device)
    result,_,_=trace_reflection_paths(cp,tracer,rays,shared=True,separate=False,source_surface=f)
    assert result.alpha.item()>.79


def test_noncontributing_source_pixel_candidate_is_preserved(backend):
    tracer,device=backend
    _,cp,rays,f=setup_scene(device,[[.01,0,2]],direction=(1,0,-.1),launch=(0,0,2.001),scale=(.001,.001))
    result,_,_=trace_reflection_paths(cp,tracer,rays,shared=True,separate=False,source_surface=f)
    assert result.alpha.item()>.79


def test_same_surfel_at_distant_hit_is_not_globally_deleted(backend):
    tracer,device=backend
    _,cp,rays,f=setup_scene(device,[[0,0,2]],direction=(1,0,-.01))
    result,_,_=trace_reflection_paths(cp,tracer,rays,shared=True,separate=False,source_surface=f)
    assert result.alpha.item()>.7


@pytest.mark.parametrize('kwargs',[
    {'source_depth_tolerance':0}, {'source_max_distance':float('nan')},
    {'source_normal_cosine':1.1}, {'source_surface_filter':True},
])
def test_invalid_source_settings(kwargs):
    with pytest.raises(ValueError):
        GlintStageSchedule(**kwargs)


@pytest.mark.parametrize('kind', ['torch', 'optix'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='gsplat requires CUDA')
def test_actual_expected_depth_launch_self_hit_regression(kind):
    device=torch.device('cuda')
    pcd=_gaussian_set([[0,0,2],[0,0,2.02]],color=(1,0,0),scale=(10,10),opacity=.8,role='interface',device=device)
    trans=_gaussian_set([[0,0,4]],role='transmission',device=device)
    zero=torch.zeros(3,device=device)
    cp=GlintCheckpoint(pcd,None,trans,zero,zero,zero,0,shared_reflection=True,separate_reflection_enabled=False)
    camera=GlintCamera(K=torch.tensor([[10.,0,4],[0,10.,4],[0,0,1]],device=device),
                       R=torch.eye(3,device=device),T=torch.zeros(3,1,device=device),image_width=8,image_height=8)
    tracer=OptixSurfelTracer() if kind=='optix' else TorchSurfelTracer()
    before=render_glint_transport(camera,cp,tracer,render_secondary_reflection=False)
    assert 2.0 < before.interface.surf_depth[0,4,4] < 2.02
    assert before.shared_reflection_trace.alpha[4,4]>.79
    cp.source_surface_filter=True
    after=render_glint_transport(camera,cp,tracer,render_secondary_reflection=False)
    assert after.shared_reflection_trace.alpha[4,4]<1e-6
    after.render.sum().backward()
    assert torch.isfinite(pcd._features_dc.grad).all()


def test_source_filter_checkpoint_roundtrip(tmp_path):
    from test_glint_training import _training_model
    m=_training_model()
    s=GlintStageSchedule(shared_reflection_start=1000,reflection_start=3000,
                        source_surface_filter=True,source_depth_tolerance=.003,
                        source_max_distance=.01,source_normal_cosine=.98)
    path=tmp_path/'filter.pt'
    torch.save({'model':m.legacy_state_dict(),'step':2000,'schedule':asdict(s)},path)
    cp=load_glint_checkpoint(path)
    assert cp.source_surface_filter and cp.source_depth_tolerance==.003
    assert cp.source_max_distance==.01 and cp.source_normal_cosine==.98
