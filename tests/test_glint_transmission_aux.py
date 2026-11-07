"""Disabling standalone RGB must preserve joint transport and geometry gradients."""
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from glint.dataset import load_easyvolcap_config
from glint.model import GlintStageSchedule
from glint.trainer import GlintLossConfig, GlintTrainer

@pytest.mark.parametrize('aux_weight', [0., 1.])
def test_joint_and_direct_geometry_gradients_survive(aux_weight):
    t=torch.full((3,16,16),.3,requires_grad=True)
    r=torch.full((3,16,16),.6,requires_grad=True)
    normal=torch.tensor([.3,0.,1.]).reshape(3,1,1).expand(3,16,16).clone().requires_grad_()
    surface=torch.tensor([0.,0.,1.]).reshape(3,1,1).expand(3,16,16)
    direct=SimpleNamespace(render=t,rend_normal=normal,surf_normal=surface)
    output=SimpleNamespace(render=.96*t+.04*r,transmission_direct=direct,
        norm_map=surface.permute(1,2,0),trans_map=torch.ones(1,256,1)*.9,
        interface=SimpleNamespace(rend_normal=surface,surf_normal=surface))
    sample=SimpleNamespace(rgb=torch.ones(16,16,3)*.8,
        camera=SimpleNamespace(image_height=16,image_width=16,R=torch.eye(3)),
        stable_normal=None,normal=None,depth=None)
    config=GlintLossConfig(transmission_rgb_aux_weight=aux_weight,
        normal_consistency_depth_weight=False)
    trainer=SimpleNamespace(loss_config=config,_multi_view_terms=lambda *args:None)
    loss,terms=GlintTrainer._loss(trainer,output,sample,0)
    assert ('transmission_rgb_l1' in terms)==(aux_weight>0)
    assert ('transmission_rgb_ssim' in terms)==(aux_weight>0)
    assert 'rgb_l1' in terms and 'transmission_normal_consistency' in terms
    loss.backward()
    for value in [t,r,normal]:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all() and value.grad.abs().sum()>0

def test_aux_config_defaults_and_validation():
    assert GlintLossConfig().transmission_rgb_aux_weight==1.
    for value in [-1.,float('inf'),float('nan')]:
        with pytest.raises(ValueError):GlintLossConfig(transmission_rgb_aux_weight=value)

def test_experiment_doubles_material_first_boundaries():
    root=Path(__file__).resolve().parents[1]
    cfg=load_easyvolcap_config(root/'configs/exps/glint/3d-front-t/scene_3_no_trans_aux_2x.yaml')
    s=cfg['model_cfg']['sampler_cfg'];loss=GlintLossConfig.from_easyvolcap(cfg)
    schedule=GlintStageSchedule(transmission_start=s['render_transmission_start_iter'],
        shared_reflection_start=s['shared_reflection_start_iter'],
        transmission_pause_start=s['transmission_pause_start'],
        transmission_resume=s['transmission_resume'],separate_reflection_enabled=False)
    assert [schedule.stage(i) for i in [1999,2000,5999,6000,11999,12000,17999]]==[
        'interface','transmission','transmission','shared_reflection_only',
        'shared_reflection_only','shared_reflection','shared_reflection']
    assert schedule.transmission_training_step(12000)==2000
    assert loss.transmission_rgb_aux_weight==0
    assert loss.rgb_l1==1.2 and loss.rgb_ssim==.4
    assert loss.transparency_guidance_start==2000
    assert loss.perceptual_start==60000 and loss.multi_view_start==20000
    assert cfg['runner_cfg']['epochs']*cfg['runner_cfg']['ep_iter']==18000
