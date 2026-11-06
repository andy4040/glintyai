from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from glint.checkpoint import load_glint_checkpoint
from glint.dataset import load_easyvolcap_config
from glint.model import GlintStageSchedule
from glint.refinement import GlintRefiner, RefinementConfig
from glint.train import _load_resume_model
from glint.trainer import GlintTrainer, GlintTrainerConfig, GlintLossConfig, _make_optimizers
from glint.transport import compose_glint_radiance
from test_glint_training import _training_model


def schedule():
    return GlintStageSchedule(transmission_start=1000, shared_reflection_start=3000,
        transmission_pause_start=3000, transmission_resume=6000, separate_reflection_enabled=False)


def trainer_stub(model, output):
    trainer = object.__new__(GlintTrainer)
    trainer.model = model
    trainer.schedule = schedule()
    trainer.config = GlintTrainerConfig(max_steps=9000)
    trainer.device = torch.device('cpu')
    trainer.output_dir = output
    trainer.tracer = SimpleNamespace(_states={})
    trainer.optimizers = _make_optimizers(model, scene_scale=1.)
    trainer.refiners = {name: GlintRefiner(getattr(model,name), opts,
        RefinementConfig(reset_every=100000), scene_scale=1.) for name,opts in trainer.optimizers.items()}
    trainer._transmission_restart_applied = False
    trainer._transmission_initial_state = {f'initial.{k}':v.detach().clone()
        for k,v in model.transmission.state_dict().items()}
    return trainer


@pytest.mark.parametrize('step,stage,transmission', [
    (0,'interface',False),(999,'interface',False),(1000,'transmission',True),
    (2999,'transmission',True),(3000,'shared_reflection_only',False),
    (5999,'shared_reflection_only',False),(6000,'shared_reflection',True),
    (8999,'shared_reflection',True)])
def test_stages_and_saved_render_semantics(tmp_path,step,stage,transmission):
    curriculum = schedule()
    assert curriculum.stage(step) == stage
    assert curriculum.train_transmission(step) == transmission
    assert not curriculum.train_reflection(step)
    assert ('transmission' in GlintTrainer._active_sets(SimpleNamespace(schedule=curriculum),step)) == transmission
    model = _training_model()
    path = tmp_path/'state.pt'
    torch.save(dict(model=model.legacy_state_dict(),step=step,schedule=asdict(curriculum)),path)
    cp=load_glint_checkpoint(path)
    assert cp.render_stage == stage
    assert cp.preserve_transparency_without_transmission == (3000<=step<6000)
    assert model.as_checkpoint(step,schedule=curriculum).preserve_transparency_without_transmission == cp.preserve_transparency_without_transmission


def test_reset_discards_transmission_and_optimizer_but_keeps_interface_and_resume(tmp_path):
    trainer=trainer_stub(_training_model(),tmp_path)
    initial={k:v.clone() for k,v in trainer.model.transmission.state_dict().items()}
    # Simulate learned transmission including populated Adam momentum.
    trainer.model.transmission._features_dc.sum().backward()
    trainer.optimizers['transmission']['sh0'].step()
    interface={k:v.clone() for k,v in trainer.model.interface.state_dict().items()}
    interface_id=id(trainer.model.interface)
    before=trainer.save(2999,latest=False)
    resumed=trainer_stub(_load_resume_model(before,torch.device('cpu'),False),tmp_path)
    resumed.restore_training_state(before)
    assert resumed.step == 3000
    resumed._prepare_material_stage(3000)
    trainer._prepare_material_stage(3000)
    assert id(trainer.model.interface)==interface_id
    for k,v in interface.items(): torch.testing.assert_close(trainer.model.interface.state_dict()[k],v)
    for candidate in (trainer,resumed):
        for k,v in initial.items(): torch.testing.assert_close(candidate.model.transmission.state_dict()[k],v)
        assert all(not opt.state for opt in candidate.optimizers['transmission'].values())
        assert candidate.refiners['transmission'].grad_accum is None
        assert candidate.model.interface._transmission_coeff.requires_grad
        assert candidate.model.interface._specular.requires_grad
        assert candidate.model.interface._features_dc.requires_grad
    after=trainer.save(4000,latest=False)
    resumed=trainer_stub(_load_resume_model(after,torch.device('cpu'),False),tmp_path)
    resumed.restore_training_state(after)
    ident=id(resumed.model.transmission)
    resumed._prepare_material_stage(4001)
    assert id(resumed.model.transmission)==ident # No second reset on resume.
    assert resumed.model.interface._transmission_coeff.requires_grad
    assert resumed.model.interface._specular.requires_grad
    resumed._prepare_material_stage(6000)
    assert id(resumed.model.transmission)==ident
    assert resumed.model.interface._transmission_coeff.requires_grad
    assert resumed.model.interface._specular.requires_grad
    resumed.model.transmission._features_dc.sum().backward()
    resumed.optimizers['transmission']['sh0'].step()
    assert not torch.equal(resumed.model.transmission._features_dc,initial['_features_dc'])


@pytest.mark.parametrize('step',[3000,4001,5999])
def test_material_parameters_receive_rgb_gradients_and_update_during_pause(tmp_path,step):
    trainer=trainer_stub(_training_model(),tmp_path)
    interface=trainer.model.interface
    interface._transmission_coeff.requires_grad_(False)
    interface._specular.requires_grad_(False)
    before=[p.detach().clone() for p in (interface._transmission_coeff,interface._specular)]
    trainer._prepare_material_stage(step)
    assert trainer._active_sets(step)==('interface',)
    composed=compose_glint_radiance(
        interface_rgb=torch.full((1,1,3),.6),
        reflection_rgb=torch.full((1,1,3),.2),
        transmission_rgb=torch.zeros(1,1,3),
        view_directions=torch.tensor([[[0.,0.,1.]]]),
        normals=torch.tensor([[[0.,0.,-1.]]]),
        transparency=interface.get_transmission_coeff[:1].reshape(1,1,1),
        specularity=interface.get_specular[:1].reshape(1,1,1),
    )
    composed.rgb.square().mean().backward()
    for name,parameter,previous in zip(('transparency','specular'),
            (interface._transmission_coeff,interface._specular),before):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
        trainer.optimizers['interface'][name].step()
        assert not torch.equal(parameter,previous)
    assert all(p.grad is None for p in trainer.model.transmission.parameters())


def test_reset_transmission_training_clock(tmp_path):
    trainer=trainer_stub(_training_model(),tmp_path)
    trainer._prepare_material_stage(3000)
    trainer._update_sh_degree(5999)
    assert trainer.model.transmission.active_sh_degree.item()==0
    trainer._prepare_material_stage(6000)
    trainer._update_sh_degree(6000)
    trainer._update_learning_rates(6000)
    assert trainer.model.transmission.active_sh_degree.item()==1
    assert trainer.model.interface.active_sh_degree.item()==3
    assert trainer.optimizers['transmission']['means'].param_groups[0]['lr']==pytest.approx(1.6e-4*.01**(1000/30000))
    assert trainer.schedule.transmission_training_step(8999)==3999


def test_recipe_material_guidance_and_reset_protection():
    cfg=load_easyvolcap_config('configs/exps/glint/3d-front-t/scene_3_material_first.yaml')
    assert GlintLossConfig.from_easyvolcap(cfg).transparency_guidance_start==1000
    assert cfg['model_cfg']['sampler_cfg']['opacity_reset_interval']>9000


@pytest.mark.parametrize('kwargs',[
    dict(transmission_pause_start=3000),
    dict(transmission_resume=6000),
    dict(transmission_pause_start=3000,transmission_resume=2000),
    dict(transmission_pause_start=3000,transmission_resume=6000,shared_reflection_start=2000),
])
def test_invalid_partial_or_mismatched_pause(kwargs):
    with pytest.raises(ValueError):
        GlintStageSchedule(**kwargs)
