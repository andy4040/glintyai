"""Export actual, fixed-weight shared-reflection ON/OFF renders and a web viewer.

For checkpoints with separate reflection disabled. OFF removes both primary
and secondary shared-reflection contributions without changing any weights,
geometry, or transmission radiance. It is an ablation, not a retrained model.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from glint.checkpoint import load_glint_checkpoint
from glint.dataset import GlintDataset
from glint.evaluate_dataset import _write_image
from glint.optix_backend import OptixSurfelTracer
from glint.transport import render_glint_transport


PAGE = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLINT 공유 반사 ON / OFF</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101820;color:#e8eef5;font:16px system-ui,sans-serif}
main{max-width:1300px;margin:auto;padding:24px}h1{margin:0 0 12px;font-size:27px}
p{line-height:1.65;color:#b9c9da}.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:16px;background:#202e3d;padding:16px;border-radius:10px}
select,button{background:#30465c;color:white;padding:8px;border:1px solid #668097;border-radius:5px;font:inherit}
button{cursor:pointer}input{accent-color:#58d6c2;transform:scale(1.4);margin-right:10px}
.toggle{padding:10px 16px;background:#124a46;border-radius:6px;font-weight:700}
figure{margin:18px 0}img{display:block;width:100%;background:#000;border-radius:6px}
figcaption{padding:10px 0;color:#c2d2e2}.sub{display:grid;grid-template-columns:1fr 1fr;gap:20px}
#state{font-weight:700;color:#6fedd4}a{color:#8edcff}#error{color:#ff9797}
@media(max-width:650px){.sub{grid-template-columns:1fr}main{padding:12px}}
</style><main>
<h1>공유 반사 ON / OFF</h1>
<p>동일한 체크포인트·카메라에서 공유 반사 기여분만 비교합니다. OFF는 재학습하거나 다른 경로의 비중을 늘리지 않습니다. 별도 반사는 비활성화 상태입니다.</p>
<div class="toolbar">
<label>체크포인트 <select id="step"></select></label>
<button id="prev" aria-label="이전 시점">◀</button>
<label>시점 <select id="camera"></select></label>
<button id="next" aria-label="다음 시점">▶</button>
<label class="toggle"><input type="checkbox" id="reflection" checked>공유 반사</label>
<span id="state"></span></div>
<p id="notice"></p><p id="error" role="alert"></p>
<figure><img id="main" alt="공유 반사 ON/OFF 합성 렌더링"><figcaption id="caption"></figcaption></figure>
<div class="sub"><figure><img id="gt" alt="정답 영상"><figcaption>정답 이미지</figcaption></figure>
<figure><img id="shared" alt="최종 영상에 더해지는 공유 반사 기여분"><figcaption>공유 반사 기여분 (1차 + 2차) · 표시 밝기
<select id="gain"><option value="1">1배 · 원래 밝기</option><option value="4">4배 · 확인용</option><option value="10">10배 · 확인용</option></select>
<br>밝기 증폭은 이 성분 이미지에만 적용됩니다. 위 ON/OFF 영상은 보정하지 않습니다.</figcaption></figure></div>
<p>3,000스텝 저장본은 정기 opacity 초기화 직후라 어둡습니다. 2,999스텝은 초기화 직전이며 아직 투과 경로가 꺼져 있습니다. 4,000스텝에는 투과도 학습 중입니다.</p>
<p><a href="manifest.json">렌더링 정보</a></p></main>
<script>
'use strict';
const $=id=>document.getElementById(id);
let runs=[];
function active(){return runs[Number($('step').value)];}
function update(){
 const run=active(),view=run.views[Number($('camera').value)],on=$('reflection').checked;
 const base=`${run.step}/${view.camera}/`;
 $('main').src=base+(on?'on.png':'off.png');$('gt').src=base+'gt.png';$('shared').src=base+'shared.png';
 $('state').textContent=on?'ON':'OFF';
 $('caption').textContent=`${run.step.toLocaleString()}스텝 · 시점 ${view.camera} · 공유 반사 ${on?'ON':'OFF'}`;
 $('notice').textContent=run.step===3000?'주의: 정기 opacity 초기화 직후 저장된 체크포인트입니다.':
  run.step<3000?'인터페이스 + 공유 반사 단계 · 투과 비활성':'인터페이스 + 공유 반사 + 투과 단계';
}
function selectRun(){
 const old=$('camera').selectedOptions[0]?.textContent;
 $('camera').replaceChildren();active().views.forEach((v,i)=>$('camera').add(new Option(v.camera,i)));
 const i=active().views.findIndex(v=>v.camera===old);if(i>=0)$('camera').value=String(i);update();
}
$('reflection').onchange=update;$('camera').onchange=update;$('step').onchange=selectRun;
function advance(delta){const n=active().views.length;$('camera').value=String((Number($('camera').value)+delta+n)%n);update();}
$('prev').onclick=()=>advance(-1);$('next').onclick=()=>advance(1);
$('gain').onchange=()=>{$('shared').style.filter=`brightness(${$('gain').value})`;};
$('main').onerror=()=>{$('error').textContent='이미지를 불러오지 못했습니다. 새로고침하거나 파일 저장 상태를 확인하세요.';};
fetch('manifest.json').then(r=>{if(!r.ok)throw Error('manifest HTTP '+r.status);return r.json();}).then(m=>{
 runs=m.runs.sort((a,b)=>b.step-a.step);if(!runs.length)throw Error('렌더링 결과 없음');
 runs.forEach((r,i)=>$('step').add(new Option(r.step.toLocaleString()+'스텝',i)));selectRun();
}).catch(e=>{$('error').textContent=String(e);});
</script></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--view-indices', type=int, nargs='+')
    args = parser.parse_args()
    dataset = GlintDataset.from_config(
        args.config, split='val', data_root=args.data_root, ratio=args.ratio,
        load_neighbors=False, load_sky_mask=False, load_stable_normal=False,
        diffren_params=(),
    )
    indices = list(range(len(dataset))) if args.view_indices is None else args.view_indices
    if not indices or any(i < 0 or i >= len(dataset) for i in indices):
        parser.error('view-indices must refer to existing validation views')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'mode': 'fixed_weight_ablation', 'secondary_shared_included': True, 'runs': []}
    for path in args.checkpoints:
        checkpoint = load_glint_checkpoint(path, device='cuda')
        if checkpoint.separate_reflection_enabled or not checkpoint.shared_reflection:
            raise ValueError('Requires shared reflection active and separate reflection disabled')
        step = int(checkpoint.epoch)
        tracer = OptixSurfelTracer().eval()
        run = {'step': step, 'checkpoint': str(path), 'stage': checkpoint.render_stage, 'views': []}
        with torch.inference_mode():
            for i in indices:
                sample = dataset[i].to('cuda')
                output = render_glint_transport(sample.camera, checkpoint, tracer)
                assert output.extra_reflection_trace is None
                assert output.secondary_extra_reflection_trace is None
                # Material weights and all non-reflection paths stay identical.
                off = output.dif_render + output.trans_render
                shared = output.shared_ref_render + output.secondary_ref_render
                torch.testing.assert_close(output.render, off + shared, rtol=1e-5, atol=1e-6)
                assert torch.isfinite(output.render).all() and torch.isfinite(off).all()
                destination = args.output_dir / str(step) / sample.camera_name
                for name, value in [('on', output.render), ('off', off), ('shared', shared)]:
                    _write_image(destination / (name + '.png'), value.permute(1, 2, 0))
                _write_image(destination / 'gt.png', sample.rgb)
                run['views'].append({'camera': sample.camera_name, 'index': i,
                                     'mean_shared_contribution': float(shared.mean())})
                print(f'step={step} camera={sample.camera_name} ON/OFF verified', flush=True)
                del output, off, shared, sample, value
        manifest['runs'].append(run)
        del checkpoint, tracer
        gc.collect()
        torch.cuda.empty_cache()
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (args.output_dir / 'index.html').write_text(PAGE, encoding='utf-8')
    print(f'Viewer ready: {args.output_dir / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
