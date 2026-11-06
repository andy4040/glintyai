# GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport

## Server-side checkpoint and viewer backup

`scripts/launch_archived_training.sh` launches the material-first experiment
with `nohup` on the GPU server. After successful training, it first uploads the
final checkpoint to a GitHub Release, then exports and uploads the interface /
transmission 3D viewer, actual shared-reflection ON/OFF renders, source snapshot,
configuration and SHA-256 manifest. Shared reflection reuses the interface set;
there is no third independent shared-reflection Gaussian set. The downloadable
viewer uses WebGL2 and a local HTTP server, not GitHub Pages hosting.

The GPU server needs a repository-scoped GitHub credential with Contents write
permission. Set `GH_TOKEN` in the server environment, or store it privately in
`~/.config/glint/github-token` (directory mode 700, file mode 600). Never commit
that file or paste its contents into logs. Verify server authentication first:

```bash
python -m glint.archive_run --check-auth
export DATA_ROOT=/root/glint-datasets/3D-FRONT-T/scene_3
export OUTPUT_DIR=/root/GLINT-material-opacity-test/results/new-experiment
export RELEASE_TAG=scene3-new-experiment
bash scripts/launch_archived_training.sh
```

Activate the environment containing the rebuilt OptiX extension before launch.
Use a new output directory and unique tag. The launch fails before training if
server credentials are missing; a laptop's Git login is not sufficient. The
laptop can disconnect after launch, but the GPU server must remain running.
Read `OUTPUT_DIR/server-job.log` for progress and
`OUTPUT_DIR/github-archive/uploaded.json` for the completed Release URL.

Uploads retry every minute for up to 24 hours, skip assets already verified by
size and GitHub SHA-256 digest, and never overwrite an existing completed asset.
Assets exceeding 1900 MiB are split; the manifest records their original names,
order and hashes for reassembly. Failed training is not published as a final
result. If viewer export fails, the checkpoint upload receipt remains in
`github-archive/checkpoint-uploaded.json`. Rerun the following command to resume
packaging/upload without training:

```bash
python -m glint.archive_run --run-dir "$OUTPUT_DIR" --tag "$RELEASE_TAG" \
  --config configs/exps/glint/3d-front-t/scene_3_material_first.yaml \
  --data-root "$DATA_ROOT"
```

Add `--prepare-only` to export/package locally without GitHub credentials or
network writes. Existing saved material/filter settings are respected during
rendering; a source snapshot documents the renderer used for the export.

[![arXiv](https://img.shields.io/badge/arXiv-2603.26181-b31b1b.svg)](https://arxiv.org/abs/2603.26181)
[![Project page](https://img.shields.io/badge/GLINT-Project%20Page-blue.svg)](https://youngju-na.github.io/GLINT)
[![Dataset](https://img.shields.io/badge/Dataset-Download-green.svg)](https://drive.google.com/drive/folders/1626l8RXKLpCqRhGLdGSYhcumm3HjOnYg?usp=sharing)

Research fork of **GLINT: Modeling Scene-Scale Transparency via
Gaussian Radiance Transport**, based on upstream commit
`a20591b20ef3365c230e3c95f749e65c498c68c4`.

This fork adds sequential shared-interface and separate-environment reflection,
an optional source-surface self-hit filter, and a 40,000-step 3D-FRONT-T schedule
with interface geometry freezing at 20,000 and LPIPS enabled after 30,000.
See [the shared-reflection training section](#optional-shared-interface-reflection-curriculum).
The original project attribution and documentation follow below.

Validation: 79 regression tests passed on RTX 3090, plus GPU stage-transition,
checkpoint resume/evaluation, and LPIPS forward/backward checks. These verify
execution, not final reconstruction quality. Source filtering is a geometric
heuristic; very close parallel surfaces remain ambiguous. Shared-reflection
hit gradients are not yet included in interface densification statistics, and
manual resume must repeat `--interface-geometry-freeze 20000` (the supplied
training script does so). Separate reflection uses background completion rather
than a depth-sorted union or recursive material transport.

[Youngju Na](https://youngju-na.github.io/)<sup>1,2,*</sup>,
[Jaeseong Yun](mailto:jaeseong.yun@naverlabs.com)<sup>2</sup>,
[Soohyun Ryu](mailto:soohyun.ryu@naverlabs.com)<sup>2</sup>,
[Hyunsu Kim](https://blandocs.github.io/)<sup>2</sup>,
[Sung-Eui Yoon](https://sgvr.kaist.ac.kr/~sungeui/)<sup>1</sup>,
[Suyong Yeon](mailto:suyong.yeon@naverlabs.com)<sup>2</sup>

_<sup>1</sup>KAIST, <sup>2</sup>NAVER LABS_

> [!NOTE]
> The `main` branch contains the new
> [gsplat](https://github.com/nerfstudio-project/gsplat)-based implementation.
> The original EasyVolCap-based release is preserved on the
> [`easyvolcap`](https://github.com/youngju-na/GLINT/tree/easyvolcap) branch.

## News

- **[2026-07-16]**: Released the
  [gsplat](https://github.com/nerfstudio-project/gsplat)-based implementation of
  GLINT. The original implementation remains available on the `easyvolcap`
  branch.
- **[2026-06-07]**: Updates with bug fixes and minor improvements are coming soon.
- **[2026-06-07]**: 🎉 Our paper has been selected as an Award Candidate!
- **[2026-04-09]**: 🎉 Our paper has been selected for an Oral presentation at CVPR 2026.
- **[2026-03-30]**: Initial code release.

## Installation

We tested the code on Linux with Python 3.11, PyTorch 2.9.1+cu126, the CUDA
12.9 toolkit, and an NVIDIA RTX 4090.

Clone this fork; its modified OptiX tracer and headers are included:

```bash
git clone https://github.com/andy4040/glintyai.git
cd glintyai
```

Create a separate environment for the gsplat implementation:

```bash
conda create -n splat python=3.11 cmake ninja -y
conda activate splat
```

Install PyTorch for your CUDA runtime. The tested combination is:

```bash
pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu126
```

Compiling gsplat and the OptiX tracer requires a CUDA toolkit with `nvcc` and
the CUDA development headers. Set `CUDA_HOME` before installing them. On
toolkits that place headers under `targets/x86_64-linux`, also expose that
directory to the host compiler:

```bash
export CUDA_HOME=/path/to/cuda
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
```

Install the GLINT-compatible gsplat commit separately. gsplat is an external
dependency rather than a copy embedded in this repository:

```bash
BUILD_2DGS=1 NUM_CHANNELS=3,4,6,7 BUILD_EXPERIMENTAL=0 \
  pip install --no-build-isolation \
  "git+https://github.com/youngju-na/gsplat.git@83a0dcd3f850cccdc7b92b50fd6a568e3260eae1"
```

GLINT uses gsplat's 2DGS rasterizer; the other gsplat kernels are not required.

Then install the remaining Python packages and GLINT:

```bash
pip install -r requirements.txt
pip install --no-deps -e .
```

Build the differentiable OptiX tracer using the same CUDA toolkit:

```bash
python -m glint.install_optix
```

Check the installation and run the tests:

```bash
python -m glint.verify_installation
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_glint_*.py
```

The PyTorch reference tracer is intended only for unit tests and tiny scenes.
Full-resolution training requires the OptiX backend.

## Data preparation

The Ref-DL3DV and 3D-FRONT-T datasets used by GLINT are available from the
[dataset download](https://drive.google.com/drive/folders/1626l8RXKLpCqRhGLdGSYhcumm3HjOnYg?usp=sharing).
Each extracted scene follows the released EasyVolCap-style layout:

```text
<scene>/
├── images/
├── intri.yml
├── extri.yml
├── sparse/0/points3D.ply
├── envs/points3D.ply
├── normals/                    # stable-normal priors, when used
└── diffrens/
    ├── normal/
    ├── depth/
    ├── diffuse_albedo/
    ├── basecolor/
    ├── roughness/
    └── metallic/
```

`diffrens` contains the G-buffer priors used for supervision. The downloadable
datasets already include the [DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer)
priors used in the paper, and the official configurations use
`diffrens/normal`. No additional normal estimator is required to reproduce the
released setup.

For custom data or alternative priors, normal maps can also be generated with
[StableNormal](https://github.com/Stable-X/StableNormal),
[TransNormal](https://github.com/longxiang-ai/TransNormal), or
[Metric3D](https://github.com/YvanYin/Metric3D), while
[UniRelight](https://github.com/nv-tlabs/UniRelight) provides intrinsic
decomposition and video relighting. Convert normal predictions to GLINT's
camera-space convention and store them under `normals/<camera>/<frame>` before
selecting `use_normal_type: stable`. The `normals` directory is optional;
missing stable-normal predictions are ignored by the data loader.

Official view splits and scene hyperparameters are provided under
[`configs/exps/glint`](configs/exps/glint). A local scene path can always be
provided with `--data-root`; no source or YAML edit is required.

## Training

### Optional shared-interface reflection curriculum

To test shared reflection before enabling transmission, with the separate
reflection set disabled throughout, run:

```bash
DATA_ROOT=/path/to/3d-front-t/scene_3 bash scripts/train_shared_reflection_first.sh scene_3
```

This experiment performs 6,000 updates: interface at steps 0–999,
interface plus shared reflection at 1,000–2,999, then interface plus shared
reflection plus transmission at 3,000–5,999. The shared-only stage does not
render or optimize transmission; its unavailable transmission weight stays
in the interface path. `--no-separate-reflection` disables separate tracing,
optimization and refinement even if training is later extended. The setting
and render stage survive checkpoint loading. Start from fresh initialization
or an interface-only checkpoint to avoid reusing already learned transmission.
Checkpoints are saved every 1,000 steps, at stage boundaries (999 and 2,999),
and after the final update (`step_005999.pt` and `latest.pt`).

Enable the new path with both `--shared-reflection-start 3000` and
`--reflection-start 6000`. Omitting `--shared-reflection-start` preserves the
original three-stage renderer and optimizer schedule.

```bash
python -m glint.train \
  --config configs/exps/glint/3d-front-t/scene_4.yaml \
  --data-root /path/to/3d-front-t/scene_4 \
  --output-dir results/shared-reflection/scene_4 \
  --max-steps 40000 \
  --interface-geometry-freeze 20000 \
  --transmission-start 1000 \
  --shared-reflection-start 3000 \
  --reflection-start 6000 \
  --source-surface-filter
```

The example uses interface-only training until step 1000, interface plus
transmission until step 3000, shared-interface reflection until step 6000,
then shared-interface reflection plus the separate reflection set. These
boundaries are configurable experimental defaults, not tuned results.
The shared-reflection experiment runs for 40,000 steps (0–39,999), freezing
interface geometry at step 20,000. Color/material parameters and the other
Gaussian sets continue training after that boundary. Other loss and refinement
schedules are unchanged except the 3D-FRONT-T perceptual-loss start, moved to
30,000. The existing strict `step > perceptual_start` condition activates LPIPS
at step 30,001 through 39,999; its weight remains 0.05. This is not a proportional
compression of all schedules.
Use `DATA_ROOT=/path/to/3d-front-t/scene_4 bash scripts/train_shared_reflection.sh`
to run this schedule; an optional first argument selects another scene.
The separate reflection set is allocated at initialization but is neither
traced nor optimizer-stepped/refined before its activation. The shared set
continues normal interface optimization; this option does not add a geometry
freeze or separate color head for the shared reflection path.

Both first-surface reflection and reflection from the transmitted surface use
the same policy. Rays first query the existing interface set with a black
background. After the separate path activates, reflection radiance is composed
as `shared.rgb + (1 - shared.alpha) * separate.rgb`, before applying GLINT's
existing reflection weights. This explicitly treats the separate set as a
background completion layer, **not** as a depth-sorted union of interleaved
surfaces. Shared opacity is coverage, not material transparency; recursive
transport through glass inside a reflected scene is not added here.

`--reflection-ray-offset` (default `1e-4`, in world units) offsets the ray origin
along the normalized reflected direction to reduce self-intersections. It may
need adjustment for different scene scales or inaccurate interface depths.
LPM, new point allocation rules, and opacity reset changes are not included.

Shared-scene reflection now uses material-aware opacity: each interface surfel
has effective opacity `opacity * (1 - transparency)` before its Gaussian
footprint, alpha clamp and front-to-back accumulation. This applies to both
primary and secondary shared reflection, on either side of a surfel. Gradients
flow through the product to both opacity and transparency. Direct camera,
transmission-set and separate-reflection rendering retain their original
opacity. This is a straight-through visibility approximation; it does not add
refraction or recursive Fresnel transport at the encountered transparent surface.

Source-surface filtering (`--source-surface-filter`) rejects shared-scene
self-hits using a detached per-ray candidate test in **both** OptiX forward and
backward traversal. Candidates must overlap the original camera pixel's ray
within the finite surfel footprint, exceed alpha 1/255 there, agree with the
source normal, and lie near the launch depth. Only intersections near the launch
position are rejected. Gaussians are never globally deleted; the separate
reflection set and transmission tracing are not filtered.

This computes geometric source-pixel candidates lazily at intersections rather
than exporting exact contributor IDs from gsplat. It is a local heuristic, not
semantic surface segmentation or exact visibility-weight provenance. Parallel
surfaces closer than the configured tolerance may still be ambiguous; large
depth errors can exceed the rejection band. Defaults are world-space
`--source-depth-tolerance 0.005`, `--source-max-distance 0.02`, and normal
agreement `--source-normal-cosine 0.9` (about 25.84 degrees). Adjust for scene scale. Invalid/zero
source normals cause no rejection. Use `--no-source-surface-filter` for an
ablation. These settings are saved in the checkpoint and restored on resume.
An older checkpoint retains its stored threshold; pass `--source-normal-cosine
0.9` on resume to override it. Source-pixel membership still uses the original
geometric opacity, independently of material-aware reflection opacity, so
transparent camera contributors are not accidentally removed from the filter.

The native OptiX extension **must be rebuilt** for material-aware reflection
with source filtering (the new `source_opacities` input). Build and run
in a separate virtual environment when another training process is using the
old extension; do not overwrite its binaries/PTX in place. The CPU reference
tracer implements the same filter for regression tests. Supported native
settings are the GLINT default `sigma_extent=3` and explicit single-bounce
tracing (`max_trace_depth=0`).

Schedules and the offset are saved in checkpoints. Resume inherits the saved
schedule unless explicitly overridden by CLI flags. Evaluation and
`glint.render_transport` automatically use the saved stage for this optional
mode, so a shared-only checkpoint does not render the untrained separate set.
`glint.render_transport --components-dir PATH` also writes
`shared_reflection_weighted.png` and `extra_reflection_weighted.png` for the
primary reflection path. Their sum is `reflection_weighted.png`; the existing
secondary reflection output remains separate.

For a checkpoint with separate reflection disabled, export an actual-render
shared-reflection ON/OFF viewer (serve the output directory over HTTP):

```bash
python -m glint.compare_shared_reflection \
  --config configs/exps/glint/3d-front-t/scene_3.yaml \
  --data-root /path/to/3d-front-t/scene_3 \
  --checkpoints results/shared-reflection-first/scene_3/checkpoints/step_004000.pt \
  --output-dir results/reflection-comparison --view-indices 0 4 8 12 16 20
```

Pass multiple checkpoints to compare stages. OFF removes both primary and
secondary shared-reflection radiance while keeping geometry, material weights,
interface and transmission contributions fixed. It does not retrain or
renormalize the remaining paths. The viewer defaults to the latest checkpoint
and original brightness; its optional gain affects only the reflection-component
preview. Step 3,000 checkpoints may be dark immediately after opacity reset;
the stage-boundary checkpoint at 2,999 records the state before that reset.

Activate the `splat` environment and run commands from the repository root.

### Ref-DL3DV

```bash
SCENE=6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f

python -m glint.train \
  --config configs/exps/glint/ref-dl3dv/${SCENE}.yaml \
  --data-root /path/to/ref-dl3dv/${SCENE} \
  --output-dir results/glint/ref-dl3dv/${SCENE} \
  --max-steps 60000 \
  --interface-geometry-freeze 31000
```

The [`train_ref_dl3dv_6b42.sh`](scripts/train_ref_dl3dv_6b42.sh)
script provides a short validation run, training, resume, and evaluation
commands:

```bash
DATA_ROOT=/path/to/ref-dl3dv/${SCENE} \
  bash scripts/train_ref_dl3dv_6b42.sh quick

DATA_ROOT=/path/to/ref-dl3dv/${SCENE} \
  bash scripts/train_ref_dl3dv_6b42.sh train
```

### 3D-FRONT-T

```bash
SCENE=scene_4

python -m glint.train \
  --config configs/exps/glint/3d-front-t/${SCENE}.yaml \
  --data-root /path/to/3d-front-t/${SCENE} \
  --output-dir results/glint/3d-front-t/${SCENE} \
  --max-steps 60000 \
  --interface-geometry-freeze 31000
```

The ten-scene sequential benchmark wrapper accepts dataset roots through
environment variables:

```bash
REF_DL3DV_ROOT=/path/to/ref-dl3dv \
SYNTHETIC_ROOT=/path/to/3d-front-t \
  bash scripts/train_benchmark_10scenes.sh all
```

## Evaluation and rendering

Evaluate the official validation split and save component visualizations:

```bash
python -m glint.evaluate_dataset \
  --config configs/exps/glint/ref-dl3dv/${SCENE}.yaml \
  --data-root /path/to/ref-dl3dv/${SCENE} \
  --checkpoint results/glint/ref-dl3dv/${SCENE}/checkpoints/latest.pt \
  --output-dir results/glint/ref-dl3dv/${SCENE}/eval \
  --save-images \
  --save-visualizations
```

### Geometry evaluation

The 3D-FRONT-T geometry benchmark reports normal MAE and angular accuracy,
depth AbsRel, normalized RMSE and δ < 1.25, and mesh Chamfer distance and F1.
First render the validation views with raw geometry output enabled:

```bash
SCENE=scene_4

python -m glint.evaluate_dataset \
  --config configs/exps/glint/3d-front-t/${SCENE}.yaml \
  --data-root /path/to/3d-front-t/${SCENE} \
  --checkpoint results/glint/3d-front-t/${SCENE}/checkpoints/latest.pt \
  --output-dir results/glint/3d-front-t/${SCENE}/eval \
  --save-geometry
```

Evaluate the saved z-depth and camera-space normal maps against the geometry
ground truth:

```bash
python -m glint.evaluate_geometry maps \
  --prediction-root results/glint/3d-front-t/${SCENE}/eval/geometry \
  --ground-truth-root /path/to/geometry_gt/${SCENE} \
  --output results/glint/3d-front-t/${SCENE}/eval/geometry_metrics.json
```

The evaluator applies the paper protocol: the released geometry view split,
per-view median depth alignment, macro-averaging over validation views, and
RMSE normalization by mean valid GT depth. The JSON report also includes RMSE
in the input depth units. Use `--all-views` only for a custom split.

After evaluating all five scenes, reproduce the table-level macro-average with:

```bash
python -m glint.evaluate_geometry summary \
  --reports results/glint/3d-front-t/scene_{1,2,3,4,5}/eval/geometry_metrics.json \
  --output results/glint/3d-front-t/geometry_metrics.json
```

For mesh evaluation, install the optional CPU dependencies and compare the
post-processed interface mesh produced by TSDF fusion:

```bash
pip install -r requirements-geometry.txt

python -m glint.evaluate_geometry mesh \
  --prediction /path/to/tsdf_fusion_interface_post.ply \
  --ground-truth /path/to/geometry_gt/${SCENE}/gt_mesh.ply \
  --output results/glint/3d-front-t/${SCENE}/eval/mesh_metrics.json
```

Chamfer distance is computed from mesh surfaces sampled at 1.5 cm spacing. The
report contains meters and decimeters; Table 1 uses decimeters. F1 uses a 50mm
distance threshold. The geometry ground-truth package is expected in this
layout:

```text
geometry_gt/<scene>/
├── depths_gt/val_depthZ_XXXX.npy
├── normals_gt/val_normalCam_XXXX.npy
└── gt_mesh.ply
```

Training writes the following under `--output-dir`:

```text
<output>/
├── checkpoints/
│   ├── latest.pt
│   └── step_*.pt
├── train.log
└── train_visualizations/
    ├── PANELS/
    ├── RENDER/
    ├── DEPTH/
    ├── NORMAL/
    ├── TRANSPARENCY/
    └── ...
```

The visualization includes direct and transported RGB, depth, normal, alpha,
specularity, conditional material transparency, the composited transparency
gate, diffuse/transmission/reflection contributions, guidance masks, and
combined visualization panels.

## Material-first shared-reflection experiment (scene 3)

```bash
DATA_ROOT=/path/to/3D-FRONT-T/scene_3 bash scripts/train_material_first.sh
```

This opt-in experiment runs 9,000 updates: interface at `[0,1000)`, interface
and transmission at `[1000,3000)`, interface and shared reflection at
`[3000,6000)`, then all three paths at `[6000,9000)`. Separate reflection is
disabled throughout. At step 3000, the entire interface is retained; only
transmission is restored to its initial SfM state with fresh Adam and
refinement state. The original learned transmission remains in the step 2999
checkpoint. Resuming a new experiment checkpoint preserves this one-time reset.

During the middle phase, interface transparency and specularity parameters
continue training along with interface geometry and SH color. The pretrained
material values are retained as initialization, not frozen targets. Learned
transparency remains in the composition; transmission is neither rendered nor
optimized. Thus missing transmitted radiance is deliberately **not** reassigned
to interface diffuse radiance. This is an experimental ablation, not a claim
that full-image RGB supervision isolates reflection. At step 6000, the fresh
transmission set joins the ongoing interface/material training; its LR, SH, and
refinement clocks restart from the initial transmission training age.

The recipe starts material guidance at step 1000 and moves interface opacity/
transparency resets beyond the run so they cannot erase the warm-up material.
Other losses retain their scene configuration (including transmission's direct
RGB supervision and LPIPS starting at 30000, outside this short experiment).
Stage-boundary and final checkpoints are `step_000999.pt`, `step_002999.pt`,
`step_005999.pt`, and `step_008999.pt`; regular saves also occur every 1000 steps.

## Citation

```bibtex
@inproceedings{na2026glint,
  title     = {{GLINT}: Modeling Scene-Scale Transparency via Gaussian Radiance Transport},
  author    = {Na, Youngju and Yun, Jaeseong and Ryu, Soohyun and Kim, Hyunsu and Yoon, Sung-Eui and Yeon, Suyong},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

This implementation is built on [gsplat](https://github.com/nerfstudio-project/gsplat).
Please also cite gsplat when this backend is used; its citation is included in
[`CITATION.bib`](CITATION.bib).

## License and acknowledgements

GLINT is released under the MIT license in [`LICENSE`](LICENSE). gsplat is an
external dependency distributed under Apache-2.0, and recursive third-party
submodules retain their own licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.

We thank the authors and maintainers of gsplat, EasyVolCap, 2D Gaussian
Splatting, and the differentiable surfel tracing implementation on which this
release builds.
