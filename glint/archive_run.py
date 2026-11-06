"""Server-side final checkpoint/viewer packaging and resumable Release upload.

Run in a detached server process; no laptop connection is needed after launch.
Credentials are read at runtime and never included in the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def token():
    value = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if not value:
        path = Path(os.environ.get('GLINT_GITHUB_TOKEN_FILE', '~/.config/glint/github-token')).expanduser()
        if not path.is_file():
            raise RuntimeError('Server GitHub authentication missing: configure GH_TOKEN or ~/.config/glint/github-token (Contents: write for the target repository).')
        if os.name == 'posix' and path.stat().st_mode & 0o077:
            raise RuntimeError('GitHub token file must be private: chmod 600')
        value = path.read_text().strip()
    return value.strip()


class GitHub:
    def __init__(self, repo):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
            raise ValueError('Expected owner/repository')
        self.repo = repo
        self.headers = {'Authorization': 'Bearer ' + token(), 'Accept': 'application/vnd.github+json',
                        'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'GLINT-result-archive'}

    def request(self, method, suffix, data=None):
        body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request('https://api.github.com/repos/' + self.repo + suffix,
            data=body, headers={**self.headers, 'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            return json.loads(payload) if payload else None

    def check_auth(self):
        result = self.request('GET', '')
        if not result.get('permissions', {}).get('push'):
            raise RuntimeError('Server GitHub identity lacks repository write access')
        return {'repository': result['full_name'], 'private': result['private'], 'write_access': True}

    def release(self, tag):
        try:
            return self.request('GET', '/releases/tags/' + urllib.parse.quote(tag, safe=''))
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        return self.request('POST', '/releases', {'tag_name': tag, 'name': tag,
            'make_latest': 'false', 'body': 'GLINT completed-run archive. Checkpoint, downloadable 3D viewer, shared-reflection ON/OFF renders, source snapshot and SHA-256 manifest. Shared reflection reuses the interface set; it is not a separate Gaussian set.'})

    def upload(self, release, path):
        digest = sha256(path)
        name = path.stem + '-' + digest[:16] + path.suffix
        assets = []
        page = 1
        while True:
            batch = self.request('GET', f'/releases/{release["id"]}/assets?per_page=100&page={page}')
            assets.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        for asset in assets:
            if asset['name'] == name:
                if asset.get('state') == 'uploaded' and asset['size'] == path.stat().st_size and asset.get('digest') == 'sha256:' + digest:
                    return asset
                if asset.get('state') == 'starter' and asset['size'] == 0:
                    # GitHub may leave an empty failed-upload placeholder.
                    self.request('DELETE', f'/releases/assets/{asset["id"]}')
                    break
                raise RuntimeError('Existing asset has unverified content; retained without overwriting: ' + name)
        base = release['upload_url'].split('{')[0]
        parsed = urllib.parse.urlsplit(base)
        if parsed.scheme != 'https' or parsed.hostname != 'uploads.github.com':
            raise RuntimeError('Unexpected GitHub upload endpoint')
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=300)
        try:
            connection.putrequest('POST', parsed.path + '?name=' + urllib.parse.quote(name))
            for key, value in self.headers.items():
                connection.putheader(key, value)
            connection.putheader('Content-Type', 'application/octet-stream')
            connection.putheader('Content-Length', str(path.stat().st_size))
            connection.endheaders()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    connection.send(chunk)
            response = connection.getresponse()
            payload = response.read()
            if response.status != 201:
                raise RuntimeError(f'GitHub upload HTTP {response.status}')
            asset = json.loads(payload)
        finally:
            connection.close()
        if asset.get('digest') != 'sha256:' + digest or asset['size'] != path.stat().st_size:
            raise RuntimeError('GitHub asset digest/size verification failed')
        return asset


def export_geometry(checkpoint, config, data_root, output):
    import numpy as np
    import torch
    from .checkpoint import load_glint_checkpoint
    from .dataset import GlintDataset
    from .geometry import quaternion_to_rotation_matrix
    cp = load_glint_checkpoint(checkpoint, device='cpu', gaussian_sets=('pcd', 'trans_env'))
    output.mkdir(parents=True, exist_ok=True)
    counts = {'scene': 0, 'optical': 0, 'unresolved': 0}
    with torch.no_grad(), (output / 'gaussians.bin').open('wb') as stream:
        for role, g in enumerate((cp.pcd, cp.trans_env)):
            if g is None:
                continue
            rotation = quaternion_to_rotation_matrix(g.get_rotation)
            xyz = g.get_xyz.numpy()
            u = (rotation[:, :, 0] * g.get_scaling[:, :1]).numpy()
            v = (rotation[:, :, 1] * g.get_scaling[:, 1:2]).numpy()
            rgb = np.clip(.5 + .28209479177387814 * g.get_features[:, 0].numpy(), 0, 1)
            alpha = g.get_opacity.numpy()
            block = np.concatenate((xyz, u, v, rgb, alpha, np.full_like(alpha, role), np.zeros_like(alpha), np.zeros_like(alpha)), axis=1).astype('<f4')
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite Gaussian viewer data')
            block.tofile(stream)
            counts[['scene', 'optical'][role]] = len(xyz)
        bounds = np.quantile(cp.pcd.get_xyz.numpy(), [.01, .99], axis=0)
    center = bounds.mean(0)
    dataset = GlintDataset.from_config(config, split='val', data_root=data_root, ratio=.5,
        load_neighbors=False, load_stable_normal=False, load_sky_mask=False, diffren_params=())
    cameras = {}
    for sample in dataset:
        c = sample.camera
        eye = c.camera_center.numpy().reshape(3)
        cameras[sample.camera_name] = dict(eye=eye.tolist(), up=(-c.R[1]).tolist(), forward=c.R[2].tolist(),
            distance=float(np.linalg.norm(center-eye)), tan_fov=float(c.image_height/(2*c.K[1,1])))
    meta = dict(step=int(cp.epoch), count=sum(counts.values()), counts=counts, bounds=bounds.tolist(), cameras=cameras,
        note='SH0 approximate geometry viewer. Shared reflection reuses interface Gaussians; see actual rendered ON/OFF comparisons.')
    (output / 'scene.json').write_text(json.dumps(meta))
    html = (Path(__file__).parent / 'templates/gaussian_viewer.html').read_text(encoding='utf-8')
    (output / 'index.html').write_text(html, encoding='utf-8')


def split_asset(path, limit=1900 * 1024 * 1024):
    if path.stat().st_size <= limit:
        return [path]
    pieces = []
    with path.open('rb') as source:
        index = 0
        while source.tell() < path.stat().st_size:
            part = path.with_name(path.name + f'.part{index:03d}')
            with part.open('wb') as target:
                remaining = limit
                while remaining:
                    chunk = source.read(min(8 * 1024 * 1024, remaining))
                    if not chunk:
                        break
                    target.write(chunk)
                    remaining -= len(chunk)
            pieces.append(part)
            index += 1
    return pieces


def prepare(args):
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    checkpoints = sorted((run / 'checkpoints').glob('step_*.pt'))
    if not checkpoints:
        raise RuntimeError('No completed checkpoint found')
    checkpoint = checkpoints[-1]
    archive = run / 'github-archive'
    archive.mkdir(exist_ok=True)
    ready = archive / 'ready.json'
    if ready.exists():
        manifest = json.loads(ready.read_text())
        if manifest['repo'] != args.repo or manifest['tag'] != args.tag:
            raise RuntimeError('Prepared archive belongs to a different repository/release tag')
        if manifest['checkpoint_sha256'] != sha256(checkpoint):
            raise RuntimeError('Run changed since packaging; use a new run directory/archive tag')
        for item in manifest['assets']:
            if sha256(archive / item['name']) != item['sha256']:
                raise RuntimeError('Packaged asset was changed')
        return archive, manifest
    bundle = archive / 'viewer'
    export_geometry(checkpoint, args.config, args.data_root, bundle / 'geometry')
    subprocess.run([sys.executable, '-m', 'glint.compare_shared_reflection', '--config', str(args.config),
        '--data-root', str(args.data_root), '--checkpoints', str(checkpoint), '--output-dir', str(bundle / 'reflection')], check=True)
    (bundle / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>GLINT results</h1><p><a href="geometry/">Interface / transmission 3D Gaussian viewer</a></p><p><a href="reflection/">Shared reflection ON / OFF</a></p><p>Shared reflection queries the interface set; it has no independent Gaussian set.</p>', encoding='utf-8')
    (bundle / 'OPEN_VIEWER.txt').write_text('Extract the viewer ZIP, run: python -m http.server 8765 --bind 127.0.0.1\nOpen http://127.0.0.1:8765/ in a WebGL2 browser. Shared reflection is a rendered path, not a third Gaussian set.\n', encoding='utf-8')
    shutil.copy2(args.config, bundle / 'experiment.yaml')
    if (run / 'config.json').is_file():
        shutil.copy2(run / 'config.json', bundle / 'training-config.json')
    viewer_zip = Path(shutil.make_archive(str(archive / 'viewer'), 'zip', bundle))
    # Save only repository source, never credentials, datasets, logs or .git.
    source_zip = archive / 'source.zip'
    with zipfile.ZipFile(source_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for directory in ('glint', 'scripts', 'configs', 'tests', 'third_party'):
            for file in (root / directory).rglob('*'):
                if file.is_file() and not any(p in {'build', '__pycache__', '.git', '.venv'} or p.endswith('.egg-info') for p in file.parts) and (file.suffix in {'.py', '.sh', '.html', '.js', '.json', '.yaml', '.yml', '.toml', '.cu', '.cpp', '.c', '.h', '.hpp', '.cmake', '.md', '.txt'} or file.name in {'LICENSE', 'CMakeLists.txt'}):
                    z.write(file, str(file.relative_to(root)))
        for name in ('README.md', 'pyproject.toml', 'requirements.txt', 'LICENSE', 'THIRD_PARTY_NOTICES.md'):
            if (root / name).is_file():
                z.write(root / name, name)
    copied = archive / checkpoint.name
    shutil.copy2(checkpoint, copied)
    files = [part for file in (copied, viewer_zip, source_zip) for part in split_asset(file)]
    manifest = {'checkpoint': checkpoint.name, 'checkpoint_sha256': sha256(checkpoint),
        'repo': args.repo, 'tag': args.tag, 'created_unix': time.time(),
        'shared_reflection': 'Interface reuse, not an independent Gaussian set',
        'reassembly': 'For split assets, concatenate parts in numeric order before opening. GitHub asset names include a SHA suffix; original names and hashes are listed below.',
        'assets': [{'name': f.name, 'bytes': f.stat().st_size, 'sha256': sha256(f)} for f in files]}
    ready.write_text(json.dumps(manifest, indent=2))
    return archive, manifest


def publish_files(github, tag, files):
    for attempt in range(1440):
        try:
            release = github.release(tag)
            assets = [github.upload(release, path) for path in files]
            return {'url': release['html_url'], 'assets': assets}
        except Exception as exc:
            print(f'Upload attempt {attempt+1} failed ({type(exc).__name__}); retrying in 60s', flush=True)
            if attempt == 1439:
                raise RuntimeError('Upload retries exhausted; rerun archive command to resume') from None
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='andy4040/glintyai')
    parser.add_argument('--tag')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--check-auth', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--train', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.check_auth:
        print(json.dumps(GitHub(args.repo).check_auth()))
        return
    if not all((args.tag, args.run_dir, args.config, args.data_root)):
        parser.error('--tag, --run-dir, --config, --data-root required')
    if args.prepare_only and args.train:
        parser.error('--prepare-only cannot launch training')
    github = None if args.prepare_only else GitHub(args.repo)
    if github:
        print(json.dumps(github.check_auth()), flush=True)
    if args.train:
        if list((args.run_dir / 'checkpoints').glob('step_*.pt')):
            raise RuntimeError('Detached training requires a new run directory to avoid archiving stale results')
        args.run_dir.mkdir(parents=True, exist_ok=True)
        # Failed/interrupted training never proceeds to publish a final result.
        subprocess.run(args.train, check=True)
        (args.run_dir / 'training.completed').write_text(str(time.time()))
    if github:
        # Protect the expensive checkpoint before GPU viewer export can fail.
        checkpoints = sorted((args.run_dir / 'checkpoints').glob('step_*.pt'))
        if not checkpoints:
            raise RuntimeError('No completed checkpoint found')
        backup_dir = args.run_dir / 'github-archive'
        backup_dir.mkdir(exist_ok=True)
        copied = backup_dir / checkpoints[-1].name
        shutil.copy2(checkpoints[-1], copied)
        receipt = publish_files(github, args.tag, split_asset(copied))
        (backup_dir / 'checkpoint-uploaded.json').write_text(json.dumps(receipt, indent=2))
    archive, manifest = prepare(args)
    print('PREPARED ' + str(archive), flush=True)
    if args.prepare_only:
        return
    receipt = publish_files(github, args.tag,
        [archive / item['name'] for item in manifest['assets']] + [archive / 'ready.json'])
    (archive / 'uploaded.json').write_text(json.dumps(receipt, indent=2))
    print('UPLOADED ' + receipt['url'], flush=True)


if __name__ == '__main__':
    main()
