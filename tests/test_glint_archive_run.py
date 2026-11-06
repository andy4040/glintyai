"""No network writes: archive integrity and retry behavior."""
import json
from types import SimpleNamespace
import pytest
from glint.archive_run import GitHub, prepare, sha256, split_asset, token


def test_split_reassembles_exactly(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    data = bytes(range(256)) * 5
    path.write_bytes(data)
    parts = split_asset(path, limit=300)
    assert len(parts) == 5
    assert b''.join(p.read_bytes() for p in parts) == data
    assert all(p.stat().st_size <= 300 for p in parts)
    assert split_asset(path, limit=2000) == [path]


def test_upload_retry_skips_verified_asset_and_rejects_collision(tmp_path):
    path = tmp_path / 'checkpoint.pt';path.write_bytes(b'checkpoint bytes')
    digest = sha256(path)
    asset = dict(name=f'checkpoint-{digest[:16]}.pt', size=path.stat().st_size,
                 state='uploaded', digest='sha256:'+digest)
    api = GitHub.__new__(GitHub)
    api.request = lambda *args: [asset]
    assert api.upload({'id': 1}, path) is asset
    asset['digest'] = 'sha256:wrong'
    with pytest.raises(RuntimeError, match='without overwriting'):
        api.upload({'id': 1}, path)


def test_ready_archive_detects_changed_checkpoint_or_asset(tmp_path):
    run = tmp_path / 'run';(run/'checkpoints').mkdir(parents=True)
    checkpoint=run/'checkpoints/step_000009.pt';checkpoint.write_bytes(b'final')
    archive=run/'github-archive';archive.mkdir()
    asset=archive/'viewer.zip';asset.write_bytes(b'viewer')
    manifest=dict(repo='owner/repo',tag='run-test',checkpoint_sha256=sha256(checkpoint),
                  assets=[dict(name=asset.name,sha256=sha256(asset))])
    (archive/'ready.json').write_text(json.dumps(manifest))
    args=SimpleNamespace(run_dir=run,repo='owner/repo',tag='run-test')
    assert prepare(args)[1] == manifest
    asset.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='asset was changed'):
        prepare(args)
    checkpoint.write_bytes(b'new run')
    with pytest.raises(RuntimeError,match='Run changed'):
        prepare(args)


def test_missing_auth_is_actionable_and_not_silently_skipped(monkeypatch,tmp_path):
    monkeypatch.delenv('GH_TOKEN',raising=False);monkeypatch.delenv('GITHUB_TOKEN',raising=False)
    monkeypatch.setenv('GLINT_GITHUB_TOKEN_FILE',str(tmp_path/'absent'))
    with pytest.raises(RuntimeError,match='Server GitHub authentication missing'):
        token()
