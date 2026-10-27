"""Owned offline bytes exercise transport integrity without any network access."""
import hashlib
import io
from dataclasses import replace

import pytest

from poc.assets import AssetRequest, resolve_asset, fetch_asset, AssetError


def request(payload=b'owned fixture'):
    return AssetRequest('owned/model', 'a' * 40, 'config.json', 'https://example.invalid/owned/' + 'a'*40 + '/config.json',
                        hashlib.sha256(payload).hexdigest(), len(payload), 'owned-fixture', 'https://example.invalid/terms', True, 1024)


def test_revisions_terms_and_url_are_immutable():
    for changes in ({'revision': 'main'}, {'sha256': '0'}, {'terms_permitted': False}, {'url': 'http://example.invalid/x'}, {'url': 'https://example.invalid/main/x'}):
        with pytest.raises(AssetError):
            resolve_asset(replace(request(), **changes))


def test_fetch_checks_bytes_and_reuses_only_verified_files(tmp_path):
    source = resolve_asset(request())
    result = fetch_asset(source, tmp_path / 'asset', opener=lambda *_a, **_k: io.BytesIO(b'owned fixture'))
    assert result.sha256 == source.sha256 and result.bytes == 13
    (tmp_path / 'asset').write_bytes(b'corrupt')
    with pytest.raises(AssetError, match='existing'):
        fetch_asset(source, tmp_path / 'asset', opener=lambda *_a, **_k: pytest.fail('network accessed'))


@pytest.mark.parametrize('payload', [b'bad', b'owned fixture extra'])
def test_changed_hash_or_size_never_publishes(tmp_path, payload):
    with pytest.raises(AssetError):
        fetch_asset(resolve_asset(request()), tmp_path / 'asset', opener=lambda *_a, **_k: io.BytesIO(payload))
    assert not (tmp_path / 'asset').exists()
    assert not list(tmp_path.glob('*.part'))


def test_cap_and_interrupted_download(tmp_path):
    with pytest.raises(AssetError, match='cap'):
        resolve_asset(replace(request(), max_bytes=2))
    class Interrupted(io.BytesIO):
        def read(self, n=-1):
            raise OSError('interrupted')
    with pytest.raises(OSError, match='interrupted'):
        fetch_asset(resolve_asset(request()), tmp_path / 'asset', opener=lambda *_a, **_k: Interrupted())
    assert list(tmp_path.iterdir()) == []


def test_range_transport_checks_identity_bounds_caps_and_retains_hashes(tmp_path):
    from poc.assets import RangeAsset, TransferBudget
    payload = b'0123456789'
    def opener(req, **_kwargs):
        a,b=map(int,req.get_header('Range').removeprefix('bytes=').split('-'))
        result=io.BytesIO(payload[a:b+1]); result.status=206
        result.headers={'Content-Range':f'bytes {a}-{b}/10'}
        return result
    source=resolve_asset(request(payload))
    budget=TransferBudget(5)
    stream=RangeAsset(source,tmp_path,budget,opener=opener)
    assert stream.read(3)==b'012'
    stream.seek(8)
    assert stream.read(2)==b'89'
    assert budget.bytes==5 and len(stream.ranges)==2
    assert all(len(row['sha256'])==64 for row in stream.ranges)
    stream.seek(0)
    with pytest.raises(AssetError,match='cap'): stream.read(1)


def test_range_refuses_full_body_or_wrong_content_range(tmp_path):
    from poc.assets import RangeAsset, TransferBudget
    def opener(*_args,**_kwargs):
        result=io.BytesIO(b'owned fixture'); result.status=200; result.headers={}
        return result
    stream=RangeAsset(resolve_asset(request()),tmp_path,TransferBudget(100),opener=opener)
    with pytest.raises(AssetError,match='Range'): stream.read(3)


def test_git_blob_verification_for_non_lfs_public_files(tmp_path):
    from poc.assets import fetch_git_blob
    data=b'owned Git fixture\n'
    oid=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    m=fetch_git_blob('alexwarstadt/blimp','a'*40,'data/owned.jsonl',oid,len(data),tmp_path/'owned',
                     license='owned',terms_url='https://example.invalid/terms',max_bytes=100,
                     opener=lambda *_a,**_k:io.BytesIO(data))
    assert m.sha256==hashlib.sha256(data).hexdigest()
    with pytest.raises(AssetError,match='Git blob'):
        fetch_git_blob('alexwarstadt/blimp','a'*40,'data/owned.jsonl','b'*40,len(data),tmp_path/'bad',
                       license='owned',terms_url='https://example.invalid/terms',max_bytes=100,
                       opener=lambda *_a,**_k:io.BytesIO(data))
