"""Explicit immutable public-asset resolution and capped atomic retrieval.

No authentication, executable dataset scripts, remote code, pickle, or home cache.
"""
from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import urlopen


class AssetError(ValueError):
    """Source qualification or transport integrity failed."""


@dataclass(frozen=True)
class AssetRequest:
    repository: str
    revision: str
    path: str
    url: str
    sha256: str
    bytes: int
    license: str
    terms_url: str
    terms_permitted: bool
    max_bytes: int


@dataclass(frozen=True)
class SourceRecord(AssetRequest):
    pass


@dataclass(frozen=True)
class AssetManifest:
    source: SourceRecord
    local_path: str
    sha256: str
    bytes: int


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_asset(request: AssetRequest) -> SourceRecord:
    if not re.fullmatch('[0-9a-f]{40}', request.revision) or not re.fullmatch('[0-9a-f]{64}', request.sha256):
        raise AssetError('full immutable revision and SHA-256 required')
    if (urlparse(request.url).scheme != 'https' or request.revision not in urlparse(request.url).path
            or not request.terms_permitted or not request.license or urlparse(request.terms_url).scheme != 'https'):
        raise AssetError('public immutable HTTPS source and permitted terms required')
    if type(request.bytes) is not int or type(request.max_bytes) is not int or not 0 <= request.bytes <= request.max_bytes:
        raise AssetError('source size exceeds byte cap')
    if Path(request.path).is_absolute() or '..' in Path(request.path).parts:
        raise AssetError('unsafe source path')
    return SourceRecord(**asdict(request))


def fetch_asset(source: SourceRecord, destination: Path, *, opener=urlopen) -> AssetManifest:
    source = resolve_asset(source)
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size != source.bytes or file_sha256(destination) != source.sha256:
            raise AssetError('existing asset has changed size/hash')
        return AssetManifest(source, str(destination), source.sha256, source.bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + '.part')
    # Exclusive partial: interrupted attempts leave no publishable file; no unsafe resume.
    created = False
    try:
        with part.open('xb') as output:
            created = True
            with opener(source.url, timeout=60) as response:
                h, size = hashlib.sha256(), 0
                while block := response.read(min(1024 * 1024, source.max_bytes - size + 1)):
                    size += len(block)
                    if size > source.max_bytes or size > source.bytes:
                        raise AssetError('download exceeds source size/byte cap')
                    output.write(block)
                    h.update(block)
            if size != source.bytes or h.hexdigest() != source.sha256:
                raise AssetError('download size/SHA-256 mismatch')
            output.flush()
            os.fsync(output.fileno())
        # Refuse replacing a concurrently published file.
        os.link(part, destination)
        part.unlink()
    except BaseException:
        if created:
            part.unlink(missing_ok=True)
        raise
    return AssetManifest(source, str(destination), h.hexdigest(), size)


@dataclass
class TransferBudget:
    max_bytes: int
    bytes: int = 0

    def reserve(self, count: int):
        if count < 0 or self.bytes + count > self.max_bytes:
            raise AssetError('transfer byte cap exceeded')
        self.bytes += count


class RangeAsset:
    """Seekable immutable HTTPS range reader for bounded parquet row groups.

    Every actual range is retained with SHA-256. The parent object's pinned
    SHA-256 identifies provenance; it is NOT claimed locally verified in full.
    """
    def __init__(self, source: SourceRecord, directory: Path, budget: TransferBudget, *, max_request_bytes=64*1024**2, opener=urlopen):
        import io
        self.source = resolve_asset(source)
        self.directory, self.budget = Path(directory),budget
        self.max_request_bytes,self.opener = max_request_bytes,opener
        self.directory.mkdir(parents=True,exist_ok=True)
        self.position,self.ranges,self.closed = 0,[],False

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def tell(self): return self.position
    def close(self): self.closed=True
    def seek(self, offset, whence=0):
        position = offset if whence==0 else self.position+offset if whence==1 else self.source.bytes+offset if whence==2 else -1
        if position < 0 or position > self.source.bytes: raise AssetError('range seek outside source')
        self.position=position
        return position

    def read(self, size=-1):
        from urllib.request import Request
        if self.closed: raise ValueError('closed range reader')
        size = self.source.bytes-self.position if size < 0 else min(size,self.source.bytes-self.position)
        if size == 0: return b''
        if size > self.max_request_bytes: raise AssetError('range request byte cap exceeded')
        self.budget.reserve(size)
        start,end=self.position,self.position+size-1
        req=Request(self.source.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        with self.opener(req,timeout=60) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.source.bytes}':
                raise AssetError('Range server ignored or changed immutable byte interval')
            data=response.read(size+1)
        if len(data) != size: raise AssetError('range size mismatch')
        digest=hashlib.sha256(data).hexdigest()
        path=self.directory/f'{start}-{end}-{digest}.range'
        if path.exists():
            if file_sha256(path) != digest: raise AssetError('existing range hash mismatch')
        else:
            with path.open('xb') as stream: stream.write(data)
        self.ranges.append({'start':start,'end_inclusive':end,'bytes':size,'sha256':digest,'local_path':str(path)})
        self.position += size
        return data


def fetch_git_blob(repository: str, revision: str, path: str, blob_sha1: str, expected_bytes: int,
                   destination: Path, *, license: str, terms_url: str, max_bytes: int, opener=urlopen) -> AssetManifest:
    """Verify pinned upstream Git object, then record SHA-256 of actual bytes.

    Small Git-tracked tokenizer/benchmark files lack upstream SHA-256 metadata.
    Git's object ID plus full commit verifies transport without guessing a hash.
    """
    if not re.fullmatch('[0-9a-f]{40}',revision) or not re.fullmatch('[0-9a-f]{40}',blob_sha1):
        raise AssetError('immutable commit/Git blob required')
    if expected_bytes > max_bytes or expected_bytes < 0: raise AssetError('Git asset cap exceeded')
    if repository == 'alexwarstadt/blimp':
        url=f'https://raw.githubusercontent.com/{repository}/{revision}/{path}'
    elif repository == 'HuggingFaceTB/SmolLM2-135M':
        url=f'https://huggingface.co/{repository}/resolve/{revision}/{path}'
    else: raise AssetError('unapproved Git asset repository')
    destination=Path(destination)
    if destination.exists(): data=destination.read_bytes()
    else:
        with opener(url,timeout=60) as response: data=response.read(max_bytes+1)
    if len(data) != expected_bytes or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() != blob_sha1:
        raise AssetError('Git blob size/hash mismatch')
    source=resolve_asset(AssetRequest(repository,revision,path,url,hashlib.sha256(data).hexdigest(),len(data),license,terms_url,True,max_bytes))
    return fetch_asset(source,destination,opener=lambda *_a,**_k: __import__('io').BytesIO(data))
