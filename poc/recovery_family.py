"""Count declared retained study outputs once, without following aliases."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path


def _absolute(value):
    if not isinstance(value,str) or not value or not Path(value).is_absolute():
        raise ValueError('absolute canonical retained-output paths required')
    p=Path(value)
    if p.is_symlink() or p.resolve()!=p:raise ValueError('symlink or noncanonical retained-output path')
    return p


def _directory(path,*,required):
    if path.is_symlink() or path.resolve()!=path:raise ValueError('retained root changed or became a symlink')
    if not path.exists() and not required:return
    if not path.is_dir():raise ValueError('retained dependency root must be an existing directory')


@dataclass(frozen=True,init=False)
class RetainedFamily:
    output: Path
    dependencies: tuple[Path,...]
    required_sources: tuple[Path,...]
    retained_files: frozenset[Path]
    initial_sizes_sha256: str

    def __init__(self,output,dependencies,*,required_sources=()):
        own=_absolute(output)
        if not isinstance(dependencies,(list,tuple)) or not dependencies:raise ValueError('explicit retained dependencies required')
        roots=tuple(sorted(_absolute(p) for p in dependencies))
        for p in roots:_directory(p,required=True)
        _directory(own,required=False)
        all_roots=(own,*roots)
        for i,left in enumerate(all_roots):
            for right in all_roots[i+1:]:
                if left.is_relative_to(right) or right.is_relative_to(left):raise ValueError('duplicate or overlapping retained roots')
        sources=tuple(_absolute(p) for p in required_sources)
        for p in sources:
            if not p.is_file() or not any(p.is_relative_to(root) for root in roots):raise ValueError('retained roots omit a required source artifact')
        retained=set(sources)
        for root in roots:
            for path in root.rglob('*'):
                if path.is_symlink():raise ValueError('symlink inside retained dependency root')
                if path.is_file():retained.add(path)
        object.__setattr__(self,'output',own);object.__setattr__(self,'dependencies',roots);object.__setattr__(self,'required_sources',sources)
        object.__setattr__(self,'retained_files',frozenset(retained))
        sizes=sorted((str(path),path.stat().st_size) for path in retained)
        object.__setattr__(self,'initial_sizes_sha256',hashlib.sha256(json.dumps(sizes).encode()).hexdigest())

    def identity(self):
        return {'schema_version':1,'output':str(self.output),'dependencies':[str(p) for p in self.dependencies],
                'required_sources':[str(p) for p in self.required_sources],
                'initial_retained_file_count':len(self.retained_files),
                'initial_retained_sizes_sha256':self.initial_sizes_sha256,
                'initial_retained_paths_sha256':hashlib.sha256(json.dumps(sorted(str(p) for p in self.retained_files)).encode()).hexdigest(),
                'completeness':'caller declares all ancestor-study roots; census guarantees only the declared scope',
                'policy':'logical file bytes, canonical disjoint roots; no symlink following or cached mutable sizes'}

    def disk(self):
        total=0;seen=set()
        for root in (self.output,*self.dependencies):
            _directory(root,required=root!=self.output)
            for path in root.rglob('*'):
                try:
                    if path.is_symlink():raise ValueError('symlink inside retained-output family')
                    if path.is_file():
                        total+=path.stat().st_size;seen.add(path)
                except FileNotFoundError:
                    if path in self.retained_files:raise ValueError('required retained source disappeared during census: '+str(path))
                    # Only new/rolling output files may be pruned during census.
                    continue
        missing=self.retained_files-seen
        if missing:raise ValueError('required retained source files disappeared: '+str(sorted(missing)[0]))
        return total
