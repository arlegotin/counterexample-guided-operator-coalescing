"""Recompute the retained measured-source to pinned metadata-source AST proof."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if (isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef)) and node.body
                and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant)
                and isinstance(node.body[0].value.value,str)):
            node.body.pop(0)


def verify_binding(path: Path) -> dict:
    binding=json.loads(Path(path).read_text())
    if binding['schema_version']!=1: raise ValueError('unsupported source binding schema')
    measured=binding['measured_source'];metadata=binding['metadata_source']
    old=(ROOT/measured['path']).read_bytes()
    new=subprocess.check_output(['git','show',metadata['git_commit']+':'+metadata['path']],cwd=ROOT)
    if (hashlib.sha256(old).hexdigest()!=measured['sha256']
            or hashlib.sha256(new).hexdigest()!=metadata['sha256']):
        raise ValueError('source binding SHA-256 mismatch')
    a,b=ast.parse(old),ast.parse(new)
    for tree in (a,b): _strip_docstrings(tree)
    imports=[n for n in a.body if isinstance(n,ast.ImportFrom) and n.module=='collections.abc']
    if len(imports)!=1 or [n.name for n in imports[0].names]!=['Iterable','Mapping']:
        raise ValueError('unexpected measured import; cannot normalize AST')
    imports[0].names.insert(1,ast.alias(name='Iterator'))
    donors=[n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='generate_donors']
    if len(donors)!=1 or ast.unparse(donors[0].returns)!='Iterable[ProbeRecord]':
        raise ValueError('unexpected measured return annotation; cannot normalize AST')
    donors[0].returns.value.id='Iterator'
    if ast.dump(a)!=ast.dump(b):
        raise ValueError('executable AST differs after only declared metadata normalization')
    return dict(executable_ast_unchanged=True,measured_sha256=measured['sha256'],
                metadata_sha256=metadata['sha256'],metadata_commit=metadata['git_commit'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--binding',type=Path,default=ROOT/'provenance/probes-source-binding-2026-10-27.json')
    print(json.dumps(verify_binding(parser.parse_args().binding),sort_keys=True))
