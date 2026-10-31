"""Build audited source and portable numeric evidence assets from a clean commit."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tomllib
import zipfile

VERSION='0.2.0'
ROOT=Path(__file__).resolve().parents[1]
EVIDENCE='provenance/continuation-checkpoint-2026-10-31'
INDEX='provenance/releases/v0.2.0/checkpoint-sha256.json'
LIMIT=128*1024**2
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
LICENSES={'LICENSE':'0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0',
          'licenses/CC-BY-4.0.txt':'9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411'}


def require(value,message):
    if not value:raise ValueError(message)


def digest(body):return hashlib.sha256(body).hexdigest()


def safe_name(name,*,prefix=None,directory=False):
    require(type(name) is str and name and '\\' not in name,'safe POSIX archive path required')
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '.' not in p.parts,'archive traversal rejected')
    if prefix is not None:
        require(name.startswith(prefix),'archive member outside declared prefix');relative=name[len(prefix):]
    else:relative=name
    parts=PurePosixPath(relative).parts
    require(not any(x in ('.git','.aws','.ssh','.codex') or x.startswith('.env') for x in parts),'private path rejected')
    require(not parts or parts[0] not in ('data','artifacts') or relative=='artifacts/release_manifest.json' or directory and relative=='artifacts','raw data/model payload rejected')
    require(not relative.endswith(('.safetensors','.pt','.pth','.ckpt','.gguf','.bin','.pem')),'model/key payload rejected')
    return relative


def audit_tar(body,*,prefix=None,depth=0):
    require(depth<=2 and len(body)<=LIMIT,'archive depth/size cap exceeded')
    count=0;total=0;nested=0;names=[]
    with tarfile.open(fileobj=io.BytesIO(body),mode='r:') as archive:
        members=archive.getmembers();require(len({m.name for m in members})==len(members),'duplicate archive member')
        require(sum(m.size for m in members)<=LIMIT,'archive declared size cap exceeded')
        for member in members:
            relative='' if prefix is not None and member.isdir() and member.name.rstrip('/')==prefix.rstrip('/') else safe_name(member.name,prefix=prefix,directory=member.isdir())
            names.append(relative)
            require(member.isdir() or member.isfile(),'archive links/special files rejected')
            if member.isdir():continue
            data=archive.extractfile(member).read(LIMIT+1)
            require(len(data)==member.size and not SECRET.search(data),'credential pattern or truncated payload rejected')
            count+=1;total+=len(data)
            if relative.endswith(('.tar.gz','.tgz','.zip','.tar','.gz','.npz')):
                require(re.fullmatch(r'provenance/dense-baseline-expansion-v2/final-source-v[123]\.tar\.gz',relative) is not None,'unqualified nested archive rejected')
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:raw=stream.read(LIMIT+1)
                sub=audit_tar(raw,depth=depth+1);count+=sub['regular_files'];total+=sub['uncompressed_bytes'];nested+=1+sub['nested_source_archives']
                require(total<=LIMIT,'aggregate nested archive size cap exceeded')
    if any(name.startswith('installed-source/') and '/mlx_lm/' in name for name in names):
        require('licenses/mlx-lm-MIT.txt' in names,'retained MLX-LM source missing MIT notice')
    return {'regular_files':count,'uncompressed_bytes':total,'nested_source_archives':nested,'forbidden_payloads':0}


def evidence_zip(root):
    root=Path(root);files=[EVIDENCE+'/'+p.name for p in (root/EVIDENCE).iterdir()]
    files+=['provenance/verify_checkpoint.py',INDEX,'LICENSE','licenses/CC-BY-4.0.txt','licenses/mlx-lm-MIT.txt',
            'LICENSING.md','NOTICE','THIRD_PARTY_NOTICES.md','CITATION.cff']
    output=io.BytesIO();prefix='cgoc-v0.2.0-evidence/'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in sorted(files):
            safe_name(name);path=root/name;require(path.is_file() and not path.is_symlink(),'regular evidence input required')
            data=path.read_bytes();require(not SECRET.search(data),'credential pattern in evidence archive')
            info=zipfile.ZipInfo(prefix+name,date_time=(2026,10,31,0,0,0));info.create_system=3;info.external_attr=0o100644<<16
            info.compress_type=zipfile.ZIP_DEFLATED;archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
        text=('CGOC v0.2.0 public numeric evidence\n\nRun from this directory:\n'
              '  python3.12 provenance/verify_checkpoint.py\n'
              '  python3.12 provenance/verify_checkpoint.py --bootstrap\n\n'
              'The default uses only the standard library; optional resampling uses NumPy (recorded version2.5.4).\n'
              'No model, corpus, protected data, training or inference replay is included.\n'
              'Native shared artifacts belong to the earlier256-update model; long1024states are a separate diagnostic.\n'
              'Full source, scope and license details: https://github.com/arlegotin/counterexample-guided-operator-coalescing/tree/v0.2.0\n')
        info=zipfile.ZipInfo(prefix+'VERIFY.txt',date_time=(2026,10,31,0,0,0));info.create_system=3;info.external_attr=0o100644<<16
        archive.writestr(info,text,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return output.getvalue()


def build_assets(root,*,commit,out):
    root=Path(root).resolve();out=Path(out).resolve()
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None,'full release commit SHA required')
    git=lambda *a:subprocess.check_output(['git',*a],cwd=root,text=True).strip()
    require(git('rev-parse','HEAD')==commit and not git('status','--porcelain'),'exact clean release commit required')
    require(not out.exists(),'release output already attempted; preserve its artifacts')
    require(not out.is_relative_to(root),'release assets must be outside source checkout')
    project=tomllib.loads((root/'pyproject.toml').read_text())['project'];require(project['version']==VERSION,'package version differs')
    require(re.search(r'^version: 0\.2\.0$',(root/'CITATION.cff').read_text(),re.M),'citation version differs')
    for name,expected in LICENSES.items():require(digest((root/name).read_bytes())==expected,'standard license text changed')
    subprocess.run([__import__('sys').executable,'-I','provenance/verify_checkpoint.py','--bootstrap'],cwd=root,check=True,capture_output=True)
    prefix=f'cgoc-v{VERSION}/';source_tar=subprocess.check_output(['git','archive','--format=tar','--prefix='+prefix,commit],cwd=root)
    audit=audit_tar(source_tar,prefix=prefix);source=gzip.compress(source_tar,compresslevel=9,mtime=0);evidence=evidence_zip(root)
    out.mkdir(parents=True,exist_ok=False);assets={}
    for name,data in ((f'cgoc-v{VERSION}-source.tar.gz',source),(f'cgoc-v{VERSION}-evidence.zip',evidence)):
        (out/name).write_bytes(data);assets[name]={'sha256':digest(data),'bytes':len(data)}
    manifest={'schema_version':1,'version':VERSION,'commit':commit,'assets':assets,'source_audit':audit,
        'numeric_evidence_files':42,'license_hashes':LICENSES,'scope':'source and saved numeric evidence; no model/corpus/optimizer payload or new scientific execution'}
    body=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();(out/'release-manifest.json').write_bytes(body)
    sums={**{name:v['sha256'] for name,v in assets.items()},'release-manifest.json':digest(body)}
    (out/'SHA256SUMS').write_text(''.join(f'{value}  {name}\n' for name,value in sorted(sums.items())))
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    print(json.dumps(build_assets(ROOT,commit=a.commit,out=a.out),sort_keys=True))


if __name__=='__main__':main()
