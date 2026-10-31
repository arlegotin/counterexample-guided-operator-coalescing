"""Release archives must be portable, complete, deterministic and data-free."""
from pathlib import Path
import importlib.util,io,tarfile,subprocess,sys,zipfile
import pytest
ROOT=Path(__file__).resolve().parents[1]


def api():
 p=ROOT/'provenance/build_release.py';assert p.is_file(),'release archive builder missing'
 spec=importlib.util.spec_from_file_location('cgoc_release_builder',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tar_payload(name,body=b'owned source',kind=None):
 out=io.BytesIO()
 with tarfile.open(fileobj=out,mode='w') as archive:
  info=tarfile.TarInfo(name);info.size=len(body)
  if kind:
   info.type=kind;info.linkname='outside';info.size=0;archive.addfile(info)
  else:archive.addfile(info,io.BytesIO(body))
 return out.getvalue()


@pytest.mark.parametrize('name,body,kind',[
 ('cgoc-v0.2.0/../outside.py',b'owned',None),('/outside.py',b'owned',None),
 ('cgoc-v0.2.0/model.safetensors',b'owned',None),('cgoc-v0.2.0/data/tokens.json',b'[]',None),
 ('cgoc-v0.2.0/.env',b'owned',None),('cgoc-v0.2.0/linked.py',b'',tarfile.SYMTYPE),
 ('cgoc-v0.2.0/token.txt',b'ghp_'+b'a'*36,None)])
def test_unsafe_or_private_archive_payload_is_rejected(name,body,kind):
 with pytest.raises(ValueError):api().audit_tar(tar_payload(name,body,kind),prefix='cgoc-v0.2.0/')


def test_owned_source_archive_passes_without_neural_or_corpus_access():
 report=api().audit_tar(tar_payload('cgoc-v0.2.0/poc/owned.py'),prefix='cgoc-v0.2.0/')
 assert report['regular_files']==1 and report['forbidden_payloads']==0


def test_evidence_zip_is_standalone_and_deterministic(tmp_path):
 m=api();a=m.evidence_zip(ROOT);b=m.evidence_zip(ROOT);assert a==b
 out=tmp_path/'bundle'
 with zipfile.ZipFile(io.BytesIO(a)) as z:
  assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(out)
 root=out/'cgoc-v0.2.0-evidence'
 result=subprocess.run([sys.executable,'-I','provenance/verify_checkpoint.py'],cwd=root,text=True,capture_output=True)
 assert result.returncode==0,result.stderr
 assert '"hashed_files": 42' in result.stdout
 assert (root/'LICENSE').is_file() and (root/'licenses/CC-BY-4.0.txt').is_file()
 assert not any(p.suffix=='.safetensors' for p in root.rglob('*'))


def test_git_archive_root_directory_is_allowed_but_root_file_is_not():
 out=io.BytesIO()
 with tarfile.open(fileobj=out,mode='w') as archive:
  root=tarfile.TarInfo('cgoc-v0.2.0');root.type=tarfile.DIRTYPE;archive.addfile(root)
  info=tarfile.TarInfo('cgoc-v0.2.0/poc/owned.py');info.size=5;archive.addfile(info,io.BytesIO(b'owned'))
 assert api().audit_tar(out.getvalue(),prefix='cgoc-v0.2.0/')['regular_files']==1
 with pytest.raises(ValueError):api().audit_tar(tar_payload('cgoc-v0.2.0'),prefix='cgoc-v0.2.0/')


def test_only_the_public_artifact_manifest_is_allowed_under_artifacts():
 out=io.BytesIO()
 with tarfile.open(fileobj=out,mode='w') as archive:
  folder=tarfile.TarInfo('cgoc-v0.2.0/artifacts');folder.type=tarfile.DIRTYPE;archive.addfile(folder)
  info=tarfile.TarInfo('cgoc-v0.2.0/artifacts/release_manifest.json');info.size=2;archive.addfile(info,io.BytesIO(b'{}'))
 assert api().audit_tar(out.getvalue(),prefix='cgoc-v0.2.0/')['regular_files']==1
 with pytest.raises(ValueError):api().audit_tar(tar_payload('cgoc-v0.2.0/artifacts/model.json'),prefix='cgoc-v0.2.0/')
