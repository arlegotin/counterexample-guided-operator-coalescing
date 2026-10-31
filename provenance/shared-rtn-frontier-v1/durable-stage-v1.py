"""Qualified process-group cleanup with a durable supervisor output stream."""
from pathlib import Path
from types import ModuleType
import hashlib,os,signal,subprocess
CONTROL=Path(__file__).resolve().with_name('qualified-controller-v4.py');image=CONTROL.read_bytes()
if hashlib.sha256(image).hexdigest()!='1bc3309d950e5c70bd0f6e12b7e4b396593901013e14268997e82ba079a0555c':raise ValueError('qualified process-group controller changed')
qualified=ModuleType('qualified_cleanup');qualified.__file__=str(CONTROL);exec(compile(image,str(CONTROL),'exec'),qualified.__dict__)
def run_stage(command,*,cwd,log_path):
 previous=signal.signal(signal.SIGTERM,qualified._cancel);child=None
 try:
  # An append-free file is owned before launching, so a disconnected client
  # cannot turn a completed supervisor into Python shutdown status120.
  with Path(log_path).open('x') as stream:
   child=subprocess.Popen(command,cwd=cwd,start_new_session=True,stdout=stream,stderr=subprocess.STDOUT)
   return child.wait()
 finally:
  try:
   if child is not None:
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=2)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()
  finally:signal.signal(signal.SIGTERM,previous)
