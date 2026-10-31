"""Finite guard for an explicitly frozen scientific preparation command."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,time,psutil

def guarded(command,*,cwd,env,log_path,wall_seconds,memory_bytes,scratch,disk_bytes):
 start=time.monotonic();peak=0;disk_peak=0;failure=None
 with Path(log_path).open('x') as log:
  child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
  try:
   while child.poll() is None:
    try:
     process=psutil.Process(child.pid);rss=0
     for p in [process,*process.children(recursive=True)]:
      try:rss+=p.memory_info().rss
      except psutil.NoSuchProcess:pass
     peak=max(peak,rss)
    except psutil.NoSuchProcess:pass
    size=0
    if Path(scratch).exists():
     for p in Path(scratch).rglob('*'):
      try:
       if p.is_file():size+=p.stat().st_size
      except FileNotFoundError:pass
    disk_peak=max(disk_peak,size)
    if time.monotonic()-start>wall_seconds:failure='stage_wall_cap'
    if peak>memory_bytes:failure='stage_process_rss_cap'
    if disk_peak>disk_bytes:failure='stage_scratch_cap'
    if failure:
     try:
      for p in psutil.Process(child.pid).children(recursive=True):p.terminate()
     except psutil.NoSuchProcess:pass
     child.terminate();break
    time.sleep(.25)
   try:rc=child.wait(timeout=10)
   except subprocess.TimeoutExpired:child.kill();rc=child.wait()
  except BaseException:
   if child.poll() is None:
    child.terminate()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill();child.wait()
   raise

 return {'exit_code':rc,'failure':failure or ('nonzero_stage_exit' if rc else None),
  'wall_seconds':time.monotonic()-start,'sampled_process_tree_rss_bytes':peak,'sampled_scratch_peak_bytes':disk_peak}

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);a=p.parse_args();raw=a.protocol.read_bytes();c=json.loads(raw);out=Path(c['provenance']);out.mkdir(parents=True,exist_ok=True)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 for ref in c['frozen_files']:
  if sha(ref['path'])!=ref['sha256']:raise ValueError('frozen stage input changed: '+ref['path'])
 if sha(__file__)!=c['supervisor_sha256']:raise ValueError('stage supervisor changed')
 env=os.environ.copy();env.update(PYTHONPATH=c['root'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for key,value in c['execution_env'].items():
  if value is None:env.pop(key,None)
  else:env[key]=str(value)
 (out/'stage-start.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'command':c['command'],'protocol_sha256':sha(a.protocol),'scope':c['scope']},indent=2))
 record=guarded(c['command'],cwd=c['root'],env=env,log_path=out/'execution.log',scratch=c['output'],**c['caps'])
 record.update(protocol_sha256=sha(a.protocol),scope=c['scope']);(out/'attempt.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
 if record['failure']:raise SystemExit(2)

if __name__=='__main__':main()
