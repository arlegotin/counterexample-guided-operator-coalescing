"""Study disk census with explicitly mutable, append-only exposure metadata."""
import hashlib,json
from pathlib import Path
from poc.recovery_family import RetainedFamily,_absolute


class StudyFamily:
 def __init__(self,output,dependencies,*,required_sources=(),input_files=(),mutable_ledgers=()):
  self.base=RetainedFamily(output,dependencies,required_sources=required_sources)
  self.output=self.base.output
  self.inputs=frozenset(_absolute(str(p)) for p in input_files)
  self.mutable=frozenset(_absolute(str(p)) for p in mutable_ledgers)
  if self.mutable & (self.inputs|set(self.base.required_sources)):
   raise ValueError('mutable exposure ledger overlaps an immutable input/source')
  for p in self.inputs:
   if not p.is_file():raise ValueError('missing immutable input: '+str(p))
  files=(self.base.retained_files|self.inputs)-self.mutable
  self.fixed_files=frozenset(files)
  self.fixed_sizes=tuple(sorted((str(p),p.stat().st_size) for p in files))

 def identity(self):
  return {'schema_version':1,'kind':'reconstruction_input_footprint_v1','output':str(self.output),
   'dependencies':[str(p) for p in self.base.dependencies],
   'required_sources':[str(p) for p in self.base.required_sources],
   'immutable_file_count':len(self.fixed_files),
   'immutable_paths_and_sizes_sha256':hashlib.sha256(json.dumps(self.fixed_sizes).encode()).hexdigest(),
   'mutable_exposure_ledgers':sorted(str(p) for p in self.mutable),
   'policy':'all current logical bytes counted once; exact exposure-ledger appends require preserved recorded prefixes; other dependencies remain immutable',
   'completeness':'declared ancestors plus metadata-listed teacher/token/text inputs; historical lineage remains exploratory'}

 def disk(self):
  count=self.base.disk()
  for p in self.inputs|self.mutable:
   _absolute(str(p))
   if not p.exists() and p in self.mutable:continue
   if not p.is_file():raise ValueError('missing immutable input: '+str(p))
   if not any(p.is_relative_to(root) for root in (self.output,*self.base.dependencies)):count+=p.stat().st_size
  return count

 def exposure_snapshot(self):
  rows=[]
  for p in sorted(self.mutable):
   _absolute(str(p))
   if not p.exists():rows.append({'path':str(p),'bytes':0,'sha256':hashlib.sha256(b'').hexdigest()});continue
   if not p.is_file():raise ValueError('regular exposure ledger required')
   with p.open('rb') as stream:
    size=0;h=hashlib.sha256()
    while block:=stream.read(1024*1024):h.update(block);size+=len(block)
   rows.append({'path':str(p),'bytes':size,'sha256':h.hexdigest()})
  return rows

 def verify_exposure_prefixes(self,rows):
  if {r.get('path') for r in rows}!={str(p) for p in self.mutable} or len(rows)!=len(self.mutable):raise ValueError('exposure prefix paths differ')
  for row in rows:
   p=_absolute(row['path']);n=row['bytes']
   if type(n) is not int or n<0:raise ValueError('invalid exposure prefix size')
   if not p.exists():
    if n:raise ValueError('missing retained exposure prefix')
    data_hash=hashlib.sha256(b'').hexdigest()
   else:
    if not p.is_file() or p.stat().st_size<n:raise ValueError('truncated exposure prefix')
    h=hashlib.sha256();remaining=n
    with p.open('rb') as stream:
     while remaining:
      block=stream.read(min(remaining,1024*1024))
      if not block:raise ValueError('truncated exposure prefix')
      remaining-=len(block);h.update(block)
    data_hash=h.hexdigest()
   if data_hash!=row['sha256']:raise ValueError('exposure prefix changed')
