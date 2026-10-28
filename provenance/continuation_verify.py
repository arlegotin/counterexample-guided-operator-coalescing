"""Record exact bounded WP1 commands, exit codes and log hashes sequentially."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    base=[sys.executable,'provenance/continuation_recovery.py','--config','configs/continuation-recovery-v1.yaml']
    commands=[('help',[*base,'--help'],0),
        ('validate',[*base,'--mode','validate-config','--out',str(args.out/'validate.json')],0),
        ('owned',[*base,'--mode','owned-controls','--out',str(args.out/'owned.json')],0),
        ('profile',[*base,'--mode','profile','--out',str(args.out/'profile.json')],0),
        ('pilot-blocked',[*base,'--mode','pilot','--out',str(args.out/'pilot-blocked.json')],3),
        ('cpu-suite',[sys.executable,'-m','pytest','-q'],0)]
    rows=[]
    receipt=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),commands=rows)
    for name,command,expected in commands:
        row=dict(name=name,argv=command,cwd=str(root),expected_exit_code=expected,
                 started=datetime.now(timezone.utc).isoformat(),status='running')
        rows.append(row)
        (args.out/'commands.json').write_text(json.dumps(receipt,indent=2)+'\n')
        log=args.out/(name+'.log')
        with log.open('wb') as stream:
            result=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        row.update(finished=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode,
            status='expected' if result.returncode==expected else 'unexpected',
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        (args.out/'commands.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(name,result.returncode,flush=True)
        if result.returncode!=expected:return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
