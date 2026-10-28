"""Exercise the executable runner without corpus/network access or pretrained assets."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path('provenance/continuation_recovery.py')


def test_cli_help_validate_and_blocked_pilot(tmp_path):
    assert SCRIPT.exists(), 'continuation runner missing'
    base=[sys.executable,str(SCRIPT),'--config','configs/continuation-recovery-v1.yaml']
    result=subprocess.run([*base,'--help'],capture_output=True,text=True)
    assert result.returncode==0 and 'owned-controls' in result.stdout
    for mode,code in [('validate-config',0),('pilot',3)]:
        out=tmp_path/(mode+'.json')
        result=subprocess.run([*base,'--mode',mode,'--out',str(out)],capture_output=True,text=True)
        assert result.returncode==code,result.stderr
        receipt=json.loads(out.read_text())
        assert receipt['mode']==mode
        if mode=='pilot':
            assert receipt['status']=='BLOCKED' and 'data.exposure_lineage' in receipt['blockers']


def test_owned_runner_uses_real_full_update_without_corpus_or_network(tmp_path,monkeypatch):
    assert SCRIPT.exists(), 'continuation runner missing'
    spec=importlib.util.spec_from_file_location('continuation_runner',SCRIPT)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    import socket
    import poc.data.access
    import poc.assets
    def forbidden(*_,**__):raise AssertionError('fixture opened corpus/network')
    monkeypatch.setattr(socket,'socket',forbidden)
    monkeypatch.setattr(poc.data.access,'open_records',forbidden)
    monkeypatch.setattr(poc.assets,'fetch_asset',forbidden)
    out=tmp_path/'owned.json'
    assert runner.main(['--config','configs/continuation-recovery-v1.yaml',
                        '--mode','owned-controls','--out',str(out)])==0
    receipt=json.loads(out.read_text())
    assert receipt['status']=='complete_owned_only'
    assert receipt['controls']['fitting']['targets']==8192
    assert receipt['controls']['fitting']['updates']==1
    assert receipt['controls']['fitting']['teacher_frozen']
    assert receipt['controls']['fitting']['restored_digest_equal']
    assert receipt['archive_gates']['acquisition']=='BLOCKED'
    assert receipt['numeric_fixture']['comparison_status']=='complete_numeric_only'
    assert receipt['aggregate_cost']['updates']==1
    assert receipt['aggregate_cost']['valid_target_presentations']==8192
    assert receipt['aggregate_cost']['teacher_forward_calls']>0
