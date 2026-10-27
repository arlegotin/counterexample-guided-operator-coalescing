"""Calibration cannot read dev/protected data or trigger upstream defaults."""
import pytest
from poc.data.manifest import DataManifest,DocumentRecord

pytestmark=pytest.mark.cpu


def test_project_calibration_uses_only_explicit_fit_components(tmp_path):
    from poc.quantize import project_calibration
    class Tokenizer:
        def encode(self,text): return [int(x) for x in text.split()]
    records=tuple(DocumentRecord(str(i),'fixture','r','s',i,text='1 2 3 4 5 6') for i in range(3))
    fit=DataManifest(records,{str(i):'fit' for i in range(3)},{str(i):str(i) for i in range(3)},'s',str(tmp_path/'exposure.jsonl'))
    source=project_calibration(Tokenizer(),fit,2,4)
    assert source.tokens == ((1,2,3,4),(1,2,3,4))
    assert len(source.component_ids)==len(set(source.component_ids))==2
    assert len((tmp_path/'exposure.jsonl').read_text().splitlines())==4
    bad=DataManifest(records,{str(i):'dev' for i in range(3)},{str(i):str(i) for i in range(3)},'s',str(tmp_path/'denied.jsonl'))
    with pytest.raises(PermissionError): project_calibration(Tokenizer(),bad,2,4)
    assert not (tmp_path/'denied.jsonl').exists()


def test_actual_import_binding_and_network_are_denied():
    import types,socket
    from poc.quantize import isolated_quantizer
    def default(): raise AssertionError('default reached')
    module=types.SimpleNamespace(load_data=default,load_dataset=default)
    with isolated_quantizer(module):
        for binding in (module.load_data,module.load_dataset,socket.create_connection):
            with pytest.raises(PermissionError): binding()
    assert module.load_data is default
