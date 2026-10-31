"""A backend parity gate must reject loss/gradient signals replaced by zeros."""
import pytest
import torch


def test_kl_gate_rejects_zeroed_dispatch_signal(monkeypatch):
    import poc.recovery_controls as controls
    from poc.contracts import MetricSums
    def zeros(student,teacher,update,config,recorder,**kwargs):
        for p in student.parameters():
            if p.requires_grad:p.grad=torch.zeros_like(p)
        return [MetricSums(kl=0.,nll=0.,valid_targets=update.valid_targets)]
    monkeypatch.setattr(controls,'backward_update',zeros)
    with pytest.raises(ValueError,match='parity'):
        controls.qualify_backend('cpu')


def test_kl_reference_signal_exceeds_its_fixed_envelope():
    from poc.recovery_controls import qualify_backend
    result=qualify_backend('cpu')
    for row in result['objectives'].values():
        assert row['reference_value']>1e-6
        assert row['reference_gradient_max_abs']>1e-5
