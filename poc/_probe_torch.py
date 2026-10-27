"""Deterministic eager Llama-shaped tiny fixtures; only imported for execution."""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def values(shape, offset=0):
    """Backend-independent bounded values; no PRNG/version-dependent fixture."""
    return ((np.arange(np.prod(shape)).reshape(shape) + offset) % 17 - 8) / 16


def execute(operation, dtype, device):
    kind = getattr(torch, dtype)
    details = {"execution_device": device}
    leaves = []
    def tensor(shape, offset=0, grad=True):
        result = torch.tensor(values(shape, offset), dtype=kind, device=device, requires_grad=grad)
        if grad:
            leaves.append(result)
        return result
    def norm(x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
    def attention(q, k, v):
        size = q.shape[-2]
        mask = torch.ones(size, size, dtype=torch.bool, device=device).tril()
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        return torch.softmax(scores.masked_fill(~mask, -float("inf")), dim=-1) @ v
    if operation == "attention":
        q, k, v = (tensor((1, 3, 4), i) for i in (0, 3, 7))
        y = attention(q, k, v)
        changed = v.detach().clone()
        changed[:, -1] += 10
        causal_error = (attention(q, k, changed)[:, :-1] - y[:, :-1]).abs().max().item()
        assert causal_error == 0, "causal attention leaked a future value"
        details["future_edit_max_abs_error"] = causal_error
    elif operation == "rmsnorm":
        y = norm(tensor((2, 4)))
    elif operation == "silu":
        y = torch.nn.functional.silu(tensor((2, 4)))
    elif operation == "gather_patch":
        x, patch = tensor((4, 4)), tensor((2, 4), 5)
        indices = torch.tensor([0, 2], device=device)
        y = x.clone().index_copy(0, indices, patch).index_select(0, torch.tensor([2, 1, 0], device=device))
    elif operation == "checkpoint":
        x, w = tensor((2, 4)), tensor((4, 4), 2)
        calls = []
        def block(a, b):
            calls.append(1)
            return torch.nn.functional.silu(a @ b)
        y = checkpoint(block, x, w, use_reentrant=False)
        details["calls"] = calls
    elif operation == "shared_use":
        x1, x2, w = tensor((2, 4)), tensor((2, 4), 4), tensor((4, 4), 2)
        y = torch.cat((x1 @ w, x2 @ w), dim=0)
        g1 = torch.autograd.grad((x1 @ w).square().sum() / y.numel(), w, retain_graph=True)[0]
        g2 = torch.autograd.grad((x2 @ w).square().sum() / y.numel(), w, retain_graph=True)[0]
        actual = torch.autograd.grad(y.square().mean(), w, retain_graph=True)[0]
        error = (actual - g1 - g2).abs().max().item()
        assert error <= 8 * torch.finfo(kind).eps, "shared gradient differs from explicit sum"
        details["summed_gradient_max_abs_error"] = error
    elif operation == "frozen_suffix":
        x, w = tensor((2, 4)), tensor((4, 4), 2, grad=False)
        y = torch.nn.functional.silu(x @ w)
        details["teacher_parameter_gradients_absent"] = w.grad is None and not w.requires_grad
        if dtype == "float64":
            h = 1e-5
            analytic = torch.autograd.grad(y.square().mean(), x, retain_graph=True)[0]
            estimates = []
            for i in range(x.numel()):
                plus, minus = x.detach().clone(), x.detach().clone()
                plus.reshape(-1)[i] += h
                minus.reshape(-1)[i] -= h
                estimates.append(((torch.nn.functional.silu(plus @ w).square().mean() - torch.nn.functional.silu(minus @ w).square().mean()) / (2 * h)).item())
            details["finite_difference_max_abs_error"] = float(np.max(np.abs(np.array(estimates) - analytic.detach().cpu().numpy().ravel())))
    elif operation == "tiny_decoder":
        # Dimensions derive from tensor shapes; query/KV heads differ and head is tied.
        embedding = tensor((16, 12), 1)
        wq, wk, wv, wo = tensor((12, 12), 2), tensor((12, 4), 3), tensor((12, 4), 4), tensor((12, 12), 5)
        wg, wu, wd = tensor((12, 24), 6), tensor((12, 24), 7), tensor((24, 12), 8)
        x = embedding[torch.tensor([[1, 3, 2, 4]], device=device)]
        batch, length, hidden = x.shape
        head_dim = wk.shape[-1]
        heads, kv_heads = wq.shape[-1] // head_dim, wk.shape[-1] // head_dim
        def project_rope(w):
            z = (norm(x) @ w).reshape(batch, length, -1, head_dim).transpose(1, 2)
            position = torch.arange(length, device=device, dtype=kind).reshape(1, 1, length, 1)
            frequency = torch.tensor([1, .01], device=device, dtype=kind).reshape(1, 1, 1, -1)
            angle = position * frequency
            a, b = z.chunk(2, dim=-1)
            return torch.cat((a * angle.cos() - b * angle.sin(), b * angle.cos() + a * angle.sin()), dim=-1)
        q, k = project_rope(wq), project_rope(wk).repeat_interleave(heads // kv_heads, dim=1)
        v = (norm(x) @ wv).reshape(batch, length, kv_heads, head_dim).transpose(1, 2).repeat_interleave(heads // kv_heads, dim=1)
        a = attention(q, k, v).transpose(1, 2).reshape(batch, length, hidden) @ wo
        z = x + a
        z = z + (torch.nn.functional.silu(norm(z) @ wg) * (norm(z) @ wu)) @ wd
        y = norm(z) @ embedding.T
        details.update(tied_head=True, query_heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                       shape=list(y.shape), attention="eager causal GQA with split-half RoPE")
    else:
        raise ValueError(f"unknown operation {operation}")
    grads = torch.autograd.grad(y.square().mean(), leaves)
    if operation == "checkpoint":
        calls = details.pop("calls")
        details["recomputed_forward_calls"] = len(calls)
        assert len(calls) > 1, "checkpoint did not recompute"
    if operation == "frozen_suffix":
        details["input_gradient_norm"] = grads[0].norm().item()
        assert details["input_gradient_norm"] > 0
        if dtype != "float64":
            details["finite_difference_max_abs_error"] = execute(operation, "float64", "cpu")[2]["finite_difference_max_abs_error"]
    if device == "mps":
        torch.mps.synchronize()
    # Two-stage transfer avoids accelerator->CPU float64 conversion regressions.
    output = y.detach().cpu().to(torch.float64).numpy().ravel()
    gradient = np.concatenate([g.detach().cpu().to(torch.float64).numpy().ravel() for g in grads])
    assert np.isfinite(output).all() and np.isfinite(gradient).all(), "nonfinite fixture result"
    return output, gradient, details
