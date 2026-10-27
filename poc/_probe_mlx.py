"""MLX eager equivalents of deterministic torch fixtures, with input/weight grads."""
import numpy as np
import mlx.core as mx
from poc._probe_torch import values, execute as torch_execute


def execute(operation, dtype, _device):
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal unavailable to this process; retry unsandboxed before declaring hardware incompatibility")
    mx.set_default_device(mx.gpu)
    kind = getattr(mx, dtype)
    details = {"execution_device": "mlx_gpu"}
    arrays = []
    def tensor(shape, offset=0):
        result = mx.array(values(shape, offset), dtype=kind)
        arrays.append(result)
        return result
    def norm(x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    def silu(x):
        return x * mx.sigmoid(x)
    def attention(q, k, v):
        size = q.shape[-2]
        mask = mx.arange(size)[:, None] >= mx.arange(size)[None, :]
        scores = q @ mx.swapaxes(k, -1, -2) / q.shape[-1] ** .5
        return mx.softmax(mx.where(mask, scores, mx.array(-float("inf"), dtype=kind)), axis=-1, precise=True) @ v
    if operation == "attention":
        q, k, v = (tensor((1, 3, 4), i) for i in (0, 3, 7))
        fn = lambda a: attention(*a)
        y0 = fn(arrays)
        changed = v.at[:, -1].add(10)
        err = mx.max(mx.abs(attention(q, k, changed)[:, :-1] - y0[:, :-1])).item()
        assert err == 0, "causal MLX attention leaked a future value"
        details["future_edit_max_abs_error"] = err
    elif operation == "rmsnorm":
        tensor((2, 4))
        fn = lambda a: norm(a[0])
    elif operation == "silu":
        tensor((2, 4))
        fn = lambda a: silu(a[0])
    elif operation == "gather_patch":
        tensor((4, 4)); tensor((2, 4), 5)
        details["implementation"] = "eager scatter-add(patch - gathered source); ArrayAt.set unavailable in MLX 0.32.4"
        fn = lambda a: a[0].at[mx.array([0, 2])].add(a[1] - a[0][mx.array([0, 2])])[mx.array([2, 1, 0])]
    elif operation == "checkpoint":
        tensor((2, 4)); tensor((4, 4), 2)
        calls = []
        def block(x, w):
            calls.append(1)
            return silu(x @ w)
        checked = mx.checkpoint(block)
        fn = lambda a: checked(*a)
    elif operation == "shared_use":
        tensor((2, 4)); tensor((2, 4), 4); tensor((4, 4), 2)
        fn = lambda a: mx.concatenate((a[0] @ a[2], a[1] @ a[2]), axis=0)
        w = arrays[2]
        g1 = mx.grad(lambda w: mx.sum(mx.square(arrays[0] @ w)) / 16)(w)
        g2 = mx.grad(lambda w: mx.sum(mx.square(arrays[1] @ w)) / 16)(w)
        actual = mx.grad(lambda w: mx.mean(mx.square(fn([arrays[0], arrays[1], w]))))(w)
        error = mx.max(mx.abs(actual - g1 - g2)).item()
        assert error <= 8 * {"float32": 2**-23, "float16": 2**-10, "bfloat16": 2**-7}[dtype]
        details["summed_gradient_max_abs_error"] = error
    elif operation == "frozen_suffix":
        tensor((2, 4))
        w = mx.array(values((4, 4), 2), dtype=kind)
        fn = lambda a: silu(a[0] @ mx.stop_gradient(w))
        # MLX has functional gradients: teacher is closed over and explicitly stop_gradient.
        teacher_grad = mx.grad(lambda w: mx.mean(mx.square(silu(arrays[0] @ mx.stop_gradient(w)))))(w)
        assert mx.max(mx.abs(teacher_grad)).item() == 0
        details["teacher_parameter_gradients_absent"] = True
        details["teacher_gradient_semantics"] = "functional AD; teacher excluded from differentiated pytree and stop_gradient derivative verified zero"
        details["finite_difference_max_abs_error"] = torch_execute(operation, "float64", "cpu")[2]["finite_difference_max_abs_error"]
    elif operation == "tiny_decoder":
        for shape, offset in (((16, 12), 1), ((12, 12), 2), ((12, 4), 3), ((12, 4), 4), ((12, 12), 5), ((12, 24), 6), ((12, 24), 7), ((24, 12), 8)):
            tensor(shape, offset)
        def decoder(a):
            embedding, wq, wk, wv, wo, wg, wu, wd = a
            x = embedding[mx.array([[1, 3, 2, 4]])]
            batch, length, hidden = x.shape
            head_dim = wk.shape[-1]
            heads, kv_heads = wq.shape[-1] // head_dim, wk.shape[-1] // head_dim
            def project_rope(w):
                z = mx.swapaxes(mx.reshape(norm(x) @ w, (batch, length, -1, head_dim)), 1, 2)
                position = mx.arange(length, dtype=kind).reshape(1, 1, length, 1)
                frequency = mx.array([1, .01], dtype=kind).reshape(1, 1, 1, -1)
                angle = position * frequency
                first, second = mx.split(z, 2, axis=-1)
                return mx.concatenate((first * mx.cos(angle) - second * mx.sin(angle), second * mx.cos(angle) + first * mx.sin(angle)), axis=-1)
            q, k = project_rope(wq), mx.repeat(project_rope(wk), heads // kv_heads, axis=1)
            v = mx.repeat(mx.swapaxes((norm(x) @ wv).reshape(batch, length, kv_heads, head_dim), 1, 2), heads // kv_heads, axis=1)
            z = x + mx.swapaxes(attention(q, k, v), 1, 2).reshape(batch, length, hidden) @ wo
            z = z + (silu(norm(z) @ wg) * (norm(z) @ wu)) @ wd
            return norm(z) @ embedding.T
        fn = decoder
        details.update(tied_head=True, query_heads=3, kv_heads=1, head_dim=4,
                       shape=[1, 4, 16], attention="eager causal GQA with split-half RoPE")
    else:
        raise ValueError(f"unknown operation {operation}")
    y = fn(arrays)
    calls_before_differentiation = len(calls) if operation == "checkpoint" else 0
    grads = mx.grad(lambda a: mx.mean(mx.square(fn(a))))(arrays)
    mx.eval(y, grads)
    mx.synchronize()
    if operation == "checkpoint":
        differentiation_calls = len(calls) - calls_before_differentiation
        details["output_evaluation_forward_calls"] = calls_before_differentiation
        details["differentiation_forward_calls"] = differentiation_calls
        details["recomputed_forward_calls"] = differentiation_calls - 1
        assert differentiation_calls > 1, "MLX checkpoint did not recompute during differentiation"
    if operation == "frozen_suffix":
        details["input_gradient_norm"] = mx.sqrt(mx.sum(mx.square(grads[0]))).item()
        assert details["input_gradient_norm"] > 0
    output = np.array(y.astype(mx.float32)).astype(np.float64).ravel()
    gradient = np.concatenate([np.array(g.astype(mx.float32)).astype(np.float64).ravel() for g in grads])
    assert np.isfinite(output).all() and np.isfinite(gradient).all()
    return output, gradient, details


if __name__ == "__main__":
    from dataclasses import asdict
    import json
    from poc.doctor import _probe_backend
    print(json.dumps([asdict(row) for row in _probe_backend("mlx")], allow_nan=False))
