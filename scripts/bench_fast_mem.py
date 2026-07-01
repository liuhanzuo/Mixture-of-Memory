"""Benchmark FastMem delta rule implementations."""
import torch
import torch.nn.functional as F
import time
import sys

sys.path.insert(0, "/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory")


@torch.jit.script
def delta_rule_jit(
    k: torch.Tensor,    # [B, T, H, d_k]
    v: torch.Tensor,    # [B, T, H, d_v]
    q: torch.Tensor,    # [B, T, H, d_k]
    gate: torch.Tensor, # [B, T, H, d_k]
    beta: torch.Tensor, # [B, T, H]
    S: torch.Tensor,    # [B, H, d_k, d_v]
) -> tuple[torch.Tensor, torch.Tensor]:
    """JIT-compiled sequential delta rule."""
    B = k.shape[0]
    T = k.shape[1]
    H = k.shape[2]
    d_k = k.shape[3]
    d_v = v.shape[3]

    outputs = torch.empty(B, T, H, d_v, device=k.device, dtype=k.dtype)

    for t in range(T):
        k_t = k[:, t]       # [B, H, d_k]
        v_t = v[:, t]       # [B, H, d_v]
        q_t = q[:, t]       # [B, H, d_k]
        gate_t = gate[:, t] # [B, H, d_k]
        beta_t = beta[:, t] # [B, H]

        # 1. Forget gate
        S = gate_t.unsqueeze(-1) * S  # [B, H, d_k, d_v]

        # 2. Delta rule with error correction
        # S @ k_t -> [B, H, d_v]
        retrieved = torch.einsum("bhkv,bhk->bhv", S, k_t)
        error = v_t - retrieved
        # error outer k_t -> [B, H, d_k, d_v]
        delta = torch.einsum("bhv,bhk->bhkv", error, k_t)
        S = S + beta_t.unsqueeze(-1).unsqueeze(-1) * delta

        # 3. Retrieve
        o_t = torch.einsum("bhkv,bhk->bhv", S, q_t)
        outputs[:, t] = o_t

    return outputs, S


def main():
    device = "cuda"
    dtype = torch.bfloat16
    B, T, H, d_k, d_v = 1, 1024, 4, 128, 128

    k = F.normalize(torch.randn(B, T, H, d_k, device=device, dtype=dtype), dim=-1)
    v = torch.randn(B, T, H, d_v, device=device, dtype=dtype)
    q = torch.randn(B, T, H, d_k, device=device, dtype=dtype)
    gate = torch.sigmoid(torch.randn(B, T, H, d_k, device=device, dtype=dtype))
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    S = torch.zeros(B, H, d_k, d_v, device=device, dtype=dtype)

    # JIT warmup
    print("JIT warmup (3 calls)...")
    for _ in range(3):
        o, S_new = delta_rule_jit(k, v, q, gate, beta, S)
    torch.cuda.synchronize()

    # JIT forward benchmark
    print("Benchmarking JIT forward only...")
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.time()
        o, S_new = delta_rule_jit(k, v, q, gate, beta, S)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    print("JIT forward (1024 tokens): %.1f ms" % (avg * 1000))
    print("Times:", [round(t * 1000, 1) for t in times])
    print("Any NaN:", o.isnan().any().item())
    print("State max: %.4f" % S_new.abs().max().item())

    # JIT fwd+bwd
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    q2 = q.clone().requires_grad_(True)
    print("\nBenchmarking JIT fwd+bwd...")
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.time()
        o, S_new = delta_rule_jit(k2, v2, q2, gate, beta, S.clone())
        loss = o.sum()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.time() - start)
        k2.grad = None
        v2.grad = None
        q2.grad = None

    avg = sum(times) / len(times)
    print("JIT fwd+bwd (1024 tokens): %.1f ms" % (avg * 1000))
    print("Times:", [round(t * 1000, 1) for t in times])

    # Compare with non-JIT (plain Python)
    from src.memory.mem_space.fast_mem import _delta_rule_sequential
    print("\nBenchmarking plain Python forward...")
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.time()
        o2, S2 = _delta_rule_sequential(k, v, q, gate, beta, S)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    avg = sum(times) / len(times)
    print("Python forward (1024 tokens): %.1f ms" % (avg * 1000))
    print("Times:", [round(t * 1000, 1) for t in times])

    # Verify outputs match
    o_jit, S_jit = delta_rule_jit(k, v, q, gate, beta, S)
    o_py, S_py = _delta_rule_sequential(k, v, q, gate, beta, S)
    diff_o = (o_jit - o_py).abs().max().item()
    diff_S = (S_jit - S_py).abs().max().item()
    print("\nOutput diff (JIT vs Python): %.6f" % diff_o)
    print("State diff (JIT vs Python): %.6f" % diff_S)


if __name__ == "__main__":
    main()
