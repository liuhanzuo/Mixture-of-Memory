"""Benchmark FastMem with truncated BPTT (chunk_size=64)."""
import torch
import torch.nn.functional as F
import time
import sys

sys.path.insert(0, "/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory")
from src.memory.mem_space.fast_mem import FastMemModule


def main():
    device = "cuda"
    dtype = torch.bfloat16

    # Instantiate with chunk_size=64 (BPTT window)
    m = FastMemModule(d_model=4096, num_heads=4, d_state=128, chunk_size=64, fusion_init=-2.0)
    m = m.to(device=device, dtype=dtype)
    m.train()

    B, T, d = 1, 1024, 4096
    x = torch.randn(B, T, d, device=device, dtype=dtype)

    # Warmup (3 iterations)
    print("Warmup...")
    for _ in range(3):
        output, S = m(x, memory_state=None)
        loss = output.sum()
        loss.backward()
        m.zero_grad()
    torch.cuda.synchronize()

    # Benchmark forward+backward (5 iterations)
    print("Benchmarking fwd+bwd with truncated BPTT (chunk=64)...")
    times = []
    for i in range(5):
        S_prev = torch.zeros(B, m.H, m.d_k, m.d_v, device=device, dtype=dtype)
        torch.cuda.synchronize()
        start = time.time()
        output, S = m(x, memory_state=S_prev)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        m.zero_grad()

    avg_time = sum(times) / len(times)
    print("Average fwd+bwd: %.1f ms" % (avg_time * 1000))
    print("Times:", [round(t * 1000, 1) for t in times])

    # Forward only benchmark
    print("\nBenchmarking forward only...")
    m.eval()
    times = []
    with torch.no_grad():
        for i in range(5):
            S_prev = torch.zeros(B, m.H, m.d_k, m.d_v, device=device, dtype=dtype)
            torch.cuda.synchronize()
            start = time.time()
            output, S = m(x, memory_state=S_prev)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    avg_time = sum(times) / len(times)
    print("Average forward: %.1f ms" % (avg_time * 1000))
    print("Times:", [round(t * 1000, 1) for t in times])

    # Correctness check
    print("\n--- Correctness ---")
    m.eval()
    output, S = m(x, memory_state=None)
    print("Output shape:", output.shape)
    print("State shape:", S.shape)
    print("Output max: %.6f" % output.abs().max().item())
    print("State max: %.6f" % S.abs().max().item())
    print("State norm (mean/head): %.4f" % S.norm(dim=(-2, -1)).mean().item())
    print("Any NaN: %s" % (output.isnan().any() or S.isnan().any()).item())

    # Stability over 10 chunks
    S_curr = None
    for i in range(10):
        out_i, S_curr = m(x, memory_state=S_curr if S_curr is not None else None)
    print("\nAfter 10 chunks (10240 tokens):")
    print("State max: %.6f" % S_curr.abs().max().item())
    print("State norm: %.4f" % S_curr.norm(dim=(-2, -1)).mean().item())
    print("Any NaN: %s" % (out_i.isnan().any() or S_curr.isnan().any()).item())

    # Gradient check
    print("\n--- Gradient flow check ---")
    m.train()
    output, S = m(x, memory_state=None)
    loss = output.sum()
    loss.backward()
    has_grad = all(
        p.grad is not None and p.grad.abs().max() > 0
        for name, p in m.named_parameters()
        if 'W_' in name
    )
    print("All projection params have non-zero gradient: %s" % has_grad)
    fusion_grad = m.fusion_gate.grad
    print("fusion_gate gradient max: %.6f" % (fusion_grad.abs().max().item() if fusion_grad is not None else 0.0))


if __name__ == "__main__":
    main()
