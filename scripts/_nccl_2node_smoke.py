import os, torch, torch.distributed as dist
lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
t = torch.ones(1024, 1024, device=f"cuda:{lr}") * (r + 1)
dist.all_reduce(t)
expected = w * (w + 1) / 2
ok = abs(t[0, 0].item() - expected) < 1e-3
if r == 0:
    print(f"[smoke] world_size={w} all_reduce sum={t[0,0].item():.1f} expected={expected:.1f} -> {'PASS' if ok else 'FAIL'}", flush=True)
dist.barrier()
dist.destroy_process_group()
