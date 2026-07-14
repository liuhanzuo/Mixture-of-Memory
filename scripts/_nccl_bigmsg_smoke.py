import os, time, torch, torch.distributed as dist
lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
# large messages like FSDP all-gather of a MoE layer (~hundreds of MB to GBs)
for mb in (64, 256, 1024, 2048):
    n = mb * 1024 * 1024 // 4  # float32 count for ~mb MB
    t = torch.ones(n, device=f"cuda:{lr}")
    dist.all_reduce(t)
    torch.cuda.synchronize()
    if r == 0:
        print(f"[bigmsg] all_reduce {mb}MB ok val={t[0].item():.0f}", flush=True)
    del t; torch.cuda.empty_cache()
if r == 0:
    print("[bigmsg] ALL PASS", flush=True)
dist.barrier(); dist.destroy_process_group()
