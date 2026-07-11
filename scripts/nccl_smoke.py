import os, torch, torch.distributed as dist, datetime, socket
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
local=int(os.environ.get("LOCAL_RANK","0"))
print(f"[rank{rank}] host={socket.gethostname()} local={local} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=40))
torch.cuda.set_device(local)
t=torch.ones(1024*1024, device="cuda")*(rank+1)  # 4MB tensor
dist.all_reduce(t)
torch.cuda.synchronize()
expected=sum(range(1,world+1))
ok = abs(t[0].item()-expected)<1e-3
print(f"[rank{rank}] all_reduce result={t[0].item()} expected={expected} OK={ok}", flush=True)
dist.barrier()
if rank==0: print("=== NCCL 2-NODE SMOKE PASS ===", flush=True)
dist.destroy_process_group()
