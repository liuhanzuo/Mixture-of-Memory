# TRAINER ACTIVE — 2026-06-07 22:05 +08:00

## Local disk-A node (29.162.227.178), 8× H20 — P11 chunk1024 arm-1 (RUNNING, HEALTHY)
- **run**: mem_space_p11_chunk1024_deltarule_normreadout, pid 4061522 (etime ~18h07m), master_port=29794
- **progress**: step 3670/5000, lm=1.32, route_aux=1.09, nf=0, skip=0. top1_sim=0.087 chunk_idx_jaccard=0.80 usage_cov=0.34 (addressing healthy). 8 GPU 78-83% ~82 GiB. ckpt step500-3000 saved. HEALTHY.

## Remote .196, 8× H20 — chunk512_l3recontoken_w0.3 train (RUNNING, HEALTHY)
- **run**: `mem_space_p11_chunk512_l3recontoken_w0.3`, master_port=29793. step 2120/5000 lm=2.30 route_aux=2.32 l3recon=7.00 nf=0. 8 GPU 81-91% ~90 GiB. step500 ckpt on disk A.

## diskB .249, 8× H20 — chunk512_l3recontoken_w1.0 train (RUNNING, HEALTHY)
- **run**: `mem_space_p11_chunk512_l3recontoken_w1.0`, master_port=29794. step 2120/5000 lm=2.20 route_aux=2.09 l3recon=6.12 nf=0 skip=0. 8 GPU 81-91% ~90 GiB. step500 ckpt on disk B. Train continues to 5000 only for lm/recon curves — BABILong already adjudicated ❌ (see below).

## diskB .76, 8× H20 — BABILong evals
- **★ w1.0 step500 eval DONE (21/21 CSV) → SCORED THIS HEARTBEAT ❌**: qa5 0k-32k=67/22/16/8/3/1/0; qa1=77/4/6/8/3/2/1; qa2=43/4/5/3/1/2/3. vs no-aux P11 chunk512 baseline (qa5=82/86/83/64/50/35) → **L3 token-recon aux weight=1.0 catastrophically destroys long-range addressing.** Real result (full n=100, not silent-fail). Locked into RUN_REGISTRY §3.
- **w0.3 step500 eval (driver pid 242122, etime ~33m): RUNNING, HEALTHY** — 17 CSV, all lengths 0k-32k producing output (32k=2/3 filling), no network-unreachable. GPU0/1/2 busy.

## H800 .247(master)/.130.90(worker), 16× H800 DDP — ❌ DEAD (lease reclaimed)
- All H800 IPs dead (port 36000 refused). H800 stable-ladder suspended until fresh lease.

## GPU UTILIZATION: 4 live H20 nodes — 3 training runs (chunk1024 arm-1 + l3recon w0.3/w1.0 sweep) all FULL/HEALTHY + .76 eval node (w1.0 DONE+scored ❌, w0.3 running). H800 dead. HEARTBEAT_OK (busy-healthy; w1.0 eval adjudicated token-recon aux ❌; w0.3 eval will close the sweep next cycle).
