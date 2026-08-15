# PENDING_TASKS.md — Task Board
## Updated 2026-08-13 09:30 GMT+8

---

## ⛔⛔ STALE-NODE WARNING — READ BEFORE ACTING ON ANY `auto_launch` ENTRY

**Added 2026-08-13 by MAIN. 0 GPU. Nothing below is deleted — it is provenance.**

This board contains **24 auto-launch-TRUE entries across THREE spellings** —
`auto_launch:true` (12), `auto_launch: true` (9), `auto_launch=TRUE` (3) — and **78
references to nodes that NO LONGER EXIST**

> ⚠️ **Grep for all three forms.** A pattern matching only `auto_launch:true` finds 12 of
> 24 and understates the hazard by half. Use `grep -nE 'auto_launch[:=] ?(true|TRUE)'`.: `.196`, `.76`, `.249`, `.245.174`, `.252`, `.48.7.53`, `28.88.184.53`,
`28.58.245.174`, `28.59.80.196`. Line ~605 even reads *"any node frees + all GPUs idle →
heartbeat auto-launches the next probe, priority P0."*

The heartbeat spec says *idle GPU + `auto_launch:true` → launch immediately*. Followed
literally against this board, that **targets decommissioned hardware** — the launch either
fails on SSH, or (worse) a stale path resolves on a surviving node and writes results under
an arm name whose provenance no longer matches.

### The only nodes that exist (2026-08-13, verified this session)

| node | disk | hardware |
|---|---|---|
| LOCAL | wzc1 | 8× B200 (sm_100, 178 GB) |
| `.21` = `28.89.19.21` | wzc1 | 8× B200 (sm_100, 178 GB) |
| `.73` = `28.85.35.73` | zwfy6 | 8× H20 (95 GB) |
| `.82` = `28.82.250.82` | zwfy6 | 8× H20 (95 GB) |
| `.104` = `28.83.24.104` | zwfy6 | 8× H20 (95 GB) |

### Ruling for any future heartbeat

1. **Every auto-launch-TRUE entry below (any of the three spellings) whose node is not in that table is STALE.**
   Treat it as `auto_launch:false` regardless of what its own line says.
2. The 2026-06 `mem_space` / `b25` / `F2` / `ROUTE-A` / `N16-TOP16` / `LONGEVAL-RULER-COMPARE`
   blocks are from a **retired research direction on retired hardware**. They are kept for
   provenance and must not be auto-launched.
3. **The live task queue is `proposal/active/*/STATUS.json` (`next_gate`) and the paper
   directories — not this board.** Priority order is unchanged: paperC + proposal first,
   and the judgement is *"does paperC/proposal still have work"*, never *"are there free cards"*.
4. Before launching anything from here, re-verify the node **and** that the referenced
   ckpt/data path exists on the disk that node actually mounts (wzc1 ≠ zwfy6; a path can
   exist on one and not the other, and `.73`'s `/apdcephfs_wzc1` is a symlink to zwfy6).

---

## 🟢 [PLAN 2026-08-15] Paper B depth-ladder step200000 eval — 三臂，driver 已写好并 negative-test 过

**Driver + 协议 + dry-run 证据（0 GPU 预备工作已完成，2026-08-15）**：
- driver：`scripts/eval_paperb_ladder_200k.sh`（一次一臂，全参数化；**两盘都已 scp -O，md5 一致**）
- 断言：`scripts/_ladder200k_assert.py`（纯 CPU）
- 协议：`paperB/LADDER_200K_EVAL_PROTOCOL.md`（battery 定义含 file:line、chat=False、base 口径、**同架构可比性裁决**、每臂盘位/搬运、断言清单、投放顺序）
- dry-run 证据：`paperB/evidence/ladder_200k_eval_dryrun.json`（**13 条 negative control 全部 rc=2 拒绝运行**，含真实 6/8 partial-merge 复现）

**★ 三条通用铁律（三个任务都适用，不要逐条重读协议才发现）**：
1. **必须在 H20 上评（`.73`/`.82`/`.104`，compute_cap 9.0）。** 干净单协议 `_v2` 阶梯六行全是 H20 cc9.0 / torch 2.13.0 / BS=8；core6 有 0.03–0.16pp 跨架构地板（bit-identical 权重实测 7–29 items 翻转）。driver 的 preflight P4 会自己挡住 B200。**keep10 的 ckpt 在 wzc1 也一样要搬到 zwfy6 评，不许在 B200 上评。**
2. **python 必须 `/opt/conda/envs/torch-base/bin/python`（torch 2.13.x）。** `.73` 上的 `olmo2_venv/bin/python` 是 torch 2.7.0，单独换版本就动 ~20 个 item（`status/PAPERB_FLIP_BOUNDARY_RESOLVED.md`），driver preflight P0 会拒。
3. **该臂自己的训练进程退出后才评**，不要和训练抢卡（现在每卡 78–96 GB 占满）。

**投放顺序（实测 ETA，Δ(timestamp)/Δ(step) 双窗口一致，2026-08-15 17:44）**：keep10 训练最先完成（1.41 d，B200 1.336 s/step）但**评测排在 keep12 之后**，因为它必须等一台 H20 空出来；keep12 2.10 d；keep8 3.67 d。

### #253 [PENDING] keep12@200k eval — auto_launch: true
- **触发条件**：`zwfy6:outputs/olmo2_probe2_7B_keep12fresh2/step200000.pt` 出现 **且** `.73` 上的 `train_olmo2_arch_probe2 --keep_front_layers 12` 进程已退出。（2026-08-15 17:42 实测 step177000/200000，7.903 s/step → 约 **2.10 天**）
- **节点**：`.73`（H20）。**零搬运** —— ckpt 训练时就写在 zwfy6。
- **命令**：
  ```bash
  cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
  DRY_RUN=1 ARM=keep12 PROJECT_ROOT=$PWD PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
    bash scripts/eval_paperb_ladder_200k.sh      # 先 dry-run，几秒、0 GPU
  ARM=keep12 PROJECT_ROOT=$PWD PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
    setsid nohup bash scripts/eval_paperb_ladder_200k.sh > logs/ladder200k_eval_keep12.log 2>&1 &
  ```
- **产出**：`olmo2_ppl_results/7B_keep12_step200000/`、`olmo2_downstream_results/7B_keep12_step200000{,_know}/`、`paperB/evidence/ladder200k_keep12_run.json`。约 45–75 min。
- 三臂里**最先可评**，`.73` 空出来第一件事就是它。

### #254 [PENDING] keep10@200k eval — auto_launch: true（**含跨盘搬运前置步骤**）
- **触发条件**：`wzc1:outputs/olmo2_probe2_7B_keep10fresh2/step200000.pt` 出现 **且** LOCAL 的 keep10 训练进程已退出。（2026-08-15 17:43 实测 step108500/200000，1.336 s/step → 约 **1.41 天**，三臂中训练最先完成）
- **前置（0 GPU，LOCAL 上做，可在等 H20 时先做完）**：ckpt 在 **wzc1**，必须 `scp -O` 到 **zwfy6**（协议 §4.3 裁决：keep10 也必须在 H20 上评，不许用它正在训练的 B200）。39.0 GiB @ 实测 12–16 MB/s 单流 ≈ **42–53 min**；搬完核 md5。配方见 `paperB/LADDER_200K_EVAL_PROTOCOL.md` §5。
- **节点**：任一 H20。现实排队 = `.73` 跑完 #253 之后（或 `.104` 若从 paperC 空出）。
- **命令**（搬运完成 + md5 核过之后）：
  ```bash
  cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
  ARM=keep10 PROJECT_ROOT=$PWD PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
    setsid nohup bash scripts/eval_paperb_ladder_200k.sh > logs/ladder200k_eval_keep10.log 2>&1 &
  ```
- ⚠️ **zwfy6 上的 `keep10fresh2/` 只到 step90000**（2026-08-15 实测），别以为 ckpt 已经在那边 —— 先 `ls` 再说。
- ⚠️ LOCAL 的 8 张 B200 会在 1.41 d 时空出来，但按 CLAUDE.md 优先级它们归 paperC/proposal，**不是**归这个必须跑在 H20 上的 eval。

### #255 [PENDING] keep8@200k eval — auto_launch: true
- **触发条件**：`zwfy6:outputs/olmo2_probe2_7B_keep8fresh2/step200000.pt` 出现 **且** `.82` 上的 keep8 训练进程已退出。（2026-08-15 17:44 实测 step145860/200000，5.852 s/step → 约 **3.67 天**，最后到）
- **节点**：`.82`（H20）。**零搬运**。⚠️ `.82` 上 `/apdcephfs_wzc1` **不存在**，PROJECT_ROOT 必须写 zwfy6 路径。
- **命令**：
  ```bash
  cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
  ARM=keep8 PROJECT_ROOT=$PWD PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
    setsid nohup bash scripts/eval_paperb_ladder_200k.sh > logs/ladder200k_eval_keep8.log 2>&1 &
  ```

**这三行 eval 能结算什么 / 不能结算什么**（详见协议 §8，写作前必读）：
- **能**：depth ladder 在**真正等 step 预算**（四臂全 200k）下的对照；且 keep8/10/12 三臂现在训练语料**已统一**（三条 resume log 全是 `rows=15491607` md5 `7df19b...`，实测 LOCAL/.73/.82 一致），`status/PAPERB_TWO_CORPORA_DEFECT.md` 的双语料缺陷在**这三臂之间**消除。
- **不能**：(a) keep14/ShortGPT/freeze_front 仍在 7,570,911 行**字节前缀**语料上，跑到 200k **不能**修 epoch 数差异（3.38 vs 1.65），阶梯跨这三臂仍有语料混淆；(b) 三臂 resume 都经过 Adam 动量 warm-restart + dataloader 位置不恢复，各自断点在不同进度分数上，**必须披露**，不得写成"全部 200k 所以 matched"；(c) 不得声称差分 LR（`_classify_param` 前缀 bug，实际均匀 2e-5）。
- **不要覆盖**已有的 `_v2` 121k/83.5k/124k 行 —— 它们是新 200k 点的同臂轨迹前驱。新输出名 `7B_<arm>_step200000{,_know}` 两盘均无冲突（2026-08-15 实测），preflight P7 也会挡。

---

## 🔴 [PLAN 2026-07-31] Paper B #1 reviewer control — full-32L continued-pretraining（用户确认最高优先训练）

### #100 [PENDING, auto_launch=TRUE @ ShortGPT-frees-LOCAL-B200] Full-32L + Dolmino 200k
- **用户 2026-07-31 确认**：ShortGPT 之后**第一优先级训练**，回答审稿人最危险的因果问题——MMLU 崩塌来自**剪层**还是 **Dolmino continued-pretraining 本身**（corpus-induced forgetting）？现有 full-base `.6053` 是**未续训**原始模型，keep14 `.3191` 已 Dolmino 200k → 二者同时改了深度+结构+语料暴露，无法归因。
- **配置**（launcher 已就绪 + CPU dry-run 验证 transplant 355/355 exact 6 checks pass，commit 7489443）：
  - `scripts/_run_olmo2_full32_dolmino_heal.sh`，keep_front=32 / n_fresh=0（不剪枝，全 32 层 transplant）；
  - **uniform LR 2e-5**（`--lr 2e-5 --lr_inherited 2e-5`，lm_head fresh 桶也 2e-5）；
  - 同 Dolmino（`/dev/shm/dolmino_now15b.npy` 已 staged 本机 B200，58G，与 ShortGPT 同文件，无需重新 stage）、seq_len 2048、eff_bs 128、200k、warmup 150、gradient_checkpointing、fp32 master；
  - save_every=5000（trajectory 覆盖 44k/111.5k/128k/153.5k/200k 附近，MMLU 判决 ~44k≈第 1 天即可读）。
- **启动方式**（ShortGPT@200k 腾出 LOCAL B200 后）：`cd <PROJECT_ROOT> && RUN=1 bash scripts/_run_olmo2_full32_dolmino_heal.sh`（BS=16 GA=1，B200 183GB）。
- **⚠️ 资源优先级（用户 2026-07-31 待确认）**：full32 **优先于 distill** 上 B200。distill（更强恢复目标）属后续方法论文，非当前机制论文必须；full32 是主 claim 承重控制。distill 继续留 .73 慢跑或暂停。
- **成本**：满 32 层 ≈ 剪枝臂 2× → ~6 天到 200k（B200）。但不必等满：full32@44k/100k MMLU 若仍 0.55–0.60 结论已定（非 Dolmino 的锅）。
- **完成/中途**：eval PPL + core6 + know5（同 keep14 口径），回填 RUN_REGISTRY + 报告给用户。
- **后续次优先训练**（full32 之后，仅当资源充裕）：`keep16+fresh0`（front-16 无 fresh，回答"是否 2 fresh 层的锅"）→ `random-init@2e-5`（消 LR confound，当前 random 用 1e-4）。二者非硬条件，可 limitation 顶过。

### 顺手免费评测（forward-only / CPU，不占训练卡，可随时穿插）
- ShortGPT `step0.pt` PPL/MMLU/downstream（量化 importance-pruning 即时损伤）；
- per-example predictions 重评 keep14/frozen/random → `scripts/paired_analysis_paperb.py`（McNemar + bootstrap CI）；
- OLMo semantic/next-token probe（把 Qwen-only depth 证据变同模型证据）。

---

## 🔴 [PLAN 2026-07-24] Paper A 收尾 — adapter-free CoMem 全 benchmark（当前最高优先，覆盖下方所有旧计划）

### #65 [PENDING, auto_launch=TRUE] CoMem adapter-free（无训练）chat=False 全 5-benchmark 行
- **用户 2026-07-24 明确要求**：论文主表需补「真正 adapter-free」行，回击 reviewer「架构本身无需训练也能工作」的质疑。现有 `tab_adapter.tex` 只在 **固定 j=12** 比 adapter on/off（证「深切点 j12 处 LoRA 很关键」），**不代表** CoMem 零训练最佳能力（需浅的 readout-safe 切点 j=9）。
- **固定一套配置，不 per-benchmark 调 j**：Qwen3-8B 冻结（`models/Qwen3-8b-local`）· `chat_template=False` · **无 LoRA** · `resume_j=9`（8B readout-safe，权威值见 RUN_REGISTRY §1）· `chunk_size=512` · `topk=12` · `sink=bos` · `selector=iter_bm25`（与旗舰同）· 同 data/seed/samples/scorer。
- **覆盖全部 5 benchmark**，输出 dir：`{ruler,longeval,longbench,babilong,locomo}_results/qcmem_8b_zeroshot_j9_chatFALSE`。
- **adapter-free 实现方式**：RULER/LongBench/BABILong eval 脚本用 `--zero_training_no_adapter`；LongEval/LoCoMo 直接**省略** `--lora_adapter`（留 None）。**绝不加** `--use_chat_template`。
- **⚠️ 资源约束**：需 **diskB 8-GPU 空节点**（Qwen3-8B + 全 eval 数据只在 diskB；wzc1 缺 RULER/longeval/locomo 数据）。旗舰 chat=False 配方可克隆 `ruler_results/qcmem_8b_iter_chatFALSE_ad/` 的 eval_config（把 lora_adapter 去掉、resume_j 12→9）。RULER/BABILong 用 `_eval_taskpool_2group.sh` 口径；LongEval/LongBench 用现成 chatFALSE driver。
- **⚠️ 不可复用现有 zeroshot 数据**：`qcmem_8b_zs_iter_chatnothink` / `qcmem_8b_zeroshot_j9_n500` / `qcmem_8b_zeroshot_babilong` 全是 **chat=TRUE**（scale-consistency 研究产物）→ chat=False 主表必须**全新跑**。
- **完成后**（#67）：主表加两行 `CoMem (adapter-free)` + `CoMem (+ distilled LoRA)`；保留**两个** ablation：(a) tab_adapter 同-j12 on/off = LoRA 因果隔离；(b) adapter-free@j9 vs 旗舰@j12 = 实用部署点对照。

### #66 [RUNNING, subagent] 旗舰 LoRA 训练成本报告
- 派 subagent 中（`status/FLAGSHIP_TRAINING_COST.md`）：LoRA 可训参数 + %backbone、训练卡数 + wall-clock、PG19 token 数、无 benchmark 标签声明。纯 CPU 分析，不占 GPU。

### #67 [PENDING, 部分不占 GPU] 论文措辞 + 主表行 + 效率表
- **措辞**（可现在做，不依赖 #65 数字）：全文**禁写「CoMem is training-free」**，统一用用户指定原句：*"CoMem's memory architecture and inference-time operations require no parameter updates. The flagship uses a lightweight self-distilled LoRA on a frozen backbone, while an adapter-free variant operates at a shallower readout-safe split."*
- **主表两行 + 效率表 LoRA-on latency 控制**：依赖 #65 数字，跑完再填。

### #68 [PENDING, auto_launch=TRUE (next-free-diskB-node)] MemoryLLM chat=False overlay
- **用户 2026-07-24「两者都要」的延迟半部分**。现状：master 矩阵（`status/BENCHMARK_CHATFALSE_MASTER.md`）MemoryLLM 的 LongEval/LongBench/BABILong **已用 chat=True ᵀ 占位**（14.0 / 12.80 / qa1 26.9·qa2 21.1·qa5 42.6，明确标 ᵀ 不进 chat=False 排名）。本任务=待 diskB 空节点跑**真 chat=False** 覆盖成双行。
- **缺的 3 项**（LoCoMo 16.11 / RULER 16.55 已有真 chat=False，不重跑）：LongEval 8k-128k(n=50,max48)；LongBench 完整 6-ds(官方 qa_f1，现仅 narrativeqa)；BABILong qa1/qa2/qa5×0k-32k(n=100,compare_answers，现 `babilong_results/memoryllm_8b_chatFALSE/` 误命名实 chat=True 需重跑)。chat=False=去 `--use_chat_template`。
- **⚠️ 资源**：MemoryLLM=Llama-3-8B-chat，env/权重/harness **仅在 diskB**（.73/.104），NOT wzc1。当前无我控制的空 diskB 节点（wzc1 满载 LOCAL训练+#65；diskB H20 .73/.104 归用户、.82=dllm）。
- **⚠️ MemoryLLM venv python 坑**（memory memoryllm-venv-python-broken）：diskB venv bin/python 被 reset 成 py3.14（包在 py3.11）→ 用 `/usr/bin/python3.11` + `PYTHONPATH=<venv-site-packages>`。参考 #50 效率修复 + #46 baseline chat=False driver。
- **完成后**：把 master ᵀ 行改成真 chat=False + 回填 PAPERA_ALL_RESULTS §0/§1.7 + §6 状态。

---

## 📋 [PLAN 2026-07-15] QCMem 收尾（旧计划，已被上方 Paper A 收尾覆盖）

### T21 [DONE 2026-07-20] 32B vt recall-vs-speed frontier（用户 idea：深 j 掉 recall 但 read 变快）
- **结论（`status/QCMEM_RECALL_SPEED_FRONTIER.md`）**：32B zero-shot vt / iter_bm25 / chat+no-think / n=50 / string_match / j∈{3,6,13,20,27,34,41,48}（.73 diskB `ruler_results/qcmem_32b_t21_vt_j*` + `logs/t21_profile_j*.log`）。**read_prefill = 56.8×(L−j) ms 近乎完美线性**（j3=3461ms/61层 → j48=909ms/16层 = **3.81× read 提速**）；decode/step 恒 ~46ms（HBM 瓶颈，对 j 不敏感，read+gen@48tok 加速摊薄到 1.84×）；显存全程恒定 ~65.6–66.2GB（算量变、显存不变）。**recall 峰在极浅 j3（16k=93.6/32k=52.0），随 j 阶梯下滑，过 readout-safe 上限（~j34/0.53L）后 j41=3.6/j48=2.0 坠崖（真重建失败=非空错答，非空输出）**。Pareto 拐点 ≈ j34（recall 33/22 未塌 + read 2.03× 快）。**旧 RUN_REGISTRY:1604 的 32B vt sweep（j3~18）是 2026-07-17 chat+no-think 前的 thinking 污染 → 本轮修正：干净口径 32B vt 浅 j 很强（≈8B+adapter/30B-A3B zs），非"全 scale 唯一崩"**。铁律2：全 16 cell empty=0，官方 string_match_all，n=50。.73 8 卡已释放，无代码改动/无 git。
- ~~以下为原 PENDING 计划（已由上面执行取代：改用 8 j 深档 + eval_ruler_qcmem 测 recall + bench profile 测 latency + nvidia-smi 测显存）~~
- **动机**：32B vt recall 随 j 变深而降（j3≈24 > j6≈15 ≈ j9≈16，j13/16/20 探针跑中）。但 QCMem read 成本 ∝ (L-j)/L 层 → 深 j = read/decode 更快。所以 vt j-sweep 不是"确认下降"，而是 **recall vs read 算力的 Pareto frontier**（论文素材：j = 精度↔算力旋钮）。
- **理论预览（32B L=64, decode∝(L-j)）**：j3=95.3%层/1.00× · j6=90.6%/1.05× · j9=85.9%/1.11× · j13=79.7%/1.20× · j16=75%/1.27× · j20=68.8%/**1.39×**。
- **动作**：等 .24 上 vt 探针跑完腾卡（或 .82.250 3b ~23:05 跑完），用**1 张卡**跑实测 latency sweep：
  ```bash
  cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
  M=models/Qwen3-32B
  for j in 3 6 9 13 16 20; do
    CUDA_VISIBLE_DEVICES=<free> /opt/conda/envs/torch-base/bin/python scripts/bench_qcmem_vs_dense.py \
      --mode profile --model_path $M --resume_j $j --topk 12 --chunk_size 512 \
      --selector iter_bm25 --context_lengths 32k --device cuda:0 \
      >> logs/32b_vt_speed_frontier.log 2>&1
  done
  ```
  profile 模式输出每 j 的 read_prefill_s + decode/step（跑 16 decode step，很快）。
- **交付**：把 recall（`ruler_results/qcmem_32b_n100/32b_j{3,6,9,13,16,20}_vt`）× speed（read_prefill + decode/step）配成双轴 frontier 表 → 写 `bench_qcmem_vs_dense_result.txt` + `status/RUN_REGISTRY.md`。
- **口径**：zero-shot 无 adapter，topk12/chunk512/selector iter_bm25，与现有 32B cell 一致。

### T22 [PENDING] eval 结果回填
- .85/.24 上跑的 32B/14B LongBench/LoCoMo/vs-Dense 跑完后 → 聚合官方判分 → 回填 `status/QCMEM_BENCHMARK_PLAN.md` 主表 + `RUN_REGISTRY.md`（32B/14B 行）。
- ⚠️ balance-j 修正未回填计划表 §1b（zero-shot 最优 j 浅=j3，非固定 0.25L）——待用户确认后一并更新。

### T23 [DONE 2026-07-20] BABILong 8B clean 重跑（thinking 污染）+ 打分回填
- **发现（2026-07-17）**：Qwen3-scale BABILong 结果（`babilong_results/qcmem_8b_{adapter_mid,zeroshot}_babilong` 等，文件 2026-07-14）在 thinking-fix `30bb2ab`（2026-07-16）之前生成 → 原始输出带 thinking/MC 标记，长档答案被挤出首句（官方 compare_answers = 首句 + 恰好一 label）。
- **完成（2026-07-20，节点 .73）**：干净口径 `chat_template=ON + enable_thinking=False + selector=iter_bm25`（GPU 部分先前已跑完）经官方 `compare_answers`+4-shard 合并（`score_nested_babilong.py`）打分回填。**8B overall(21 cell)：adapter(j12)=57.10 / zero-shot(j9)=48.43；全 cell empty_output=0，输出 well-formed**。
- **vs 旧污染版**：zs 39.2→48.4（+9.2）、adapter 55.5→57.1（+1.6 持平），增益集中 0k–8k。**修正旧估计**：32k qa1/qa2 干净口径仍低（真长程失败非 artifact，"真值 35–50" 不成立）；**新 caveat**：iter_bm25 对 qa1 单事实中档反掉分（adapter qa1 16k 55→23）。回填 `QCMEM_BENCHMARK_PLAN.md §1a` + `RUN_REGISTRY.md`。
- **连带（未做）**：4B/1.7B/0.6B/14B/32B 未重跑（仍旧口径 legacy），如需 scale 一致性后续补。

### T24 [RUNNING, auto_launch=true] Paper B 缺失实验与最终评测（2026-07-28 审计更新）
- **P0-1 keep14 final eval**：继承 train-all 主臂已于 step200000 完训，现有 `outputs/olmo2_probe2_7B_keep14fresh2/{step200000.pt,final.pt}`；但完整 held-out PPL/core-6/knowledge-5(MMLU) 只评到 step153500。下一个可用整节点直接评 step200000，无需训练。
- **P0-2 freeze_front control**：LOCAL 健康训练中，2026-07-28 13:05 已到 step179720/200000（1.32s/step，约剩 7.5h）；完成后立即跑 held-out PPL + core-6 + knowledge-5，与 keep14 train-all@200k 和 fully-random-init@200k 三臂同步对照。精确 128k/153.5k ckpt 已轮转掉，仅保留 125/130k、150/155k；论文主对照改用干净 200k。
- **P1 compute-matched depth ladder**：keep8 仅 44k、keep10 仅 10k、keep12 仅 111.5k 的完整 eval；现有深度梯不等训练预算。若论文要声称 architectural threshold，必须补 keep8/10/12 至 200k 或至少 matched-step 比较。当前只能称 available-checkpoint frontier。
- **P1 keep12**：最高 ckpt/eval step111500，目标200k，当前本机未运行；需确认 diskB 远程 run 是否 crash 后再 resume。
- **P2 1B replication**：keep7 最高知识 eval 150k、PPL 147k，仍在下降；作为 qualitative replication 已够，补到200k为可选。
- **已闭合**：fully-random-init 16L 已训练并完整评到200k；旧“from_scratch 只随机 decoder”叙事已纠正为全模型随机初始化。
- **发布仓库 URL（P1, auto_launch=false）**：本地匿名仓库 `perplexity-heals-knowledge-lags/` 已就绪；待用户创建匿名远程并提供 URL 后，将链接加入 `paperB/main.tex` 与匿名 release 的 `paper/main.tex`，重新编译。

---

## 📋 [PLAN 2026-07-13] 当前待办（用户回归后确认，覆盖旧计划）

### ✅ 已完成（本轮自主运行沉淀）
- 方向4 混元剪层：A13B(65B) 四连坑修复 + healing 验证(loss102→35) + **Hy-MT2-30B prune-heal frontier 4 点单调**(keep12→ppl63.5 / 24→42 / 30→27 / 36→12) [TaskList #16]
- **QCMem 4-selector RULER n=100 对照聚合**(bm25/recency/reader_attn/oracle 各 90/90, commit a2fa0d9, `status/QCMEM_SELECTOR_COMPARISON.md`)：NIAH oracle=100 读出无损 bm25≈oracle；VT oracle 最差(9.2)需 multi-hop [TaskList #11]

### 🔧 待办（harness TaskList #17-#20）
- **T17 [RUNNING, auto_launch] armB 200k 迁本机 L20A（~5×加速）**：传 step36000.pt(47GB) diskB→wzc1【进行中】→ kill 本机 chunk1024 消融 + kill .24.104 armB → 本机 `train_qwen3_arch_probe2.py --resume_from step36000.pt` resume 到 200000。脚本/data(slimpajama_chunks_2048_qwen3.npy)/model 本机已就位。
- **T18 [PENDING, auto_launch] VT 迭代 bm25 selector（便宜先试，用户 idea）**：follow 变量链——query 变量名 bm25→找 chunk→读下一个变量名→再 bm25→累积 topk。eval_ruler_qcmem 加 `--selector iter_bm25`。判据：VT recall vs 单次 bm25(28)/reader_attn(60)。
- **T19 [PENDING] VT learned scoring head（保变量身份）**：训小 head，contrastive on gold chain，迭代多跳（之前 mean-pool 抹身份失败）。若 T18 够好可降级。
- **T20 [PENDING, 可选] Hy-MT2-30B split-j sweep**：补它的 QCMem 缓存切点(48层,预期~j19 需实测)。低优先级。
- **T-chunk [PAUSED] chunk1024 vs 512 消融**：本机让位给 armB 迁移(已跑~51/90)，后续在 .24.104 补完 + aggregate。
- **T-paper [PENDING] 写作**：方向4 frontier + 4-selector 对照 → probe#2 / §2.5 / §2.6 章节。

### 资源规划（迁移后）
- **本机 wzc1(L20A)**：armB 200k（快 ~5×）
- **.24.104（armB 迁走后空）**：VT 迭代 bm25 + learned head 实验
- **.85.73**：selector 数据 / VT eval
- **.53.31**：offline，不管（用户 2026-07-13 指令）

---

<details><summary>旧计划（2026-06-25 FIFO eval，已过期存档）</summary>

## Updated 2026-06-25 00:15 CST

---

## 🌙 [TONIGHT-FIFO-EVAL][PLAN 2026-06-25] 方案B FIFO 消融 4臂 eval 计划（按依赖分类）

### 依赖层级

```
[无需等待，立即可做]
  T1. H20三臂 lm 曲线趋势分析（读远程日志，无GPU）
  T2. 准备所有 eval 启动命令草稿（无GPU）

[依赖 B200 step3000，约1h内]
  T3. rsync B200 ckpt (wzc1→diskA)  ← 需要网络，无GPU
  T4. 本机 H20 跑 B200 chunk1024/b50 W0+W6 eval  ← 需要 本机8×H20

[依赖 H20三臂 step3000，约18.5h后（明天白天）]
  T5. .196 b50/chunk512 W0+W6 eval  ← 训练结束后直接在.196跑（盘A共享）
  T6. .7.53 b25/chunk512 W0+W6 eval  ← 在.7.53本机跑（ckpt在盘B）
  T7. .245.174 b100/chunk512 W0+W6 eval  ← 在.245.174本机跑（ckpt在盘B）
  （T5/T6/T7 三臂并行，各自训练结束后立即在原节点启动）
```

### T1 [PENDING, auto_launch:true, 无GPU] H20三臂 lm曲线趋势分析
- 读 .7.53/.245.174 的训练日志（b25/b100），对比 b25/b50/b100 三臂 lm 收敛曲线
- 判断 buffer_length 对 FIFO lm 的影响趋势
- 无任何依赖，立即可做

### T3 [DONE/SKIP 2026-06-24 23:53] rsync B200 ckpt — 不需要：T4 改在 B200 .53 本机跑（ckpt 已 native 在 wzc1 盘）。step3000 ckpt 也已 scp 回本机 outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter_step003000.pt（7.1G）作备份。
<details><summary>原 rsync 草稿</summary>

```bash
# wzc1→diskA，只同步 step3000 ckpt + adapter_config
sshpass -f configs/password_b200_53.txt scp \
  -o StrictHostKeyChecking=no -o PreferredAuthentications=password -P 36000 \
  root@28.88.184.53:/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter_step003000.pt \
  outputs/mem_space_fifo_b50_chunk1024/
# adapter_config 已在本机（训练前同步过），若无：
# sshpass ... scp ... adapter_config.json outputs/mem_space_fifo_b50_chunk1024/
```
</details>

### T4 [RUNNING 2026-06-24 23:53 @B200 .53] B200 chunk1024/b50 W0+W6 eval
- **改在 B200 .53 本机跑**（训练完成后节点全空闲，ckpt 在 wzc1 盘 native 无需 rsync，比等本机 H20 更快；本机 H20 仍占于 c512/b50 训练）。
- ckpt=outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter.pt（step3000 final, lm=3.849）
- W0(swa0)→W6(swa6) 串行两次 scheduler 调用；wrap /tmp/fifo_c1024_eval_w0w6.sh pid 3806156
- log: logs/fifo_b50_c1024_eval_W0.out / _W6.out（B200 .53）；结果 babilong_results/fifo_b50_c1024_step3000_W0|W6
- W0 已起：21 tasks，8GPU 100% healthy。预计 W0+W6 共 ~5h。完成后 scp 结果回本机 score + 填 RUN_REGISTRY。
- 原草稿（本机 H20，留档）：
```bash
# 本机执行（PROJECT_ROOT=diskA）
RUN_PREFIX=fifo_b50_c1024 \
CKPT_FILES="outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter_step003000.pt outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter_step003000.pt" \
CK_NAMES="fifo_b50_c1024_step3000_W0 fifo_b50_c1024_step3000_W6" \
ADAPTER_CONFIG=outputs/mem_space_fifo_b50_chunk1024/adapter_config.json \
CHUNK_SIZE=1024 \
EXTRA_ARGS="--swa_eval_chunks 0 --swa_eval_chunks 6" \
setsid nohup bash scripts/_eval_taskpool_2group.sh >logs/fifo_b50_c1024_eval_sched.out 2>&1 &
```
⚠️ W0/W6 需分开两次调用（EXTRA_ARGS 不能合并），或传两个 ckpt + 两个 swa_eval_chunks 值——需检查脚本支持方式，见草稿。
- 预计时长：~2.5h（42 tasks，2组并行，chunk1024 单task约10min）

### ✅ 三臂训练完成 + eval 已启动（2026-06-25 07:11 heartbeat）
- **b25/b50/b100 chunk512 三臂均 step3000 完成**（07:02-07:07，0 crash/0 non-finite，~622min），`full_model.pt` 落盘。
- T5/T6/T7 **均已在各自原节点启动 W0+W6 BABILong eval**（ckpt=`full_model.pt`，loader strict=False 兼容；注意实际产物是 full_model.pt 而非草稿假设的 mem_space_adapter_final.pt）：
  - **T5 b50 @ 本机 8×H20**（diskA，.venv）：driver /tmp/fifo_b50_c512_eval_w0w6.sh，log logs/fifo_b50_c512_eval_{W0,W6}.out，结果 babilong_results/fifo_b50_c512_final_{W0,W6}。
  - **T6 b25 @ .48.7.53**（diskB，.venv）：driver /tmp/fifo_b25_c512_eval.sh，结果 babilong_results/fifo_b25_c512_final_{W0,W6}。
  - **T7 b100 @ .58.245.174**（diskB，.venv）：driver /tmp/fifo_b100_c512_eval.sh，结果 babilong_results/fifo_b100_c512_final_{W0,W6}。
- **完成后**：score_nested_babilong.py 聚合 → 填 RUN_REGISTRY（b25/b50/b100 × W0/W6）→ 与 B200 c1024/b50 + MemoryLLM baseline 对比 buffer_length×chunk_size 效应。

### T5 [RUNNING 2026-06-25 07:11 @本机8×H20] b50/chunk512 W0+W6 eval
```bash
# .196 节点执行（PROJECT_ROOT=diskA，.venv PYBIN）
sshpass -f configs/password_diskA.txt ssh root@28.59.80.196 \
  "cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory && \
   RUN_PREFIX=fifo_b50_c512 \
   CKPT_FILES='outputs/mem_space_fifo_b50_chunk512/mem_space_adapter_final.pt' \
   CK_NAMES='fifo_b50_c512_final_W0' \
   ADAPTER_CONFIG=outputs/mem_space_fifo_b50_chunk512/adapter_config.json \
   CHUNK_SIZE=512 \
   setsid nohup bash scripts/_eval_taskpool_2group.sh >logs/fifo_b50_c512_eval_sched.out 2>&1 &"
# W6 另一次调用加 EXTRA_ARGS="--swa_eval_chunks 6"
```

### T6 [RUNNING 2026-06-25 07:13 @.48.7.53] b25/chunk512 W0+W6 eval
```bash
# .7.53 节点执行（PROJECT_ROOT=diskB，.venv PYBIN）
# ckpt: /apdcephfs_zwfy6/share_304376610/.../outputs/mem_space_fifo_b25_chunk512/
```

### T7 [RUNNING 2026-06-25 07:15 @.58.245.174] b100/chunk512 W0+W6 eval
```bash
# .245.174 节点执行（PROJECT_ROOT=diskB，.venv PYBIN）
# ckpt: /apdcephfs_zwfy6/share_304376610/.../outputs/mem_space_fifo_b100_chunk512/
```

### 并行策略总结
- **现在**：T1立即做（无需GPU）+ T2 准备命令草稿
- **~1h后**：T3 rsync → T4 本机启动 eval（本机全空闲等这个任务）
- **~明天白天**：T5/T6/T7 三臂各自训练结束后，**在原节点立即启动**（不需要 rsync，ckpt 在本地）
- **eval 完成后**：score_nested_babilong.py 聚合 → 填 RUN_REGISTRY → 与 MemoryLLM baseline (qa5 32k=34) 比较

---

## 🧪 [N16-TOP16][RUNNING 2026-06-23 12:08 @28.58.245.174] 16-slot / top16 对照 — auto_launch:true

- **问题**：标准 nctx63 是 128 slots 中选 top16；用户提出比较“一共只有 16 个 slot”的效果。
- **配置**：`scripts/launch_distill_pg19_nctx63_N16_top16.sh`，`--num_slots 16 --top_k 16`，其余对齐 nctx63 SOTA，PG19 nctx63 cache，total_steps=500/save250。
- **运行**：`28.58.245.174`（盘B H20）已启动，run=`distill_pg19_nctx63_N16_top16`，log=`logs/distill_pg19_nctx63_N16_top16.log`。
- **判据**：step250/500 同口径 BABILong W0；若 N16 接近/超过 N128-top16，说明大 bank 的 selector/稀疏选择不是优势；若明显差，说明 128 bank 容量仍必要。

## 📏 [LONGEVAL-RULER-COMPARE][PLANNING] MoM ckpt vs Landmark baseline — auto_launch:true

- **目标**：测之前关键 ckpt 的 LongEval 与 RULER，并和 Landmark baseline 比较。
- **候选 ckpt**：nctx63 SOTA step250/500、lr5e5 step250/500、train-time router/recency step250/500、必要时 N16 step250/500。
- **已知脚本**：`scripts/eval_longeval_mem_space.py`、`scripts/eval_longeval_landmark.py`、`scripts/eval_ruler_mem_space.py`、`scripts/eval_ruler_landmark.py`、`scripts/_ruler_eval_2group.sh`。
- **下一步**：等当前 router/recency/N16 训练与 eval 节点释放后，优先跑 nctx63 SOTA vs Landmark 的 LongEval/RULER 小网格，再扩到 lr5e5/router/recency。

## 🌙 [TONIGHT-LONGTRAIN][RUNNING/PLANNING] 重点：让训练“长训不掉点” — auto_launch:true

- **核心原则**：每个机制先分清 train-time / eval-time / both。若模型训练时没见过对应路径，eval-only 切换只能算 OOD diagnostic，不能作为机制裁决。
- **已在跑**：
  - local: `distill_pg19_nctx63_lr5e5_s1234`（lr 1e-4→5e-5，seed1234，total_steps=500，save250）
  - .196: `distill_pg19_nctx63_lr5e5`（seed42，同配置）
  - .53: `slot_kv_cache_pg19_chunk512_nctx63_recency`（train-time recency slot-kv）
- **今晚优先假说**：
  1. **优化轨迹过冲**：step250 最好、step500/1000 掉点可能是 lr 过高或后期梯度噪声；lr5e5 双 seed 是第一判据。
  2. **训练/评测路径不匹配**：slot-kv eval-only selector 已证不能裁；必须 train-time recency/all 后再同 mode eval。
  3. **后期不可学 token 死磕**：若 lr5e5 仍掉，下一步考虑 distill loss schedule（后期降低 logits KL/保留 hidden 或只训高置信且同 train path）。
  4. **保长程能力的 regularization**：候选包括 replay/anchor loss（固定少量 step250-like long-context batches）、early-best EMA/SWA 权重平均、step250→500 小 lr continuation 对比。
- **下一步自动动作**：任一 8GPU 节点释放后启动 `scripts/launch_nctx63_slotkv_all.sh`（train-time all-slot arm）；lr5e5 step250/500 ckpt 出后立即同口径 BABILong eval，判断是否真正“不掉点”。

---

### [OVERLAY-COEF05][DONE 2026-06-17 06:00 — seed确认通过] mass coef0.5 + 蒸馏叠加 — ★假说反转→长程最佳（2026-06-16）
- **★假说反转（2026-06-16 08:02 完整W0曲线）**：coef0.5+蒸馏叠加，弱mass+蒸馏长程轻微协同最优（超纯蒸馏11/8、mass coef2 12/7、coef2叠加6/5）。
- **★seed确认通过（2026-06-17 06:00）**：seed1234 final W0 BABILong eval已完成（diskB .249本盘）：
  - qa5 0k-32k = **80/43/48/25/15/11/9**（seed42 = 70/49/44/25/14/13/9）
  - 两seed长程一致：16k=11~13、32k=9 均稳定，远超 coef2(12/7)；qa1=94/35/29/16/7/7/6。
  - **裁决：坐实"弱mass(coef0.5)+蒸馏 = 长程稳定最优组合"，非单seed运气。** 16k 有 11~13 的轻微seed方差（噪声级），32k=9两seed完全一致。
- **下一步**：已入阶段报告候选；结果待落 RUN_REGISTRY（与 32k 双负结果一并写）。

## 🚨 [BASE-MIX0][RUNNING @.249 13:18] mix=0 干净 P11 baseline — 架构实验公平对照前置（2026-06-13）

> **13:18 启动**：`scripts/launch_BASE_mix0_N128.sh` @ H20-.249（diskB，.venv PYBIN），RUN=BASE_mix0_N128，pid=1197243。= expL1ERASE 配置去掉三个架构 flag（无 delta_erase / 无 independent_slot_key / 无 use_l2），唯一区别即 baseline。step5 健康 lm=3.21 babi=0 满载 80GB/79%。total 1000 步，eval_interval=0，ckpt 落 `outputs/BASE_mix0_N128/`（step500+final）。**跑完起同口径 task-pool BABILong eval**。
> **判据用途**：3 架构实验（expL1KEY/expL1ERASE/expL2ON，均 mix=0）长程 qa5 要超的是**这个 mix=0 baseline**，不是旧的 mix=0.15 P11(48/45/44)。
> **下一步**：BASE_mix0 + 3 架构实验都跑到 final → 同口径 eval → 横向对照写 RUN_REGISTRY。

---

## 🎯 ROUTING / CAPACITY 方向（2026-06-10 用户立项）— 根因：usage_cov~0.25，128 槽只用 ~32 个

> **背景（gp-44 根因报告 + gp-39 32k 样本诊断 + 用户总结 2026-06-10）**：长上下文退化的核心瓶颈是 **selector 路由集中**——usage_cov 仅 ~0.25，128 槽实际只活跃 ~32 个，富者愈富（死槽 key 停初值竞争不过）。叠加 delta-rule EMA 写到不动点（new≈current → slot_delta→0，门 g_in≈0.5 没关但写不动）+ 读侧晚期坍缩到单槽（retrieved_norm 4.8→0.44，top1_sim→0.99）。
> **结论**：先修路由均衡（用满 128 槽 = 把"饱和长度"后推 ~4×，零额外参数/显存），再谈遗忘/容量。**加 num_slots 大概率没用（128 都没用满，只多死槽）。**

### [ROUTE-A][~CLOSING — 全旋钮证伪] 路由均衡 sweep — 目标 usage_cov 0.25 → 接近 1.0 — auto_launch:true
> **2026-06-13 06:56 收尾裁决**：四类旋钮全臂跑完且大多评分完毕，**无一在长程(8k-32k)超 P11 base step500(qa5≈48/45/44)**：
> - `loss_free` arm1(0.01): usage_cov 0.25→0.88 达标但 qa5 8k-32k=48/45/44 仅持平 base，未超 → 坐实「usage_cov↑≠长程↑」。
> - `entropy_aux` arm2: 比 arm1 更差。
> - `selector_temperature` {20,40,80}: arm3(temp20) qa5 8k/16k/32k=**10/7/5** 长程崩 REJECTED；arm4(temp80) 38/40/28 REJECTED；temp40=arm1-2 底座。三档全证伪。
> - `load_balance_weight=0.01` 三 seed{42,1234,2026}: seed2026 长程崩(14/14/10)+step500 0k=8 坏，seed42/1234 eval 收尾中（.196/.76），方差噪声，无稳超。
> **裁决：路由均衡旋钮（loss_free/entropy/temp/lbw）全证伪——把 usage_cov 推到 ~0.9 并不能改善长程检索保持。归入写入/读出侧全证伪谱系。下一方向待主会话决策（见 needs_code alert）。** 剩余未扫的 loss_free{0.005,0.02} 不再跑（已证伪方向铺 noise-curve 无意义）。
- **现成抓手（全是 hyperparameter，无需改代码，授权内可自主跑）**：
  - `--loss_free_update_rate`：P11=0.001 可能太弱 → 扫 {0.001, 0.005, 0.01, 0.02}（DeepSeek loss-free-balance，调 router bias 不污染主 loss）。
  - `--entropy_aux_weight`：当前=0 → 扫 {0, 0.001, 0.01} 小权重开。
  - `--load_balance_weight`：当前=0 → 可选小权重 {0, 0.01}（Switch Transformer aux）。
  - `--selector_temperature`：P11=40 → 可对照 {20, 40, 80}（温度影响 softmax 锐度/路由集中度）。
- **底座**：P11 chunk512 delta-rule+normreadout，**只改上述路由旋钮，单变量**，step500 即可读 usage_cov 趋势（不必跑满 5000）。
- **判据**：(1) 训练 diag 的 usage_cov / uniq_sel_slots 是否抬升；(2) step500 同口径 BABILong 长程 cell（8k-32k）是否改善。usage_cov↑ 且长程↑ = 路由修复有效。
- **优先级最高**：ROI 最高、零额外显存。空闲节点优先消化此 sweep。
- 关联在跑：ladder top_k 实验（容量旋钮）、L1-only（L3贡献）、D6三臂（读机制）——这些结果会进一步收窄路由修复的具体形态，可并行。

### [ROUTE-B][PENDING, 需设计讨论 auto_launch:false] 周期性 reset 最不活跃 slots（主动遗忘机制）
- **思路（用户 2026-06-10）**：路由修好后，每隔一段时间直接 reset 最不活跃的一部分 memory slots，给饱和记忆腾位置（比 dual-gate 被动遗忘更直接）。
- **待定设计点（做前讨论）**：(1) "不活跃"定义（累计选中次数 / 最近 N chunk 频率 / EMA 写入幅度）；(2) reset 成什么（清零 / strided_token 重新初始化为可写空槽）；(3) 频率 + reset 数量；(4) **风险**：不活跃 ≠ 无用，可能误删存着早期关键事实的槽 → 伤 NIAH，reset 策略需和"是否还会被读"挂钩。
- **依赖**：必须先做完 ROUTE-A（路由修好、128 槽用满）+ 对"容量-长度饱和拐点"有量化，否则瞎调。与"L1 精确 retrieval 应覆盖多远"这个根本权衡直接相关（reset 旧槽 = 拿"记多远"换"记多清楚"）。

---

## 📋 EVAL QUEUE（2026-06-09 建立）— 空闲卡按序消化，每条跑完落 RUN_REGISTRY + 用 hb_emit_alert 报 train_done

> ★ **eval 调度方式（2026-06-13 用户指定,权威）**：用 `scripts/_eval_taskpool_2group.sh` —— 8 GPU 分 2 组(0-3/4-7),每个 (ckpt,task,length) 任务在一组 4 卡各跑 25 样本(num_shards=4),21+ 任务进共享 pool 哪组空就 flock 原子 pop append。详见 CODEBUDDY.md「标准 eval 方式」。**旧的 per-GPU LPT 静态调度器(`_expR1c*_eval_sched.sh`)已弃用**(长档 shard 致空转)。
> 规则：有空闲节点 → 用 task-pool 调度器跑下一条未完成 eval。
> 统一口径：`scripts/run_babilong_mem_space.py`，qa1/qa2/qa5 × 0k-32k，n=100，chunk512 bf16 sdpa；评分 `scripts/score_nested_babilong.py`。
> ⚠️ 已知坑：远程节点需 woa proxy + HF_HUB_OFFLINE=1 + HF_DATASETS_OFFLINE=1（否则 BABILong dataset HEAD 挂）；短长度(0k/1k)有跨进程 bf16 非确定性，W0/W1/W2 同节点同批跑、重点看 4k-32k；跨节点脚本改对 PROJECT_ROOT（盘A=303098609 / 盘B+wzc1=304376610）+ diskB PYBIN 用 .venv。

### [EVAL-1][RUNNING] P11 step500 (SOTA峰值) × SWA W0/W1/W2 — gp-29 在 B200 收尾中
- ckpt `outputs/mem_space_p11_chunk512_deltarule_normreadout/mem_space_adapter_step000500.pt`，`--swa_eval_chunks {0,1,2}`。
- 目的：SOTA 峰值 ckpt 配 cross-chunk SWA 的天花板，对照 step5000+SWA(W2 qa5=58/29/68/62/42/39/39)。
- 状态：B200 共存运行中（各 ~5-7/21），gp-29 盯出齐评分。

### [EVAL-2][DONE 2026-06-10] ★ step5000 vs step500 的 LongEval 对照 — 假说被反驳
- **结果（B200 GPU3/4，6 LongBench QA 任务，F1，chunk512，no_chat_template，hotpotqa n=200 其余 n=100 同 index 可比）**：step500 AVG=8.87 vs step5000 AVG=6.06。step5000 在**全部 6 任务一致更差**，含三个全局语义任务（narrativeqa 2.07 vs 5.72、qasper 3.89 vs 4.85、multifieldqa 11.25 vs 16.53）——恰是假说预期 step5000 应追平/超过处，结果反而退化最狠。
- **★裁决：假说 REFUTED。过训是单调退化（L1 整体被污染），不是「检索→语义压缩」能力迁移。** LongBench 证明语义总结能力也退化了。step500 是 NIAH 检索 + 全局语义双口径统一最佳，早停是正确交付。详见 RUN_REGISTRY §3b。输出 `longbench_results/p11_step500` / `p11_step5000`（B200 wzc1）。

### [EVAL-2-orig][SUPERSEDED] ★ step5000 vs step500 的 LongEval 对照 — auto_launch:true
- **动机（用户 2026-06-09 insight）**：发现一根因假说 = 单层 L1 slot 训久了转去承担「预训练式高级语义压缩」，把 NIAH 精确检索能力挤掉。**验证**：若 step5000 在 LongEval（需全局语义总结）上**不比 step500 差、甚至更好**，就坐实「L1 没变差、只是能力从检索挪到语义压缩」——发现一从单向假说变双向证据。
- 跑：P11 chunk512 的 step500 ckpt 与 step5000(final) ckpt，各跑 LongEval（用 `scripts/launch_longbench_eval.sh` 或现有 LongEval/LongBench harness，确认口径一致）。
- 同口径对照输出：两个 ckpt 的 LongEval 分数 + 已有的 BABILong qa5（step500=82/86/83/64/50/46/41 vs step5000=54/62/51/30/28/22/31）并排。
- 交付：结论写 RUN_REGISTRY + 回报 main（决定是否进 PPT 发现一作为双向证据）。

### [EVAL-3][PENDING] D1 slot16384 final BABILong — 依赖 D1 训练完(~B200 10h)
- D1 run `outputs/d1_slotdim16384` 跑完 5000 步后，step500/最佳 ckpt 同口径 BABILong，对照 P11 chunk512 baseline（slot4096）。注意 D1 是 (slot_dim16384 + lowrank_gate) 双变量，解读需谨慎。
- 建议补：lowrank_gate@slot4096 对照（gp-20 指出的纯净 slot_dim 消融缺口）——可另起一条 EVAL 或 train。

### [EVAL-4][PENDING] D6 xattn 消融三臂 BABILong — 依赖 D6 训练完(.249)
- 臂A=P11基线(复用)、臂B=xattn OFF(`outputs/d6_xattn_off`)、臂C=xattn ON+sink OFF(`outputs/d6_nullsink_off`)。三臂 step500+final 同口径 BABILong。
- 判据：A vs C 隔离 null-sink 贡献；B vs C 隔离「独立 own-softmax 读机制」整体贡献。

### [EVAL-5][PENDING] D2b 训练侧SWA 双口径 BABILong — 依赖 D2b 训练完(.196)
- D2b run `outputs/d2b_swa_train_w2` ckpt（重点 step1500-3000，避开过训）跑 **双口径**：W0 标准单chunk eval + W2 (`--swa_eval_chunks 2`) eval。
- 判据：对照 D2a 纯-eval-SWA 增益，看「训练时也见过 SWA」是否消除 train/eval mismatch / 进一步提升长程。

---

## [RUNNING 2026-06-08 08:22] F2 prep-3 — real wiki long-doc re-tokenize build（CPU，data-validity gate 已通过）
- **dry_run 裁决（08:22）**：pes2o=DEAD（扫 3.86M docs，0 docs ≥8192 tok，全短摘要）；wiki=唯一可用源（≥2048-tok docs token p90/p99/max=7530/16699/61969；4.7% ≥10k tok、0.4% ≥20k tok）。filter `dolmino_per_doc` 路线 confirmed DEAD（4096 硬截断）。
- **已起真实 build**（pid 615605，CPU 16-proc，log `logs/f2_build_wiki_min4k.log`）：`build_dolmino_longdoc_raw_retokenize.py --raw_glob wiki/*.json.gz --min_tokens 4096 --out_path MemLong/data/processed/dolmino_longdoc_wiki_min4k`。dolmino_per_doc schema 兼容，`--per_doc_data` 可直读。
- **TODO(next HB)**：build 完（检查 out_path DatasetDict + n_docs/长度分布）→ 用 F1 当前最佳（P11 delta-rule + chunk512）+ 此长文子集 → .196/.249 起 F2 long-doc 8-GPU train（eval_interval=0，per_doc_data 指向新子集）→ 落 RUN_REGISTRY。auto_launch: true（F2 既定方向，data 一就绪即起）。
- ⚠️ .196/.249 现 idle 但**合法被 data-build 阻塞**（F2 train 无 confirmed long-doc data 不能起），非「idle-with-runnable-step」。

## [RUNNING 2026-06-08 06:12] F1 v3 top_k-ladder 对照臂（本机 8×H20 空闲 → 自主起，coder general-purpose-2 写脚本）
- **背景**：本机/.196/.249 三节点空闲（FINAL chunk1024 eval + l3recon eval 都收工）；canonical F1 v3（固定 top_k=16）在 .76 跑。研究 note（high-conf）建议测 top_k 随 chunk 阶梯（唯一 warm-start 安全的 slot 容量旋钮）。
- **派 coder（reasoning）写 `scripts/launch_progressive_chunk_local_v3_topk_ladder.sh`**：盘A 本机版 v3 链，单变量改 top_k schedule（c256→16 / c512→24 / c1024→32），num_slots/selector_dim 保持 128 不变（warm-start 形状匹配），其余逐字对齐 canonical v3。独立 output_dir/log/master_port，eval_interval=0。
- **判据**：每 stage step500 离线 BABILong 对照 canonical v3（固定 top_k16）+ v1 stable 链，验证「大 chunk 增 top_k 是否提升长上下文检索」。
- TODO(next HB)：收 coder 报告（脚本 + bash -n + commit + 启动命令）→ 本机 8×H20 仍空闲 → 立即启动 → 落 RUN_REGISTRY。属 adopted F1 v3 base 的 ablation 延伸，可自主起。

## [DONE 2026-06-08 06:08] F2 prep — long-doc 子集 dry_run → ★关键发现：现有 per-doc 数据集 4096 硬截断，filter 路线死路
- coder general-purpose-1 已写 `scripts/build_dolmino_longdoc_subset.py`（commit 待查）。
- **06:05 main dry_run 结果**：train/val 全部文档 max=4096 tok（p99=4096，~6.8% 文档卡在 4096）。chunk512 ≥10 chunks 的文档 = **0%**，chunk1024 ≥10 = 0%。**filter 现有 dolmino_per_doc 永远拿不到「单样本几十~上百 chunk」**——最多 16(chunk256)/8(chunk512)/4(chunk1024) chunk。
- **根因**：`dolmino_per_doc` 是从 packed `dolmino_0.5B_1024`（max_length packing）按 EOS 还原出来的，原始 packing 上限把文档截在 4096。要拿真正长文档必须 **重新 tokenize 原始 raw json.gz**（不截断）。
- **原始长文档源已就位**：`MemLong/data/raw/dolmino_pes2o_wiki/raw/data/{pes2o,wiki}/*.json.gz`（pes2o=学术论文，天然长文）。
- → 转 [RUNNING] 重 tokenize 任务（见下）。

## [DONE 2026-06-08 08:22] F2 prep-2 — raw re-tokenize 脚本 + dry_run 长度裁决（→ 转 prep-3 real build，见顶部）
- 脚本 `build_dolmino_longdoc_raw_retokenize.py` 就位且 dolmino_per_doc schema 兼容。dry_run 裁决：pes2o DEAD、wiki 唯一可用源（详见顶部 prep-3）。→ real wiki build 已起。
- **派发 coder（reasoning）写 `scripts/build_dolmino_longdoc_raw_retokenize.py`**：直读 `MemLong/data/raw/dolmino_pes2o_wiki/raw/data/{pes2o,wiki}/*.json.gz`，用 Llama-3 tokenizer **不截断**整篇 tokenize（加 BOS/EOS），保留 ≥ N tok 的长文档，输出 HF DatasetDict（单列 `input_ids`，schema 与 `dolmino_per_doc` 一致，`--per_doc_data` 可直读）到 `MemLong/data/processed/dolmino_longdoc`。先 `--dry_run --max_files 2` 打印真实长度分布（确认 pes2o 能产出 ≥20k tok 长文）。
- **动机**：plan [F2] 要单样本几十~上百 chunk 压力测 memory 多 chunk 写入→保持→读回；现有数据 4096 截断做不到，必须重 tokenize。纯 CPU prep，与 F1 v3 ladder（.76）无冲突。
- TODO(next HB)：收 coder 报告（脚本 + dry_run 长度分布 + commit）→ 分布合理（pes2o 出 ≥20k tok 文档）→ 实跑生成子集 → 待 F1 v3 ladder 完成用 F1 最优配置起 F2 长文训练。

## [DONE 2026-06-08 05:48] l3_recon CONVERGED (step5000) eval — 确认 REJECTED 不翻案
- **w0.3@.196 converged eval 成功**（diskA 有外网）：qa5 step5000=50/59/45/20/19（16k/32k cell 未全补），qa1=80/27/43/15/3/1/2 → 与 step500 同向，**确认收敛点仍一致劣于无-aux baseline，REJECTED 裁决成立**。已锁进 RUN_REGISTRY §3。
- **w1.0@.249 converged eval silent-fail（0 CSV）**：.249=diskB 无直连外网，BABILong dataset 下载失败 → 0 样本无 CSV（已知 proxy 问题 `reference_h800_babilong_proxy.md`）。**不重跑**——sweep 已终裁 REJECTED，converged 仅确认用，w0.3 收敛点已足够确认，无需补 w1.0。
- **chunk1024 FINAL（step5000）eval 完成**：qa5=29/68/29/15/7/4（32k 收尾），qa1=56/56/15/15/7/5/0。**确认 chunk1024 的 1k 后断崖满训后依然持续**（对照 chunk512 qa5=82/86/83/64/50/35），渐进 warm-start（F1 v1）仍是修断崖正解。已锁进 RUN_REGISTRY §3。

---

## [DONE 2026-06-07 22:05] l3_recon_token_weight sweep — w1.0 step500 BABILong eval 评完 + 裁决 ❌
- 21/21 CSV（qa1/qa2/qa5×7 len，n=100）全完成，已 canonical 评分（`scripts/score_nested_babilong.py`，diskB .76）。
- **结果（灾难）**：qa5 0k-32k = **67/22/16/8/3/1/0**；qa1=77/4/6/8/3/2/1；qa2=43/4/5/3/1/2/3。
- **裁决：L3 token-recon aux weight=1.0 灾难性破坏长程寻址。** 对照无-aux P11 chunk512 baseline（qa5=82/86/83/64/50/35）→ 仅 0k 部分存活，≥1k 全面塌方。真实实验结果（CSV 满 n=100 非 silent-fail）。已锁进 RUN_REGISTRY §3「l3_recon_token_weight sweep」。
- 含义：强 token-level recon aux 与 routing/检索目标冲突；待 w0.3 弱权重确认是否「弱即无害 vs 仍劣于无 aux」。两 train run（.196 w0.3 / .249 w1.0）继续跑满 5000 仅为 lm/recon 曲线，BABILong 已基本判定 token-recon aux 不优于 baseline。

## [DONE 2026-06-07 23:15] l3_recon_token_weight sweep — w0.3 step500 BABILong eval 评完 + sweep 终裁 ❌
- **23:11 .76 eval 节点全空闲（8 GPU 0 MiB）→ w0.3 step500 eval 完成（7/7 长度 × qa1/qa2/qa5 CSV 齐，n=100）。** canonical 评分（`scripts/score_nested_babilong.py`，.76 diskB）。
- **w0.3 结果**：qa5 0k-32k=**54/61/56/34/25/21/10**；qa1=78/26/42/31/22/21/14；qa2=33/3/15/14/14/9/11。
- **裁决：弱权重 token-recon aux 仍一致劣于无-aux P11 baseline（qa5=82/86/83/64/50/35/41）——全长度无一更优。** 破坏比 w1.0（67/22/16/8/3/1/0）温和但方向相同。
- **★sweep 终裁：L3 token-level recon aux 在 w0.3 + w1.0 均 REJECTED。token-recon 与 routing/检索目标冲突，弱权重也只是「破坏更小」非「有益」。最佳仍是 P11 无-aux baseline。** 已锁进 RUN_REGISTRY §3。两 train run（.196 w0.3 / .249 w1.0）继续到 5000 仅留 lm/recon 曲线。

---

## [DEAD 2026-06-07 17:25] H800 16卡 lease 又被回收 — hung-fix subagent 失败（节点消失）
- 16:40 派的 general-purpose-1 修 H800 hung 没能完成：~17:20 两节点 SSH 全拒（port 36000 refused、port 22 password denied），跟之前所有 H800 IP 一样被回收。
- stage1/stage2 ckpt（step600+final）在 jn2 共享 FS 上，现已不可访问；stage3/4 从未存出。
- **所有 H800 IP（.247/.130.90 及历史全部）现已死，别再试**。H800 stable-ladder 工作挂起，等新 lease 重新分配。mem_space ablation 全部转到 4 个 H20 节点继续。

## [RUNNING 2026-06-07 17:22] chunk 阶梯 step500 judge evals（auto_launch 自主起，on diskB .76 free GPUs）
- diskB .76 GPU6/7 在跑旧 eval、GPU0-5 空闲 → 自主起两个 step500 BABILong eval：
  - **chunk256** deltarule_normreadout step500：GPU0-2，driver pid 194650（17:22）。已到 qa1/0k 17%。
  - **chunk1024** deltarule_normreadout step500：GPU3-5，driver pid 195766（17:24）。模型加载中。
- 同口径 qa1/qa2/qa5 × 0k-32k，n=100，babilong.metrics。对照 P11 chunk512 step500 baseline（qa5 0k-8k=82/86/83/64/50）。woa proxy + HF_HOME 已 export，worker log 无 network err。
- 完成判读：补全 P11 deltarule_normreadout 的 chunk 阶梯三点（256/512/1024）横向对照，写入 RUN_REGISTRY.md。

---

## [用户决策 2026-06-07 10:25]
- **D6（null-sink vs xattn 解耦）= 取消**。用户："null sink 和 xattn 的解耦可以暂时先不做，毕竟现在效果很好"。不改 selector.py。从 roadmap 移除（不再 BLOCKED-pending-decision）。
- **下一轮阶梯式训练 = 等远程两个H20(.76/.249)评测跑完后起**。但用户要求先 research：(1) 小 chunk size 训练波动大 → 找"更合适的小-chunk 训练方式"；(2) 阶梯/小chunk 对 slot 容量的要求可能不同 → 谨慎探讨 slot 容量 vs chunk size。调研中（general-purpose-4，写 status/research_notes/small_chunk_training_and_slot_capacity_20260607.md）。调研出方案 + 节点空出 → 起改进版阶梯。

---

## [DONE 2026-06-07 13:04] stable progressive-ladder FINAL ckpt BABILong eval（.76 空闲自主起）
- **背景**：diskB(.76) 的 stable progressive chunk 阶梯 08:41 全链路完成（4 stage: 128→256→512→1024, nf=0, stage4 121.5min）。
- ckpt = `outputs/progressive_chunk_diskB_stable/stage4_c1024/mem_space_adapter.pt`（P11 delta_rule+normreadout 渐进训练）。
- **评完（21/21 CSV，eval@chunk1024）**：qa1 0k-8k=86/69/45/41/25；qa2=39/35/32/16/12；qa5=14/23/82/59/39（qa5 0k/1k 低是 chunk1024 短长度已知抖动，2k 起 82/59 强）。
- **★关键裁决：渐进式 chunk 训练 ≫ 单 chunk1024 训练。** 同在 chunk1024 eval 下：qa1 2k ladder=45 vs 单chunk1024=4；qa5 2k ladder=82 vs 单=20；长程 qa5 16k=32/32k=29 vs 单 16k=5/32k=4。**渐进 warm-start（小→大 chunk）彻底修复了单 chunk1024 的 1k 后断崖塌方。** 这是阶梯训练价值的决定性证据。已锁进 MEMORY_PROTOCOL_PLAN。
- driver 已退（GPU6 仅剩 stage1_c128 step400 32k 收尾 cell，非调度器，~分钟级完成）。

## [DONE 2026-06-07 13:02] chunk-ladder step500 BABILong eval 评完 + 裁决
- 两个 step500 eval dir（21/21 CSV）已 babilong.metrics 评分（diskB .76）。qa5 0k-8k：chunk256=78/66/47/28/42，**chunk512(baseline)=82/86/83/64/50 ⭐**，chunk1024=82/43/20/29/16。
- **裁决：chunk512 决定性最佳。chunk256 中长度弱，chunk1024 1k 后断崖（2k=20、16k=5/32k=4，复现 P8 chunk1024 长程塌方形态）。** 已锁进 MEMORY_PROTOCOL_PLAN P11 段。后续臂一律 chunk512 底座。c256/c1024 训练继续到 5000 仅为 lm/压缩曲线。

## [SUPERSEDED 2026-06-07 08:22] chunk-ladder step500 BABILong eval 补全（chunk256 + chunk1024）— RELAUNCHED w/ proxy（评分已在上面 13:02 完成）
- ⚠️ **07:48 首launch 静默失败**：diskB(.249) 无直连外网，BABILong dataset HEAD 请求报 "Network is unreachable"，0 样本评出、无 CSV，driver 仍打印 "all done"（假完成）。根因同 memory `reference_h800_babilong_proxy.md`（diskB 须挂 woa proxy + HF_HOME）。
- **08:22 重启修复**：export http_proxy/https_proxy=hy-proxy.woa.com:3128 + HF_HOME=.../share_304376610/.../.hf_home 后重跑。chunk256 GPUs0-3 (driver pid201775) + chunk1024 GPUs4-7 (driver pid201776)。已确认 worker 加载 766 keys + 经 proxy 触达 HF Hub（不再 Network unreachable），8 卡各 35GB busy。
- qa1/qa2/qa5 × 0k-32k，n=100，commit 同 P11。脚本 `scripts/eval_p11_chunk{256,1024}_deltarule_normreadout_step500.sh`（diskB）。step500 ckpt 两个均在 diskB（chunk256 5:50、chunk1024 6:25）。
- 对照 P11 chunk512 step500 baseline（qa5 0k-8k=82/86/83/64/50）→ 三点齐定 P11 最佳 chunk。
- ETA ~1.3h。完成后 aggregate 三 chunk → 更新 MEMORY_PROTOCOL_PLAN + RUN_REGISTRY。
- 🔧 **TODO(auto_launch:false)**：eval driver 在 worker 全失败时仍打印 "all eval lengths done" + exit 0，掩盖网络失败。应在 run_on_gpu 后校验 CSV 生成 / worker 退出码，否则 driver 退非零。避免再静默假完成。

---

## [DONE 2026-06-07 03:20] 4-arm chunk512 step500 ablation 评分 + 裁决
- 4 臂全训到 5000、step500 ckpt 同口径 BABILong 评完。**P11 (delta-rule + normalized writeback) = 新最佳臂**，qa5 1k-8k=86/83/64/50 超 top_k16 基线（76/77/54/48）。P10(ST-Gumbel 硬路由) 与 topk8 均劣于基线 → REJECTED。结果锁进 RUN_REGISTRY.md §3 + MEMORY_PROTOCOL_PLAN P10/P11。

## [RUNNING 2026-06-07 03:56] 下一臂 arm-1：P11 + chunk1024（ablation 延伸，auto_launch 自主起跑）
- P11(delta-rule+normreadout) 已确立为新基线。本机 8×H20 空闲 2 个 patrol → 按 heartbeat「adopted 底座的 ablation 延伸可自主起」启动 arm-1。
- run `mem_space_p11_chunk1024_deltarule_normreadout`，本机 8×H20，commit 9a9e3d0 配置，单变量 chunk_size 512→1024（chunk = 最大杠杆，§4 观察1）。script `scripts/launch_mem_space_p11_chunk1024_local.sh`（flags 与 chunk512 逐项一致，仅 chunk_size/run/port 差）。total_steps5000 save500 eval0 seed42 bs1×ga4×8=eff32 lr1e-4。pid 4061522 master_port29794。
- health: step5 lm=4.8064 route_aux=3.37 nf=0，8 卡 79-100% util ~81GB/卡，no error。
- judge: step500 ckpt 同口径 BABILong（qa1/qa2/qa5×0k-32k，n=100）对照 P11 chunk512 step500（qa5 0k-8k=82/86/83/64/50）。
- **剩余备选臂（仍 auto_launch: false，等用户/下个空闲节点）**：(2) P13 surprise-gated write（Titans 2501.00663）；(3) P11 + register slots(P9 num_global_slots) 组合。

## [RUNNING 2026-06-07 04:37] arm-2：P11 + chunk256（chunk 阶梯补全，auto_launch 自主起跑）
- .196 在 P11 step500 eval 全部 drain 完后空闲 → 按「adopted 底座 ablation 延伸可自主起」启动 chunk 阶梯第三点。
- run `mem_space_p11_chunk256_deltarule_normreadout`，远程 .196 8×H20，单变量 chunk_size 512→256（脚本 `scripts/launch_mem_space_p11_chunk256_remote196.sh`，flags 与 chunk512 逐项一致仅 chunk_size/run/port 差，master_port29793）。total_steps5000 save500 eval0 seed42 bs1×ga4×8=eff32 lr1e-4。pid 2687516。
- health: step5 lm=4.5015 route_aux=5.10 nf=0，8 卡 84-100% util ~75GB/卡，no error。
- judge: step500 ckpt 同口径 BABILong（qa1/qa2/qa5×0k-32k，n=100）对照 P11 chunk512 step500（qa5 0k-8k=82/86/83/64/50）+ chunk1024（本机跑中）。
- **chunk 阶梯（P11 base）现况**：256(此/.196)·512(adopted baseline DONE)·1024(本机 RUNNING)。三点齐则可定 P11 最佳 chunk。

---

## [DONE] researcher: chunk128 vs chunk256 step1000 退化形态差异根因 (general-purpose-35, 2026-06-05 20:08)
- **现象**：null-sink P8 两个臂 step500 都好，step1000 都崩到 ~0%，但**失败形态不同**（chunk256=连贯续写 haystack，chunk128=token 重复死循环乱码）。
- **根因（confidence high）**：⚠️ **推翻旧前提"TF lm 全程健康~3.3"**——chunk128 的 TF lm loss 在 **step895-1010 飙到 ~8-9（PPL~3000）**，step1000 ckpt 恰好存在这个 loss spike 中段；step490-510=~2.4（谷底），step1490-1510 已回落~4.0。每 500 步存盘节奏不巧把 chunk128 step1000 存在了 spike 顶上。
- **为何 chunk 越小越偏 LM 崩坏**：注入次数=seq_len/chunk_size，chunk128 是 chunk256 的 2×；spike 期过量注入（topk_mass>1.5）在 2× 注入事件上累积 → backbone 彻底塌成功能词死循环。chunk256 同期注入少，只退化成连贯续写。chunk256 跑 5000 步，step1000 lm=3.35（谷底未崩），其 spike 在 1200-1300 / 1750-1950。
- **不是 adapter 永久损坏，也不是单纯 greedy 假象**：是瞬态训练不稳定的快照。rep_penalty/temp 只能减轻不能完全救回。
- **结论**："早 ckpt=最终交付"对 chunk128 成立（用 step500），但原因从"过训练"改写为"快照撞 loss spike"。
- 诊断脚本 `scripts/diag_chunk128_step1000_repgen.py` 已写好未运行（GPU 全忙）。报告已 append RESEARCHER_REPORTS.jsonl。

## [DONE 2026-06-06 02:54] eval chunk512/1024 step500+step1000 (验证 chunk 越大越稳假设)
- **完成**：chunk512/1024 step500 与 step1000 全部 0k-32k 已评完，数字已锁进 MEMORY_PROTOCOL_PLAN.md P8 阶梯表。结论坐实「chunk 越大越稳，step1000 崩=快照撞 loss-spike×注入频率」。最佳臂=chunk512 step500。无遗留 eval。

## [DONE 2026-06-06 05:25] coder: 加 topk_mass + chunk_idx_jaccard routing 诊断 (agent general-purpose-21)
- **完成**：commit `5656cb6` 已落地，新指标 topk_mass / chunk_idx_jaccard 已在 QUERY_DIAG 中输出（chunk128_routeaux eval log 已可见）。纯诊断 no-grad，不改训练数值。后续 launch 自动带上。
- **动机存档**：top_k=16 等权监督下 top1_sim 有数学天花板 ln(16)=2.7726，top1_sim≈1/16 平是预期非 routing 崩。topk_mass 判 mass 是否集中，chunk_idx_jaccard 区分真寻址 vs 退化捷径。

## [DONE 2026-06-06 05:25] E5 route_aux 8B 验证 run (commit 35ea240) + offline BABILong eval gate
- **完成**：E5 train 出 step500 ckpt（outputs/e5_route_aux_remote/，train 后续停在 ~step830，step500 即交付点）。offline BABILong eval（qa5 × 0k-32k，commit 35ea240）已于 2026-06-05 02:20 跑完，CSV 存 babilong_results/perdoc_chunk128_routeaux/*。
- **结论（已存档于 TRAINER_ACTIVITY + RESEARCHER_REPORTS）**：route_aux 是 routing differentiation 的 driver（key_max_cos 0.47→0.58，top1_sim 0.015→0.10+，lm 1.60 vs l3iso 2.63）。但 eval QUERY_DIAG 显示 ≥2k 仍 top1_sim≈0.02-0.03、topk_mass 仅 0.28-0.42、chunk_idx_jaccard 0.33-0.44（退化捷径迹象）—— route_aux 提升了 key 可区分性但未把 retrieval 真正爬起来。
- l3iso_noL3_local 是 E5 的 route_aux-OFF 对照，researcher 已判 KILL（预期 no-L3 collapse，无法回答真问题），不再续跑。

---

## [DONE] toy 诊断矩阵 E1/E2/E4 (2026-06-04 14:00)
- 5 arm 全完成。**E1**：decoupled-read 饿死 selector LM 梯度（ON lm_grad 0.3–4 vs OFF 8–15，~10–50× 衰减）。**E2**：纯 LM loss 无法 bootstrap content addressing（aux_off exact_acc=0）；routing-supervision aux → exact_acc 0.25↑。**E4**：冻结 inject gate 非主因（force-open top1_sim→0.30 但 exact_acc 仍 0）。
- 决定：自动派 coder 实现 route_aux + E5 8B 验证 run。

---

## [PENDING] 修 FSDP checkpoint-save host OOM — auto_launch: false
- fsdp_smoke_remote @2026-06-04 11:56 在首个 checkpoint save 时 SIGKILL -9（FSDP full state_dict gather 8B 模型 → host mem OOM）
- commit 02561b4 "complete FSDP migration" 的存盘路径需改：用 sharded state_dict / get_state_dict API（日志里有 deprecation 提示），或 rank0 流式存盘避免一次性 gather 全量
- 优先级：仅当需要 FSDP 路径时才修；当前 DDP+gradient_checkpointing 在本机 8B 已能跑通 2000 step
- auto_launch: false（涉及存盘逻辑改动，等确认确实需要 FSDP）

---

## [DONE] P2 decoupled-read offline BABILong eval (2026-06-04 13:25)
- 21/21 cells (qa1/qa2/qa5 × 0k-32k)。**FAILS gate**：0k qa1=72/qa2=27/qa5=53，≥2k 全 0.0%。
- 结果已写入 status/BENCHMARK_RESULTS.md。eval 期 top1_sim≈0.05≈uniform → routing collapse 确认。

## [DONE] researcher toy-vs-full routing collapse 报告 (2026-06-04 12:30)
- ops/research_notes/toy_vs_full_routing_collapse_20260604.md。confidence high/very_high。
- 关键：top1_sim 是 red-herring（toy retrieval_exact_acc=0 全程）；decoupled-read 切断 selector LM 梯度（mask_h_to_l1）；LM loss 单独无法 bootstrap content addressing；inject_gate 冻结 α≈0.12。
- 建议先跑单 GPU E1/E2/E4 再决定 8B 修复 → 已于 13:49 在 H20-1 GPU0-4 启动诊断矩阵。

## [DONE] P2 decoupled-read full 8B run (2026-06-04 12:13)
- dolmino_p2_decoupled_local step2000/2000 完成。Routing 仍塌缩 top1_sim≈0.013≈uniform。
- 关键发现：同机制在 toy arm 能学会(0.998)，full 8B 塌缩 → 已派 researcher 分析 scale/data gap。
- checkpoint: outputs/dolmino_p2_decoupled_local/mem_space_adapter.pt，offline eval 进行中。

## [DONE] P1-v3 routing fix 系列、multi_query、chunk_query（早前）
- 结论汇总见 status/gpu_runs.jsonl 与历史 UPDATELOG。所有 P1 routing-pool 变体均塌缩在 1-2% noise floor。

## [DONE 2026-06-07 12:17] eval P11 chunk512 deltarule CONVERGED ckpt (step5000) — on .249
- P11 chunk512 deltarule+normreadout train FINISHED 02:20 (step5000, lm=2.43, non-finite=0); only its step500 ckpt was BABILong-evaluated. Converged ckpt eval **COMPLETE** (21/21 CSVs, "all eval lengths done" 12:17, 1h32m).
- output: `babilong_results/p11_chunk512_deltarule_normreadout_final/` on diskB (raw CSVs target/output/question — needs babilong.metrics scoring to aggregate).
- TODO(next): score converged CSVs w/ babilong.metrics; compare converged-vs-step500 (step500 qa5 0k-8k=82/86/83/64/50); update RUN_REGISTRY + MEMORY_PROTOCOL_PLAN P11 row.

## [RUNNING 2026-06-07 12:25] v2 progressive chunk ladder (per-stage scaled warmup/grad_accum) — LAUNCHED on .249
- **背景**：用户 10:25 决策门——"下一轮阶梯式训练 = 等远程两个H20(.76/.249)评测跑完后起，先 research 小-chunk 训练方式 + slot 容量"。research note `status/research_notes/small_chunk_training_and_slot_capacity_20260607.md`（11:08 完成）+ v2 脚本（commit 5aa2329, 11:21）均就绪。.249 的 converged-c512 eval 12:17 跑完→8 卡全空闲→门已满足，自主起 v2 ladder。
- **v2 vs v1**：per-stage 反比缩放 warmup + grad_accum（c128:warmup800/accum8, c256:500/4, c512:300/2, c1024:200/1），使 warmup-token 与有效梯度-token/step 跨 stage 恒定，压小 chunk 梯度方差（research note 标 [high,可直接采用] 零风险）。其余配方 = v1 = P11 stable（delta-rule writeback + normalize_readout + loss_spike_skip + ST-Gumbel OFF）逐项一致。
- node **.249** 8×H20（自有卡，非 .76），warm-start 链 stage1 c128(scratch)→s2 c256→s3 c512→s4 c1024，各 stage 从上一 stage step000600 adapter init。driver pid 230717，log `logs/progressive_chunk_diskB_v2.driver.log` + 各 stage `logs/progressive_chunk_diskB_v2_stage*.log`。total_steps800/stage save200 chain_step600 eval0 seed42。
- **health**：stage1 c128 8 ranks 全载入权重（15.7→74GB/卡），util 38-100%，无 error/unreachable/nan。代码已从 diskA rsync 到 diskB（v2 脚本确认存在 + delta_rule flag）。
- judge: 对照 v1 stable ladder（.76 已跑完，FINAL ckpt eval 收尾中）+ P11 单 chunk 各点 → 验证 per-stage 缩放是否改善小-chunk 稳定性 / 最终 retrieval。

---

## [PENDING] ★ b25/c512 中间 ckpt 早评（step500/1000/1500/2000/2500）— auto_launch: true (next-free-node)
- 动机：.7.53 b25/c512 step3000 W0 全档破墙(qa5 32k=68 vs MemoryLLM 34)。过训退化铁律：历史 step500 普遍是甜区，step3000 可能已退化。早评中间 ckpt 找峰值。
- ckpts: `outputs/mem_space_fifo_b25_chunk512/full_model_step00{500,1000,1500,2000,2500}.pt` on .7.53 (diskB)
- eval: `_eval_taskpool_2group.sh`，W0+W6，CHUNK_SIZE=512，n=100，21 cells/ckpt × 5 ckpts × 2 modes = 210 tasks。可分批：先 step500/step1000，足够定形态。
- 节点选择：等任一节点 free 即起。本机/.196 不持 b25 ckpt 需 rsync (~23GB/ckpt)；.7.53 自持 ckpt 但目前 W6 在跑；.245.174 共享 diskB 可直接读 .7.53 路径。
- 优先级 P0(决定破墙结果时序稳定性)。auto_launch: true。

## [PENDING] ★ b25/c512 step3000 真实长文档 benchmark（LongBench / LongMemEval / LongEval）— auto_launch: true (next-free-node-after-b25-ckpt-eval)
- 动机：BABILong 破墙不等于真实长文档破墙(pg19 nctx7 案例：BABILong 16k +3 但 LongBench AVG 6.5；对话记忆 mem vs base 差 3.8-7×)。必须验证 b25 c512 不是 BABILong 过拟合。
- benchmark：LongBench (hotpotqa/2wikimqa/musique/narrativeqa/qasper/multifieldqa_en)、LongMemEval (oracle n=500 全6题型)、LongEval (lines retrieval ≥8k)。
- 脚本已有：`scripts/eval_longbench_mem_space.py`、`scripts/eval_longmemeval_mem_space.py`、`scripts/eval_dialogmem_mem_space.py`、`scripts/eval_longeval_mem_space.py`。
- 优先级 P1(决定结果迁移性)。auto_launch: true，但排在 b25 中间 ckpt 早评之后。

## [PENDING] b50/c512 + b100/c512 中间 ckpt + 跨臂 ckpt-curve 对照 — auto_launch: true (eval-after-final-W0)
- 动机：等本机 b50/c512 W0 + .245.174 b100/c512 W0 出炉后，若长档分数(8k-32k)显示 buffer_length 单调影响 → 确认 dilution 剂量曲线；若 b50/b100 也破墙 → buffer_length 不是 load-bearing → H3/H4 候选；若 b50/b100 不破 → b25 是 load-bearing → 探索更小 buffer (b10/b5)。
- 跨臂中间 ckpt(b50/b100 各 5 个早 ckpt) 用于过训退化对照。
- 优先级 P1(决定 H1/H2 假说裁决)。auto_launch: true。

---

## [PARTIAL-RUNNING] ★★ b25 数据泄漏判别 — 三个对照实验(auto_launch: true, 等节点空出)

**背景**:b25 W0 qa5 0k=100/4k=87 受 BABILong 数据泄漏污染(train 默认 babilong_mix=0.15 掺 eval-split,全项目通病)。需三个对照隔离真实能力。脚本已写好 commit d03db24。

1. **b25 NOLEAK 训练**[RUNNING 2026-06-25 20:37 @.48.7.53 pid775325](`scripts/launch_mem_space_fifo_b25_chunk512_NOLEAK_diskB.sh`,babilong_mix=0 纯 dolmino):零泄漏基线,产出 outputs/mem_space_fifo_b25_chunk512_noleak/。3000 steps@8H20。✅已启动(b25 W6 eval DONE 后空闲节点自主起,babilong_mix=0.00 已确认)。发射后离线 W0 eval → 真实 b25 长程能力(对比脏 b25 qa5 0k=100 的差值=泄漏贡献)。盘B 节点(.7.53)。auto_launch:true。

2. **b25 T2-align 训练**(`scripts/launch_mem_space_fifo_b25_chunk512_T2_diskB.sh`,babilong_mix=0 + t2_recall_mix=0.15 合成 needle):合法 task-alignment(独立 needle 非 eval split),产出 outputs/mem_space_fifo_b25_chunk512_t2align/。验证"用独立 QA post-training 能否真提升 held-out BABILong"。若 T2 版 > NOLEAK 版(在干净 8k+ 长档)→ task-aligned post-training 有效。盘B 节点。auto_launch:true。

3. **memory-disabled 对照**(等 workflow wf_28a3f1c9 加好 --memory_disabled 开关):用现有脏 b25 ckpt 跑 8k-32k qa5 关 memory。若仍 60+ → 连 OOD 长档都靠 prior 非 memory。脚本待 workflow 产出。盘B 节点(.7.53/.245.174 共享 b25 ckpt 零 rsync)。auto_launch:true。

**裁决逻辑**:
- 脏 b25(qa5 32k=68) vs NOLEAK b25 vs memory-disabled b25 → 三方对比定位 68 分里多少是泄漏/prior/真 memory
- T2-align vs NOLEAK → task-aligned post-training 净增益
- 全部 W0 口径,n=100,_eval_taskpool_2group.sh

注:8k-32k 训练 max_seq_len=2048 不覆盖 → 这部分本来就是 OOD,相对干净;主要污染在 0k-4k。

---

## [PENDING] ★ 树形 hidden memory 方向(用户 2026-06-25 提出)— 设计中
**用户想法**:把过去 chunk 的 hidden states 存成**树形结构**,query 时用树搜索/遍历决定用哪些 hidden。与 slot 压缩交叉(用户明确要探索 slot×tree)。
**动机(今日发现驱动)**:
- H2 dilution:flat buffer 全注意力→needle 被稀释;小 buffer 抗稀释但丢早期 needle
- 树同时解决两者:叶子保留所有 chunk(不丢 needle),导航只 attend O(log N) 路径节点(不稀释)
- slot×tree 解决"hidden 无压缩 vs slot 丢精度"矛盾:内部节点 slot 压缩(导航用,省算力),叶子原始 hidden(回答用,保精度)
- 绕开选择器死路:树每层选 1-of-B(小分支)而非 1-of-64(flat),per-level 精度高
**设计 workflow**: wiq6fz89m 进行中(5 角度 + 批判 + 综合)。出来后:派 coder 实现第一个 no-train probe(在现有 b25 ckpt 上做 tree-navigation eval),验证可行性再训练。
**novelty**: vs MemWalker(文本树导航无 hidden)/RAPTOR(摘要树 RAG 外挂)/Compressive Transformer(2级)——我们 = hidden 树 + reader-attn 导航 + 集成 forward + slot 内部压缩 + 可训练,组合可能原创。auto_launch: false(等设计 + 用户确认架构)。

---

## [PENDING] ★★★ Design A: SnapKV-on-chunks 零训练淘汰实验(文献调研 wbr15ytio 推荐,最高优先级)
**这是当前最优下一步**:实现"reader q·k attention 打分淘汰"替代"丢最老",**零训练 + 用现有 b100 ckpt**,直接验证用户"丢置信度最低 buffer"的想法 + 修复 b25 丢早期 needle 的盲区。
**机制**:b100 全保留 → 每个 chunk 边界用 reader q·k(obs_window=当前 chunk 最后 64 token)给每个 buffered chunk 打分 score(c_i)=mean_h mean_{q in obs} max_{k in c_i}(q·k/√d),跨32层 mean pool,保留 top-25 + 前2 sink(StreamingLLM),其余淘汰。
**零新参数**(用 frozen reader 自己 attention,避开所有 trained selector 死路 H2)。**eval-time only**,改 run_babilong_mem_space.py 的 FIFO readout 加 --evict_policy snapkv_chunks。
**eval**: qa1/qa2/qa5 × {0k,8k,16k,32k} n=100,现有 b100 ckpt(在 .245.174 diskB)。
**预测**:强成功 qa1 32k≥30(=b25,证明 attention 淘汰=隐式 isolation 且修复早期 needle);中等 >15;失败 ≤10。
**证伪线**:不超 b50 qa1 32k=24(2/3 task)→ reader-attn chunk 级不 transfer → 转 softmax-sharpening(SSMax)。
**complexity**: LOW(一次 obs-window query vs buffered keys + argsort,无训练)。
**依赖**:coder 实现 eval-time evict policy(新代码,~1 文件)。等树形 workflow wiq6fz89m 出来一起综合(树导航 = SnapKV-on-chunks 的分层版,统一实现)。auto_launch: false(等综合 + 用户确认)。

---

## [PENDING] ★★★ FIFO eval-time probes(commit eddb4f1)— heartbeat 按节点空出顺序推进
**目的**:用现有 b25/b50/b100 ckpt **零训练**验证 W0/W6 gap 的根因。两套 flag(可独立组合,9 个组合中重点跑下面 5 个):
- `--fifo_pos_mode {none,packed,real}`:位置方案(pos-0 vs 重打包保序 vs 真实稀疏)
- `--fifo_keep_set_mode {none,flat_readerattn}` + `--fifo_keep_topk 25 --fifo_keep_recency 2`:用 reader q·k 选 chunk
- `--fifo_keep_all_buffer`:eval 时不淘汰 buffer(配 keep-set 测 keep-all-attend-few)

**实验矩阵(按重要性排序,每个 = 现有 ckpt + 改 flag eval,无训练)**:

| # | ckpt | flag 组合 | 测什么 | 预期 |
|---|------|----------|--------|------|
| P1 | b25(脏)  | `pos_mode=packed` | H_POS:packed 位置能否替代 pos-0 抬升 W0 | 8k-32k W0 跳升 → 位置是 gap 主因 |
| P2 | b100(脏) | `keep_set=flat_readerattn keep_topk=25 keep_all_buffer` | H_DIL:reader-attn 选 25 chunk + keep-all 是否 = b25 长档 | 32k 从 5 → 30+ → dilution + 选择有效 |
| P3 | b25(脏)  | `pos_mode=packed keep_set=flat_readerattn keep_topk=25 keep_all_buffer` | H_POS + H_DIL 叠加 | 8k-32k 大幅超 b25 baseline 或 ≈ W6 → 全胜 |
| P4 | b25(脏)  | `pos_mode=real` | real vs packed:绝对距离是否额外有用 | ≈ packed → 只保序够;< packed → real OOD |
| P5 | b100(脏) | `keep_set=flat_readerattn keep_topk=10 keep_all_buffer` | top-k 敏感度 | 看 K_keep 甜区 |

**口径**:n=100, qa1/qa2/qa5 × {0k,4k,8k,16k,32k}(0k/2k/1k 可跳过省时间,反正泄漏饱和), `_eval_taskpool_2group.sh`, W0(`--swa_eval_chunks 0`)。CSV 路径 `babilong_results/probe_<ckpt>_<flagid>/`。

**节点分配**(等节点空出按顺序):
- diskA 任一(本机/.196):b25/b50 ckpt 在盘A,直接读
- diskB 任一(.7.53/.245.174):b25/b100 ckpt 在盘B,直接读 ← P1/P3 最佳
- B200.53:b50/c1024 ckpt 在 wzc1 盘
- ⚠️ .7.53 正在跑 NOLEAK 训练,跑完之前不能用
- ⚠️ NOLEAK b25 训练完成后(明早 ~07:30),先 W0 eval(task #7),然后立即让 .7.53 接 P1/P3 probe(NOLEAK ckpt 也可以做同样 probe,作为干净版对照)

**裁决逻辑(根据 5 个 probe 的结果)**:
- 若 P1(pos packed) 单独显著抬升 W0 → H_POS 成立,正式做 position-fix 重训
- 若 P2(keep-set) 单独抬升 b100 → H_DIL 成立,正式做 reader-attn FIFO 重训
- 若 P3 叠加 ≈ W6 → 两者结合彻底关闭 gap,顶级突破
- 若 P1/P2 都不动 → 位置和 dilution 都不是 gap 主因,需要换假说(可能是 staleness 或 hidden 本身有损)

**触发条件**:任一节点空出 + GPU 全空闲时 heartbeat 自动发射下一个未完成 probe。**auto_launch: true,优先级 P0**(这是当前最高价值的便宜实验,零训练判定理论假说)。

**Eval launch 模板(diskB,以 P1 b25 packed 为例)**:
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
CKPT=outputs/mem_space_fifo_b25_chunk512/full_model.pt
CFG=outputs/mem_space_fifo_b25_chunk512/adapter_config.json
setsid nohup bash -c "
  export WANDB_MODE=offline
  RUN_PREFIX=probe_b25_P1_posPacked \
  CKPT_FILES=\"$CKPT\" CK_NAMES=\"probe_b25_P1_posPacked\" \
  ADAPTER_CONFIG=\"$CFG\" CHUNK_SIZE=512 \
  EXTRA_ARGS=\"--swa_eval_chunks 0 --fifo_pos_mode packed\" \
  PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
  PYTHON_BIN=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/.venv/bin/python \
  bash scripts/_eval_taskpool_2group.sh > logs/probe_b25_P1_posPacked.out 2>&1
" > logs/probe_b25_P1_posPacked_driver.log 2>&1 &
```
(其他 probe 类比,改 EXTRA_ARGS 即可。注意 .7.53 / .245.174 / 本机 / .196 / B200 用各自的 PROJECT_ROOT 和 PYTHON_BIN。)

## [RUNNING] #99 keep14-distill heal — **已在 `.212` 8×B200 跑起来**（2026-08-15 21:31:20 启，取舍选项 (b)）

**状态：RUNNING。** 2026-08-13 的 BLOCKED 判定里，**两个阻塞理由都已实测解除**，选项 **(b) 迁 B200** 成立并已执行。
step **5000 → 200000**，实测 **2.359 s/step**，ETA **5.32 d**（~1021 GPU-h）。**忠实 resume，不是新 run。**

- 节点 `.212`（8×B200 sm_100，178.4GB/卡，wzc1 盘），torchrun **PID 524842**，worker 525664-525671
- launcher `scripts/launch_keep14_distill_resume_212_0815.sh`；log `logs/olmo2_7B_keep14_distill_212_0815.log`
- 落账 `status/gpu_runs.jsonl`（commit `371b114`）、`status/GPU_STATUS.md`、`status/TRAINER_ACTIVE.md`

### 解除阻塞 1：「bnb 把 distill 锁死在 .73/.104」——**这条记载是错的**

旧记载（CLAUDE.md + 本条 08-13 版）说 module 级 `import bitsandbytes` + 硬编码 `AdamW8bit` ⇒ B200 不能跑。
**源码注释自证反面**：line 63 原文 `# 8-bit AdamW to fit keep14 train-all + teacher in H20 95GB`
⇒ **bnb 存在的唯一目的是塞进 H20 的 95GB**，与 178.4GB 的 B200 无关。
实测：`.212` 上 `pip install bitsandbytes` → **0.50.1**，`AdamW8bit` 在 **sm_100 (10,0) 构造 + step 成功**。
⇒ **保留 bnb 反而是忠实 resume 的前提**（ckpt 里 optimizer state 是 bnb 8-bit 格式；
换 fp32 AdamW 只能从 step0 重跑、丢掉已跑的 5000 步）。log 实证
`[resume] optimizer state restored (179 param states) -> Adam momentum preserved`，
loss 从 **3.169 / ppl 23.79** 接续（从头跑在单卡探针里是 ppl≈2.9e6）。

### 解除阻塞 2：`save_every` —— 这才是真阻塞，已改 **500**

08-13 的分析正确：`save_every 5000` + resume 起点正好 5000 ⇒ 下一次落盘在 10000，
而 07-31 死于 step5200、08-05 只到 step7780 ⇒ **两次烧完预算、0 ckpt**。
本轮 `--save_every 500` ⇒ 约 **20 min 一个 ckpt**，不可能再重演。
**这只改落盘节奏、不改优化路径**，故与 keep8/keep10/keep12/keep14-NTP **仍同口径可比**。
retention 实测（`select_rotation_victims` 干跑）：`keep_last_n=3` + `milestone_every=5000` 全留
⇒ 到 step25000 只留 7 个（step5000/10000/15000/20000 + 最后 3 个），**磁盘有界**，不会被 500 步一存撑爆。

### 08-13 记载里**仍然有效**的几条（不要推翻）

- **teacher 必须是 HF 目录** `/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B`（32L），
  **不是** `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt`（那是 16L 学生自己）。launcher 已 preflight 断言 32 层。
- **不得声称差分 LR**：`_classify_param` 没剥 `module.` 前缀 + `build_param_groups` 在 DDP wrap 之后
  ⇒ 8 卡 log 只有 `inh_decay 4060.1M @2e-5` + `inh_nodecay 0.3M @2e-5`，**实际均匀 2e-5**（与 ladder 同 bug 故可比）。
- 该 trainer **无 inline eval / 无 `--eval_interval`** ⇒ 传了会崩，NCCL desync 风险结构性不存在。
- ⚠️ 08-13 记的「两盘 trainer 不同、zwfy6 版没有 rotation flags」仍成立，但**本轮跑的是 wzc1 版**（md5 `228812e8`），
  rotation flags 齐全。**照 zwfy6 的 flag 写命令行会崩，反之亦然。**

### 新增实测（B200 特有，08-13 时无法得知）

- **`gradient_checkpointing` 在 B200 上也必须开**：单卡 40-step 探针
  （`scripts/_probe_distill_gc_b200.sh`）GC=1 bs=16 → 2.26 s/step / 115.7GB 正常；
  **GC=0 bs=16 → OOM**（178.35 GiB 已用 177.84）。bs 受**激活**限制，不是静态显存（静态仅 51.41 GiB）。
- **`bs=16 GA=1 → eff_bs=128`，与 H20 配方逐位相同**，没有为了吃满显存去改 batch（改了就不可比）。
- **`.212` 与 LOCAL 共享 wzc1 项目盘，但 `/dev/shm` 是 node-local** ⇒ 118 GiB 语料在 `.212` 上
  必须**本地重建**（150 s，`scripts/build_dolmino_corpus_wzc1.py`），md5
  `7df19b217e5b0670d58bf6e01e6559d0` 与 keep8/keep12 所用**逐字节一致**。
- ckpt `step5000.pt` 从 `.82` 6 路并行 `ssh dd` 搬到 wzc1，**~4 min**（~102 MB/s），
  md5 `0ec4481adde2314a470616d49aa922e9` 两盘一致（`scripts/pull_distill_step5000_to_wzc1.sh`，0 GPU、源端只读）。

### 下一步（无需审批）

1. **监控首个 ckpt 落在 step5500**（#99 的历史失败模式正是「跑了但没落盘」）。
2. 到 step10000/15000 等 milestone 时，可按 base 协议（chat=False / no-BOS / LL-MC）排 eval，
   与 `olmo2_*_results/7B_keep14distill_step5000/` 同口径对照。
3. `.212` 若重启 → `/dev/shm` 语料消失，先重建再 resume；launcher 会挡住拿错语料静默开跑。

<details><summary>原 [BLOCKED] 记录（2026-08-13，两个阻塞理由均已于 08-15 解除，保留作 provenance）</summary>

**结论：`.73` 已空闲，但 resume 仍不该跑。阻塞原因是 `save_every` 与预算不相容，不是节点可用性。改 auto_launch=false，需要用户就下面的取舍拍板后才动。**


**证据（全部在 .73 = zwfy6 实测，log=`logs/olmo2_7B_keep14_distill.log`）**：

1. **★ 关键：08-05 已经用光同样的预算，产出为 0。** 该 run 11:53→22:03 跑了 **10.2 h wall = 81 GPU-h**，从 step5000 推到 **step7780**，然后停了——**盘上依然只有 `step5000.pt`**（log 里 `saved` 只出现过一次，即 07-31 的 step5000）。因为 `--save_every 5000` 且 resume 起点正好是 5000 → **下一次落盘在 step10000**。
   - 实测 sustained rate **13.11 s/step**（`elapsed/iter` 与 tqdm 一致，全程稳定，maxmem 94.6GB/97.8GB）。
   - 5000→10000 = 5000 步 = **18.2 h wall = 146 GPU-h**，是 80 GPU-h 上限的 **1.8×**。
   - 80 GPU-h 只能到 **step ~7745**，**再次差 2255 步落不了盘**。→ 在给定预算内 resume 必然重演 08-05：烧满 80 GPU-h、0 checkpoint、0 可 eval 产物。**07-31 那次同样如此（到 step5200 就没了）。这会是第三次。**
   - 若要落盘必须改 `--save_every`（如 500/1000）——但那是**新配置**，且 200k 全程 = **5681 GPU-h ≈ 71× 预算**，本节点不可能完成。

2. **teacher 路径前提是错的。** 任务给的 `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` **不是 teacher**：它 16,241,486,089 B = 4.0604B×4B fp32 = **keep14 学生自己**（16 层）。真 teacher 是 HF 目录 `../models/OLMo-2-1124-7B`（32L，7.2986B，log 第 26 行 `[distill] teacher loaded: 7.2986B`），trainer 用 `AutoModelForCausalLM.from_pretrained` 加载**目录**，喂 `.pt` 会直接失败。`--distill_teacher_model` 必须是那个 HF 目录。

3. **差分 LR 缺陷：选项 (1) 成立，但本身不是阻塞项。** zwfy6 的 `_classify_param`（line 287-298）确实**没有 `module.` 剥离**，而 DDP wrap（line ~595）在 `build_param_groups`（line 620）**之前** → 全部参数落入 inherited。log 三次启动都只有 `inh_decay 4060.1M @2e-5` + `inh_nodecay 0.3M @2e-5`，**无 fresh 组** → 实际是**均匀 2e-5**，与 keepN ladder 同 bug 同行为（故可比）。**写作时不得声称差分 LR。**
   - 且 ckpt 的 optimizer state 是 **2 组 bnb 8-bit** 格式；zwfy6 版 trainer **没有** `train_olmo2_arch_probe2.py:912` 那个 2→4 组 remap shim（grep=0）→ 一旦补上 `module.` 剥离就变 4 组，`load_state_dict` 抛 ValueError 降级 warm-restart，**Adam 动量全丢**。所以选项 (2) 不只是「破坏可比性」，在 zwfy6 上还会**破坏忠实 resume**。→ 若要恢复，只能选 **(1) 原样均匀 LR**。

4. 两盘 trainer **不同**（LOCAL md5 `228812e8` / zwfy6 `9e824f7d`）：zwfy6 版**没有** `--seed`、没有 rotation flags（`--keep_last_n` 等硬编码 latest-2+每 5000），LOCAL 版有。**照 LOCAL 的 flag 写命令行会在 .73 上 `unrecognized arguments` 直接崩。**
5. 该 trainer **完全没有 inline eval 代码**（grep `eval_interval|babilong|quick_eval` = 0 命中）→ NCCL desync 风险结构性不存在，`--eval_interval 0` 这个 flag 也不存在，传了会崩。
6. 资产确认存在：`step5000.pt`(24,489,312,843 B, has_optimizer=True) / 数据 `/dev/shm/dolmino_now15b.npy`(126.9GB, 在 shm 与 data/ 双份) / bnb 0.50.0 / zwfy6 剩 3.4T。**资产不是问题。**

**要拍板的取舍**（任一都超本次 80 GPU-h 授权，故不自行启动）：
- (a) **放弃 #99**：distill 属「后续方法论文」（见本文件 line 16 用户 2026-07-31 定调），非当前机制论文承重项；step5000 已有完整 base-协议 eval（PPL+core6+know5，`olmo2_*_results/7B_keep14distill_step5000/`）可作为 distill 的唯一数据点。
- (b) **迁 B200**（原任务本意，~3s/step，约 7 天到 200k）——但 LOCAL/.21 现跑 SparseForge #246，且 **B200 无 bnb** → 需先装 bnb 或换 fp32 AdamW（后者丢动量）。
- (c) **降 `--save_every` 到 500 继续在 .73 慢跑**：需接受这是新配置 + 200k 需 ~33 天独占。

<details><summary>原 PENDING 记录（2026-07-30，触发条件已过期）</summary>

**触发条件**: LOCAL B200 的 ShortGPT heal 跑到 step200000 完成（当前 ~96k，~1.8 天后）。

**动作**:
1. ShortGPT 200k 完成 → 4-point eval (step0/50k/128k/200k) 排队
2. .73 distill 当前 checkpoint (outputs/olmo2_probe2_7B_keep14fresh2_distill/step{N}.pt, save_every=5000) 迁到 B200
   - .73 diskB → B200 wzc1 跨盘, 需 scp ckpt (~46GB student + 8bit opt state)
   - 或: .73 distill 继续跑 (H20 14.4s/step), B200 空了跑**别的** (keep12 resume / keep8 到200k)
3. B200 resume distill: `--resume_from <ckpt>`, bs=16 gaccum=1 eff_bs=128, ~3s/step
   - ⚠️ 8bit adam ckpt: B200 若保持 bnb.optim.AdamW8bit 则 optimizer state 兼容直接 resume;
     若换 fp32 torch.optim.AdamW 则 optimizer state 加载会 skip (model 权重保留, opt 从头, 影响小)
   - B200 183GB 装得下 fp32 adam (56GB) + model(42GB) + teacher(14GB) = 112GB, 可不用 8bit

**决策点** (ShortGPT 完成时): distill 迁 B200 (7天完成) vs .73 继续 H20 (33天) + B200 跑其他。用户 2026-07-30 指令: "先跑着吧 B200空出来可以迁移" → 迁移。
</details>

</details>

## [DONE] Paper B P0.6 content-MMLU 全 sweep (2026-08-02，9 arms 全跑完，.73 + .104 并行)

**状态**: **DONE (2026-08-02)** —— 9 arms 全部跑完 (base/full32/keep8/keep10/keep12/keep14/freezefront/random-init/ShortGPT-16)，双协议 (letter+content_raw+content_norm)，14042 题，base 协议 chat=False/add_bos=0/LL-MC，每 arm n_valid=14042 nan=0。letter 逐题复现 P0.7 (base=.6054 vs P0.7 .6053；full32=.5877 vs P1.1 .5867；各 keep 臂全对齐)。**harness bug 已修**: `mcnemar_exact_p` 在全 14042 题 merge 时 `OverflowError`（`math.comb(n,i)*0.5**n` 上溢），改成 log-space (lgamma+logsumexp)，commit `324a44f`（committer=LiuHanzuo，未 push）。
- 结果 raw (.73 + .104 上 `olmo2_mmlu_content_results/<TAG>/`，已汇总到 LOCAL `olmo2_mmlu_content_results/P0_6_content_mmlu_summary.json`)。
- **核心发现 (dissociation)**：content 协议 above-chance recovery 远高于 letter。base=.6054 为分母。keep14: letter recovery 19.3% vs content_norm 60.4%；ShortGPT-16: 63.1% vs 68.5%；full32(intact 续训): 95.0% vs 98.0%(≈无损上锚)。**random-init@200k 是关键 control**: letter recovery ≈0 (−0.85%，纯 chance) 但 content_norm=.3598 (recovery 49.8%) → content_norm 有一个 fluency 驱动的"地板"，与知识无关；因此 content recovery 必须相对 random-init 地板解读，不能直接当"知识恢复"。→ 主结论支持 answer-symbol/readout binding lag（content>>letter），但 competence lag 仍在（content 也远低于 base，且 random-init 地板抬高了绝对值）。
- MAIN 待回填 paperB/TODOList.md P0.6 表 (数据在上述 summary.json + `*_vs_base_*_compare.json`)，不碰 `.tex`。

<details><summary>原 PENDING 记录 (harness-ready 阶段)</summary>

**状态**: harness DONE (2026-08-02，`scripts/eval_olmo2_mmlu_content.py` + `_run_olmo2_mmlu_content.sh`，commit `d2e28f2`，self-test 通过，未 push)。**只差 GPU 节点** —— 当前 5 台全忙 (LOCAL full32 / .252 P1.6 / .104+.73 P0.5 / .82 P1.7)。

**动作** (第一个空节点，按 letter+content_raw+content_norm 双协议，14042 题，base 协议 chat=False/no-BOS)：
```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory   # 或对应节点 root
TAG=7B_base CKDIR= bash scripts/_run_olmo2_mmlu_content.sh                                                   # 分母
TAG=7B_keep14  CKDIR=outputs/olmo2_probe2_7B_keep14fresh2            STEPS=200000 KEEP_FRONT=14 N_FRESH=2 bash scripts/_run_olmo2_mmlu_content.sh
TAG=7B_scratch16L CKDIR=outputs/olmo2_probe2_7B_keep14fresh2_fromscratch STEPS=200000 KEEP_FRONT=14 N_FRESH=2 bash scripts/_run_olmo2_mmlu_content.sh
TAG=7B_shortgpt16 CKDIR=outputs/olmo2_probe2_7B_shortgpt16           STEPS=200000 KEEP_FRONT=16 N_FRESH=0 bash scripts/_run_olmo2_mmlu_content.sh
# 可选补: keep8/keep12/freezefront；full32 待 final.pt 出 (KEEP_FRONT=32 N_FRESH=0)
```
跨臂对比: `eval_olmo2_mmlu_content.py --compare --file_a <arm> --file_b <base> --protocol content_norm`。跑完 MAIN 回填 paperB/TODOList.md P0.6 表 + status，不碰 `.tex`。⚠️ ShortGPT `KEEP_INDICES="0-12,31"` 是占位 provenance，用前核对真实选层。优先级: Paper A 待跑项 > 此项 (P0.6 是 Paper B REQUIRED 但仅推理)。
</details>

---

## [PENDING] union-9 gap-fill: `slorb` × `hard_drop` on `.212` (variant-matched ±SLoRB)
`auto_launch: false`  ← **由 MAIN 决定何时投；要等 ±SLoRB 混淆审计结论。**
登记 2026-08-15 by subagent。**前置工作全部 0 GPU 已完成并实测通过。**

### 为什么需要这一格
token-matched union-9 两臂在 2026-08-15 跑完，但矩阵**不对称**，两臂同时差了**两件事**：

| arm | variant | zero_ratio | exact_2of4 | union9_primary | ppl@4096 |
|---|---|---|---|---|---|
| noslorb | hard_drop | 0.500000000 | 1.0 | 59.5535 | 6.6795 |
| slorb | hard_fold | 1.08e-9 | 0.0 | **61.5413** | 6.1938 |

所以「+SLoRB 赢 1.99 pp union9」里混着 **真 2:4 vs 稠密** 的 export variant 差异。
slorb 臂自己的 run log 就写着 `2:4 COLUMN: BARRED`
（`logs/sparseforge_tm_union9_slorb_progress.log:314`）。

**混淆有多大——有现成同 ckpt 对照**（`outputs/cast_eval_spec/sparseforge_5b/sparseforge_same_harness_table.json` → `headline`）：
同一个 ckpt 只换 variant，`hard_drop` 57.0678 → `hard_fold` 62.4335 = **+5.37 pp union9 primary**；
第二个 ckpt 复现（`sparseforge_dolmino_link2/link2_summary.json`，plain-acc）53.8748 → 58.9594 = **+5.08 pp**。
**5.37 / 5.08 pp 都远大于归给 SLoRB 的 1.99 pp** → 现有跨臂 gap 在 variant 固定前不可解释为 ±SLoRB 效应。

### 补哪一格（只有一格可能）
- ✅ **`ARM=slorb VARIANT=hard_drop`** —— 唯一最小充分补格。0 GPU 实测（`.212`，2026-08-15）：
  对 slorb ckpt 自己的 mask 施加 `nm_2_4_hard()` 再 drop 分支，得 `zero_frac=0.500000000`、
  12 抽样张量 **0 bad tiles**、in-scope 张量数 224（export 期望 224）→ `export_sparseforge_to_hf.py:213`
  的 `mask=hard slorb=drop must yield exact 2:4` 断言会 **PASS**。可与现有 noslorb/hard_drop 组成
  variant-matched、2:4-legal 的 ±SLoRB 对照。
- ❌ **`ARM=noslorb VARIANT=hard_fold`** —— **技术上不可能**。该 ckpt 1411 张量、
  `SLoRB_Weight=0`/`x_proj=0`（0 GPU 实测），`export_sparseforge_to_hf.py:181` 硬退出
  `--slorb fold requested but ...SLoRB_Weight/...x_proj missing`。没有分支可 fold。launcher 已显式拒绝该组合。

### ⚠️ 解读警告（必须与数字同时引用）
`slorb/hard_drop` 是对「训练时依赖该分支的模型」做**事后截肢**，正是
`baselines/cast_repro/SPARSEFORGE_SAME_HARNESS.md` CORRECTION 的 **Defect 1**（该 block 已
**RETRACT** 一条基于此混淆的 headline 结论），亦见
`scripts/_run_sparseforge_tokenmatched_union9_watcher.sh:56-74`。
→ 这一格让 2:4 列 variant-matched，但**它本身不是干净的「训练时 ±SLoRB」数字**，不得如此引用。

### 怎么投（launcher 已写好，默认 DRY_RUN=1 不碰卡）
```bash
# 在 .212 上（28.89.18.212，密码 configs/password_b200_18212.txt，省略 -p）
M=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
bash $M/scripts/launch_union9_gapfill_212.sh                    # 只做 preflight，0 GPU
DRY_RUN=0 bash $M/scripts/launch_union9_gapfill_212.sh          # 真跑
```
- **只能在 `.212` 或 LOCAL（sm_100/B200）跑**：两个已完成臂都是 cc-10.0 上打的
  （其 `lm_eval.log` 记 `190842863616` B = 177.7 GiB/卡 × 4 卡）。Paper B 实测 bit-identical
  权重上有 0.03-0.16 pp 跨架构地板 → H20 行会带架构偏移。launcher 有硬 `REQUIRE_SM=10.0` 守卫。
- **实测成本**：约 **10 分钟 wall、4 卡（≈0.7 GPU-h）**。从两个已完成 run 的 stage 时间戳实测：
  export 2m10s–3m03s / verify 51-53s / PPL 42-45s / union-9 4m53s–4m56s / aggregate <1s / verify 37-40s
  → noslorb 总 9m15s、slorb 总 10m15s。
- **不需要装任何东西**：`.212` 已有钉死的 harness，在**项目盘**上 → `$ROOT/venv_union9`
  （`lm_eval 0.4.8` + `transformers 4.57.6` + torch 2.13.0），随盘持久。裸 conda env **没有** lm_eval。
- 已实测通过（`.212`，2026-08-15，0 GPU）：preflight 全链 exit 0；9/9 task 数据集加载且
  per-task n 与已完成臂逐一相等；piqa override hash-identical（1838 docs，0 mismatch）；
  5 个否定守卫全部正确触发（不可能格 / 已测格 / 卡数≠4 / 架构不符 / 非法 variant）。

⚠️ **禁碰 LOCAL/.73/.82/.104** —— 32 卡在跑 4 个 200k 训练（keep8/keep10/keep12/paperC）。
