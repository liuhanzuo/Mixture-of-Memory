# QCMem (CoMem) Paper — Benchmark 计划表（协作分工用，全量方案）

> 建立 2026-07-14。QCMem 用 **instruct 模型**（需 follow instruction）。**Qwen3 全家做 scale，全 benchmark 全量跑，每个 scale 都测全 baseline。**
> 用法：认领后在「Owner」列写名字，跑完在状态列更新（✅=n100完成 / 🟡=部分 / ⬜=待测 / ⏳=模型/harness未就位）。

---

## 0. 固定协议（所有 cell 统一，保证同口径）

| 项 | 规定 |
|---|---|
| 模型 | **Qwen3-Instruct 全家 7 档**：0.6B / 1.7B / 4B / 8B / 14B / 32B / 30B-A3B(MoE) |
| split-depth **j** (≈0.33L) | 0.6B→**9**(L28) / 1.7B→**9**(L28) / 4B→**12**(L36) / 8B→**12**(L36,已验证) / 14B→**13**(L40) / 32B→**21**(L64) / 30B-A3B→**16**(L48) |
| chunk_size | 512 |
| selector | 主 `bm25`；vt 加 `iter_bm25`(hop4,k16)；对照 `reader_attn`/`oracle`/`recency` |
| topk | 主 12（RULER 网格扫 4/8/12/16/24） |
| adapter | 自蒸馏 LoRA（in-window 追平 dense）；报告标注 with/without |
| 样本 n | **RULER/BABILong = n=500/cell**（对齐官方 RULER，2026-07-14 用户定）；LongBench/LoCoMo/LongMemEval/∞Bench/HELMET = **全测试集**（固定大小，无 n 选择）；sanity n=30。⚠️现有 8B 的 RULER/BABILong 是旧 n=100，需**重跑到 n=500** |
| 判分 | RULER=`string_match`(官方)；BABILong=`TASK_LABELS`+`compare_answers`(禁 re.search)；LongMemEval/LoCoMo=QA acc(judge/EM) |
| 长度档 | 合成(RULER)：8k/16k/32k/64k/128k(**含超窗口=卖点**)；BABILong：0k-32k；真实任务按数据集原长 |

## 0b. 模型就位状态（下载队列）
| 模型 | L | j | 位置 | 状态 |
|---|---|---|---|---|
| Qwen3-0.6B | 28 | 9 | 本地+diskB | ✅就位 |
| Qwen3-1.7B | 28 | 9 | 本地+diskB | ✅就位 |
| Qwen3-4B | 36 | 12 | — | ⏳需下载 |
| Qwen3-8B(instruct) | 36 | 12 | 本地+diskB | ✅就位(主) |
| Qwen3-14B | 40 | 13 | diskB | ✅就位 |
| Qwen3-32B | 64 | 21 | — | ⏳需下载(~65GB) |
| Qwen3-30B-A3B(MoE) | 48 | 16 | — | ⏳需下载(~60GB) |

---

## 1. 主表：Benchmark × 模型（每格 = 一个 eval 任务；每格内跑全 baseline，见 §2）

| Benchmark | 测什么 | 0.6B | 1.7B | 4B | **8B(主)** | 14B | 32B | 30B-A3B | Owner |
|---|---|---|---|---|---|---|---|---|---|
| **RULER** niah_single | 单针 8k-128k | ⬜ | ⬜ | ⏳ | 🟡n100→需500 | ⬜ | ⏳ | ⏳ | |
| **RULER** niah_multikey | 多针消歧 | ⬜ | ⬜ | ⏳ | 🟡n100→需500 | ⬜ | ⏳ | ⏳ | |
| **RULER** vt | 多跳链 iter_bm25 | ⬜ | ⬜ | ⏳ | 🟡n100→需500 | ⬜ | ⏳ | ⏳ | |
| **BABILong** qa1/2/5 | 长噪声推理 0-32k | ⬜ | ⬜ | ⏳ | 🟡n100→需500(缺0-4k) | ⬜ | ⏳ | ⏳ | |
| **LongBench** | 真实任务 QA/summ | ⬜ | ⬜ | ⏳ | ⬜ | ⬜ | ⏳ | ⏳ | |
| **LongEval** | 行检索 | ⬜ | ⬜ | ⏳ | 🟡 | ⬜ | ⏳ | ⏳ | |
| **LoCoMo** | 长对话记忆 | ⬜ | ⬜ | ⏳ | 🟡 | ⬜ | ⏳ | ⏳ | |
| **LongMemEval** ⏳harness | 长期记忆5能力 | ⬜ | ⬜ | ⏳ | ⬜ | ⬜ | ⏳ | ⏳ | |
| **∞Bench/InfiniteBench** ⏳harness | 100k+ 综合12任务 | ⬜ | ⬜ | ⏳ | ⬜ | ⬜ | ⏳ | ⏳ | |
| **HELMET** ⏳harness | 全面长上下文7类可控128k | ⬜ | ⬜ | ⏳ | ⬜ | ⬜ | ⏳ | ⏳ | |
| **vs-Dense 效果** | 128k崩塌对比 | ⬜ | ⬜ | ⏳ | ✅(mk 128k=0/100) | ⬜ | ⏳ | ⏳ | |
| **vs-Dense 速度** | prefill/decode TPS+显存 | ⬜ | ⬜ | ⏳ | ✅(57×/32-68×) | ⬜ | ⏳ | ⏳ | |
| **split-j sweep** | 验证 j≈0.33-0.40L 通用 | ⬜ | ⬜ | ⏳ | ✅ | ⬜ | ⏳ | ⏳ | |

## 1b. 蒸馏策略（2026-07-14 用户定：先只 8B）

### 表1 — 每模型是否蒸馏（每 cell 的 QCMem 报法）
| 模型 | zero-shot(免训练) | +self-distill adapter | adapter 路径 | 说明 |
|---|:---:|:---:|---|---|
| Qwen3-0.6B | ✅报 | ✗ | — | 只 zero-shot |
| Qwen3-1.7B | ✅报 | ⬜可选(P2) | 待蒸 | scale 一致性可补 |
| Qwen3-4B | ✅报 | ✗ | — | 只 zero-shot |
| **Qwen3-8B(主)** | ✅报 | **✅报** | `outputs/qcmem_distill_qwen_j12_r32_4k` | ★唯一必蒸 |
| Qwen3-14B | ✅报 | ⬜可选(P2) | 待蒸 | scale 一致性可补 |
| Qwen3-32B | ✅报 | ✗ | — | 只 zero-shot |
| Qwen3-30B-A3B | ✅报 | ✗ | — | 只 zero-shot |
- **zero-shot** = 免训练、stock backbone 直接跑（核心卖点，所有模型都报）。
- **+adapter** 作用：支撑 abstract「in-window matches dense」+ 把可用 j 推深。仅 8B 必做，1.7B/14B 可选（P2 不阻塞）。

### 表2 — 蒸馏方法（self-distillation，无外部 teacher / 无标注）
| 角色 | 配置 | 梯度 | 作用 |
|---|---|---|---|
| **Teacher** | QCMem `resume_j=0`（adapter DISABLED, `no_grad`）= 精确全 forward | 冻结 | 取每 loss token top-k=64 logit 支撑（无损上界） |
| **Student** | QCMem `resume_j=j`（默认12）+ LoRA on `layers[j:]` | 仅 LoRA 可训, backbone 冻结 | 从第 j 层浅缓存重算上层, 学着还原 teacher |
| **同一份权重** | adapter on/off 切换充当 student/teacher | — | 不额外占显存、无需外部 teacher/标签 |

**Loss** = teacher top-k 支撑上的**双向 top-k KL** `λ·KL(p‖q)+(1-λ)·KL(q‖p)`（λ=0.6）+ 可选极小 CE-to-argmax（默认0）。纯蒸馏，让「深度 j=12 的读出」逼近「深度 j=0 全 forward 的读出」。

### 表3 — 蒸馏超参（默认，per-backbone 各一个 LoRA）
| 项 | 值 | 项 | 值 |
|---|---|---|---|
| resume_j | 12（=split 在第12层, ≈0.33L）| lora_rank | 32 |
| chunk_size | 512 | n_ctx | 7 →(7+1)×512=**4096-tok 窗口** |
| teacher_topk | 64 | distill_lambda | 0.6 |
| ce_weight | 0（默认关）| total_steps | 1000 |
| lr | 1e-4 | warmup | 50 |
| grad_ckpt | on | 成本 | ~1-2 GPU 时（很便宜）|

- **数据**：PG19 文本，on-the-fly chunk。**效果**：in-window 追平 dense + 可用 j 从 ~9 推到 ≥12（脚本注释例：BABILong qa5 .14→.67）。
- 脚本：Mixture-of-Memory `scripts/train_qcmem_distill.py` / COMem `train/distill.py`（`python -m train.distill --model <hf> --j auto`）。

## 2. 方法对照（每个 cell 内全部跑 = 全 baseline，每 scale 都测）
| 方法 | 说明 |
|---|---|
| **QCMem** bm25 / iter_bm25 / +adapter | 我们（主） |
| Dense / full-ctx | 窗口内上界，超窗口崩塌对照 |
| KV-Direct | 全深度重算无检索(resume_j=0) |
| HCache | 中层无检索 |
| StreamingLLM | recency 固定预算 |
| MemoryLLM | 外部固定记忆 |

## 3. Benchmark 就位度（harness 工程量）
| benchmark | 脚本 | 状态 |
|---|---|---|
| RULER | `eval_ruler_qcmem.py` | ✅就位(上限32k, 超窗口走 bench 脚本) |
| BABILong | `eval_qcmem_babilong.py` | ✅ |
| LongBench | `eval_qcmem_longbench.py` | ✅ |
| LongEval | `eval_qcmem_longeval.py` | ✅ |
| LoCoMo | `eval_qcmem_locomo.py` | ✅ |
| vs-Dense | `bench_qcmem_vs_dense.py` | ✅(含128k) |
| **LongMemEval** | — | ⏳需接(长期记忆chat, ~500q, 5能力: 抽取/多会话/时序/知识更新/弃权; LongMemEval_s~115k) |
| **∞Bench** | — | ⏳需接(avg 100k+, 12任务) |
| **HELMET** | — | ⏳需接(7类别, 可控到128k, 含RAG/re-rank/QA/summ) |

## 4. 分工建议
- **一人认领「一列（模型）」**最干净：跑该模型全 benchmark × 全 baseline，output `ruler_results/qcmem_<model>_<bench>/`。
- diskB 三节点共享 FS → task-pool 动态调度多卡排空。
- 大模型(32B/30B-A3B)显存：H20 97GB 单卡可放，QCMem read pack 恒定省显存；Dense baseline 长档可能吃紧，必要时分片。
- 跑完更新本表 + `status/RUN_REGISTRY.md`。

## 5. 待办（工程 prep，非 eval）
1. **下载** Qwen3-4B / 32B / 30B-A3B(instruct) → 本地+diskB。
2. **接 harness**：LongMemEval / ∞Bench / HELMET 的 QCMem 适配（loader + QCMem generate 接口 + 官方判分）。
3. 各新模型跑 QCMem **self-test**(j=0 与全 forward logit diff<1e-4) 确认代码适配。
4. 各新模型自蒸馏 **adapter**（in-window 追平 dense 需要）。

---
> ⚠️ 全量方案工作量巨大（7 模型 × 10+ benchmark × 6 方法）。建议执行顺序：①下载+接harness ②self-test各模型 ③8B 收尾(基准) ④按模型列铺开 scale。

---

## 6. 启动命令（copy-paste，权威）

> **COMem 仓库**（collaborator 用，`git@github.com:liuhanzuo/COMem.git`）：`--n` 默认已=500（RULER/BABILong），真实集全量。
> 环境：`pip install -r requirements.txt`；模型传 HF 路径或本地路径；`--j auto` 自动查 model_registry。

### 6.1 COMem 一行跑单 cell（collaborator）
```bash
cd COMem
# RULER (n=500默认): 单模型全长度
./run_cell.sh ruler <model_path> --lengths 8k,16k,32k,64k,128k --selector bm25 --j auto
# vt 用 iter_bm25
./run_cell.sh ruler <model_path> --tasks vt --selector iter_bm25 --lengths 8k,16k,32k --j auto
# BABILong (n=500默认)
./run_cell.sh babilong <model_path> --tasks qa1,qa2,qa5 --lengths 0k,1k,2k,4k,8k,16k,32k --j auto
# LongBench / LoCoMo (全测试集, 不传--n)
./run_cell.sh longbench <model_path> --j auto
./run_cell.sh locomo   <model_path> --locomo_data <path> --j auto
# 带自蒸馏 adapter (仅8B): 加 --adapter
./run_cell.sh ruler <model_path> --adapter <adapter_dir>/final --j auto
# baseline (每 scale 都测): --baseline dense|kvdirect|hcache|streamingllm
./run_cell.sh ruler <model_path> --baseline dense --j auto
# 或 dispatcher 形式
python -m eval.run --benchmark ruler --model <model_path> --j auto --n 500 ...
```
每模型 j 由 `--j auto` 自动取（0.6/1.7B→9, 4/8B→12, 14B→13, 32B→21, 30B-A3B→16）。

### 6.2 自蒸馏 adapter（仅 8B 必做，1.7B/14B 可选）
```bash
cd COMem
# 单卡
python -m train.distill --model <Qwen3-8B_path> --j auto --data <pg19.jsonl> --out outputs/comem_distill_8b_j12
# 多卡
torchrun --nproc_per_node 8 -m train.distill --model <path> --j auto --data <pg19.jsonl> --out <dir>
# 产出 <dir>/final 喂给 eval 的 --adapter
```

### 6.3 内部（Mixture-of-Memory，diskB 三节点 task-pool 批量铺 n=500）
```bash
# 8B RULER n=500 重铺(旧n100作废): 队列worker多卡排空,每卡抢cell
export MODEL_PATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
export LORA_CK=outputs/qcmem_distill_qwen_j12_r32_4k/final   # 带adapter版; zero-shot则 LORA_CK=none
export OUT_DIR=ruler_results/qcmem_n500_8b SELECTOR=bm25 NAME_PREFIX=qcmem_n500
export PYTHON_BIN=/opt/conda/envs/torch-base/bin/python
# ⚠️ drain脚本内 --limit 需从100改500 (scripts/_qcmem_n100_drain.sh 复制为 _qcmem_n500_drain.sh 改 --limit 500)
for dev in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$dev setsid nohup bash scripts/_qcmem_n500_drain.sh q 1 $dev >logs/qcw_n500_dev$dev.out 2>&1 &
done
# eval入口(单cell): scripts/eval_ruler_qcmem.py --model_path $M --lora_adapter $CK --resume_j 12 \
#   --selector bm25 --topk 12 --ruler_tasks niah_single --lengths 16k --limit 500 --chunk_size 512
# BABILong n=500: scripts/eval_qcmem_babilong.py ... --limit 500
# 超窗口崩塌(vs-Dense, 128k): scripts/bench_qcmem_vs_dense.py --mode accuracy --n_acc 100 (崩塌用n100够)
```

### 6.4 节点/环境速查
- diskB 三节点(28.85.35.73/28.82.250.82/28.83.24.104, 36000端口): `PYTHON_BIN=/opt/conda/envs/torch-base/bin/python`, 共享FS免同步。
- 本机 wzc1 L20A: `.venv/bin/python`(torch2.13)。
- 判分: RULER `string_match` / BABILong `TASK_LABELS`+`compare_answers`(禁 re.search)。
- ⚠️ RULER eval_ruler 路径长度上限 32k; 64k/128k 超窗口档走 `bench_qcmem_vs_dense.py`。
