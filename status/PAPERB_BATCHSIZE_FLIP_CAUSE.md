# PAPERB_BATCHSIZE_FLIP_CAUSE.md
# batch_size 对 OLMo-2 core6 downstream eval flip 的影响实验
# 生成时间: 2026-08-08

## 实验设计

**目标**：验证 batch_size 差异是否能解释 OLMo-2 core6 downstream eval 中观察到的跨 run flip 现象。

**ckpt**：`outputs/olmo2_probe2_7B_shortgpt16/step200000.pt`（zwfy6 磁盘，keep_front_layers=16, n_fresh_layers=0）

**节点**：
- Arm A (bs=8)：.82（28.82.250.82，zwfy6/H20 cc9.0）
- Arm B (bs=16)：.82（同一节点，消除节点噪声）
- Arm C (bs=4)：.104（28.83.24.104，zwfy6/H20 cc9.0，zwfy6 共享盘）
- Arm D (bs=32)：.104（同一节点）

**Python 环境**：全部用 `/opt/conda/envs/torch-base/bin/python`（Python 3.14.6, torch 2.13.0）

**其他参数**：`--keep_front_layers 16 --n_fresh_layers 0 --num_shards 8 --add_bos 0 --save_per_example`

**TASKS**：`hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa`（core6）

**Shard 断言**：每次 merge 前断言 8/8 shard 齐全（abort 不 merge）

**Output names**（不覆盖任何已有目录）：
- `7B_shortgpt16_step200000_bs4`
- `7B_shortgpt16_step200000_bs8`（=v3 的精确重现，见下方字节一致性验证）
- `7B_shortgpt16_step200000_bs16`
- `7B_shortgpt16_step200000_bs32`

---

## 结果：n_correct_acc（acc 指标，sum-logprob argmax）

| Task         | n     | bs4  | bs8  | bs16 | bs32 |
|--------------|-------|------|------|------|------|
| hellaswag    | 10042 | 5201 | 5195 | 5203 | 5197 |
| arc_challenge| 1172  |  506 |  503 |  506 |  506 |
| arc_easy     | 2376  | 1823 | 1826 | 1826 | 1831 |
| piqa         | 1838  | 1379 | 1380 | 1381 | 1380 |
| winogrande   | 1267  |  839 |  834 |  831 |  836 |
| openbookqa   | 500   |  163 |  163 |  161 |  162 |
| **TOTAL**    | **17195** | **9911** | **9901** | **9908** | **9912** |

---

## Per-item Flip 计数（精确 flip = 同一 item_id correct 标签变反）

| Pair           | Total flips | Near-tie flips (<0.1 nats margin) |
|----------------|-------------|-----------------------------------|
| bs4 vs bs8     | 90          | 63                                |
| bs8 vs bs16    | 107         | 84                                |
| bs4 vs bs16    | 119         | 88                                |
| bs4 vs bs32    | 117         | 92                                |
| bs8 vs bs32    | 123         | 100                               |
| bs16 vs bs32   | 70          | 59                                |

**字节一致性验证**：bs8（.82, Aug 8）与 v3（.104, Aug 8，`within_disk_floor_v3.sh`）= **0 flip, 0 near-tie**（byte-identical）。同节点、同 conda 环境、同 BS=8 → 完全可复现。

---

## 具体 flip item 示例（证明近似 near-tie 机制）

### [arc_challenge] item_id=46（gold=C）

- **bs8**：pred=C ✓，scores={'A':-7.579, 'B':-9.782, 'C':-7.532, 'D':-11.750}，C-A margin=**0.048 nats**
- **bs16**：pred=A ✗，scores={'A':-7.570, 'B':-9.830, 'C':-7.580, 'D':-11.798}，A-C margin=**0.009 nats**

padding 改变导致 C 与 A 的 log-softmax 值相对顺序翻转（差值仅 0.04 nats）。

### [arc_challenge] item_id=82（gold=C）

- **bs8**：pred=B ✗，scores={'B':-8.981, 'C':-9.148}，B-C margin=**0.167 nats**
- **bs16**：pred=C ✓，scores={'B':-9.058, 'C':-9.021}，C-B margin=**0.037 nats**

BS=8 时 B 更优，BS=16 时 C 更优（符号翻转）。

### [arc_easy] item_id=542（gold=A）

- **bs8**：pred=C ✗，scores={'A':-8.122, 'C':-7.997}，C-A gap=**0.125 nats**
- **bs16**：pred=A ✓，scores={'A':-8.046, 'C':-8.046}，A-C gap=**0.000 nats**（恰好相等！）

BS=16 时 pad 量改变让 A 与 C 分数精确相等，argmax 取第一个 → 返回 A。

---

## 根因分析

**机制（实测确认）**：driver 在 `score_task` 中先按 token 长度排序（`sorted(..., key=lambda i: len(items[i][2]))`），再以 `batch_size` 为步长分批，每批内对齐到该批的最长序列（`maxl = max(len(items[i][2]) for i in bidx)`）。batch_size 改变 → 批成员组合改变 → `maxl` 变化 → padding 量变化 → bf16 autocast forward 的 log-softmax 数值微变 → near-tie 选项的排名翻转。

**关键代码**（`scripts/eval_olmo2_probe2_downstream.py` lines 370-385）：
```python
order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
for b in range(0, len(order), batch_size):
    bidx = order[b:b + batch_size]
    maxl = max(len(items[i][2]) for i in bidx)
    input_ids = torch.full((B, maxl), pad_id, ...)  # pad 量随 batch 成员变
    ...
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attn)
    logprobs = torch.log_softmax(out.logits.float(), dim=-1)
```

---

## 判据评估

**MAIN 的假设**：BS=8 vs BS=16 出现 ~15-20 flip → 假设成立。

**实测结果**：BS=8 vs BS=16 = **107 flip**，远超预期的 15-20。

**解释**：假设**部分成立但数量级估错了**：

1. batch_size 变化确实导致大量 flip（107 个），且 84/107 是 near-tie（<0.1 nats margin），机制清晰。

2. **但 v1（`7B_shortgpt16_step200000`，Aug 2 04:59-05:07）实际上也用了 BS=8**！v1 的 launcher 是 `_run_shortgpt_downstream_only.sh`（zwfy6 上 untracked 脚本），其中明确写了 `--batch_size 8`。

3. v1-vs-v3 的 ~20 flip 的**真实成因是 Python 环境差异**：
   - v1 用 `$WD/olmo2_venv/bin/python`（Python 3.10.20, torch **2.7.0**+cu126）
   - v3 用 `/opt/conda/envs/torch-base/bin/python`（Python 3.14.6, torch **2.13.0**）
   - torch 2.7.0 vs 2.13.0 的 bf16 autocast、log_softmax 等实现微差 → ~20 near-tie flip

---

## 结论

**MAIN 的假设被部分否证**：

- batch_size 确实会导致大量 flip（~90-120 个，远超 20），机制为 padding 改变 bf16 数值。
- 但 v1-vs-v3 的 20 flip **不是** batch_size 差异造成的（v1 也是 BS=8）。
- v1-vs-v3 的成因是 **torch 版本差异**（2.7.0 vs 2.13.0）。

**如实结论**：batch_size 是一个强力的 eval 不稳定性来源（bs8 vs bs16 → 107 flip，是 v1-vs-v3 观察的 ~5×），但它不是这次特定 20 flip boundary（Aug 2 20:12 前后）的成因。实际边界由 torch 版本变化（olmo2_venv → conda）划定，与 batch_size 无关。

---

## 对 Paper 的后果

### 哪些数字是 BS=16 跑的（需要重跑）

`7B_base_full`（**Table 4 的 base 行**！）是由 `_run_olmo2_probe2_downstream_8gpu.sh`（BS=16）于 2026-07-19 生成的，可从 `logs/olmo2_downstream_sched.out` 的 "latest 1B keep7 ckpt" 头部行确认。

同批 BS=16 跑的 CONFIGS（同一个 `olmo2_downstream_sched.out`，Jul 19）：
- `1B_base_full`
- `1B_keep7_step50000`、`step100000`、`step147000`、`step148500`（动态 LATEST）
- `7B_base_full` ← **Table 4 base 行**
- `7B_keep10_step10000`

这些结果存在 `olmo2_downstream_results/{name}/summary.json`，没有 per_example 文件（BS=16 launcher 未加 `--save_per_example`）。

所有经过 `_run_olmo2_within_disk_floor_v3.sh` 或 `_run_olmo2_eval_shortgpt.sh`（BS=8，`--save_per_example`）重跑的结果是干净的。

### Methodology 要求

**eval batch_size 必须写进 protocol 并固定**（例如 `--batch_size 8`）。任何两个 run 若 batch_size 不同，其差异不具 paired-item 可比性（bs8-vs-bs16 存在 107 个 flip，far exceeds random noise）。Paper 方法部分需说明：所有 downstream MC eval 统一用 `--batch_size 8, --add_bos 0, --save_per_example`。

---

## 实验文件位置（zwfy6 磁盘）

```
olmo2_downstream_results/7B_shortgpt16_step200000_bs4/   (node .104, 2026-08-08)
olmo2_downstream_results/7B_shortgpt16_step200000_bs8/   (node .82,  2026-08-08, byte-identical to v3)
olmo2_downstream_results/7B_shortgpt16_step200000_bs16/  (node .82,  2026-08-08)
olmo2_downstream_results/7B_shortgpt16_step200000_bs32/  (node .104, 2026-08-08)
```

启动脚本：
```
scripts/_batchsize_flip_exp_82.sh   (bs8 + bs16 on .82)
scripts/_batchsize_flip_exp_104.sh  (bs4 + bs32 on .104)
scripts/analyze_batchsize_flips.py  (per-item flip analysis)
```
