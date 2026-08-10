# Recall 扫描 + 读出墙突破 — 结果总结（2026-07-05，本机 B200/L20A 自主实验链）

> 主力机 offline 期间，本机接力自主完成。全程官方 `compare_answers` 判分、n=100、
> `--babilong_mix_fraction 0`（无泄漏）。对标标杆见文末。

## 一句话结论

**破 qa5 读出墙的配方 = dense-LM 主信号（continued-pretrain 式 last-chunk LM loss）
+ 微量合成 recall。recall 比例呈倒 U，语义任务甜点在 0.02，qa5 16k=40，
首次超过 MemoryLLM-8B teacher（~38）。但 RULER 精确串检索揭示：recall 比例需按
目标任务类型调（语义 vs 精确串），无单一全能甜点。**

## 一、recall 扫描（qa5，官方判分 n=100，warm b64 buffer64，step500）

| recall | 2k | 4k | 8k | 16k | 说明 |
|-------:|----|----|----|-----|------|
| 0.00（纯 dense-LM） | 61 | 46 | 23 | 14 | 短档最高但中长档崩（没学检索动作） |
| **0.02（语义甜点）** | 52 | 50 | **47** | **40** | ★ 16k 峰值，超 MemoryLLM |
| 0.05 | 45 | 43 | 42 | 37 | |
| 0.15（recipe-fix） | 37 | 52 | 33 | 32 | |
| 0.60（主力机 SFT 旧配方） | 62 | 49 | 28 | 10 | recall 主导，过拟合合成捷径 |

**16k 倒 U 曲线**：`10 → 32 → 37 → 40(峰) → 14`。
**8k 倒 U**：`28 → 33 → 42 → 47(峰) → 23`。

## 二、三重验证（16k=40 是真突破，不是假象）

1. **满 n=100 官方判分**（非 re.search 送分；非小 n 高估：n48→27 → 满 n→32/40 反升）。
2. **buffer 隔离**：recipe-fix ckpt（训练 buffer64）用 `--fifo_buffer_chunks_eval 25`
   评，16k=34（不掉）→ 16k 高分**不是 buffer 容量功劳，是训练配方**（读出能力提升）。
   这推翻了旧结论「16k 断崖 = buffer 容量」。
3. **单调曲线**：4 个 recall 点连成干净单调关系，非单点侥幸。

## 三、RULER 跨任务泛化 → 任务权衡（最有科学价值的发现）

| ckpt | babilong qa5 8k/16k（语义读出） | RULER niah 8k/16k（精确 7 位串） |
|------|------------------------------|-------------------------------|
| recall 0.02 | 47 / 40 | 8 / 4 |
| recall 0.15 | 33 / 32 | 40 / 4 |

**诊断**（看逐样本 output）：recall0.02 在 RULER 只读出数字前缀（714→7144013）或退化重复；
recall0.15 能读全 7 位。**合成 recall 样本（NIAH，答案是精确串）恰恰训练「精确串检索」能力**。

**结论**：
- **语义读出任务（babilong qa5）**：recall 越少越好（dense-LM 主导），甜点 0.02。
- **精确串检索任务（RULER niah）**：需要更多 recall（0.15 才 40）。
- **无单一全能甜点** → recall 比例应按目标任务调。recall0.02 的 babilong 16k=40
  部分是「语义任务特化」（牺牲了精确串检索）。

## 四、论文对标（qa5 16k，有界 memory 闭卷 W0 设定）

| 方法 | qa5 16k | 设定 |
|------|--------:|------|
| **本工作 recall0.02** | **40** | 有界 memory 闭卷 ★ |
| MemoryLLM-8B teacher | ~38 | 同类，**已超越** |
| 项目干净 SOTA（pg19 nctx7） | 16 | **2.5×** |
| 主力机旧 SFT（recall0.6） | 10 | 4× |
| activation-beacon | 72 | 压缩 memory，**下一个要追的标杆** |
| Llama3.1-8B-Inst / GPT-4 | 99 / 94 | 开卷全上下文，不可比 |

## 五、可复现资产

- ckpt：`outputs/mem_space_sft_L8_recall{00,02,05,10}/full_model_step000500.pt`
  + recipe-fix(0.15) `outputs/mem_space_sft_L8_denselm_recall15_eb64/`
- launcher：`scripts/_launch_sft_L8_denselm_recall15.sh`（RECALL 环境变量参数化）
- eval：`scripts/eval_babilong_sample_sharded.sh`（样本分片负载均衡 + adaptive bs + expandable_segments）
- 结论：`status/RESEARCHER_REPORTS.jsonl` rpt_20260705_recall_sweep_readout_wall
- git：commit b8ef5d6 及之前

## 六、下一步方向

1. **recall=0.1 中点**（跑中）：测能否语义+精确两全。
2. **混合 needle 类型数据配方**：同时含词级答案 + 精确串答案，让一个 ckpt 两种读出都强。
3. **向 beacon 72 推进**：当前 40 vs 72 仍有差距（beacon 是压缩 memory + 大量训练）。
4. **LongBench / LongEval** 真实长文档验证（RULER 已做，NIAH 类）。
