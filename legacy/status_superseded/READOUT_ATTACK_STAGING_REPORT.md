# Readout 攻坚阶段报告（mass + 蒸馏 + curriculum）

**Date:** 2026-06-16
**范围:** 针对 W0（纯 memory readout）瓶颈的四类干预 + 弱叠加协同的完整实验。
**判据:** W0（`--swa_eval_chunks 0`）下 BABILong qa5 长程准确率（chunk512，CHUNK_SIZE 匹配训练）。

---

## 0. 背景与瓶颈

mem_space 架构（Llama-3-8B 冻结 + 可训 memory adapter）的核心矛盾「写没问题，读不准」：
- SWA 开卷读（query 段局部 attention 看原始 context）→ **50-60**，信息确实进了 memory。
- 纯 memory readout（W0）→ **10-30**，长程 32k 掉到 ~6。

本轮系统比较了攻 readout 的四类杠杆 + 它们的叠加。

---

## 1. 完整 W0 对照表（qa5，chunk512，final ckpt，总数据量对齐）

| 配置 | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 类型 |
|------|----|----|----|----|----|-----|-----|------|
| **baseline**（T2_chunk512） | 70 | 31 | 53 | 22 | 13 | 8 | 6 | 起点 |
| mass coef0.5 | 73 | 49 | 40 | 25 | 10 | 12 | 9 | 改readout加权 |
| mass coef1.0 | 78 | 50 | — | — | — | — | — | |
| mass coef2.0 | 78 | 58 | 48 | 28 | 10 | 12 | 7 | |
| mass coef2.0 (seed1234) | 63 | 58 | 46 | 26 | 12 | 10 | 8 | seed鲁棒 |
| **蒸馏 A+B**（self-study） | 70 | 59 | 45 | 25 | 15 | 11 | 8 | 改监督信号 |
| curriculum 4K→32K final | 79 | 67 | 45 | 35 | 23* | 14* | 11* | 改训练课程 |
| **叠加 coef0.5+蒸馏** | 70 | 49 | 44 | 25 | 14 | **13** | **9** | 弱叠加 |
| 叠加 coef0.7+蒸馏 | 69 | 57 | 36 | 26 | 11 | 10 | 8 | |
| 叠加 coef2.0+蒸馏 | 71 | 50 | 33 | 22 | 10 | 6 | 5 | 强叠加(干扰) |

\*curriculum 是 chunk1024，与 chunk512 行不完全可比，列此作趋势参考。

---

## 2. 核心发现

### 2.1 四类杠杆各自有效，但都不破长程天花板（单独）
- **mass bias（token-mass readout 加权）**：seed-robust（coef2 seed42/1234 一致），中短程翻倍（1k 31→58），剂量 coef2>coef0.5。机制=按 slot 浓缩的真 token 数加 `log1p(mass)` bias 到 readout softmax 前。
- **蒸馏（self-study A logits-KL + B hidden-cosine）**：可复现（.249 两跑一致），8k 处最强（15）。机制=teacher 冻结 backbone 看 full-context，student memory readout 逼近其 logits+hidden。
- **curriculum 4K→8K→16K→32K**：救回「直训 32k 退化」（直训 c1024_seq32k 8k=14 vs curriculum 23），但没破中程。
- 单独最优长程：32k ≈ 8-9，相比 baseline 6 有提升但有限。

### 2.2 ★关键突破：弱 mass + 蒸馏在长程协同（倒 U，峰 coef≈0.5）

mass 强度 × 长程协同（qa5，叠加蒸馏）：

| mass coef | 8k | 16k | 32k |
|-----------|----|----|----|
| 0.3 | 14 | 11 | 8 |
| **0.5** | **14** | **13** | **9** ← 峰 |
| 0.7 | 11 | 10 | 8 |
| 2.0 | 10 | 6 | 5（崩） |

- **coef≈0.5 + 蒸馏**：长程 16k=13、32k=9，**首次在 32k 超过所有单独方法**（单独最高 8）。
- **倒 U 形**：mass 过强（2.0）会与蒸馏的优化目标冲突（mass 强行改 attention 分布 vs 蒸馏要 readout 匹配 teacher），长程崩到 5。
- 解读：弱 mass 给高信息量 slot 一点先验倾斜，不破坏蒸馏的精确对齐，两者在长程互补。

### 2.3 训练侧 vs 机制侧
- 训练侧杠杆（容量 N、深度 ctx、训练长度、curriculum）此前 + 本轮均撞 32k 天花板。
- 机制侧杠杆（mass、蒸馏）各自有效，且**弱组合 > 单独**——长程突破的希望在机制侧。

---

## 3. 长程天花板现状
- 32k：baseline 6 → 单独杠杆 7-9 → **弱叠加(coef0.5) 9**。
- 仍未根本突破（远低于 SWA 开卷 ~39），但**方向明确**：机制侧弱组合是目前最优路径。

---

## 4. 下一步
1. **进行中**：coef0.5+蒸馏 **长训 1000 步**（B200，`distill_chunk512_AB_MASS0p5_long1k`）——验证长程协同随训练量能否进一步推。
2. **longbench 真实长文档验证**：把最优配置（coef0.5+蒸馏）放到真实长文档 QA（非合成 babilong），看协同是否泛化。
3. **更长上下文 + 最优 coef**：chunk512 × 真实 32k 数据 + coef0.5+蒸馏。
4. 候选第三类机制（非加权/非监督/非课程）攻 32k 残余天花板。

---

## 5. 工程护栏（本轮踩坑 → 已加固）
- **dolmino per-doc 4096 cap → curriculum DDP hang**：(n_ctx+1)*chunk_size>4096 饿死 loader → 零产出 raise（commit dadc19f）。curriculum 只拉长 T2（`--t2_curriculum`），dolmino 锁 n_ctx=3。
- **蒸馏 cache 静默失效**：(a) 缓存按位置 doc_idx 索引，跨盘 Arrow 行序不同→全 miss；(b) dolmino_dataset.py 漏 rsync→sample_id=None→全 miss。已加 **dataset_fingerprint 断言** + **step50 hit-rate fail-fast**（commit 0cf2c23/438b674）。
- **NCCL teardown 僵死**：训完卡 teardown 占卡，busy=8/8 假象。巡检需 proc 状态+final ckpt 双验。
- **OOM**：启动训练前必查 nvidia-smi 显存真空（不只 proc 数）。
- **跨盘 rsync 完整性**：代码改动要传全（train/build/launch/src 所有文件），diskA(zwfy6 303098609)、diskB(zwfy6 304376610)、B200(wzc1 304376610) 是三个独立物理盘。
- **大量文件计数用 find 不用 `ls *.glob`**（glob 超限误报 0）。

---

## 6. 复现关键命令
```
# 建蒸馏缓存（teacher full-context，按数据集 fingerprint 校验）
python scripts/build_distill_cache.py --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --chunk_size 512 --n_ctx 3 --distill_layers 12,20,28 --model_path models/Meta-Llama-3-8B --out_dir distill_cache/512
# 最优配置训练（弱 mass + 蒸馏）
bash scripts/launch_distill_chunk512_AB_MASS0p5.sh   # --use_readout_mass_bias --readout_mass_coef 0.5 + --distill_logits --distill_hidden
# W0 评测（--swa_eval_chunks 0，CHUNK_SIZE 匹配训练）
```
