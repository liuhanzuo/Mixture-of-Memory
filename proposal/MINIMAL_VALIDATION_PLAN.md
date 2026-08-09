# Proposal 最小验证计划（MAIN 亲自核实版）

> 2026-08-08 22:5x CST。**执行者：MAIN 本人**（不是 subagent）。
> 背景：今晚三次 workflow/agent 都在 100 分钟内静默 stuck 未交付
> （`af7aa54f` / `wcd4z00jb` / `wvb0oo7sq`），见
> `.claude/.../memory/long-running-subagents-stall-silently.md`。
> 因此改为 MAIN 直接查盘。**本文只包含我亲自跑过命令验证的结论。**

---

## §1 一句话结论

**A01 是唯一现在就能推进的方向**，其四个 gate 里 gate-4（C4 aggregation 预注册）
是纯 CPU 且数据齐备。**B02 的「用现有 sweep 算 per-example oracle」这一前提已被我实测
证伪**（见 §3），是今晚最重要的负面发现 —— 它把 B02 从「今天能跑」降级为「需要新 GPU run」。

---

## §2 实测证据表

| 检查项 | 结果 | 我跑的命令/证据 |
|---|---|---|
| A01 四个 evidence json + regenerator | **全部存在** | `evidence/null_calibration_{p1,obs4}_nperm2000.json` 1004/2442 行；`code/build_null_calibration_table.py` 1056 行 |
| A01 外部输入 `per_example_mmlu.jsonl` | **18 个** | `ls olmo2_mmlu_content_results/*/per_example_mmlu.jsonl \| wc -l` = 18 |
| A01 `data/squad_val.jsonl` | **存在，2000 行** | `wc -l` |
| A01 `results/p1_2/p1_2_summary.json` | **存在** | `ls` |
| A01 三个「不得复活的旧数字」 | **未被误用** ✅ | grep `4.8` / `.2822` / `58/91` 于 `code/` `evidence/` `claims/`：全部只出现在**纠正说明**中（"does not reproduce"、"must not be quoted"、"→ 4.69×"、"canonical 为 split-tie .2845"） |
| B02 六个 SOURCES | **全部存在** | `new_propositions_20260725.md` 217 行 / `P0_20_PHASEB_NOTES.md` 151 / `P0_20_EQLAT_NOTES.md` 92 / `tab_distilled_depth_curve.tex` 28 / `QCMEM_RECALL_SPEED_FRONTIER.md` 133 / `bench_p2_4_depth_quality_latency.py` 210 |
| B02 T21 sweep 是否 per-example | **是**，51 行 = 50 样本 + header，列 `target,output,question,recall` | zwfy6:`ruler_results/qcmem_32b_t21_vt_j3/t21/variable_tracking_16k.csv` |
| **B02 跨 config 能否配对 per-example** | ❌ **不能** —— 见 §3 | 8 个 j 全部 50 样本齐，但 question md5 交集 **0/50**，target 交集 **0/50** |

---

## §3 ★ 关键负面发现：B02 Stage 0 的前提不成立

B02 的 `PROPOSAL.md` Stage 0 原文：

> 使用现有 sweep 计算 per-example oracle action 和 regret：
> - 若 oracle 相对最佳 fixed config 的收益不足，方向关闭；

`SOURCES.md` 指向的核心 sweep 是 T21（`status/QCMEM_RECALL_SPEED_FRONTIER.md`），
Qwen3-32B × vt × `resume_j ∈ {3,6,13,20,27,34,41,48}` × {16k,32k}，n=50/cell。

**我实测**（zwfy6 上逐个读 csv、用 question 全文与 target 列做 md5 集合求交）：

```
j3 unique questions = 50
  j3 ∩ j6  = 0/50  (0% 重叠)
  j3 ∩ j13 = 0/50
  j3 ∩ j20 = 0/50
  j3 ∩ j27 = 0/50
  j3 ∩ j34 = 0/50
  j3 ∩ j41 = 0/50
  j3 ∩ j48 = 0/50
target 列同样 j3 ∩ j6 = 0/50, j3 ∩ j48 = 0/50
```

即 **8 个 config 各自用了完全不同的 50 个样本**（RULER vt 现场合成、样本身份未固定/未记录 seed）。

**后果**：per-example oracle 的定义要求「同一 example 在不同 config 下的表现」，
从而算 `oracle = max_config(quality)` 与 `regret = oracle − best_fixed_config`。
样本不配对 → **oracle 与 regret 都算不出来**，只能算 cell-level 的
「最佳固定 config」，那恰恰是 oracle 要超越的对照，不是 oracle 本身。

**因此 B02 的 compute 等级从 `recompute_existing_data` 修正为 `needs_new_gpu_run`**：
必须先重跑一次 **固定样本集**（同一批 n 个 example 跑全部 config），才能进 Stage 0。

这条修正必须写回 `backlog/B02-adaptive-depth-and-read-budget/PROPOSAL.md`，
否则下一个 agent 会照着「用现有 sweep」再白跑一次。

---

## §4 今天就能跑的最小验证（经我实测前置）

| # | proposal | gate | 决定什么 | 算力 | 前置 | 能否判死 |
|---|---|---|---|---|---|---|
| 1 | **A01** | gate-4 C4 aggregation 预注册 | A01 自己说「不要把恰好超过 10× 作为 headline，因为 C4 aggregation 的合理变体会得到约 7–10×」。预注册一个 aggregation 口径并报告该口径下的 residual fraction，消掉选择性报告嫌疑 | **cpu_only** | ✅ `shared/representation/repr_alignment_results.json` 38820 行在盘 | 否（但去掉一个 reviewer 必攻点） |

**只有这一条通过了「前置实测就绪 + 纯 CPU」双重门槛。** 我不虚报第二条。

---

## §5 需要 GPU 的 gate（现在全部阻塞）

集群现状：LOCAL / .73 / .82 / .104 全满载跑 Paper B resume（keep14 @~45k、keep8 @~124k、
keep10 @~87k、keep12 @~127k），剩余 2.8–8.9 天；`.21` 空闲但**已预留给 SparseForge CAST**。

| proposal | gate | 需要 | 阻塞在 |
|---|---|---|---|
| **B02** | Stage 0 前置：固定样本集重跑 j-sweep | 1 卡 × 8 config × n 样本 | §3 的样本不配对；且需先定 n 与固定 seed 协议 |
| **B04** | 补齐 6/6 bs ladder（现 2/6） | 4 rung × bs16 eval | 已有 watchdog 挂在 .73，等 keep8 训练结束（~5 天）自动跑 |
| **A01** | gate-1 第三模型家族 MC interface | forward pass | 无空闲卡 |
| **A01** | gate-3 OLMo full-fp32 forward | forward pass（fp32 显存大） | 无空闲卡 |
| **A02** | 第一阶段 5 config × 5 benchmark | 25 cell | 未核实 Write-LoRA/Read-LoRA 是否在盘（见 §7） |
| **A03** | 1B floor 判死实验 | 1B × 3 benchmark | 未核实 1B ckpt 是否在盘（见 §7） |
| **A04** | 4 结构 × 3 budget × 2 seed | 24 个 1B 训练 | 显然最贵，且 A04 自己承认现有 ladder 不是 controlled |

---

## §6 promising 评估（我读完 4 个 active PROPOSAL.md 后的判断）

**1. A01（最成熟，风险最低）** —— 四个 construct 的 null calibration 已完成，
residual fraction 8%–77%。最强的点不是「做了 null calibration」（那不新），
而是**用这个协议撤回自己的 headline**：MMLU 两 interface 的 ranking flip 里，
参与 flip 的三个 arm 全部处于或低于 letter interface 的 best-constant floor
`always-D = 0.2689`；限制到两 interface 都有效的四个 arm 后 6 个 pair、0 sign flip、
0 显著 flip。**关键在于用 `.2689` 而非 `.25` chance line** —— 用 `.25` 会错误认为
keep10 有效。这个自我证伪叙事是它最难被 scoop 的部分。
风险：它自己划的新颖性边界很窄（不能主张首创 permutation null / BH / similarity null），
kill 条件之一是「论文只能退化为已有 similarity-null 方法的案例集合」。

**2. A02（ROI 最高但依赖未核实）** —— 第一阶段明写"零新增训练"，若 adapter 真在盘上
就是纯 eval。它的 kill gate 写得很诚实：「若 paired quality CI 仍显著低于 0，
则停止 CoMem 优于 RAG 的叙事，定位为高复用 workload 的 storage/read-compute 方案」。
风险：现有证据「只在 synthetic multikey 8k/16k 上成立」，自然任务迁移是真实的未知。

**3. A03（上限最高但最贵）** —— 「结构压缩后丢失的知识应该住在哪」这个问题
连接 Paper A 与 Paper B，framing 好。但要 6 arm × 4 知识轴，且 kill 条件
「1B pilot 所有知识指标处于 floor」需要先测。

**4. A04（更像 protocol 论文）** —— 它自己说现有 depth ladder 混入五个 confound
（两语料 2.046×、不同 steps、LR grouping bug、未记录 seed、partial-shard），
不能当 scaling law。这反而提示它**可以先只写预注册 protocol 而不跑 24 个训练**。

**backlog 里 B04 最接近成熟**（已有 acc_norm 口径上 Spearman +1.00 p=.0028 的
margin compression），但它自己写明「失败则作为 A01 appendix 的 negative result，
不单独成篇」—— 这个自我定位是对的。

---

## §7 我没能核实的（诚实清单）

> **§7 补充（写完后我立刻补查了 1 与 3，结果如下）**
>
> | 项 | 结果 | 证据 |
> |---|---|---|
> | **A02 Write-LoRA** | ✅ **在盘**（zwfy6） | `outputs/qcmem_writepath_distill_qwen_j12_r32/` **556 MB**，含 `distill_args.json` + `step{500,1000,1500,2000,2500}` —— 正是 task #142「P1.10 write-path distillation, Qwen3-8B, lower-12 Write LoRA」的产物，5 个 checkpoint 齐 |
> | **A02 Read-LoRA** | ⚠️ **可疑** | `outputs/lora_best_ref/` 只有 **512 字节**（可能是 symlink 或占位），**不是真的 adapter 权重**。真正的 CoMem flagship LoRA 路径待定 |
> | **A03 1B pruned ckpt** | ✅ **在盘**（zwfy6） | `outputs/olmo2_probe2_1B_keep7fresh2/` 与 `..._16card/` 两个 |
> | ⚠️ 三者**全部只在 zwfy6**，wzc1 侧 `ls` 全空 | | 跨盘规则：要在 wzc1/.21 用需 `scp -O` |
>
> **对计划的修正**：
> - **A03 的判死实验前置已就绪**（1B keep7fresh2 ckpt 在盘），只缺空闲卡 → 是 §5 里
>   最该在卡空出来时先跑的一条（最便宜、能直接判死）。
> - **A02 的"零新增训练"仍未成立** —— Write-LoRA 有了，但 Read-LoRA 侧
>   `lora_best_ref` 只有 512 B，需要先定位真正的 flagship adapter。
> - 这也意味着 A02 若要在 `.21`（wzc1）上跑，还得先跨盘搬 556 MB Write-LoRA。

1. ~~**A02 的 adapter 是否真在盘上**~~ → **已查，见上方补充**（Write-LoRA 在，Read-LoRA 可疑）
2. **A02 的 `931.9→664.4 ms` 与 RULER `99.19→96.07` provenance** —— 项目历史有
   task #167「paperA latency 931.9/664.4 provenance 三选一」，说明这两个数字曾追不到源。
3. ~~**A03 的 1B pruned ckpt 是否在盘**~~ → **已查，在盘**（见上方补充）
4. **B01/B03/B05–B09 我只读了 B01/B02/B04 三个**，其余六个未读。
   特别是 **B03 有 ~2200 行 SKEPTIC/KILLCHECK 文献审计**，我没读其结论，
   因此**不能判断它是否已被文献判死**。
5. **A01 gate-2「非 MMLU 的 MC benchmark」的前置** —— 我确认了 18 个
   `per_example_mmlu.jsonl` 存在，但**没有逐个打开确认字段齐全**
   （是否含 `option_scores` / `norm_scores` / `label`），也没确认其中哪些是非 MMLU 的。
6. **根目录整理白名单完全没做**（68 目录 + 41 文件的引用检查）。

---

## §8 立即建议

1. **修 B02 的 PROPOSAL.md**，把「使用现有 sweep」改为「需先重跑固定样本集」，
   并附 §3 的实测证据。否则这个错误前提会继续误导。
2. **A01 gate-4 可以现在做**（纯 CPU，数据在盘）。
3. §7 的 1/3/5 三条各是一条 `ls` 命令的事，值得补上再定优先级。
