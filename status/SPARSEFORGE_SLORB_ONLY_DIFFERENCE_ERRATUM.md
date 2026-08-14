# SparseForge token-matched ±SLoRB：「唯一变量」声明的勘误

**日期**：2026-08-14 21:35 GMT+8
**触发**：为回答用户「每个 GPU 节点在训练什么」而跑的 5 节点探针 workflow（`wf_88b00ed6-3e2`），
其 `.212` 与 LOCAL 两条 lane 独立指出了同一个问题。MAIN 已自行复核（见下「我自己的复核」）。

---

## 1. 被勘误的声明

`scripts/_run_sparseforge_tokenmatched_resume.sh:310`：

```bash
`# ---- SLoRB: THE ONLY DIFFERENCE BETWEEN THE TWO ARMS ----` \
--SLoRB $SLORB --SLoRB_k 16 --SLoRB_init_type sum --trainable_projection True \
```

**这句话在启动时是对的，在 2026-08-13 22:2x 节点重启之后不再成立。**

## 2. 实测证据（MAIN 自己复核，不是只信 subagent）

两臂的 `[RESUME]` banner：

| 臂 | 节点 | resume 自 | log |
|---|---|---|---|
| `noslorb`（arm 3） | LOCAL | **iter_num=6700** | `logs/sparseforge_tm_noslorb_RESUME_0814_123843.log` |
| `slorb`（arm 2） | `.212` | **iter_num=6500** | `logs/sparseforge_tm_slorb_RESUME_0814_124313.log` |

复核命令（任何人可重跑）：

```bash
grep -aoE "iter_num[ =][0-9]+" logs/sparseforge_tm_{noslorb,slorb}_RESUME_0814_*.log | head
```

⇒ 两臂**除 `--SLoRB` 外，还差「重启时所处的 iter」**，因而也差：
- cosine 尾部所处的位置（两者 `lr_decay_iters=7125` 相同，但进入重启的点不同）；
- 重启瞬间的 `hardening_x`（它是 `iter_num` 的函数）；
- 物理主机（LOCAL vs `.212`）。

**另**：LOCAL 有实测的单调 +7.9% s/it 漂移，`.212` 没有（`status/GPU_STATUS.md` 20:40 节）。
这只影响墙钟，不影响数值，但它是「两臂并非同机同境」的又一条证据。

## 3. 为什么这**不**毁掉 #246 的科学结论

**读出点的可比性是完好的**，三条独立理由：

1. **resume 是忠实的**。实测 `[RESUME]` 打印：
   `Optimizer state restored` / `Scaler state restored` / `Resuming from iter_num=6700` /
   `Restored eval_count=68` / `Restored best_wiki_ppl=5.176220417022705`，且 **8 个 rank 全部 synced**。
   不是「拿权重重新起跑」，optimizer 动量与调度计数都接上了。
2. **`hardening_x` 是 `iter_num` 的函数，不是「重启后第几步」的函数**。
   脚本 :127 实测注释：`hardening_x = 1.0 at 5293, 0.5 at 6397, 0.0 at 7500`。
   ⇒ **两臂在 iter 7500 都恰好落在 `hardening_x = 0.0`**，即都是**完全 2:4 硬掩码**的终点模型。
3. **预注册的读出点是 iter 7500 的终点 ckpt**（`scripts/_run_sparseforge_tokenmatched_resume.sh:97`
   `MAX_ITERS=7500`，理由见 :90-103 的 budget 块：7500×256×4096 = 7,864,320,000 token，
   对齐 `outputs/cast_repro_zero2/run_manifest.json` 的 `total_tokens`）。
   两臂在该点**token 预算相同、hardening 终态相同**。

⇒ 结论：**这是一个「文档/记账」缺陷，不是「实验」缺陷。**

## 4. 因此，写作时的硬约束

- ❌ **不得**写「SLoRB 是两臂唯一的区别」/「the only difference between the two arms is SLoRB」。
  该句在 artifact 记录里已为假。
- ✅ 应写：**两臂 token 预算、语料、seed(1234)、LR 调度、hardening 终态、拓扑均相同；
  唯一的设计变量是训练期 `--SLoRB True/False`。二者曾于 08-13 因节点重启分别自 iter 6500 / 6700
  忠实 resume（optimizer+scaler+调度计数全部恢复），故重启时刻不同；因 `hardening_x` 由 `iter_num`
  决定，两臂在读出点 iter 7500 均处 `hardening_x = 0.0` 的同一硬掩码终态。**
- ⚠️ 该 resume 差异**必须在方法/附录里出现一次**。它不改变结论，但隐去它就等于让「唯一变量」这句
  错话继续留在论文里 —— 而这正是本项目已经吃过教训的一类（把 flag 的存在当作行为、把默认值当作实际值）。

## 5. 为什么**没有**直接改那行脚本

`scripts/_run_sparseforge_tokenmatched_resume.sh` **当前正被 PID 40229 这个活 bash 执行**
（实测 `pgrep -af _run_sparseforge_tokenmatched_resume.sh`）。bash 是**边跑边读**脚本文件的，
改动一个在跑的脚本可能破坏它剩余 249 iter 的收尾路径（含 finalize + 存盘）。

⇒ 依项目铁律「**不要编辑正在运行的 bash 脚本**」，本轮**只写勘误、不动脚本**。
**两臂都跑完之后**（noslorb ETA ~3.9 h、slorb ~6.7 h，即约 08-15 04:30 之后）应把 :310 那行改成：

```bash
`# ---- SLoRB: the only DESIGN variable. NOTE: the two arms resumed from` \
`#      different iters (noslorb 6700 / slorb 6500) after the 08-13 restart;` \
`#      see status/SPARSEFORGE_SLORB_ONLY_DIFFERENCE_ERRATUM.md ----` \
```

## 6. 同轮发现的其它未记账差异（来自探针，**MAIN 尚未逐条复核**，标注为待核）

探针在把本臂与 **CAST-repro**（`cast_7500`）对比时，指出若干**无任何文档标注**的未匹配旋钮。
这些**不影响** ±SLoRB 的**臂间**对比（两臂彼此一致），只影响「vs CAST-repro 隔离了 mask machinery」
这一说法的强度：

| 旋钮 | 本臂 | CAST | 状态 |
|---|---|---|---|
| `weight_decay` | 0.1（resume 脚本 :284） | 0（`SPEC.md:37`；`cast/train_cast_llama.py` 无该 flag） | **待 MAIN 复核** |
| 蒸馏损失形式 | 未归一化 `1.0*CE + 1.0*KL` | 凸组合 `eta*KL+(1-eta)*CE`，eta=1/3 | **待 MAIN 复核** |
| micro-batch | 8（grad_accum 4/rank） | 1（grad_accum 32/rank） | **待 MAIN 复核** |
| 并行方式 | FSDP hybrid_shard | DDP + ZeRO-1 | **待 MAIN 复核** |

⚠️ 另一条更重要、且**此前已在 `SPARSEFORGE_TOKENMATCHED_PREP.md:23-32` 记录过**的：
**`--mask_metric hessian_obd` 是惰性的** —— `hessian_diag` 被 warm-start 成 1.0 且从不更新，
故打分退化为 `W^2` 的正倍数，即**普通 magnitude 剪枝**。写作中**不得**声称用了 Hessian/OBD 重要性。

> **本节是待办，不是结论。** 在把任何一条写进论文之前，MAIN 必须像 §2 那样自己复核到
> 「文件:行 + 实测输出」。探针给的路径是线索，不是证据 —— 本项目已有先例：这类报告
> 会给出看似精确却偶有编造的细节。

---

## 附：读出点速查（防止把 `max_steps` 当决策点）

| 节点 | run | 读出点 | 依据 |
|---|---|---|---|
| LOCAL | sparseforge noslorb | **iter 7500** 终点 ckpt | resume 脚本 :97 + :90-103 budget 块 |
| `.212` | sparseforge slorb | **iter 7500** 终点 ckpt | 同上 |
| `.73` | paperB keep12 | **step 200000**（此处 `max_steps` **就是**决策点） | `status/PAPERB_TABLE4_BUDGET_DEFECT.md` |
| `.82` | paperB keep8 | **step 200000**（同上） | 同上 |
| `.104` | paperC heal | **step 121000**，**不是** 200000 | `paperC/HEAL_CONFOUND_PREREGISTRATION.md:87` |
