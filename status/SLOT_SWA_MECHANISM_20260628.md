# slot + SWA(W0→W6) 读出鸿沟机制 — 干净复算 + 论文论证（2026-06-28）

> 研究 agent。只读 + 复算 + 分析，不跑训练/eval、不改代码。
> 全部分数由 `babilong.metrics.compare_answers`（`third_party/babilong-pkg/babilong/metrics.py`）复算
> 自 `babilong_results/*/...csv`，与 registry 数字逐格核对一致。
> 诚实标注【读码/复算确证】vs【推断/假说(置信度)】。厘清【干净 mix=0】vs【泄漏 mix>0】。

---

## TL;DR（一句话）

**「slot/memory + SWA 远强于纯 W0 读出」是一个干净、可复算、跨两条独立技术线（slot 路由 / FIFO hidden）一致复现的真实现象，其机制 = memory 里存了长程信息但压缩 hidden 读出有损，SWA/token-reforward 把最近(或选中)几块的【原始 token】在 query 在场下重新过全 32 层、重新 contextualize，直接补回读出精度（干净增益 ~2-3×）。这强力支撑论文中心论点「memory 存储有效，读出是瓶颈」，且 slot 线与 FIFO 线指向同一结论。**

★**两处必须诚实厘清（对 brief 的修正）**：
1. 用户印象里的 slot+W6≈40 **确来自泄漏 ckpt**，但泄漏的**只有 P11**（`babilong_mix=0.15`，读码确证）。
2. ★**brief 把 mass_coef / lr5e5 也列为泄漏，这是错的**：二者训练 `babilong_mix=0.00`（读码 + 训练 log 双重确证），是**干净**的。它们的 W6 分数（mass_coef2 qa5 8k=**54**、16k=36）是**合法的干净 slot+SWA 结果**，不是泄漏放大。selfstudy(rawkv) 的干净 swa2 ≈ qa5 16k=26/32k=18 是另一个更保守的干净锚点。

---

## 0. 数据纯净度审计（读码 + 训练 log 确证）

| ckpt 系列 | 训练脚本 | `babilong_mix` | 其它数据 | 纯净度 | 证据 |
|---|---|---|---|---|---|
| **selfstudy** (rawkv grouped readout) | `launch_mem_space_selfstudy_rawkv_chunk512.sh` | **0.0** | pg19 真长文蒸馏 | ✅ 干净 | 脚本 :75 `--babilong_mix_fraction 0.0 --t2_recall_mix_fraction 0.0` |
| **lr5e5** (slot 路由 + l3) | `launch_distill_pg19_nctx63_lr5e5_s1234.sh` | **0.0** | pg19 nctx63 蒸馏 | ✅ 干净 | 脚本 + log `babilong_mix=0.00` |
| **mass_coef1/2** (= T2_recall_MASS, slot) | `launch_T2_recall_chunk512_MASS_coef2_s1234.sh` | **0.0** | `t2_recall_mix=0.5`（pg19 背景的**合成** name→code needle，与 BABILong **不同源**） | ✅ 干净（对 BABILong） | 脚本 :babilong_mix 0.0 + log `babilong_mix=0.00`；T2 不同源见 `RUN_REGISTRY.md:109` |
| **HARDOBJ ctx7/ctx3** (slot) | `HARDOBJ_lastchunk_*` | **0.0** | pg19 | ✅ 干净 | log `babilong_mix=0.00` |
| **★P11** (deltarule/massbias) | `launch_mem_space_p11_chunk512_massbias.sh` 等 | **0.15** | dolmino + **BABILong qa1/qa2/qa5 0k-4k SFT** | ❌ **泄漏** | log `babilong_mix=0.15`（chunk512/256/1024/INSTRUCT/massbias 全 0.15） |

**泄漏机理**（`RUN_REGISTRY.md:22-27` 读码确证）：`babilong_mix=0.15` 时 15% 训练步是 BABILong SFT，且 `--babilong_tasks=qa1,qa2,qa5`、`--babilong_lengths=0k-4k` 与 eval **完全同任务同数据集**，HF 数据集无 train/test 隔离 → 训练池与 eval 池重叠 → 模型背过答案。**P11 的 0k-4k 高分 + 长档放大都不能引用为干净能力。**

---

## 1. 干净 slot/memory ckpt 的完整 W0 vs SWA 阶梯（复算 csv，全 mix=0）

### 1a. selfstudy（rawkv grouped readout，n=100，step500，最佳 ckpt）— brief 主线锚点
csv: `selfstudy_w0_ladder_20260621_0540/` + `selfstudy_step500_swa_20260621_0705/`。**与 RUN_REGISTRY.md:126-134 逐格一致。**

| 任务 | 4k W0 | 4k swa1 | 4k swa2 | 8k W0 | 8k swa1 | 8k swa2 | 16k W0 | 16k swa1 | 16k swa2 | 32k W0 | 32k swa1 | 32k swa2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **qa1** | 17 | 34 | 36 | 19 | 15 | 21 | 12 | 16 | 19 | 8 | 10 | 15 |
| **qa5** | 29 | 41 | 49 | 16 | 31 | (–) | 11 | 19 | **26** | 7 | 11 | **18** |

- ★ qa5 16k: W0=11 → swa2=**26**（2.4×）；qa5 32k: W0=7 → swa2=**18**（2.6×）。qa1 16k: 12→19；qa1 32k: 8→15。
- 注：selfstudy ckpt **只跑了 swa0/1/2**（无 W6/swa6，已确认 csv 不存在）。brief 里"干净真实值~26"即此 qa5 16k swa2。

### 1b. lr5e5 s1234（slot 路由 + l3_summary，**纯 slot 读出**，n=100，step500）— 最干净的"纯 slot+SWA 阶梯"
csv: `lr5e5_s1234_step500{,_swa2,_swa4,_swa6}/`。**与 RUN_REGISTRY.md:60-66 一致。**
（lr5e5/mass 无 `--use_rawkv_readout`，是默认 **TopK selector → slot bank** 读出，比 selfstudy 的 rawkv readout 更纯粹是"slot 读出"。）

| 任务 | 4k W0/W2/W4/W6 | 8k W0/W2/W4/W6 | 16k W0/W2/W4/W6 | 32k W0/.../W6 |
|---|---|---|---|---|
| **qa1** | 17 / 33 / 37 / 37 | 10 / 23 / 27 / **26** | 8 / 18 / 20 / **22** | 6 / – / – / **14** |
| **qa5** | 25 / 56 / 61 / 52 | 16 / 41 / 48 / **49** | 12 / 33 / 33 / **39** | 11 / – / – / **34** |

- ★ qa5 8k: W0=16 → W6=**49**（3.1×）；16k: 12→**39**（3.3×）；32k: 11→**34**（3.1×）。**单调随窗口抬升 W0<W2<W4≈W6。**

### 1c. mass_coef2 s1234（T2_recall_MASS, slot, n=100, final）— 干净 slot+SWA 的最高分
csv: `mass_coef2_s1234_final_W6/`（W6）+ `mass_coef2_s1234_final_swa0/`（W0）。**与 RUN_REGISTRY.md:69 一致（qa5 8k=54）。**

| 任务 | 4k W0/W6 | 8k W0/W6 | 16k W0/W6 | 32k W0/W6 |
|---|---|---|---|---|
| **qa1** | 20 / 43 | 16 / 25 | 8 / 21 | 0 / 10 |
| **qa5** | 32 / 51 | 28 / **54** | 0 / 36 | 8 / 23 |

- ★ qa5 8k: W0=28 → W6=**54**（1.9×，迄今 lr5e5/mass 配方干净最高中程分）；16k: 0→36；32k: 8→23。
- **这是干净 mix=0 的合法结果**，不是泄漏。

### 1d. ★泄漏对照（P11，babilong_mix=0.15，n=100）— 用户"40"的真实来源
csv: `p11_step500_local_swa{0,2,6}/`。**仅作"泄漏放大"展示，不引用为能力。**

| 任务 | 8k W0/W2/W6 | 16k W0/W2/W6 | 32k W0/W2/W6 |
|---|---|---|---|
| **qa1** | 20 / 42 / 53 | 25 / **40** / 43 | 18 / 20 / 30 |
| **qa5** | 48 / 72 / **85** | 45 / 67 / 73 | 44 / 49 / 69 |

- **用户印象的"slot+W6≈40" = P11 泄漏的 qa1 16k swa2=40**（或 qa5 系更高的 67-85）。其 W0 基线本身已被泄漏抬高（qa5 8k W0=48 vs 干净 ~16-28），SWA 在泄漏底座上进一步放大到 85。
- ★诚实结论：**40/85 是泄漏放大**；同配方的干净真实 slot+SWA 是 mass_coef2 qa5 8k W6=54 / 16k=36（合法）、selfstudy qa5 16k swa2=26（最保守）。

---

## 2. 机制：为什么 slot/memory + SWA >> W0？（读码确证）

### 2a. W0 与 SWA 在 eval 代码里到底做了什么（`scripts/run_babilong_mem_space.py:692-768`，确证）
- **流式阶段（两者相同）**：除最后一块外的所有 chunk 逐块 `model(input_ids=chunk)`，只把信息写进 memory（slot bank / rawkv / FIFO buffer），不读 logits（:696-699）。
- **W0（swa_eval_chunks=0）**：生成窗口 = **仅最后一块**原始 token；前面所有 chunk **只能经 memory 读回**（压缩 hidden）。:766 的 SWA 分支不触发。
- **SWA（W=swa_eval_chunks>0）**：生成窗口 = **最后 (W+1) 块原始 token 拼接**，整窗重过一次 forward（:766-768 `window = torch.cat(chunks[start:])`）。于是最后 forward 的 self-attention **直接注意最近 W 块的原始 KV**（且这些 token 在窗内有正确相对 RoPE），**叠加在 memory 读回之上**（纯加法，前面更远的块仍只在 bank 里）。

### 2b. 机制裁决（确证 + 推断 HIGH）
**SWA 的增益 = 把"被压缩进 memory、读出有损"的最近几块，换成"原始 token 在 query 在场下重新过全 32 层"。** 关键差异：
1. **压缩有损 vs 无损原 token**：W0 走 memory（slot：128 槽聚合 / rawkv：grouped pooled hidden）→ 容量压缩 + 写入期固定。SWA 直接喂原始 token-id，无压缩。
2. **query-blind 写入 vs query-conditional 重读**：memory 里的 hidden 是**写入流式时**算的（query 还没出现），是 query-agnostic 的"当时觉得显著的东西"。SWA 重 forward 时 query 已在窗内，每层 needle↔query 双向重新 contextualize → 读得出多跳/消歧信息。
3. **位置**：W0 各 chunk RoPE 从 0 重启；SWA 窗内相对位置正确。（次要，见 §3。）

∴ **「slot 存了粗语义但读出有损，SWA 补原始 token 精度」假说 = 确证。** 这与 FIFO 线的 `HIDDEN_VS_SWA_ANALYSIS_20260626.md` Q1 结论同源：FIFO/W0 是"32 次 query-blind 单跳读冻结快照"，SWA/token-reforward 是"原始 token 在 query 在场下全耦合多跳读"。

---

## 3. ★关键对比：slot+SWA 增益 vs FIFO token-reforward 增益 —— 同源吗？

### 3a. 两条线的干净增益（全 mix=0，复算）

| 线 | 读出机制 | W0(纯 memory 读出) | 加 SWA / token-reforward | 增益 |
|---|---|---|---|---|
| **slot（lr5e5）** | TopK→slot bank（128 槽压缩） | qa5 8k=16 / 16k=12 / 32k=11 | W6: 8k=**49** / 16k=**39** / 32k=**34** | **~3×** |
| **slot（mass_coef2）** | 同上 + T2 训练 | qa5 8k=28 / 16k=0 / 32k=8 | W6: 8k=**54** / 16k=36 / 32k=23 | ~1.9× |
| **rawkv（selfstudy）** | grouped pooled hidden | qa5 16k=11 / 32k=7 | swa2: 16k=**26** / 32k=**18** | ~2.4-2.6× |
| **FIFO（b10 NOLEAK，新 ckpt）** | per-layer hidden FIFO（1:1，无压缩计数但冻结快照） | qa5 8k=**34**（W0，HEARTBEAT） | oracle-token: 8k=**66** / 16k=63 / 32k=58 | ~1.9× |
| **FIFO（早 NOLEAK ckpt）** | 同上 | qa1 8k=12（H_V2_PLAN clean） | hidden-oracle=20 → **token-reforward=50** | hidden 仅 ~1.7×，**token 4×** |

复算来源：`noleak_oracle_token/`（qa1 8k=50/16k=28/32k=33；qa5 8k=25/16k=15/32k=23）、`noleak_oracle/`（hidden 隔离 qa1 8k=20/16k=24/32k=22）。

### 3b. 同源 vs 差异（推断 HIGH）
**机制同源 = YES。** 两条线增益的本质都是「**原始 token 在 query 在场下重新过全 32 层** → 补回压缩/冻结读出丢失的精度」。FIFO 线已用三段判据钉死（`FIFO_FINDINGS_SUMMARY`, `MIDPOINT_CONCLUSION`）：
- 纯 memory hidden 读出 W0 ≤ ~20-34（墙）；
- 隔离正确 needle 的**死 hidden 快照** oracle 也只 ~20-24（读出墙不在"选对"，在"hidden 读不动"）；
- 同一 needle 的**原始 token 重 forward** → 50-66（破墙）。
slot+SWA 是同一现象的"近窗口免选择版"：不挑哪一块，直接把**最近 W 块**的原始 token 重 forward；只要 needle 落在最近 W 块内或其语义被近窗口锚定，就能多读出 ~2-3×。

**差异（机制细节）**：
| 维度 | slot+SWA | FIFO token-reforward |
|---|---|---|
| memory 表示 | **压缩**（128 槽 / grouped pool），写入期 query-blind | **1:1 hidden 快照**（不压缩 count），写入期 query-blind |
| SWA/reforward 选谁 | **最近 W 块**（固定窗口，免选择，免费） | **选中/oracle 的 needle 块**（需选择器；oracle 作弊知道在哪） |
| 增益上限 | 受"needle 是否在近窗口"限制（长档 needle 远则 SWA 够不到 → 32k 增益衰减） | 选对 needle 时不受距离限制（oracle 32k=58）；但**无监督选择是另一半墙**（长档 reader-attn≈随机） |
| 算力 | (W+1)×512 token，每 sample 固定 | k×512，但需先选 |

∴ **同源（原 token 重读破读出墙），差异在"怎么把 needle 弄进重读窗口"**：slot+SWA 用免费的近窗口（短中档有效、长档 needle 远会漏），FIFO-reforward 用选择器定位（潜力更大但卡在无监督选择精度）。

---

## 4. 对论文的意义：支撑"存储有效、读出是瓶颈"中心论点吗？

**支撑，且两条线交叉印证（置信度 HIGH）。**

1. **存储有效**：同一 ckpt、同一已写好的 memory，只把读出方式从"压缩 hidden 读回"换成"原始 token 重读"，分数 ~2-3× 抬升（slot：lr5e5 qa5 8k 16→49；FIFO：W0 12→token-reforward 50）。信息**已在**模型流式时见过的上下文里，否则任何读出方式都拿不出来。→ "存了"。
2. **读出是瓶颈**：W0 把同样的信息读不出来（slot W0 qa5 16k=11-12；FIFO hidden-oracle 即使隔离正确 needle 也只 ~20-24）。→ "读不出"。
3. **两条独立架构（slot 路由 / FIFO 1:1 hidden）+ 两种读出补丁（近窗口 SWA / 选中块 token-reforward）四象限一致** → 不是某一架构的 artifact，是**压缩/冻结 hidden 读出范式的共性瓶颈**。这是论文最强的机制论证骨架。
4. **诚实边界**：(a) 全部干净长档天花板仍受限（slot 32k W6=23-34，FIFO 长档卡选择）；(b) selfstudy/lr5e5/mass 的高分**不含 BABILong 泄漏**，但 P11 的 40-85 含泄漏，**论文严禁引用 P11**。

---

## 5. 对 slot+reforward 方向（E2 正在验）：slot 存储值得用 reforward 读吗？

**值得验，但路线1（chunk 级），不是路线2（孤立 token 级）**（与 `SLOT_REFORWARD_TWO_ROUTES_20260628.md` 一致，置信度 HIGH）。

- slot+SWA 的 ~2-3× 干净增益**直接证明**：slot 写入的上下文里**确有可被原 token 重读榨出的长程信息**，"slot 存储值得用 reforward 读"成立。SWA(近窗口) 已是 reforward 的"免选择特例"且免费拿到 ~2-3×；把它升级成"slot 选中相关 document chunk → reforward 整块"（路线1）是自然的下一步，上限 = FIFO oracle 的 50-66。
- ★但**关键变量从"读出"移到"选择"**：slot+SWA 之所以免费有效是因为近窗口免选择；一旦要够长档远处 needle（32k），就必须**选对块**。FIFO 线已证长档无监督选择 ≈ 随机（reader-attn recall@4≈chance）。所以 slot+reforward 能否超越 slot+SWA，**全压在"slot 的跨 chunk 全局表示选 chunk 是否比 reader-attn 单层 q·k 准"**这个未验问题上（`MIDPOINT_CONCLUSION` 下一方向、E2 Gate Probe）。
- ★机制红线：路线2（只 reforward slot 记的孤立 token）破不了墙——孤立 token 无法 contextualize + 写入期 query-blind top-k 压缩，是 reforward 本要逃离的东西（`SLOT_REFORWARD_TWO_ROUTES` §2b，HIGH）。
- ★工程红线：产出 oracle=66 的是 **FIFO ckpt（bypass slots）**，与 slot ckpt 不同源；路线1 需先加 slot→document-chunk-id 写入通道（~100-160 LOC，当前不存在）。

---

## 6. 诚实标注：确证 vs 假说

**读码/复算确证**：
- 全部 §1 表格分数 = `compare_answers` 直接复算 csv，与 RUN_REGISTRY 逐格一致。
- selfstudy/lr5e5/mass_coef = `babilong_mix=0.0`（脚本 + log）；P11 = `0.15`（log）。
- W0 = 仅最后块原 token + memory；SWA = 最近 (W+1) 块原 token 拼接重 forward（`run_babilong_mem_space.py:692-768`）。
- selfstudy 用 rawkv grouped readout；lr5e5/mass 用纯 slot 路由读出（无 rawkv flag）。
- FIFO 干净线：W0 / hidden-oracle / token-reforward 三档（`noleak_oracle{,_token}/`，b10 HEARTBEAT）。
- 用户"40"对应 P11 泄漏 qa1 16k swa2=40（复算确证）。

**推断/假说（置信度）**：
- "slot+SWA 与 FIFO token-reforward 机制同源（原 token query-conditional 重读破压缩/冻结读出墙）"——**HIGH**（四象限一致 + 读码 + FIFO 三段判据）。
- "差异在选择 vs 近窗口免选择，长档 slot+reforward 成败压在选择精度"——**HIGH**（FIFO 长档选择已证伪、SWA 长档增益衰减）。
- "slot 全局表示在长档选 chunk 是否优于 reader-attn"——**未知/活问题**（E2 待验）。

---

### 附录 — 关键引用
- 评分函数：`third_party/babilong-pkg/babilong/metrics.py`（`compare_answers`/`preprocess_output`）。
- SWA/W0 eval 路径：`scripts/run_babilong_mem_space.py:692-768`（流式 :696-699，SWA 窗口 :766-768，oracle-token 重 forward :733-745）。
- 数据纯净度：`launch_mem_space_selfstudy_rawkv_chunk512.sh:75`、`launch_distill_pg19_nctx63_lr5e5_s1234.sh`、`launch_T2_recall_chunk512_MASS_coef2_s1234.sh`（全 babilong_mix 0.0）；P11 `logs/mem_space_p11_chunk512_*.log`（babilong_mix=0.15）；泄漏机理 `RUN_REGISTRY.md:22-27`。
- 机制分析：`status/HIDDEN_VS_SWA_ANALYSIS_20260626.md`（Q1）、`status/FIFO_ORACLE_ANALYSIS_20260626.md`、`status/FIFO_FINDINGS_SUMMARY_20260627.md`、`status/MIDPOINT_CONCLUSION_20260628.md`、`status/SLOT_REFORWARD_TWO_ROUTES_20260628.md`。
- 干净分数 csv：`babilong_results/{selfstudy_w0_ladder_20260621_0540, selfstudy_step500_swa_20260621_0705, lr5e5_s1234_step500{,_swa2,_swa4,_swa6}, mass_coef2_s1234_final_{W6,swa0}, noleak_oracle, noleak_oracle_token}`；泄漏 `p11_step500_local_swa{0,2,6}`。
</content>
