# 训练稳定化方案报告（2026-06-07，纯调研，未改代码/未跑训练）

## 关键事实（已核对代码 + 日志）
- 公共配置：chunk512、5000 步、lr 1e-4、warmup 100、bf16、DDP8、grad_ckpt、grad_accum=4、batch=1、n_ctx=3、babilong_mix=0.15、grad_clip=1.0、proj_grad_clip=0.1（`train_mem_space_dolmino_cpt.py:718-723`；launch:25-41）。
- **决定性对照**（同 lr/warmup/batch，唯一变量是算法 flag）：
  - **P10**（`--use_st_gumbel_topk` + key_rep0.05）：lm p50=3.45 / **p95=5.21 / max=10.42**，巨 spike 在 step1370-1405、1495-1555、2465-2525（lm 6-10）。
  - **P11**（`--use_delta_rule_writeback --normalize_readout`，无 ST-Gumbel）：lm p50=2.19 / **p95=2.73 / max=4.81**。**P11 抖动幅度仅 P10 的一半，无大 spike。**
- 两 run 全程 **nf=0**（无 non-finite，grad_clip/bf16 没溢出）。
- lm 与 route_aux **强反相关**（step4240 lm3.47/ra1.40；step4245 lm1.58/ra3.69；step4435 lm1.85/ra3.57）→ 单步 lm 高低主要由"该 batch 命中啥"决定。

## 根因诊断
1. **ST-Gumbel 随机路由是 P10 大 spike 的主因（high）**。P10 vs P11 唯一危险变量是 ST-Gumbel（`selector.py` 训练期给 selection logits 加 Gumbel 噪声）。它让每步选中的 slot 集合随机抖，注入内容随机 → teacher-forcing lm 大幅震荡，max 冲到 10.4。P11 用确定性 topk + bounded readout，max 仅 4.8。
2. **注入频率 × 缺约束放大抖动（high）**：注入频率=seq_len/chunk_size，chunk 越小越频繁；F2/RUN_REGISTRY 已实证 chunk128 step1000 撞 spike 崩成 token 重复（PPL~3000）。P11 的 `normalize_readout`（L2-norm+rescale 到 local-attn 尺度，`layer.py`）把注入幅度钉住 → 直接压低方差，是已验证有效的稳定器。
3. **warmup=100 偏短 + 无 spike 防护（medium）**：warmup 仅占 2%，且 optimizer 无"loss 突增则 skip step"保护（`:1838-1846` 只在 step_valid_micros>0 时无条件 step）。坏 batch/随机路由的尖梯度会直接落到权重。
4. **数据是次要因素（medium）**：batch=1 + n_ctx=3，单步信号方差天然大；babilong_mix=0.15 混入异质样本进一步加噪。但这是基础噪声，不是 P10 那种 max10 的 spike 来源。
5. 已排除：bf16 数值溢出（nf=0）、grad_clip 失效（已生效，clip=1.0/0.1）、aux_loss 爆炸（aux 全程 0.01 量级，稳定）。

## 稳定化方案清单（小→大 / 快→慢）
1. **[high, 最快] 默认关 ST-Gumbel 或调低温度**：去掉 `--use_st_gumbel_topk`（或 temp 1.0→0.3）。机理=去掉路由随机源。风险：可能略降 routing 探索，但 P11 证明确定性路由更稳且 BABILong 不差。
2. **[high, 已验证] 默认开 `--normalize_readout`**（必要时 +`--use_delta_rule_writeback`）：把注入幅度钉到 local-attn 尺度。P11 实测直接半砍方差。风险：极低。
3. **[high, 小改] 加 loss-spike skip**：在 `:1800` 已有 nf 检查旁加"若 lm > running_mean+kσ（如 k=3）则本 micro 不计 backward / 跳过 optimizer.step"。机理=拦截尖梯度污染权重，正是当前缺失的防护。风险：阈值需调，过紧会丢正常 batch。
4. **[medium] warmup 100→300~500**：`--warmup_steps`。机理=让 adapter 在高 lr 前更充分对齐冻结 backbone。风险：纯收益，仅稍慢热身。
5. **[medium] 提高有效 batch 降方差**：grad_accum 4→8/16（或多机）。机理=平均掉单步坏 batch。风险：步进变慢，显存基本无增（accum）。
6. **[medium] peak lr 1e-4→5e-5 + cosine（已是 cosine）**：降尖梯度幅度。风险：收敛略慢，可能需更多步。
7. **[low, 已是事实标准] ckpt 选择按 TF-loss-min 而非固定步数**：F2 已确认 step1000 常撞 spike，交付一律取早 ckpt（step500）+ 优选大 chunk。这是兜底，不解决 spike 本身。

## Top3 推荐（按性价比）
**① 关 ST-Gumbel + 开 normalize_readout（P11 配方）** → 已实证半砍方差、无大 spike；**② 加 loss-spike skip 防护**（小改、补当前缺失的尖梯度拦截）；**③ warmup 100→300**（零风险微调）。
