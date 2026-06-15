# v21 — Self-Study 蒸馏：teacher–student 共享冻结 backbone（logits + hidden 双蒸馏）

**Date:** 2026-06-15
**Flags（设计阶段，尚未实现）:** `--distill_logits`, `--distill_hidden`, `--distill_lambda 0.6`, `--distill_layers 12,20,28`, `--distill_cache_dir <path>`, `--distill_weight 1.0`
**References:**
- Attention Matching (arXiv:2602.16284) — per-head attention output + mass 匹配，闭式解
- Cartridges / Self-Study (arXiv:2506.06266) — logits context-distillation
- KV-Distill (arXiv:2503.10337) — 双向 KL，λ=0.6

---

## 1. 背景与动机（W0 瓶颈）

当前 mem_space 架构（P8/P11，`--use_memory_xattn`）：Llama-3-8B 主干**冻结**，外挂可训练 memory adapter（slot bank + L3 summary + `MemoryCrossAttentionRead` readout，≈4.87B 可训练参数）。超长上下文切 chunk，context chunk 在 `no_grad` 下流式写入 memory bank，再由 readout 把 memory 注入回 hidden 来回答 BABILong。

**已诊断的核心矛盾「写没问题，读不准」**：
- 用 SWA 开卷读（`--swa_eval_chunks ≥1`，让 query 段的局部 attention 直接看到原始 context）能到 **50–60 分** → 信息**确实进了** memory / context。
- 纯 memory readout（`--swa_eval_chunks 0`，即 **W0**，query 段只能靠 slot readout）只有 **10–30 分**，长程 32k 掉到 ~10 → readout **没把信息精确取出来**。

**当前训练 loss 的信息量太小**：
- dolmino 走 `dolmino_train_step`（`scripts/train_mem_space_dolmino_cpt.py:1482`）：context `no_grad` 流入 → detach → target chunk 算 **next-token CE**（one-hot）。
- T2 合成 needle 走同一个 `dolmino_train_step` + `answer_mask`（`niah_chunked_dataset.py:287`）：只在 5 个 answer 数字位算 CE（one-hot）。

两条都是 one-hot 监督——每个 token 只给「正确 id」1 bit 级别的信号，对「memory 里该取什么」几乎没有形状约束。readout 学到「答对那个 token」即可，不需要在表征层面对齐 full-context 的行为。

**Self-study 蒸馏的关键洞察**：backbone 冻结，**teacher 与 student 共享同一套权重**，区别只在：
- **teacher**：原始 attention，**看 full-context**（把所有 chunk 拼成一条序列，一次 forward，无 memory）。
- **student**：memory readout（context 切 chunk `no_grad` 写入、detach，再 readout）。

所以 teacher 的 full-context 输出（answer 段每个 token 的软分布 + 每层 hidden）就是 student readout 应当复现的**金标准**——而且因为 backbone 冻结、teacher 不含可训练参数，teacher 的输出对每条固定语料是**确定的**，可以**离线缓存**，训练时零额外 forward。把 one-hot CE 升级成「逼近 full-context 行为」的稠密监督，直接对 W0 readout 施压。

---

## 2. 范式总览

```
离线阶段（预跑一遍，存盘）:
  for each (固定) 训练样本 = [ctx_chunk_0 ... ctx_chunk_{n-1}, target_chunk]:
      teacher = 冻结 Llama-3-8B（未 patch / memory off）
      flat = concat(ctx_chunks, target_chunk)              # 一条长序列
      out = teacher(flat, output_hidden_states=True)
      在 answer 段（见技术点1）抽取:
        p_t   = softmax(logits)         → top-64 稀疏存盘
        h_t^ℓ = hidden_states[ℓ]         → 仅选定层 (技术点2) 存盘
      → distill_cache/<sample_id>.npz

训练阶段（student，与现在一致的 chunked 流式）:
  student_out = memory-readout forward(target_chunk)        # 现有 dolmino_train_step
  q_t   = softmax(student logits)         # answer 段
  ĥ_t^ℓ = student hidden[ℓ]               # 同层同 token
  loss = lm_ce(现有 one-hot)                                 # 保留
       + distill_weight * [ A: 双向KL(p_t‖q_t) + B: hidden_match(ĥ, sg(h)) ]
```

A、B **共享同一份 teacher forward 缓存**（同一次 teacher forward 同时产出 logits 与 hidden）。

---

## 3. 方案 A — logits 蒸馏（双向 KL）

teacher full-context forward 在 answer 段第 t 个 token 上产出 next-token 软分布 `p_t = softmax(z_t^teacher / T)`；student memory-readout 在同一 token 上产出 `q_t = softmax(z_t^student / T)`。

$$
\mathcal{L}_A = \frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}} \Big[\;\lambda\,\mathrm{KL}(p_t \,\|\, q_t)\;+\;(1-\lambda)\,\mathrm{KL}(q_t \,\|\, p_t)\;\Big]
$$

- `𝒜` = answer 段 token 集合（技术点1）。
- **λ = 0.6**（KV-Distill 经验值）：偏重 `KL(p‖q)`（forward KL，让 student **覆盖** teacher 的 mass，不漏正确答案的概率），同时保留 `KL(q‖p)`（reverse KL，惩罚 student 把 mass 放到 teacher 认为不该有的地方 → 抑制 readout 幻觉）。
- 温度 T：top-64 稀疏存盘下，建议 **T=1.0**（与 KV-Distill 一致；若后续要软化再调）。KL 在 student 侧只对 teacher 的 top-64 support 计算（其余 mass 视为缺失），见技术点1 的稀疏化与归一化处理。
- teacher 侧 `p_t` 是缓存常量，无梯度；梯度只经 `q_t` 回流到 readout / slot / 写路径。

---

## 4. 方案 B — hidden matching

teacher full-context forward 在选定层 ℓ、answer 段 token t 的 hidden `h_t^(ℓ)`（缓存常量）；student 同层同 token 的 hidden `ĥ_t^(ℓ)`：

$$
\mathcal{L}_B = \sum_{\ell\in\mathcal{S}}\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}} d\big(\hat{h}_t^{(\ell)},\ \mathrm{stopgrad}(h_t^{(\ell)})\big)
$$

- `stopgrad` 必需：teacher hidden 是金标准，不接收梯度。
- 距离 `d` **推荐 cosine**（`1 − cos`）而不是裸 MSE——理由见技术点2(c)。
- 选层 `𝒮` 见技术点2。

**A + B 叠加**：`loss = lm_ce + distill_weight·(L_A + β·L_B)`。首批建议 `β` 让 L_B 与 L_A 量级相当（先各取 1.0，跑短 run 看两项 loss 数量级再平衡）。两项共用同一缓存，零额外 teacher forward。

---

## 5. 三个技术点的明确方案

### 技术点 1 — 缓存格式 + 序列对齐

**(a) answer 段 token 位置怎么标定**

两条训练流不同：

| 流 | answer 段定义 | 代码依据 |
|----|--------------|---------|
| **dolmino** | 整个 **target chunk**（`chunk_size` 个 token）都是被监督的 NTP target；answer 段 = target chunk 全部 token | `dolmino_train_step` 走 `answer_mask is None` 分支，`labels = target_input`（`train...cpt.py:1528-1530`） |
| **T2 needle** | 仅 `answer_mask==True` 的位置（5 个数字 token）；其余 −100 | `answer_mask` 在 `niah_chunked_dataset.py:287-294` 构造（只 mask 数字位，跳过空格位）；`dolmino_train_step` 把非 mask 位置设 −100（`train...cpt.py:1536-1539`） |

蒸馏的 `𝒜`（KL/MSE 算的位置）**严格复用同一套 mask**：dolmino = target chunk 全 token，T2 = `answer_mask`。注意 HF CausalLM 内部 shift（`labels[i]` 监督预测 token i 的 logits），所以 answer 段的 logits 位置 = answer token 自身的位置（与 one-hot CE 完全一致），不需要额外 shift 对齐。

> **首批只对 dolmino 加蒸馏**（见技术点3），所以技术点1 的缓存/对齐主要针对 dolmino：answer 段 = 整个 target chunk，每个 token 都存 teacher 软分布 + hidden。

**(b) teacher full-context forward 与 student 的对齐**

- **student**（现有路径）：context chunk 逐个 `no_grad` forward（`train...cpt.py:1518-1521`），每个 chunk **独立** forward → 每个 chunk 的 position id 从 0 重新开始；detach bank 后，target chunk 单独 forward（`:1527,1540`），其 position id 也是 `0..chunk_size-1`（chunk-local）。
- **teacher**：把 `concat(context_chunks, target_chunk)` 拼成一条 `(n_ctx+1)·chunk_size` 的扁平序列，一次 forward（原始 causal attention，无 memory），position id = 绝对 `0 .. (n_ctx+1)·chunk_size−1`。

**对齐方式：按 token 在 target chunk 内的「序内 index」对齐，不是按 position id 对齐。**
- target chunk 的第 j 个 token：在 student 里是它独立 forward 的第 j 个输出；在 teacher 里是 flat 序列最后 `chunk_size` 段的第 j 个输出（绝对位置 `n_ctx·chunk_size + j`）。
- 二者是**同一个 token id**（同一份 `target_ids`），KL/MSE 按 j 一一配对即可。token 化完全一致（同一 tokenizer、同一 `target_ids` 张量），无需重对齐。
- ⚠️ **position id 不一致是预期的、也是蒸馏的目标本身**：teacher 在深位（绝对位置）靠原始 attention 看到 full-context；student 在 chunk-local 位靠 memory readout。我们要 student 在「只有 memory」的条件下复现 teacher「看了全文」的输出。RoPE 相位差对 logits（vocab 空间）影响小，对 hidden 影响较大——这是方案 B 的已知 confound，见技术点2 + 已知风险。

**(c) top-k logits 稀疏存盘格式**

- 每个 answer token：teacher logits 取 **top-64**，存 `indices: int32[64]` + `values: bf16[64]`（存 logits 原值，训练时再除 T、做 softmax；或直接存 softmax 后的概率，二选一，建议存 logits 更灵活）。
- 单 token 大小：`64×4B + 64×2B = 384 B`。
- dolmino target chunk = `chunk_size` 个 answer token 全监督：
  - chunk512 → `512 × 384 B ≈ 192 KB / 样本`（仅 logits 部分）。
  - chunk256 → ≈ 96 KB；chunk1024 → ≈ 384 KB。
- 训练用满 `total_steps × eff_batch` 条不同样本：以 P11 为例 5000 step × eff_batch 32 ≈ 16 万样本，但 dolmino 是有限 corpus 循环复用 → 实际**唯一样本数 = corpus 切出的 (context+target) 段数**。先按 corpus 全量切段缓存，循环训练时按 sample_id 命中缓存即可。若 dolmino_per_doc 切出 ~5 万段：`5万 × 192KB ≈ 9.6 GB`（logits）。加 hidden（技术点2，仅 answer 段、仅 3 层、bf16 4096 维）：`512 × 3 × 4096 × 2B ≈ 12 MB/样本` → ❗ hidden 才是大头，见技术点2 的「只存若干检索关键 token / 用 cosine 降维」建议。

存盘建议：`distill_cache/<chunk_size>/<sample_id>.npz`，含 `logit_idx int32[A,64]`, `logit_val bf16[A,64]`, `hidden bf16[A, |𝒮|, 4096]`, `answer_mask bool[A]`（A = answer 段长度）。sample_id 由 dolmino 切段的确定性顺序给定，保证训练循环可命中。

---

### 技术点 2 — hidden matching 选哪几层

**(a) memory readout 注入在哪几层**

`apply_mem_space_to_model(model, ms_cfg, layer_indices=None)`（`train...cpt.py:1351`）→ `layer_indices=None` 时 **patch 全部 32 层**（`patch.py:81-82`）。每个 `MemorySpaceLayer` 都在 forward 末尾把 readout 加进残差：

```
next_hidden = bypass_h + g·slot_delta + fast_mem_out          # layer.py:1999
if memory_xattn_out is not None:
    next_hidden = next_hidden + memory_xattn_out              # layer.py:2002-2009  (P8 readout)
```

即 memory 信号在**每一层都注入**，随深度逐层累积。

**(b) 推荐 match 哪 2–4 层 + 理由**

推荐 **match 中后段 3 层：`{12, 20, 28}`**（0-indexed，共 32 层），可作为默认 `--distill_layers 12,20,28`；预算紧时退到 2 层 `{16, 28}`。

理由：
- **早层（0–8）主要做局部句法 / token 级特征**，长程事实还没被组装出来，此处 memory readout 和 full-context 的差异主要是噪声，匹配早层会强迫 readout 去复现「还用不到 memory 的」表征，信号噪。
- **中后层（12–28）才是长程事实被聚合进 query 表征的地方**——这正是 W0 掉分（读不准）的所在层。让 student 在这些层对齐 teacher full-context 的 hidden，直接对 readout「把对的信息搬到对的层」施压。
- **不选最后一层（31）**：最后一层 hidden 紧贴 lm_head，其信息已被方案 A（logits KL）覆盖；B 的价值在于给中间层提供更早、更稠密的对齐信号（logits 只在出口约束）。28 已足够接近出口又保留「中间层监督」的意义。
- 3 层在「监督密度」与「缓存大小 / 计算」间平衡：hidden 缓存随层数线性增长（技术点1 估算 hidden 是存盘大头）。

> 注：readout 是逐层累积的，更靠后的层 memory 贡献更大、与 teacher 差距的「可学习性」也更高，所以选层偏中后段。

**(c) hidden MSE 要不要先 normalize / 用 cosine**

`layer.py:1703` 的 `normalize_readout`（P11，`--normalize_readout`）是把 **readout 向量**重缩放到与局部 hidden 同尺度——它约束的是「注入信号的尺度」，**不直接保证 student 整层 hidden 与 teacher 同尺度**。而且：
- Llama hidden 的 per-token norm 随深度显著增长，层间尺度差异大；
- teacher（绝对位置 + full attention）与 student（chunk-local + memory）的 hidden 即便方向一致，**绝对尺度也未必对齐**（RoPE 相位、attention 归一化路径不同）。

**推荐用 cosine（`1 − cos(ĥ, sg(h))`）而不是裸 MSE**：
- cosine 只约束**方向**，规避层间 / teacher-student 的尺度 mismatch，避免 MSE 被大 norm 层主导、也避免逼 student 去复制一个它结构上达不到的绝对幅度。
- 若后续发现只对方向约束不够（幅度也想学），再退化为「先按 per-token RMS 归一化两侧 hidden、再 MSE」的方案。**首批用 cosine**。

---

### 技术点 3 — 离线缓存可行性边界

**(a) dolmino — 可离线缓存。** `--per_doc_data --dolmino_path .../dolmino_per_doc/train` 是**固定 corpus**，切段确定（给定切分逻辑 + 顺序），teacher 输出对每段确定 → 预跑一遍全量缓存，训练循环按 sample_id 命中。✅

**(b) T2 needle — 无法直接离线缓存。** `NIAHChunkedDataset` 每个样本在 `_make_sample` 里用 `rng`**动态随机**生成：随机 6 字母 name + 随机 5 位 code（`niah_chunked_dataset.py:218-224`）、随机背景 chunk 游标。每次 epoch / 每个 worker 产出的样本都不同，没有固定 sample_id ↔ teacher 输出的映射。❌

**(c) T2 要不要蒸馏 — 推荐：首批 T2 不加蒸馏，只保留现有 answer-only CE；只对 dolmino 加蒸馏。**

理由（不是为了省事，而是 T2 蒸馏边际价值低）：
1. T2 的 answer 是 5 个**数字 token**，teacher 走 full-context 时**能直接看到 needle**（`MEMORIZE: ... code ... is 8 0 4 0 2`）→ teacher 在每个数字位的软分布几乎是**对正确数字的 one-hot**。
2. 那么 `KL(p_t‖q_t)` 退化得和现有 `answer_mask` one-hot CE 几乎一样（teacher 软标签 ≈ 真实数字的 one-hot）→ **蒸馏在 T2 上提供的额外信息 ≈ 0**。
3. 蒸馏真正的增益在 **dolmino 这种「下一 token 本就多模态」的通用文本**：teacher full-context 的软分布是真正软的、含丰富的「合理续写」结构，one-hot CE 丢掉了这部分——这才是值得蒸的。

所以：
- **dolmino**：one-hot NTP CE + 双向KL(A) + hidden(B)。
- **T2**：保持 `dolmino_train_step(..., answer_mask=...)` 现状（answer-only CE），**不接蒸馏**。

> 备选（若日后想让 T2 也吃蒸馏）：把 T2 改成 `--t2_fixed_seed` 预生成固定集（固定 seed + 固定背景游标 → dump 到磁盘成定长数据集），再对它跑一遍 teacher 缓存。但鉴于上面 (c1)-(c3)，**首批不做**，避免无谓复杂度。

---

## 6. 离线缓存脚本设计（`scripts/build_distill_cache.py`，待实现）

职责：对 dolmino 固定 corpus 预跑 teacher，dump answer 段的 top-64 logits + 选定层 hidden。

```
输入: --dolmino_path, --chunk_size, --distill_layers 12,20,28, --topk 64,
      --model_path models/Meta-Llama-3-8B, --out_dir distill_cache/<chunk_size>/
流程:
  teacher = LlamaForCausalLM.from_pretrained(model_path)   # 未 patch，frozen，eval()
  teacher.to(bf16).cuda(); torch.no_grad()
  遍历 dolmino 切段（与训练同一确定性切分 → sample_id = 段序号）:
    flat = concat(context_chunks, target_chunk)            # (n_ctx+1)*chunk_size
    out  = teacher(flat, output_hidden_states=True, use_cache=False)
    ans  = 最后 chunk_size 段（= target chunk）              # answer 段 𝒜（dolmino: 全 token）
    logits_ans = out.logits[ans]                            # [chunk_size, V=128256]
    topv, topi = logits_ans.topk(64, dim=-1)                # int32 idx + bf16 val
    hid = [out.hidden_states[ℓ][ans] for ℓ in distill_layers]  # |𝒮|×[chunk_size,4096]
    np.savez(out_dir/f"{sample_id}.npz",
             logit_idx=topi.int(), logit_val=topv.bf16(),
             hidden=stack(hid).bf16(), answer_mask=...)
要点:
  * teacher 与训练同一 tokenizer / 同一 model_path / 同一 chunk_size 切分逻辑，
    否则 sample_id ↔ teacher 输出错位。
  * 多卡数据并行加速（按 sample_id 取模分片，各 rank 写各自 .npz，无需通信）。
  * 大小估算见技术点1：logits ~192KB/样本(chunk512)，hidden 是大头(~12MB/样本，3层全token)
    → 若 corpus 大，可只缓存「检索关键 token」子集或进一步降 hidden 精度 / 减层。
  * 一次性离线作业，可放任意空闲节点；产物随盘持久，训练阶段零 teacher forward。
```

---

## 7. 训练循环改动点（file:line，仅标注不实现）

1. **新增 distill 参数解析**：在 argparse 区（与 `--use_memory_xattn` 等并列，约 `train...cpt.py:1290-1339` 对应的 add_argument 区）新增 `--distill_logits/--distill_hidden/--distill_lambda/--distill_layers/--distill_cache_dir/--distill_weight`。
2. **dolmino 样本带出 sample_id + 命中缓存**：dolmino 数据集 yield 时附 `sample_id`；主循环 dolmino 分支（`train...cpt.py:2441-2467`）按 sample_id 从 `distill_cache_dir` 读 `.npz`（teacher logits/hidden）。
3. **student forward 暴露 logits + 选定层 hidden**：`dolmino_train_step`（`train...cpt.py:1482-1546`）当前只取 `out.loss`；需改为 `model(..., labels=..., output_hidden_states=True)` 并返回 `out.logits` + `out.hidden_states[𝒮]`（answer 段切片）。**只在 distill 开启时**取，关闭时 byte-identical 现状。
4. **distill loss 计算 + 折进 backward**：在 `dolmino_train_step` 内（`:1540` 的 `out=...` 之后、`:1545` 的 `.backward()` 之前）：
   - A：对 answer 段 student logits 在 teacher top-64 support 上做双向 KL（teacher 概率重归一到 top-64）。
   - B：对选定层 student hidden 与缓存 teacher hidden 做 `1−cos`（teacher `stopgrad`）。
   - `total = lm_loss + distill_weight*(L_A + β*L_B)`，替换 `:1545` 的 `(lm_loss + aux_loss).backward()`。
5. **日志**：在主循环 logging 区（`train...cpt.py:2486` 附近 step loss 累加处）新增 `step_distill_kl` / `step_distill_hidden` 累加，写入 wandb，便于「distill_loss 是否下降」监控。
6. **T2 / babilong 路径不动**：`babilong_train_step`、T2 分支保持现状（技术点3）。
7. **adapter_config 无需变更**：distill 是训练期 loss，不引入新模块参数，checkpoint / eval loader 不受影响（蒸馏只改训练目标，readout 架构与 P11 一致 → 评测脚本零改动）。

---

## 8. 最小验证计划

**先 chunk512 短 run，看两件事：distill_loss 下降 + W0 是否抬头。**

1. **离线缓存 smoke**：对 dolmino_per_doc 取前 ~500 段，跑 `build_distill_cache.py` 产 `distill_cache/512/`，校验 `.npz` shape（`logit_idx [512,64]`, `hidden [512,3,4096]`）+ teacher full-context 在 answer 段的 NTP loss 合理（应明显低于 student W0，因为 teacher 开卷）。
2. **短训练 run**（chunk512，~300–500 step，单节点 8×H20，基于 `launch_mem_space_p11_chunk512_remote196.sh` 复制改 distill 开关）：
   - 监控 `step_distill_kl` / `step_distill_hidden` 是否**单调下降**（蒸馏目标可学）；若不降 → 检查序列对齐（技术点1b）/ 选层（技术点2）。
   - 监控 `lm_loss` 不应被蒸馏项带崩（若崩 → 降 `distill_weight` 或 `β`）。
3. **W0 抬头判定**：step300/500 ckpt 离线跑 `scripts/_eval_taskpool_2group.sh`，**`--swa_eval_chunks 0`（W0）** 跑 qa1/qa2/qa5 × {1k,4k,16k}，与同步数的 P11 baseline（无蒸馏）同口径对比。**核心指标：W0 是否从 10–30 抬向 SWA 的 50–60。** 哪怕只抬 5–10 分也算方向 confirmed → 再扩 step + 全长档。
4. **消融顺序**（验证通过后）：A-only / B-only / A+B 三 arm 并行（不同节点），定位双蒸馏各自贡献。

---

## 9. 已知问题 / 风险

- **RoPE 位置 confound（方案 B 最大风险）**：teacher answer token 在绝对深位、student 在 chunk-local 位，RoPE 相位不同 → 同一 token 的 hidden 即便语义对齐、方向也带位置分量。cosine 缓解但不根治。若 B 难下降，优先信 A（logits 对位置鲁棒），B 退为「只在最后 1–2 层 + cosine」的弱辅助。
- **top-64 截断**：teacher 长尾 mass 丢失，`KL(q‖p)`（reverse）在 support 外无定义 → 实现时把 student 概率也重归一到 top-64 support，或对 support 外加小 floor。需在实现中明确（首批：teacher top-64 重归一为 `p_t`，student 同 support 上算 KL）。
- **缓存体积**：hidden 是存盘大头（技术点1）。若 corpus 大到放不下，先减到 2 层 / 降采样 answer token / 只缓存 logits（先验证 A，B 后补）。
- **teacher = 纯 backbone 还是 patch-but-memory-off**：本设计取**纯未 patch 的 frozen Llama-3-8B 全序列 forward** 作 teacher（最干净、可离线、层 ℓ 与 student 同深度对齐）。若日后想让 teacher 也带某种 full-context memory，再单议。
- **curriculum 下 n_ctx 变化**（`--curriculum 0:3`）：target chunk 始终是最后一个 chunk，answer 段定义不变；但 n_ctx 增大会改变 teacher flat 序列长度 → 缓存须按训练实际用的 n_ctx 切分口径生成，避免 sample_id 错位。
