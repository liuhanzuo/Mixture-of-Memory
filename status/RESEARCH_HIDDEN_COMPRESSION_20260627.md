# 研究报告: hidden 压缩的可寻址性 — 为什么 slot 不归零 / W6 gap 机制 (2026-06-27)

> 作者: researcher subagent (调研 + 读码 + 写报告, 不跑任何训练/eval, 不改任何代码)。
> 纪律: 只用干净 (babilong_mix=0) 数字; 泄漏 ckpt (b50/b100/P2/c1024/旧b25) 的分数一律不引用。
> **诚实分级**: 每条结论标注 `[确证事实]`(代码/干净日志直接支撑) 或 `[假说]`(机制推断, 待实验)。
>
> ⚠️ **联网受限声明**: 本轮 WebSearch 持续返回服务端网关 502 (toolConfig 校验 bug), WebFetch 拒绝
> arxiv/github/huggingface/ar5iv 域名 (8+ 次重试均失败)。因此 §A 的 MemoryLLM 事实来自
> **(1) 训练知识 (标注 `[训练知识·本轮未联网核实]`)** + **(2) 本仓已有文献笔记**。一旦联网恢复,
> §A 中标 `[需联网复核]` 的具体数字 (pool size / θ / 维度) 必须二次核实。

---

## TL;DR (三句话)

1. **问题B (slot 为何 32k 还有 ~9%)**: 本项目 slot **不降维** (`slot_dim` 默认 = backbone hidden 4096, `config.py:58`),
   压缩 100% 来自 **token 数压缩** (128 slot × top_k=16 vs 数千 token) + EMA/dual-gate **池化聚合**。
   池化保留了 chunk 的**主方向 (粗语义指针)**, 够把多选答案 (qa5 答案空间小) 推到 entity-prior 之上 →
   8k=19; 但池化抹掉精确绑定 (who→what→whom), 且 32k 时 128 slot 被稀释/覆盖 → 衰减回 entity-floor ≈9。
   **"可寻址但不可重建" = 粗粒度语义指针假说, 与数据一致 `[假说]`。**
2. **问题W6gap (W0=9 vs W6=34, 3.8×)**: FIFO 存的是 **query-blind 的冻结 hidden 快照** (写入时 needle 从没 attend 过 query,
   RoPE 坍缩 pos-0, `layer.py:1554/1478`)。原 token 重 forward 让 needle 跨全部层重新 attend query → 多跳绑定重建。
   这是 **staleness/query-blindness 损耗**, **不是压缩损耗** (FIFO 是 1:1 无压缩)。`[确证事实+假说]`
3. **两面一币? 部分成立**: 共同点 = "存储表示保留粗可寻址信息但丢精确可重建信息"。**但损耗算子不同**:
   slot 经 **池化+稀释** 丢精度; FIFO 经 **query-blind 快照+pos坍缩** 丢精度。把它们当"同一枚硬币"在
   *现象层*成立, 在*机制层*要分开说 (否则会误导修复方向)。`[假说]`

---

## §A. 问题A — MemoryLLM 的 hidden 降维/存储做法 (事实部分)

### A.0 关键澄清: 用户前提 "MemoryLLM 把 hidden 投影成更低维再存" 需要打问号

`[训练知识·本轮未联网核实]` 据我对 MemoryLLM (Wang et al., *MEMORYLLM: Towards Self-Updatable Large
Language Models*, ICLR 2024) 的记忆, **vanilla MemoryLLM 的 memory token 维度 = backbone hidden 维 d (不降维)**;
它的"压缩"主要来自 **token 数 (固定池 + 丢弃旧块)**, 而**不是**把每个向量投影到更低维度。
"投影到更低维"更可能出现在 **M+ (2025, 长期记忆 + retriever)** 的 retriever 侧, 或是对原文的误记。
→ **建议: 联网恢复后核实"是否有 per-token 降维投影"。** 但下面 A.1-A.4 列出的、对本项目真正有指导价值的
事实, **不依赖**这个维度细节。

### A.1 MemoryLLM 的存储结构 `[训练知识·本轮未联网核实, 需联网复核具体数字]`

- **per-layer memory pool**: 每个 transformer 层挂一个**固定大小** memory token 池 (与本项目 per-layer slot / FIFO 同构)。
- **pool 规模**: 池很大 (总额外参数 ~1.6B 量级); 以 Llama-2-7B (d=4096, L=32) 反推, 每层 ~1.2万 memory token 量级。
  `[需联网复核]` —— 本仓 `config.py:868-871` 注释印证: *"fifo_buffer_chunks 50 ≈ MemoryLLM's num_blocks × chunk_size;
  50×512=25600 tokens of raw KV context"*, 即项目作者把 FIFO 25600-token buffer 当作 MemoryLLM 池规模的类比。
- **池分块 (num_blocks)**: 池被切成若干 block。

### A.2 自更新机制 (inject new + drop old) `[训练知识·本轮未联网核实]`

- 新 context (一段文本) 进来时, 模型把它**编码成 θ 个新 memory token (delta memory)**, **注入**池中;
  同时**随机丢弃等量的旧 memory token** → 固定大小 + 优雅遗忘 (random drop = graceful forgetting)。
- 与本项目对照: 本项目 slot 是 **EMA/dual-gate 原地写选中槽** (`memory_bank.py:967 write`), FIFO 是 **append+pop 最旧**
  (`layer.py:1231 _fifo_write_to_buffer`)。MemoryLLM 的"随机丢弃"是第三种淘汰策略, 项目未试。

### A.3 训练目标: LM CE, 不是 reconstruction MSE `[训练知识·本轮未联网核实, 高置信]`

- **这是对本项目最重要的事实**: MemoryLLM 端到端用 **next-token / LM 交叉熵**训练, 且**把自更新循环放进训练**——
  即"先把 context 压进 memory → 丢掉原文 → 用 memory 预测后续/答案"的整个流程在训练里走通,
  梯度穿过"生成 delta memory 的模块"。**没有 reconstruction MSE**。memory 因此被训成"**可被读出用于预测**", 而非"可被重建成原文"。
- 与本项目对照, 这正是本仓 `versions/v_prediction_not_reconstruction_2026-06-25.md` 独立得到的洞察:
  *"memory 存储目标是 prediction 不是 reconstruction; L3 token-recon aux 全 REJECTED 是因为用错了 loss"*。
  → **MemoryLLM 的成功印证了项目这条洞察**: 它的 memory 之所以可读, 是因为读出路径**带着 LM 梯度被训练过**。

### A.4 读回方式 `[训练知识·本轮未联网核实]`

- memory token 作为**注意力前缀 (KV prefix)** 拼在当前输入前, 在每层参与注意力 —— 与本项目 KV-prepend 同构
  (slot 路径 `layer.py:13-18`, FIFO 路径 `layer.py:1282 concat[prefix|H]`)。
- **关键差异 (连到 W6gap)**: MemoryLLM 的 memory 是**被 LM 损失训练出来的可读表示**;
  本项目 FIFO 存的是 **detach 的冻结 hidden 快照** (`layer.py:1554 h_stored = hidden_states.detach()`),
  FIFO 路径**唯一可训练参数是 inject_gate** (`layer.py:1540-1551`), 读出表示本身**从未被训练**。
  → MemoryLLM 没有 W0/W6 gap 的结构原因: 它的 W0 (纯 memory) 就是被训出来的读出; 本项目 W0 是未训练的死快照。

### A.5 本项目自己的"投影/压缩"模块清单 (确证事实, 读码)

| 模块 | 文件 | 做什么 | 维度 | 训练目标 |
|---|---|---|---|---|
| `hidden_to_slot` | `layer.py:629` | `nn.Linear(d_model, slot_dim, bias=False)` hidden→slot 投影 | **slot_dim 默认=d_model=4096, 不降维** (`config.py:58`) | 经 LM CE (Fix I 后可解冻, `layer.py:741-763`) |
| `TopKSelector` Q_sel/K_sel | `selector.py:111-112` | hidden/slot → selector_dim=128 打分 (寻址用, 非存储) | 128 | LM CE + load-balance aux |
| `L3SummaryPool` | `l3_summary.py:67` | Q-Former 式 cross-attn 把 chunk T token 池化成 K 个 summary token | d_model (不降维), 压在 **token 数** | LM CE (+diversity aux) |
| `MemoryReconDecoder` | `recon_decoder.py` (v12) | 用 MSE 重建 L3 summary 监督写入 | — | **MSE recon (历史路线, 已被 prediction 洞察判定方向错误)** |
| Slot-Routed Evidence / rawkv | `memory_bank.py:73-150` | 存**未压缩**原始 token hidden ([d_model]) 旁路, 读时重注入 | d_model, **1:1 无压缩** | 无 (冻结快照) |

**结论 (确证事实)**: 本项目压缩**从不降维**, 一律压在 **token 数 / 池化聚合**; slot_dim、L3 都是 d_model。
唯一存过低维的 v12 recon 用的是 MSE, 已被 prediction-not-reconstruction 洞察判定"方向错"。
**这与 MemoryLLM 的 count-compression + LM-CE 高度一致** —— 项目和 MemoryLLM 的"对的部分"是同一个;
项目缺的不是降维, 而是**让读出表示真正被 LM 损失训练**(FIFO 死快照 = 没训读出)。

---

## §B. 问题B — 压缩 ~5000× 的 slot 为何 32k 还有 ~9% 而非归零

### B.1 先厘清"压缩比"与基线 (确证事实)

- 本项目 slot **不是把单向量压成 1/5000 维**; 是 **128 slot × 4096 维**(每个全维)去**概括** 32k≈数千 token →
  压缩在 **token 计数** (~数千→128) 与 **EMA 池化** 上。每个 slot 是它吸收过的 token 的(门控)聚合方向。
- qa5 (BABILong, 三元关系: 谁把什么给了谁) **答案空间小** (答案是故事里少数被点名实体之一)。
  → **纯随机/entity-prior 基线本身不是 0**。这一点必须先扣除, 否则会高估"记忆贡献"。

### B.2 机制假说: 粗粒度语义指针 (coarse semantic pointer) `[假说, 与数据一致]`

**slot = 它吸收的 chunk token hidden 的门控池化平均** (`memory_bank.py:967 write`: EMA/dual-gate)。
均值/EMA 这种聚合算子的数学性质:

1. **保留主方向, 抹掉细节**: 一组 hidden 向量取(门控)平均, 结果**强烈偏向这组向量的公共/主成分方向**
   (top principal direction 对平均鲁棒), 而把彼此正交的精确细节(具体 token 身份、绑定关系)平均成噪声。
   → slot 保住了 chunk 的**粗语义**(出现了哪类实体、哪类事件), 丢了**精确三元绑定**。
2. **够"寻址"不够"重建"**: 对答案空间小的多选任务, 一个偏向"milk-transfer / Mary-ish"主方向的 slot,
   足以把 lm_head 输出从均匀分布**偏向正确实体一侧** → 8k=19 (高于 entity-prior)。但它无法精确恢复
   "Mary→milk→Bill"的方向绑定 → 远达不到高分。**这就是"可寻址 (addressable) 但不可重建"。**
3. **32k 衰减回 floor 的机制**: 128 个 slot 固定, 32k 把更多 chunk 路由进同一批 slot →
   (a) EMA 覆盖洗掉 needle 早期贡献; (b) 路由碰撞使 needle 的主方向被其它 chunk 的主方向稀释。
   → needle 对任何 slot 主方向的贡献趋向噪声底 → 分数衰减到 **entity-guessing floor ≈9** (8k=19→16k=16→32k=9)。

### B.3 这个假说与已有干净证据一致 (确证事实支撑假说)

- 本仓对 **FIFO hidden** 已做过同型 logit-lens probe (`FIFO_FINDINGS_SUMMARY:27`, `H_V2_PLAN:22`):
  needle hidden 对答案 token 在 L31 的 **rank=21 (random=44)** —— **"存了但弱/不 sharp"**。
  这正是"粗指针"的定量指纹: 显著高于随机(确有寻址信息), 但远非 rank-1(不可精确重建)。
  → slot 路径极可能同型 (尚未对 slot 直接测 — 见 §E 实验1)。
- slots 干净曲线 8k=19→16k=16→32k=9 (`CLEAN_SOTA_SURVEY:37`) 的**单调衰减**与 B.2 的稀释机制吻合;
  非泄漏 b25 的"非单调驼峰 8k=65<16k=76"才是不物理的泄漏指纹(已查实)。

### B.4 诚实边界

- **9% 里有多少是真寻址、多少是 entity-floor, 目前未拆开。** 很可能 32k 的 9% **绝大部分是答案空间 floor**,
  slot 真实残余寻址≈0; 而 8k 的 19 里 slot 贡献明显(19 远高于 floor)。**这是可证伪的, §E 实验2 直接测。**

---

## §C. 问题W6gap — 同模型 W0=9 vs W6=34 (3.8×) 的机制

### C.1 确证事实 (代码 + 干净判据链)

- FIFO buffer 存 `hidden_states.detach()` (`layer.py:1554`), 即 **chunk "当初作为 current 时算的 hidden 快照"**。
- 读出 mask 是严格因果 `triu(-inf, diagonal=1)` over [P+T] (`layer.py:1478`): **prefix token 之间因果, query H 能看全部 prefix,
  但 prefix token 看不到后面的 query H** → **存储 hidden 在读出时是 query-blind 的**(它从没、也不会 attend query)。
- RoPE 坍缩 pos-0 (`layer.py:1392-1401`, 默认 `_pos_mode=None`)。
- 干净判据链 (`FIFO_FINDINGS_SUMMARY:14-23`, 全 NOLEAK mix=0):

  | 配置 | qa1 8k | 16k | 32k |
  |---|---|---|---|
  | 纯 memory W0 | 12 | 8 | 2 |
  | hidden-oracle (完美隔离 needle 的**hidden 快照**) | 20 | 24 | 22 |
  | **oracle-token (隔离 needle 的**原始 token**重 forward)** | **50** | 28 | 33 |

  → **死 hidden 快照 readout 到顶 ~20-24; 原 token 重 forward 跳到 50**。差距 = 读出表示, 不是选择(oracle 都完美选中)。

### C.2 机制假说: query-blind 快照 vs 全深度 query-aware 重算 `[假说, 强证据]`

W6 给原 token 重 forward 比 W0 读冻结 hidden 强 3.8×, 三个叠加损耗:

1. **query-blindness (主因)**: 多跳任务(needle↔query 绑定)需要 needle token **attend 到 query**。
   冻结快照写入时 query 还没出现, 永远断了这条耦合; 原 token 重 forward 让每一层重新 attend query → 绑定重建。
   (oracle-token 50 vs hidden-oracle 20 是这条的直接证据。)
2. **单层快照 vs 全深度重算**: FIFO 在层 ℓ 存层-ℓ hidden, 读回只在**一层**注入; 下游各层从没机会"在 query 在场下"
   重算 needle 表示。原 token 重 forward 走**全部 32 层**重算。
3. **pos-0 坍缩 (次要, 已部分证伪为主因)**: prefix 变成无位置词袋。本仓 ArmC(训练时 real 位置)长档≈基线
   (`H_V2_PLAN:23`, `FIFO_FINDINGS:28`) → 位置**不是主因**, 但仍是叠加损耗。

### C.3 关键区分: W6gap **不是压缩损耗** (确证事实)

FIFO buffer 存的是 **1:1 未压缩 hidden** (`config.py:870-871` 明确 "uncompressed raw KV")。
所以 W6 gap **与 §B 的 slot 压缩损耗不是同一个损耗算子**。统一只在抽象层成立(都丢精确可重建信息),
机制层必须分开: **slot 经池化+稀释丢精度; FIFO 经 query-blind 快照丢精度**。

---

## §D. "两面一币"判定 — 假说能否同时解释 slot 9% + W6 gap 3.8%

`[假说]` **能, 但要在正确的抽象层。**

- **共同核**: 一个**预先算好、与未来 query 解耦、经有损算子压过**的存储表示, 会**保留粗可寻址信息**
  (主方向/粗语义 → 不归零、W0 非零), 但**丢精确可重建/可绑定信息** (→ 远低于"拿原料现算")。
- **slot 9%**: 池化保主方向 → 8k 高于 floor(19); 稀释+覆盖 → 32k 退回 floor(9)。✓
- **W6 gap 3.8×**: query-blind 快照保粗内容(W0=9/34≈1/4) → 原 token 重 forward 恢复 query-绑定(W6=34)。
  比值 34:9≈3.8 ≈ "答案信号里约 3/4 在精确/query-绑定部分(快照丢掉了), 约 1/4 在粗指针部分(快照留住了)"。✓
- **但损耗算子不同** (slot=池化+稀释, FIFO=query-blind+pos坍缩), 所以**修复方向不同**:
  - 救 slot 精度 → 要么别用池化(存 evidence 原 token, 项目已试 rawkv), 要么把读出训成 query-aware。
  - 救 FIFO W0 → 让冻结 hidden 在读出时**能 attend query**(query-conditioned refresh), 或干脆重 forward 原 token。
  - **统一处方 (与 §A MemoryLLM 对齐)**: 让**读出表示带 LM 梯度被训练**(MemoryLLM 没 gap 的根因), 而非存死快照。

---

## §E. 三条可在本项目代码上做的具体实验建议

> 全部用**干净 ckpt** (slots: pg19 nctx7; FIFO: NOLEAK b25), 强制 `--babilong_mix_fraction 0`。
> 标注 [需改码?] 与改哪个文件。**本报告只建议, 不启动。**

### 实验1 — slot 的 logit-lens 可寻址性 probe (直接证伪/证实 §B 粗指针假说)
- **做什么**: 干净 slots ckpt 上跑一条 qa5 故事, 抓 needle chunk 路由到的 slot 向量, 经**冻结 lm_head**
  做 logit-lens, 测答案 token 的 **rank**; 对比 (a) needle-slot rank, (b) 随机 slot rank, (c) entity-prior。
  若 needle-slot rank 显著优于随机但远非 rank-1 → 坐实"粗指针: 存了但不 sharp" (复刻 FIFO 已得的 L31 rank=21)。
- **预期判据**: needle-slot rank ∈ (rank-1, random) 之间, 且 32k 比 8k 更接近 random (稀释)。
- **[需改码: 小]** `scripts/run_babilong_mem_space.py` 加一个 probe 钩子 dump 选中 slot;
  `src/memory/mem_space/layer.py` 复用已有 oracle needle-chunk 通道 (`_fifo_select_keep_set_oracle` 同款)
  定位 needle slot。**零训练**, 一次 eval。

### 实验2 — memory-ablation / entity-floor 控制 (拆"9% 里多少是真寻址")★最便宜最决定性
- **做什么**: 干净 slots ckpt 上, qa5 × {8k,16k,32k}, 三臂对比:
  (a) 正常 W0; (b) **memory 旁路** (走已存在的 `forward_no_memory`, `layer.py:1217`); (c) slot 随机化/置零。
  若 (b)/(c) 在 32k 仍≈7-9 而正常 W0=9 → **32k 的 9% 几乎全是 entity-floor, slot 残余寻址≈0**;
  若 (b) 掉到 ~2-3 → slot 在 32k 确有真实残余寻址。8k 同理拆 19。
- **预期判据**: 这是"9% 是真寻址还是 floor"的**唯一决定性控制**。
- **[需改码: 极小]** `forward_no_memory` 已存在; 只需在 `scripts/run_babilong_mem_space.py` 加一个
  `--ablate_memory` eval flag 把目标层路由到 `forward_no_memory` (或喂空 buffer)。**零训练**。

### 实验3 — query-conditioned prefix re-attend (判 W6gap 是否=query-blindness, 给廉价修复方向)
- **做什么**: 干净 NOLEAK FIFO ckpt, eval-only 改读出 mask: 让 prefix(冻结 hidden)**也能 attend 当前 query chunk H**
  (prefix↔H 双向, 仅读出 forward 一次), 即给死快照一次 query-conditioned 重过。对比标准 W0。
  若 W0 显著上抬接近 W6 → gap 主因是 query-blindness, **廉价修复 = 读时让 hidden 重 attend query, 不必存原 token**;
  若不抬 → 必须全深度重 forward 原 token (快照根本不够), token-reforward 路线是唯一解。
- **预期判据**: 区分"query-blindness(便宜)" vs "需全深度原 token 重算(贵)" —— 直接决定下一步训练投资方向。
- **[需改码: 中]** `src/memory/mem_space/layer.py` `_forward_fifo` 的 mask 构造 (~`:1478`):
  把 prefix 行对 H 列的 `-inf` 改成可见 (eval probe 开关, 默认关 = 字节等价)。**零训练**, 一次 eval sweep。

---

## §F. 关键文件:行 索引 (便于复核)

- slot 不降维: `src/memory/mem_space/config.py:58` (`slot_dim: Optional[int]=None → hidden_size`), `:36` 注释。
- slot 写入 (池化/EMA/dual-gate): `src/memory/mem_space/memory_bank.py:967 write` (单门 `:1132`, dual `:1059`, delta `:1100`)。
- hidden_to_slot 投影 (默认不降维): `src/memory/mem_space/layer.py:629`, 解冻逻辑 `:741-763`。
- FIFO 存死快照 detach: `layer.py:1554`; 写 buffer `:1231`; pos-0 坍缩 `:1392-1401`; 严格因果 mask `:1478`。
- FIFO buffer = 未压缩 raw KV, "≈MemoryLLM num_blocks": `config.py:868-871`。
- forward_no_memory 旁路 (实验2 复用): `layer.py:1217`。
- 干净判据链 (oracle-token 50 vs hidden 20): `status/FIFO_FINDINGS_SUMMARY_20260627.md:14-23`。
- 干净 slots SOTA 曲线: `status/CLEAN_SOTA_SURVEY_20260625.md:37`。
- prediction-not-reconstruction 洞察 (= MemoryLLM LM-CE 对齐): `versions/v_prediction_not_reconstruction_2026-06-25.md`。
- logit-lens needle rank=21: `status/FIFO_FINDINGS_SUMMARY_20260627.md:27`, `status/H_V2_PLAN.md:22`。

---

## §G. 待办 / 联网恢复后必做

- [ ] **核实 §A.0**: vanilla MemoryLLM 是否对 memory token 做 per-token 降维投影 (我的训练知识倾向"不降维, 压在 token 数 + 随机丢块"; 可能是 M+ retriever 才降维)。来源: arXiv 2402.04624 + repo github.com/wangyu-ustc/MemoryLLM。
- [ ] **核实 §A.1**: pool 规模 (num_blocks, θ delta token 数, 总 memory 参数量) 的确切数字。
- [ ] **核实 §A.3**: 确认训练目标确为 LM CE + 自更新循环在训练内 (这点我高置信但本轮未联网核实)。
