# Token Re-Forward 读出机制 — 存储代价 & 推理速度代价 定量分析 (2026-06-28)

> 只读分析报告。所有数字标注来源类型：**【码算】**=读代码算出的理论值 / **【实测】**=logs 墙钟值 / **【估算】**=外推。
> 红线遵守：不引用泄漏 b25「破墙」分数；clean 锚点用 NOLEAK ckpt(`outputs/mem_space_fifo_b25_chunk512_noleak`, babilong_mix=0)。
> 关键代码：`scripts/run_babilong_mem_space.py:597-830` (`generate_with_mem_space`)，`src/memory/mem_space/memory_bank.py`，`patch.py:108-121`，`layer.py:1231-1260`。

---

## 0. 机制确认（读码结论）

**re-forward 保留的是 token-id，不是 hidden/KV。** 这是本分析最关键的一条：
- 文档在 eval 入口被切片成原始 token：`chunks = list(tokens.split(chunk_size))`（`run_babilong_mem_space.py:690`），`tokenizer.encode(..., return_tensors="pt")` → `torch.long`（`:1533-1535`）。
- 破墙读出窗口 = 选中 chunk 的**原始 token 拼接** + last chunk：
  `pieces = [chunks[c] for c in sel] + [chunks[-1]]; window = torch.cat(pieces)`（`:743-744`），再 `model(input_ids=window)` 走全 32 层重算（`:815`）。
- 所以 re-forward 的「读出 payload」就是 token-id 本身（极便宜），KV/hidden 是**每步现算、不存**。
- **代价不在『存』而在『算』**：用「存 tiny token-id + 重算」换掉「存大 KV」。

**两个独立 forward 都是 `use_cache=False`**（`:699` streaming，`:815` 生成）：生成阶段每步都把整个 window 从头 forward 一遍，window 还逐 token 增长（`cur = torch.cat([cur, next_tok])`, `:826`）。这是速度代价的放大器（见 §2）。

**slot bank 是单一 shared bank，非每层独立**：`config.shared_memory_bank=True`（默认，`config.py:249`），`patch.py:110-121` 只 new 一个 `MemoryBank` 传给全部 32 层（`object.__setattr__` 注册，不进 state_dict）。dtype 跟随 hidden = **bf16**（`memory_bank.py:229`）。`num_slots=128, slot_dim=4096`（实测 log header `num_slots=128`；`config.py:56` `slot_dim=None`→backbone hidden=4096）。

---

## 1. 存储代价

### 1a. Slot bank 本身（纯 memory / W0 方案）

| 量 | 值 | 来源 |
|---|---|---|
| num_slots × slot_dim × dtype | 128 × 4096 × 2B | 【码算】 |
| 单个 shared bank / sample | **1.0 MiB** | 128·4096·2 = 1,048,576 B |
| 层数因子 | ×1（shared，非 ×32） | `patch.py:110` |
| 附属 `slot_token_mass` [128] fp32 | 512 B（可忽略） | `memory_bank.py:303` |

→ **纯 slot bank ≈ 1 MiB / sample，且与文档长度无关（压缩，常数）**。这是 W0「纯 hidden 读出」的全部历史存储。
（若 `shared_memory_bank=False`，则 ×32 = 32 MiB；当前默认不是这条。）

### 1b. re-forward 额外存什么

| 存储项 | 32k 文档 (64 chunk) | 来源 | 说明 |
|---|---|---|---|
| **保留原始 token-id（re-forward payload）** | **~0.13–0.26 MiB** | 【码算】 | 32768 tok × 4B(int32 下限，vocab=128256>65535 故不能 int16) = 128 KiB；代码实际 int64 = 256 KiB。**近乎免费** |
| 选择用的 key（当前实现：复用 32 层 FIFO hidden buffer） | **~6.25 GiB** | 【码算】 | 每 chunk/层 = 512·4096·2 = 4 MiB；FIFO cap=50 chunk(`config.py:879`)；50·4MiB·32层 = 6400 MiB。**这是当前实现的真实大头** |
| 选择 key（设计文档建议的生产化：只存 1 个选择层） | ~256 MiB（raw）/ ~8 MiB（池化到 16 tok/chunk） | 【码算+设计文档】 | `TOKEN_REFORWARD_DESIGN_20260627.md:54-58`：32 层 hidden 是「dead weight」，应砍到单层 |

**核心区分**：re-forward 的**读出**只需保留 token-id（0.26 MiB，免费）。当前 eval 实现额外背着一个 6.25 GiB 的 32 层 FIFO hidden buffer，但那是**选择索引**（沿用旧 hidden 路径），不是读出必需 —— 设计文档明确说它该被砍到单层（256 MiB）甚至池化（8 MiB）。

### 1c. 32k 文档下的存储对比

| 方案 | 读出 payload | 选择/索引 | 合计 | 备注 |
|---|---|---|---|---|
| **纯 slot (W0)** | slot bank 1.0 MiB | — | **~1 MiB** | 压缩，常数，与长度无关 |
| **re-forward（读出部分）** | token-id 0.26 MiB | （复用 slot 选 or 单层 key） | **~0.26 MiB** + 选择索引 | 读出 payload 本身比 slot bank 还小 |
| re-forward（当前 eval 实现，含 32 层 FIFO 选择 crutch） | 0.26 MiB | 6.25 GiB | **~6.25 GiB** | 选择索引主导，可优化 |
| re-forward（设计文档生产化，单层选择 key） | 0.26 MiB | 256 MiB | **~256 MiB** | |
| 【参考】稠密长上下文全 KV cache | 4.0 GiB | — | 4.0 GiB | 【码算】2·8kv·128·32层·2B = 128 KiB/tok ×32768 |

**结论**：re-forward 的「读出存储」（token-id）几乎为零，**比 slot bank 还便宜**。它用「存 0.26 MiB token-id + 每步重算」替代了「存 4 GiB KV」。真正的存储负担来自**选择机制**（当前 6.25 GiB 的 FIFO crutch，可优化到 8–256 MiB）。slot bank（1 MiB）和 re-forward token-id（0.26 MiB）本身都极小 —— 二者并不冲突，存储不是 re-forward 的瓶颈。

---

## 2. 推理速度代价

### 2a. 相对 FLOPs（读出窗口，【码算】）

window 长度 L = (K+1)×512。生成 20 步、`use_cache=False`、window 逐 token 增长（`:815,826`）：
每步 forward 长度 ≈ L+step，20 步总「token-forward 当量」≈ 20·L + 190 ≈ **20·L**（+190 可忽略）。

| 配置 | window L | 线性/MLP/proj 项 ∝L（相对 W0） | 注意力项 ∝L²（相对 W0） |
|---|---|---|---|
| W0 | 512 (1 chunk) | 1× | 1× |
| K=2 | 1536 (3) | 3× | 9× |
| K=4 | 2560 (5) | 5× | 25× |
| K=6 / SWA-W6 | 3584 (7) | 7× | 49× |
| K=16 | 8704 (17) | 17× | 289× |

d=4096 时 MLP 项(L·d²)与 attention 项(L²·d) 在 L≈d 处交叉；K≥4 起 window 接近/超过 4096，attention 二次项开始主导 → 实测落在线性与二次之间、并随 K 增大趋近二次（见 2b）。
注意：**streaming 阶段（forward chunks[:-1] 入 bank）W0 与各 K 完全相同**（`:696-699` 不受 K 影响），re-forward 的额外开销**纯粹**在生成窗口；长文档 streaming 占比大，会**稀释** re-forward 的相对惩罚（16k 比 8k 惩罚比更小，见下表）。

### 2b. 实测墙钟（【实测】logs s/样本）

**A 组 — NOLEAK ckpt + taskpool 同机型（可横向比，95GB GPU）**：

| 配置 | window | qa1 8k s/样本 | qa1 16k s/样本 | qa1 32k s/样本 | 相对 8k W0-class |
|---|---|---|---|---|---|
| hidden-oracle（W0 级，窗≈1 chunk）| 512 | **11.4** | 38.0 | — | 1.0× (基准) |
| reader-attn-token **K1** | 1024 | 43.1 | 144.7 | — | 3.8× |
| reader-attn-token **K2** (cprobe) | 1536 | 72.1 | 163.6 | — | 6.3× |
| reader-attn-token **K4** (L8) | 2560 | 208.8 | 305.0 | — | 18.3× |
| oracle-token (K≈needle 1–2) | ~1024-1536 | (194/部分) | 234.8 | 419.8 | ~17× |
| oracle-token **qa5** | | 151.4 | | | |
| **K6 / K8 / K16** | 3584–8704 | **大量 OOM** | OOM | — | 见 2d |
| SWA-**W6**（窗 3584，对照）| 3584 | ~504（部分）| — | — | ~44× |

来源 logs：`eval_noleak_oracle_taskpool`(11.4/38.0)、`eval_noleak_cprobe_k1_taskpool`(43.1/144.7)、`eval_noleak_cprobe_k2_taskpool`(72.1/163.6)、`eval_noleak_readerattn_L8k4_taskpool`(208.8/305.0)、`eval_noleak_oracle_token_taskpool`(234.8/419.8)、`eval_noleak_W6_taskpool`(~504 部分)。

**B 组 — a1000 机型（不同 GPU + 不同 ckpt supervised_select step1000，仅组内可比，勿与 A 组直接比）**：
- W0 qa1 8k = 60.1 s/样本，qa5 8k = 66.8；K2(local) qa1 8k = 78.6，qa5 8k = 94.9（`logs/a1000_*`）。
- 组内 K2/W0 ≈ 1.3×（注意此 W0 已是 60s，与 A 组 11.4 差异是机型，**不可跨组比**）。

**拟合**（A 组 8k）：总时间 ≈ streaming(共享, ~3-4s) + g·(K+1)²，g≈8-10 → 生成阶段**二次项主导**，K1=3.8×/K2=6.3×/K4=18.3× 落在线性(2/3/5×)与二次(4/9/25×)之间、趋近二次。16k 因 streaming(31 chunk)占比大，相对惩罚显著缩小（K2 仅 4.3×、K4 仅 8.0×）。

### 2c. 生成阶段累积代价（【码算】）

`use_cache=False` 下 20 步生成 = 20 次完整 window forward（不复用 KV）。这是**双重浪费**：
1. 每步重算整窗（标准 KV-cache 解码只算新 token）→ 20× 冗余。
2. 窗越长每步越贵 → 与 §2a 的 (K+1)/(K+1)² 叠乘。

设计文档 `:236-238` 明确：**生产化必须给 window 加 KV cache**（重算一次 prefill 后增量解码），可把生成阶段从「20·L」降到「L + 20」，约 **20× 提速**。当前 probe 实现没做（够用但慢）。

### 2d. K 太大 → OOM（【实测】）

| K | qa1 8k OOM 次数（4 shard 合计，n=100） | 完成样本 s/it |
|---|---|---|
| K6 | **85** | 263（少数小档完成）|
| K8 | **39** | 149-273 |
| K16 | **61** | 107（仅幸存小档）|

OOM 报错：`CUDA out of memory. Tried to allocate 6-14 GiB`（95GB 卡）。原因 = `use_cache=False` 全窗 forward 的激活显存随 L 增长，K≥6（窗≥3584）在长档常爆。**K6/8/16 实际不可用**（既慢又 OOM）。

---

## 3. Slot vs re-forward 本质权衡（定量）

### 3a. 准确率锚点（clean NOLEAK base，【实测】FIFO_FINDINGS_SUMMARY_20260627.md / HEARTBEAT_LATEST.md）

| 配置 (qa1) | 8k | 16k | 32k | 含义 |
|---|---|---|---|---|
| 纯 memory W0（slot hidden 读出）| 12 | 8 | 2 | FIFO 现状 |
| hidden-oracle（隔离 needle 的 **hidden 快照**）| 20 | 24 | 22 | **死快照读出墙 ~20** |
| **oracle-token（隔离 needle 的**原始 token** 重 forward）** | **50** | 28 | 33 | **读出机制被解决** |
| reader-attn-token **K4**（可部署选择）| **11** | 5 | — | **选择墙：≈W0，等于没赚** |
| reader-attn-token K6 | ~0(含OOM) | 0 | — | 更大 K 反更糟（稀释+OOM）|

qa5（多 mention）：clean base oracle-token = 25/15/22(8k/16k/32k)，reader-attn-token ≈ 20。
**训练 A 模型（learn-to-select, mix=0, 非泄漏）token-reforward 曲线**（HEARTBEAT_LATEST，用户已知）：qa5 8k C-probe 28 → step500 39 → step1000 **46** → oracle **66**；qa1 8k 14→11→20→54。

### 3b. 「每分准确率的代价」

以 qa1 8k、A 组同机型为基（W0-class=11.4s, 12 分）：

| 方案 | 准确率 | 存储 | 速度 s/样本 | 每分增量·相对成本 | 裁决 |
|---|---|---|---|---|---|
| W0 纯 slot | 12 | 1 MiB | 11.4 | 基准 | 便宜但弱 |
| **oracle-token**（作弊选择）| 50 (+38) | +0.26 MiB | ~17× | +38 分换 17× 算力，**存储几乎免费** | 读出方向真贡献，但选择作弊不可部署 |
| **reader-attn-token K4**（可部署）| **11 (≈W0)** | +6.25GiB(可优化) | **18.3×** | **+0 分换 18× 算力** | ❌ **qa1 纯亏**：同分、慢 18×、还 OOM |
| reader-attn K6+ | ~0 | 更大 | 更慢+OOM | 负收益 | ❌ |

**qa5（多 mention）则相反**：训练后 token-reforward 28→46(+18 分)，oracle 66 → 选择有真实增益空间，re-forward 的算力代价换来真分数。

### 3c. 本质结论

1. **存储维度上，slot 与 re-forward 都不贵且不冲突**：slot bank 1 MiB（压缩、常数），re-forward token-id 0.26 MiB（比 slot 还小）。re-forward 用「免费 token-id + 重算」换掉「4 GiB KV」。当前 6.25 GiB 是选择 crutch（32 层 FIFO），非读出必需，可砍到 8–256 MiB。**存储不是 re-forward 的瓶颈。**
2. **速度维度上，re-forward 的代价是真实且二次增长的**：生成阶段 ∝(K+1)~(K+1)²，叠加 `use_cache=False` 的 20× 冗余；K≥6 直接 OOM。**速度是 re-forward 的主要代价。**
3. **读出质量 vs 选择质量是两堵墙**：re-forward 解决了「读出墙」（hidden 快照 20 → token 重算 50，qa1 8k）；但**可部署的无监督选择（reader-attn）把 50 打回 11 ≈ W0**。所以当前 qa1 上「可部署 re-forward = 同分 + 18× 慢 + OOM 风险」是**纯亏**；价值全压在能否训出好选择器（A 模型 qa5 28→46 是首个 mix=0 干净正信号）。

---

## 4. 部署建议

### 4a. K 取多少性价比最高
- **K=2（窗 1536）是甜点**：恰好落在训练 curriculum `0:3` n_ctx 分布内（in-distribution，无 RoPE 外推），速度 ~6× W0（8k）/ ~4× W0（16k，streaming 稀释），不 OOM。`TOKEN_REFORWARD_DESIGN_20260627.md:97-100`。
- **K=4 上限**：18× 慢、窗 2560 轻度外推、长档偶发 OOM；只在选择召回明显吃紧且选择器够强时用。
- **K≥6 禁用**：OOM 频发（8k 已 85/100 爆）、稀释反伤分（K6 qa1≈0）。
- 注意：**当前 K 的准确率收益取决于选择器**。无监督 reader-attn 选择下 qa1 任意 K 都 ≈W0（白花算力）；只有训练出好选择器（或多 mention 的 qa5）才值得付 re-forward 的速度税。

### 4b. 生产化必做的两项优化（不改读出机制，纯工程）
1. **window 加 KV cache**：re-forward 一次 prefill + 增量解码，砍掉 `use_cache=False` 的 20× 冗余（设计文档 `:236`）。这是单项最大提速（~20×），把 K=2 的实测 72s 量级拉回个位数秒。
2. **选择索引从 32 层 FIFO hidden 砍到单选择层（或池化）**：6.25 GiB → 256 MiB（raw）/ 8 MiB（池化 16 tok/chunk），设计文档 `:54-58`。读出本身只需 token-id（0.26 MiB）。

### 4c. 长档（≥32k）怎么办
- **瓶颈是选择 recall 不是读出/存算**：reader-attn top-2 chunk 命中率随长度衰减 72/78/58/**40**% @4k/8k/16k/32k（`RUN_REGISTRY.md:1290`）。32k 下选不中 needle → re-forward 无的放矢。
- token-id 存储在 32k 仍只有 0.26 MiB，slot bank 1 MiB，**存储完全不构成长档障碍**；速度上 streaming 占比随长度上升、re-forward 相对惩罚下降（16k K2 仅 4.3×），加 KV cache 后长档 re-forward 完全可行。
- **真正要投资的是选择器**：(a) 训练 reader-attn-select + reforward（STE，mix=0，A 模型已给首个正信号 qa5 28→46）；(b) 多层投票选 chunk；(c) K=2 起步、recency floor 1-2。详见 `LEARN_TO_SELECT_DESIGN_20260627.md`（confidence MED-LOW，2 个 death-list 风险）。

---

## 附：数字溯源速查
- 模型：Llama-3-8B `hidden=4096, layers=32, kv_heads=8, head_dim=128, vocab=128256, bf16`（`models/Meta-Llama-3-8B/config.json`）。
- chunk_size=512，max_new_tokens=20（log header / `:1067`）。
- slot：num_slots=128, slot_dim=4096, shared bank, bf16（log header；`memory_bank.py:229`；`patch.py:110`）。
- FIFO：per-layer，cap=50 chunk（`config.py:879`），h_stored=detached hidden [1,512,4096]（`layer.py:1505,1554`）。
- 准确率：clean NOLEAK base（`FIFO_FINDINGS_SUMMARY_20260627.md`）；训练曲线 A 模型 mix=0（`HEARTBEAT_LATEST.md`，非泄漏 b25）。
- 速度：见 §2b 各 log 路径；A 组(taskpool 95GB) 与 B 组(a1000) **机型不同，禁止跨组比**。
