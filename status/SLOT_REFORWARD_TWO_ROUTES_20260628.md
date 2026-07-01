# slot+reforward 两条路线对立分析：粗筛指针(chunk级) vs token容器(token级)（2026-06-28）

> 架构分析 agent。只读 + 分析，不跑训练/eval，不改代码，不碰泄漏 ckpt。
> 全部 mix=0 干净数字。引用一律 `file:line`。诚实区分【读码确证】vs【推断/假说(置信度)】。
> 本报告聚焦用户提的【两条具体路线】的对立，是对正在跑 Workflow(reader-attn select → chunk reforward)的补充。

---

## TL;DR — 一句话裁决

**路线1可行（= Workflow 的"换一个选择器"，readout 复用已破墙的 chunk-reforward），路线2在机制上自毁（token级 reforward 抽掉了 reforward 赖以破墙的上下文，且继承写入期 query-blind 压缩），其唯一可救形态就是退化成路线1。**

- **路线1**：slot 当粗筛指针 → 定位 document chunk → reforward 整块原始 token。readout 复用已确证破墙的 oracle/reader-attn-token 路径（qa5 8k oracle=66）。唯一新东西 = "slot→document-chunk-id" 映射通道（当前**完全不存在**，~120-160 LOC）。这正是 `MIDPOINT_CONCLUSION_20260628.md` 收尾时提的"下一方向候选"。**可行，是 Workflow 选择墙的一个新攻法。**
- **路线2**：slot 直接记 token、reforward 这些孤立 token。**致命**：(i) reforward 破墙靠"连贯 span 在 query 在场下重新 contextualize"，孤立 token 无法被上下文化；(ii) slot_evidence 的 token 是**写入期按 routing 亲和度 top-k 选的**——这恰恰是 query-blind 压缩，是 reforward 本该逃离的东西；(iii) 要补上下文窗口就等于把整块拉回来 = 路线1。**机制上坏，置信度 HIGH。**

---

## 0. 读码确证：两个 slot 通道存了什么（核实用户给的现状，全部确认）

### 0a. `slot_kv` 通道（路线1/2都会碰）
`src/memory/mem_space/memory_bank.py:145-150`：
- `slot_kv_hidden:[B,M,d_model]` = detached 原始 token **HIDDEN**（非 token-id）
- `slot_kv_slot:[B,M]` = 每个存储 token 属于哪个 slot
- `slot_kv_pos:[B,M]` = **chunk 内 RoPE 位置**（写入点 `layer.py:3695` `_sk_pos = torch.arange(T)` = 0..T-1，**非 document 绝对位置**）

写入 `append_slot_kv_cache` `memory_bank.py:574-678`，调用点 `layer.py:3698-3701`。
**粒度确证**：`memory_bank.py:622` `_h.unsqueeze(1).expand(B,k,T,d)` 把**整个 chunk 的全部 T 个 token** 复制到**每个**被选中 slot。所以 slot_kv 实际是 slot→(它路由到的所有 chunk 的全部 token)，**不是**"slot 精确定位的少数 token"。

### 0b. `slot_evidence` 通道（路线2的核心，用户问的重点）
`memory_bank.py:82-95`：
- `slot_evidence:[B,N,Bcnt,d]` = 每 slot top-Bcnt 高 salience token 的 **HIDDEN**（默认 Bcnt=8，`config.py:534`）
- `slot_evidence_score:[B,N,Bcnt]` = salience（routing 亲和度）
- `slot_evidence_pos:[B,N,Bcnt]` = **每 token 的源位置（-1=空）**

★关键核实：`slot_evidence_pos` 存的是 document 绝对位置吗？**不是。** 写入点 `layer.py:2036` `_top_s,_top_t = torch.topk(_aff, k=_C, dim=2)`——`_top_t` 是对 **chunk 内 T 个 token** 的 affinity 取 top-k 得到的**chunk 内 token 索引 0..T-1**，原样作 `token_pos=_top_t` 传给 `write_evidence`（`layer.py:2043-2044`）。注释 `memory_bank.py:87-91` 说 "SOURCE absolute position ... the in-chunk offset" ——**确证是 chunk 内 offset，不是 document 绝对位置**（每个 streaming chunk 的 RoPE 都从 0 重启，所以 offset==它在那个 chunk 里的相位，但跨 chunk 不可区分）。

**两通道对 reforward 的三缺口（全部确证）**：
1. **无 token-id**：存的是 hidden。reforward 要原始 token-id 重喂模型；冻结 hidden reforward = 退回死快照（读出天花板 ~20-24，`MIDPOINT_CONCLUSION:1`段2）。
2. **位置是 chunk 内 0..T-1，非 document 绝对**：无法知道一个 token 来自第几个 document chunk。对比 `RawKVReadoutStore` 有 `token_chunk:[B,M]`(`rawkv_readout.py:56,64`) 用 `self.n_chunks` 计数器逐 chunk +1（`:111-113,143,149`）——slot 两通道**都没有**对应字段/计数器。
3. **MemorySpaceLayer.forward 拿不到 input_ids**：`grep input_ids src/memory/mem_space/layer.py` = **零命中**（确证）。层内只有 `hidden_states`，所以"在写入点直接存 token-id"需先把 input_ids 一路 plumb 到被包裹 decoder 的每层 forward —— 跨整个 backbone 的签名改动。

### 0c. reforward 读出接口（两路线共用）
`run_babilong_mem_space.py:734-745`：吃 **document-absolute chunk 索引集合** `oracle_token_chunks` → `chunks[c]`（`chunks=tokens.split(chunk_size)` 是原始 token-id，`:690`）→ `torch.cat` → reforward 全 32 层。**chunk 级，免费复用。**

### 0d. ★机器不同源（确证，易被忽略的前提）
产出 qa5 8k oracle=66 / K4 部署 16k=38 的干净 NOLEAK ckpt 用 `--use_fifo_memory`（`scripts/launch_mem_space_fifo_b10_chunk512_NOLEAK_diskB.sh:31`，无 `--use_slot_kv_cache`/`--use_slot_evidence`）。FIFO 路径 **bypass 全部 slot 路由**。`use_slot_kv_cache`/`use_slot_evidence` 默认 False（`config.py:533`, `layer.py:977-982`）。**两条路线都需要一个 slots 真正在跑、且 slot_kv/evidence 通道开启的 ckpt，与产出 66 的 FIFO ckpt 不是同一个模型。**

---

## 1. 路线1：slot 粗筛 → 定位 chunk → reforward 整块（chunk级）

### 1a. slot 能输出"哪个 document chunk 相关"吗？
**当前不能。**【确证】
- selector forward 返回 `idx:[B,top_k]` = 选中的 **slot 索引**（`selector.py:504,557`），是 slot-space，不含 chunk 信息。
- `slot_token_mass:[B,N]`（`memory_bank.py:117`）= 每 slot 累积吸收的 token 质量，也是 slot-space，**无 chunk 映射**（注释 `:109-117` 明确 routing 是 per-chunk 聚合）。
- routing 分布同理 slot-space。
- **结论**：要"slot→哪个 document chunk"，必须**新加 slot→chunk-id 映射**。当前无任何代码产出它。

### 1b. 改动点 + LOC（核实 `RESEARCH_SLOT_REFORWARD_20260628.md:73-88` 的估计，确认）
1. **slot_kv（或 evidence）加 document-chunk-id 通道**：
   - `memory_bank.py:148-150` 加 `slot_kv_chunk:[B,M]`，在 `append_slot_kv_cache`(`:574`) 随 hidden/pos 并行 append（~15 LOC，照抄 `rawkv_readout.py:111-113,141,147` 的 `token_chunk` 写法）。
   - 写入点 `layer.py:3695` 需传入**当前 document-absolute chunk 计数**。该计数在 slot 路径**不存在**（只 FIFO 有 `_fifo_write_seq` `layer.py:618`，且只在 oracle 路径维护）。需新增一个 per-sample 重置的 chunk 计数器（仿 `_fifo_write_seq`，~10 LOC + reset 接入 `memory_bank.reset()` `:156-180`）。
2. **新增 "slot 选择 → document-chunk 集合" 提取器**（仿 `_select_chunks_reader_attn` `run_babilong_mem_space.py:476-594`）：question chunk 跑 selector → top-k slot idx → 用新 chunk-id 字段把 slot 反映射到 document chunk 集合 → 返回索引。~60-90 LOC。
3. **喂现有窗口构造**（免费，`run_babilong_mem_space.py:734` 已接受 chunk 索引集合）。
4. CLI flag `--swa_slot_select_token` + 互斥校验，~15 LOC。

**改动量：~100-130 LOC（中等）**，核心难点 = (1) 的 chunk-id 写入通道 + per-sample 计数器、(2) 的反向映射。

### 1c. 与正在跑 Workflow 的重合度
**高度重合 readout，竞争 selector。**【确证 + 文档对照】
- Workflow（`HEARTBEAT_LATEST.md`, `MIDPOINT_CONCLUSION.md`）= **reader-attn select chunk → chunk reforward**，K4 部署 qa5 16k=38/32k=32。
- 路线1 = **slot select chunk → chunk reforward**。**readout 完全相同**（都是 §0c 的 chunk-reforward），只是把 `_select_chunks_reader_attn` 换成 slot-based 选择器。
- `MIDPOINT_CONCLUSION.md` 末尾"下一方向候选"**就是路线1的重新框定**："不是 slot 当检索器，而是用 slot 训练时持续更新+跨chunk聚合的全局表示做更强粗筛，再 reforward 选中 chunk 原 token"——slot 有 reader-attn（单层 q·k 局部打分）没有的**跨 chunk 全局信息**。
- ★旧否决理由失效：昨天否决（slot 选 needle 22% < reader-attn 55%，`RESEARCH_SLOT_REFORWARD:122`）基于 **8k**；但 MIDPOINT 实测**长档 reader-attn 已掉到 ≈ 随机**（recall@4=0.15-0.17 vs chance 0.16）。所以"slot 比 reader-attn 差"的否决在**长档不成立**——这是路线1唯一真正未知、值得 probe 的活问题。

---

## 2. 路线2：slot 直接记 token → reforward 这些 token（token级）★Workflow 漏的

### 2a. 要能 reforward 必须改的（核实用户 (a)(b)）
**(a) 存 token-id 或 document 绝对位置（而非 hidden）**：
- 写入点 `layer.py:2039-2044`：现在 `hidden_states.detach().gather(...,_top_t)` 存 hidden。
- 选项A（存 token-id）：层内**无 input_ids**（§0b 缺口3）→ 须把 input_ids plumb 进每层 forward，跨 backbone 签名大改，**不推荐**。
- 选项B（存 document 绝对位置 + 让 eval 端映射回 token-id）：eval 端有完整 `chunks`/tokens（`run_babilong_mem_space.py:690`），可由 doc 位置回查 token-id。**较便宜**，但依赖 (b)。

**(b) `slot_evidence_pos` 从 chunk 内 pos 改成 document 绝对位置**：
- 写入点 `layer.py:2043-2044` 传的是 `_top_t`（chunk 内 0..T-1，§0b 确证）。
- 改法：`doc_pos = chunk_counter * T + _top_t`，需在 evidence 写入路径加**同一个 per-sample chunk 计数器**（与路线1 §1b(1) 同一需求）。
- 改 `write_evidence`(`memory_bank.py:334`) 注释/语义 + 调用点 `layer.py:2043` + 加计数器。~25-40 LOC。
- 新增 readout 路径：选中 slot 的 evidence 位置 → 映射 token-id → reforward。~50-80 LOC。

**改动量：~80-120 LOC**，但见 §2b —— **改完也大概率不 work**。

### 2b. ★根本疑问：token 级 reforward 有意义吗？
**裁决：没有（对孤立 token）。token 级 reforward 在机制上自毁。**【推断，置信度 HIGH，有机制 + 读码 + 历史数据支撑】

reforward 破墙的机理（`MIDPOINT_CONCLUSION.md`段1.1, `TOKEN_REFORWARD_DESIGN:26-38`）= **选中 chunk 的连贯 token span 重过全 32 层，在 query 在场下每层 needle↔query 多跳重新 contextualize**。这依赖**一段连贯上下文**（needle 的完整句子 + 邻近）。把 slot 记的孤立 token 抽出来 reforward 会触发三重失效：

1. **丢局部上下文**【HIGH】：slot_evidence 存的是 affinity top-8 的**散落 token**（`layer.py:2036` topk over T）。needle 事实 "Mary moved to the kitchen" 若只剩 "kitchen" 这一个高 salience token，孤立 token 无法被正确 contextualize——这正是 reforward 想逃离的"无上下文"困境。
2. **写入期 query-blind 压缩**【HIGH，致命且 Workflow 漏点的反面】：slot_evidence 的 top-k 是**写入流式时**按 routing 亲和度选的（`layer.py:2029` `_aff = q·k`），**query 还没出现**。这恰恰是 reforward 应该绕过的 query-blind 压缩。若写入期 top-8 没存到答案 token，reforward 永远拿不回。等于把"压缩-then-读"换了个皮，**没逃出读出墙**。
3. **RoPE 位置 OOD**【MED-HIGH】：散落 token 拼接（或按 doc 位置放进小窗口）位置非连贯，超出训练窗口分布。

**唯一可救形态** = 用 token 位置回去**拉一段上下文窗口**再 reforward —— 但那就等于"定位 → 扩成 chunk → reforward 整块" = **路线1**。用户对路线2的定义明确是"只重读 slot 记的少数关键 token"，**这个纯形态破不了墙**。

**所以 Workflow"漏了路线2"不是遗漏了一条捷径，而是路线2的 token 级粒度本身是错的目标——它要么自毁，要么坍缩成路线1。**

---

## 3. 两条路线对比

| | 路线1 (slot→chunk→整块 reforward) | 路线2 (slot→孤立 token→直接 reforward) |
|---|---|---|
| readout 机制 | 已确证破墙（oracle chunk-reforward qa5 8k=66） | 自毁（孤立 token 无法 contextualize） |
| 唯一变量 | **选择精度**（slot 选 chunk 准不准） | 选择精度 **+ 写入期 query-blind 压缩 + 上下文丢失** 三重 |
| 算力 | reforward 整块（k×512 token） | 省（只重读少数 token）——但省的代价是破不了墙 |
| 与 Workflow | **重合 readout，竞争 selector**（= MIDPOINT 下一方向） | **全新但机制可疑** |
| killer 风险 | slot 长档选 chunk 精度**仍可能 ≈ 随机**（则不优于已部署 reader-attn 38）；+ §0d 机器不同源 | token 级 reforward 破坏上下文 + 继承写入期压缩 = **双重判死**；可救形态坍缩成路线1 |
| 可行性裁决 | **可行**（值得 probe） | **不可行**（纯形态机制坏） |

**哪条更可能 work：路线1。**【推断 HIGH】
- 路线1 把已破墙的 readout 配一个**未在长档测过的、带全局信息的选择器**。最坏 = slot 选择 ≈ reader-attn（38），最好 = slot 全局聚合在长档比单层 q·k 准（未知，活问题）。下行有保底（复用 readout），上行有机会。
- 路线2 即便选择完美（oracle），孤立 token reforward 也破不了墙（§2b 机制 + 写入期压缩）。下行无保底。

**路线2 各自 killer 风险**（用户问）：根本不是"省算力丢上下文"的 tradeoff——而是丢上下文 = 丢掉 reforward 的**全部威力**，省下的算力买不到任何读出收益。这不是省 vs 准的权衡，是"省了算力但东西不工作"。

---

## 4. 各自最小验证（zero-train probe，mix=0）

### 路线1 Probe — 只量 slot 长档选 chunk 精度（最便宜，先做）
- **不需要建完整 chunk-id 通道**。在一个 **slots 真正在跑** 的干净 ckpt 上，跑 BABILong **长档**（qa5 16k/32k —— reader-attn 正是在这里掉到随机，是 Workflow 的墙）。
- 加一个**纯 telemetry logger**（不改字段）：流式写入时在 `layer.py:~2002`（idx 已知处）记录每 chunk 的 `(chunk_counter, idx)`；question chunk 时记录 selector idx。离线把 question 选中的 slot 反映射到喂过它们的 chunk → 算 needle-chunk recall@k。~20-30 LOC logger，**零训练、零字段改、不动 readout**。
- **判据**：slot recall@4 > reader-attn 的 0.15-0.17（`MIDPOINT_CONCLUSION` 段2）→ slot 全局表示在长档确有信号 → 值得建完整通道(§1b ~100-130 LOC)走 full reforward。若 ≈ 随机/≈ reader-attn → 路线1 在长档也无优势，否决。**约半天。**

### 路线2 Probe — 直接证伪 token 级粒度（更便宜，用 oracle 隔离选择变量）
- **关键设计**：用 **oracle 选 needle chunk**（已知正确，剔除选择变量），但 readout 时**只 reforward 该 chunk 内 affinity-top-k 的孤立 token**（模拟 slot_evidence 存的东西），对照 **full-chunk oracle reforward(66)**。
- 实现 = 在 `run_babilong_mem_space.py:743` 窗口构造处加一个"只取 chunk 内 top-k token / 或 token±context"的切片分支。~40-60 LOC，零训练。
- **判据**：若 oracle 选对 chunk 后，孤立-token reforward 显著掉向 baseline（远低于 66）→ **确证 §2b：token 级粒度本身破坏 reforward**，路线2 与选择精度无关地被否决。若意外接近 66 → §2b 推断被推翻，路线2 复活。**约半天，且直接回答机制问题。**
- 推荐：**先做路线2 probe**——它最便宜，且若证伪 §2b 就一锤定音地砍掉路线2，让精力集中到路线1。

---

## 5. 诚实标注：确证 vs 推断

**读码确证**：
- slot_kv / slot_evidence 都存 hidden + chunk 内位置 + slot-id，**无 token-id、无 document-chunk-id**（`memory_bank.py:145-150,82-95`；`layer.py:2036-2044,3695-3701`）。
- slot_evidence_pos = chunk 内 0..T-1（topk over T 的索引 `layer.py:2036`），非 document 绝对位置。
- slot_kv 整块复制到每个选中 slot（`memory_bank.py:622`），非精确少数 token。
- 层内无 input_ids（grep 零命中）。
- reforward 接口吃 document-chunk 索引回取原始 token-id 整块（`run_babilong_mem_space.py:734-745,690`）。
- `RawKVReadoutStore.token_chunk` 是现成的 chunk-id 写法模板（`rawkv_readout.py:56,64,111-113`）。
- 产出 66 的 NOLEAK ckpt 用 `--use_fifo_memory`，bypass slots（`launch_...NOLEAK...:31`）；slot 通道默认关（`config.py:533`）。
- selector 返回 slot-space idx，slot_token_mass 是 slot-space，无 chunk 映射（`selector.py:504,557`；`memory_bank.py:109-117`）。
- 旧否决（slot 22% < reader-attn 55%）基于 8k；长档 reader-attn ≈ 随机（`MIDPOINT_CONCLUSION` 段2）。

**推断/假说（置信度）**：
- "token 级 reforward 孤立 token 破不了墙，必须 chunk 级 / 坍缩成路线1"——**HIGH**（机制 + 写入期 query-blind 压缩 + 历史 readout 墙 ~20-24）。
- "路线2 写入期 affinity-top-k 是 query-blind 压缩，等于没逃出读出墙"——**HIGH**（读码 `layer.py:2029` + 项目核心结论 inject vs reforward）。
- "路线1 在长档 slot 选择是否优于 reader-attn"——**未知/活问题**，旧 8k 否决不适用，需 probe。
- "路线1 比路线2 更可能 work"——**HIGH**（路线1 readout 有保底，路线2 readout 自毁）。
