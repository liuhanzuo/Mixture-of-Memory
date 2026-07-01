# 方案可行性分析：slot 当检索器定位 → 对选中【原始 token】做 re-forward 读出（2026-06-28）

> 架构分析 agent。只读 + 分析，不跑训练/eval，不改代码，不碰泄漏 ckpt。
> 全部 mix=0 干净数字。引用一律 `file:line`。
> 诚实区分【读码确证】vs【推断/假说】。

---

## TL;DR — 一句话裁决

**方案在概念上"对一半、错一半"，且当前组件【不能直接拼接】，需要新写代码。**

- **对的一半（读出）**：re-forward ≫ inject 是【读码 + 历史确证】的真机制差异。任何 re-forward 都能轻松破 Method A 的 +1-2.5（那是 inject 读出天花板）。用户对 "必须配 re-forward 非 inject" 的判断**正确**。
- **错的一半（选择 + 粒度）**：用户方案把 re-forward 的读出红利，套到一个【比现有 reader-attn 更差的选择器（slot 路由）】上，并隐含一个【token 级 re-forward】粒度（破坏 re-forward 赖以生效的上下文）。**slot→token 映射当前存的是 hidden + chunk 内位置，既无 token-id 也无 document-chunk-id，无法直接 re-forward。**
- **核心判断**：能破 +1-2.5（必然，因为换成 re-forward 读出），但**没有理由相信能破真正的墙**（deployable reader-attn-token = 11-12，oracle-token = 46-66）。因为 re-forward 读出与选择源无关，剩下的唯一变量是**选择精度**，而 slot 路由的选择精度历史 = ~22%（≈随机），**劣于**已部署的 reader-attn（55%）。

---

## 1. 两组件现状（读码确证）

### 1a. slot→token 映射：存的是什么？能否取回原始 token-id？

**结论：存 hidden + chunk 内位置 + slot-id；【不存 token-id，也不存 document-chunk-id】。无法直接 re-forward。**

读码确证（`src/memory/mem_space/memory_bank.py`）：
- 字段声明 `memory_bank.py:145-150`：
  - `slot_kv_hidden : [B, M, d_model]` — **detached original token HIDDEN**（不是 token-id）
  - `slot_kv_slot   : [B, M]` — 每个存储 token 属于哪个 slot
  - `slot_kv_pos    : [B, M]` — **source within-chunk RoPE position（chunk 内 0..T-1）**
- 写入 `append_slot_kv_cache` `memory_bank.py:574-678`：入参是 `token_hidden`（hidden，非 id）。位置 `_pos_dup` 来自调用方传入的 `token_pos`。
- 写入调用点 `layer.py:3695-3701`：
  ```python
  _sk_pos = torch.arange(T, ...).unsqueeze(0).expand(B, -1)   # ← chunk 内 0..T-1
  self.memory_bank.append_slot_kv_cache(
      idx.long(), hidden_states.detach(), token_pos=_sk_pos, ...)
  ```
  传入的是 `hidden_states.detach()`（层输入 hidden）+ chunk 内位置。**MemorySpaceLayer 在 forward 里只拿得到 `hidden_states`，拿不到 `input_ids`**（确证：`layer.py` 全文无 `input_ids`/`token_id` 引用，仅 hidden）。

**对 re-forward 的三个致命缺口（确证）**：
1. **无 token-id** → re-forward 需要原始 token-id 重新喂模型；slot_kv 只有 hidden。冻结 hidden re-forward = 退回死快照路径（HIDDEN_VS_SWA Q1，读出天花板 ~20），**不是** token re-forward。
2. **位置是 chunk 内 0..T-1，非 document-absolute** → 即使想"映射回所在 chunk 再 re-forward 整块"，slot_kv 也没记录 token 来自第几个 document chunk。对比 `RawKVReadoutStore` 有 `token_chunk` 字段（`rawkv_readout.py:64`），slot_kv **没有**对应字段。
3. **粒度其实是 chunk 级，不是 token 级**：写入时 `_h.unsqueeze(1).expand(B, k, T, d)`（`memory_bank.py:622`）把**整个 chunk 的全部 T 个 token** 复制到**每个**被选中的 slot 下。所以 slot→token 实际是 slot→(它路由到的所有 chunk 的全部 token)，**不是 slot 精确定位少数 token**。用户设想的"slot 精确定位的少数 token"在当前实现里不存在。

### 1b. re-forward 读出：接口接受什么粒度？

**结论：接受【document-absolute chunk 索引集合】，回取原始 token-id 整块拼接，chunk 级。**

读码确证（`scripts/run_babilong_mem_space.py`）：
- `generate_with_mem_space(... oracle_token_chunks=None ...)` `run_babilong_mem_space.py:597-611`。
- 窗口构造 `:734-745`：
  ```python
  sel = sorted(c for c in oracle_token_chunks if 0 <= c < last_idx)
  pieces = [chunks[c] for c in sel] + [chunks[-1]]   # chunks = tokens.split(chunk_size)
  window = torch.cat(pieces, dim=0)                  # 原始 token-id 整块拼接
  cur = window.unsqueeze(0).to(device)
  ```
  `chunks` 来自 `tokens.split(chunk_size)`（`:690`），是**原始 token-id**。窗口 = 选中 chunk 的原始 token-id + 最后(问题)chunk，重新 forward 全 32 层。
- **接口只吃 document-absolute chunk 索引**，回取 token-id 的工作由 `chunks[c]` 完成。它**不吃** token 级位置，也不吃 hidden。
- 已部署的可训练-free 选择器 `_select_chunks_reader_attn` `:476-594`：跑一遍 question chunk，用 reader q·k salience 在 `_fifo_buf`（FIFO 的 per-chunk hidden 快照）上打分，返回 document-absolute chunk 索引，喂给上面同一个窗口构造。这就是 `--swa_readerattn_token` 路径（`:1100`）。

---

## 2. 拼接可行性 + 需改的代码点

### 关键发现：两组件【接口不匹配】，且 slot 路径与 FIFO 路径是不同机制

re-forward 窗口要的是 **document-absolute chunk 索引（→ 原始 token-id）**。
slot_kv 状态有的是 **slot→(hidden, chunk 内位置)**，无 chunk-id、无 token-id。
中间缺一座桥：**"被选中的 slot → 它们对应哪些 document chunk"**。当前**无任何代码产出这个映射**。

补充确证（机制不同源）：现有 token-reforward 干净数字（46-66 / 11-12）全部来自 **FIFO ckpt**，选择走 `_fifo_buf`（chunk hidden 快照），**FIFO 路径 bypass 全部 slot 路由**（HIDDEN_VS_SWA Q4 `:116`：FIFO 从不填 `_cum_usage`）。slot_kv 是**另一套 config**（`use_slot_kv_cache`，`layer.py:977-982`，默认关）。所以"slot 检索 → re-forward"需要的是一个**带 slots 的 ckpt**，与产出 46-66 的 FIFO ckpt **不是同一个模型**。【这是一个容易被忽略的前提，推断置信度 HIGH】。

### 若仍要拼，需改的具体代码点

1. **给 slot_kv cache 加 document-absolute chunk-id 通道**（新字段）
   - `memory_bank.py:148-150` 加 `slot_kv_chunk: Optional[Tensor]`，在 `append_slot_kv_cache`（`:574`）里随 hidden/pos 并行 append。
   - 写入点 `layer.py:3695-3701`：需传入当前 document-absolute chunk 计数。**该计数当前在 slot 路径不存在**（只有 FIFO oracle 路径有 `_fifo_write_seq`，`layer.py:1251`）。需新增一个 per-sample 重置的 chunk 计数器（仿 `_fifo_write_seq` / `_set_fifo_oracle_needle` 的 reset，`rawkv_readout.py` 同目录 helper `:444-459`）。

2. **新增"slot 选择 → document-absolute chunk 集合"提取器**（仿 `_select_chunks_reader_attn`）
   - 新函数：跑 question chunk → selector 出 top-k slot idx（`selector.py:220` forward 返回 `idx:[B,top_k]`，确证 `selector.py:53`）→ 用新 `slot_kv_chunk` 字段把 slot 映射回 document chunk 集合 → 返回索引。
   - 位置：`run_babilong_mem_space.py`，与 `_select_chunks_reader_attn` 并列，约 60-90 LOC。

3. **喂给现有窗口构造**（免费）
   - `generate_with_mem_space:734-745` 已接受 chunk 索引集合，把 (2) 的输出赋给 `oracle_token_chunks` 即可（仿 `:731-732`）。**这一段不用改。**

4. CLI flag（`--swa_slot_select_token` 之类）+ 互斥校验（仿 `:1272-1283`），约 15 LOC。

**改动量估计**：中等（~120-160 LOC），核心难点是 (1) 的 chunk-id 写入通道 + per-sample 计数器，与 (2) 的 slot→chunk 反向映射。**没有任何现成组件直接做 (1)(2)**【确证】。

---

## 3. 粒度决策：token 级 vs chunk 级 re-forward

**裁决：必须 chunk 级（slot 定位 → 映射回所在 chunk → re-forward 整块）。token 级 re-forward 在机制上是坏的。**【推断，置信度 HIGH，有机制 + 读码支撑】

理由：
- re-forward 之所以破墙（HIDDEN_VS_SWA Q1 `:47-53`，FIFO_FINDINGS `:27`）= **joint query-conditional 重新上下文化**：原始 token 整块重过 backbone，每层 needle↔query 多跳耦合。这**依赖一段连贯的 token span**（needle 的句子上下文）。
- token 级 re-forward（把 slot 精确定位的散落 token 抽出来重 forward）会：(a) 丢失局部上下文（孤立 token 无法被正确上下文化）；(b) RoPE 位置 OOD（散落位置拼接）；(c) 正是丢掉 re-forward 的全部威力。
- 且 §1a 已确证 slot_kv 存的是**整块** chunk（slot→whole-chunk），本来就没有"少数精确 token"可抽。

所以唯一有意义的形态 = **slot 定位 → 整块 chunk → re-forward**，这恰好就是现有 oracle/reader-attn-token 路径，**只是换了个（更差的）选择器**。

---

## 4. 能否破 Method A 的 +1-2.5？关键差异是否真在 inject vs reforward？

### 4a. inject vs reforward 的差异【真实，确证】
- Method A inject：`inattn_kv.py` 把检索到的 hidden 经 k_proj/v_proj 拼进**单层** native KV（keys-only），frozen reader 读不出 → +1.0~+2.5（RUN_REGISTRY `:1056,1066-1070`）。读出天花板 ~20-22。
- token re-forward：原始 token-id 重过全 32 层，query 在场 → oracle 选对 = 46-66（HEARTBEAT_LATEST `:8-16`，FIFO_FINDINGS `:18`）。
- 机制差异是【读码 + 多次实验确证】的项目核心结论。用户"想法对但必须配 re-forward"判断**正确**。

### 4b. 但这是【对错误基线的比较】
**+1-2.5 是 inject 读出天花板。换成 re-forward 读出，破它是必然且 trivial 的——与选择源无关。**

真正的墙已经搬到【选择】（FIFO_FINDINGS `:10,29` 明确："读出已被 token-reforward 解决；选择是另一半墙"）。因为 re-forward 读出对任何选择源都一样，**用户方案的唯一变量 = slot 路由的选择精度**。

### 4c. slot 路由 vs reader-attn：谁选得准？【确证：slot 更差】
| 选择器 | needle precision | 来源 |
|---|---|---|
| Method A slot/gist 路由 (top1) | **22.5%（≈随机）** | RUN_REGISTRY `:1240` |
| trained gist (lr×30 已排除欠训) | **0.0%（比随机还低）** | RUN_REGISTRY `:1250` |
| **reader-attn q·k** | **55%（8.8×随机）** | RUN_REGISTRY `:1256` |

- slot 路由本身就是 Method A 用过的检索源（HEARTBEAT_LATEST `:19`："= 已实现的 Method A，slot_kv_pos 映射"），其 needle 命中 ~22%（随机）。
- 已部署的 reader-attn-token（55% 选择器 + re-forward 读出）只到 **11-12**（FIFO_FINDINGS `:20`，HIDDEN_VS_SWA 等）。
- **推断【置信度 HIGH】**：用 22%/随机精度的 slot 路由喂 re-forward，不会优于 55% 精度的 reader-attn-token（11-12），更到不了 oracle（46-66）。slot 选择劣于已测的 reader-attn。

### 4d. 还有 dilution 复发【确证机制】
slot 是 EMA 多 chunk 混合的吸引子；slot_kv 在一个 slot 下存了**所有路由到它的 chunk 的全部 token**（`memory_bank.py:622`）。"取回选中 slot 绑定的 chunk" = 一个**大而稀释的集合**，正是 Method A "真墙 = DILUTION" 结论（RUN_REGISTRY `:1261-1267`：full_haystack 0%，加 15 distractor 把 97.5% 打到 0%）。slot→chunk 是多对多偏多对一，**不是干净指针**。

### 裁决
- "能破 +1-2.5？" **能**（换成 re-forward 读出必然破）——但这是和错误基线比。
- "能破真墙（11-12）/逼近 oracle（46-66）？" **没有理由相信，倾向不能**。re-forward 读出红利与选择源无关；slot 选择精度（~22%/随机）劣于已测 reader-attn（55%→11-12）。slot 检索 + re-forward = "用更差的选择器跑同一个读出"。

---

## 5. 风险 + 最小验证实验

### 风险
- **(a) slot 选择精度更差**【确证】：slot/gist 路由 ~22%（随机）< reader-attn 55%。选择墙不变或更糟。这是方案的命门。
- **(b) token 级粒度破坏 re-forward 上下文**【推断 HIGH】：必须 chunk 级；token 级丢上下文 + RoPE OOD，且 slot_kv 本就只存整块。
- **(c) dilution 复发**【确证】：slot 绑定多 chunk → 取回集合大而稀释，回到 Method A 真墙。
- **(d) death-list**：若进一步**训练** slot 选择器 → H2 裁决"所有 trained selector 崩到随机精度"（RUN_REGISTRY `:1247-1259`）；T2→BABILong 不转移（`:109,207-214`，rawkv_methodA 教训）。两条都适用。
- **(e) 组件缺口 + 机制不同源**【确证】：slot_kv 无 token-id/无 chunk-id，re-forward 无法直接喂；且产出 46-66 的是 FIFO ckpt（bypass slots），slot 检索需另一套 slots ckpt。

### 最小验证实验（zero-train 优先，mix=0）

**Gate Probe（最便宜，先做）—— 只量选择精度，不用拼 re-forward**：
- 在干净 slots ckpt 上跑一条 BABILong qa（qa1/qa5）样本，记录 question chunk 路由到的 top-k slot，用 slot_kv（或 slot 路由）反推命中 needle chunk 的精度。复刻 Method A needle-precision probe（已知 ~22.5%，RUN_REGISTRY `:1240`）。
- **判据**：若 slot 选择精度 ≤ reader-attn 的 55%（几乎必然），则方案在拼 re-forward 之前就已死——因为 re-forward 读出对选择源无差别，55% 选择器已只到 11-12，更差的选不可能更好。**约 1-2h，零训练，零新代码（仅加 logger）。**

**Full Probe（仅当 Gate 出乎意料地高才做）—— 真拼 re-forward**：
- 实现 §2 的 (1)(2)(4)（chunk-id 通道 + slot→chunk 提取器 + flag），(3) 复用现有窗口构造。
- 跑 qa5 8k + qa1 8k，n=100，W0，与三个锚点对照：reader-attn-token（11-12）、oracle-token（46-66）、Method A inject（16-22）。
- **判据**：>reader-attn-token（11-12）才有意义；逼近 oracle 才算破墙。**约 1 天，零训练。**

**推荐顺序**：先做 Gate Probe。它以最低成本决定方案生死。基于现有全部确证证据，**预测 Gate Probe 会显示 slot 选择 ≈ 随机 ~22%，方案在此即被否决**。若要在"选择 + re-forward"方向投入，已有更优候选 = `LEARN_TO_SELECT_DESIGN_20260627.md` 的监督 reader-attn 选择器（55% 起点，非 22% slot 起点）+ 零训练 topk/layer/multi-layer sweep（该 doc §5 建议先做）。

---

## 6. 诚实标注：确证 vs 推断

**读码确证**：
- slot_kv 存 hidden+chunk内位置+slot-id，无 token-id、无 document-chunk-id（`memory_bank.py:145-150,574-678`；`layer.py:3695-3701`）。
- slot_kv 按整块 chunk 复制到每个选中 slot（`memory_bank.py:622`）。
- re-forward 窗口只吃 document-absolute chunk 索引，回取原始 token-id 整块（`run_babilong_mem_space.py:734-745,690`）。
- inject 机制（单层 native KV concat，frozen reader）`inattn_kv.py` 全文。
- 两组件无现成桥接；FIFO 路径 bypass slots。
- 历史数字：inject +1-2.5（RUN_REGISTRY `:1056,1066`），oracle-token 46-66 / reader-attn-token 11-12（HEARTBEAT/FIFO_FINDINGS），slot/gist 选择 22%/0% vs reader-attn 55%（RUN_REGISTRY `:1240,1250,1256`），dilution 真墙（`:1261-1267`）。

**推断/假说**（置信度标注）：
- "slot 检索 + re-forward 不会优于 reader-attn-token 11-12"——HIGH（基于 re-forward 读出与选择源无关 + slot 选择精度劣于 reader-attn 的确证）。
- "token 级 re-forward 机制上坏，必须 chunk 级"——HIGH（机制 + 读码）。
- "Gate Probe 会显示 slot 选择 ≈ 随机"——MED-HIGH（外推 Method A 的 22%，但未在当前确切 ckpt 上重测）。
- "产出 46-66 的 FIFO ckpt 与 slot 检索需要的 slots ckpt 不同源"——HIGH（读码 + Q4 FIFO bypass routing）。
