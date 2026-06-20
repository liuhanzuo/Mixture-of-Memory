# Per-layer cache_top_k → mem_space 移植接口规格

**作者**: landmark-repro  **日期**: 2026-06-20  **用途**: gap (b) selection precision —
把 Landmark 的 per-layer(+per-head)chunk 选择移植到 mem_space,替换当前"单一全局 reader-attn 打分一次喂所有注入层"(top1 仅 22.5%)。纯设计,待 methodA-eval chunk-64 (a) 粒度确认后实施。

---

## 1. Landmark cache_top_k 每层 chunk salience 的精确张量流

源：`external/landmark-attention/llama/llama_mem.py:311-436`（每个 `LlamaAttention.forward` 内，即**每层各跑一次**）。

### (1a) query / key 来源 — 每层自己的投影
```
query_states = q_proj(hidden_states).view(bsz, num_heads, q_len, head_dim)   # :311 本层 q_proj、本层 hidden
query_states = apply_rotary_pos_emb(query_states, ...)                        # :322 RoPE 后
```
- **打分用的 query = 本层每个普通 token 的 q**（不是 landmark token 的 q；是 q_len 个当前 token 各自的 q）。**per-head**（num_heads 维保留）。

### (1b) chunk key = 该 block 的 landmark-token 的 key
```
mem_key_nopos = past_key_mem.select(dim=3, index=mem_freq)   # :381  每 block 取它最后一位(landmark token)的 K
                                                             # past_key_mem: [bsz,nh,num_mems,mem_freq+1,hd]
mem_key = apply_rotary_pos_emb(mem_key_nopos, mem_pos)       # :390  给 landmark-key 上它所在位置的 RoPE
```
- **chunk 的"代表 key" = 该 block 末尾那个专门 landmark token 的 K 投影**（不是 block 内 token 的 mean/pool）。landmark token 是训练时每 mem_freq 个 token 插一个的特殊 token，full-FT 让它学成"本 block 的概括索引"。

### (1c) salience = 本层 per-head 的 query·landmark-key
```
mem_attn_weights = (query_states @ mem_key.transpose) / sqrt(head_dim)
   # :391  shape [bsz, num_heads, q_len, num_mems]  —— 每 (head, 当前token) 对每个历史 block 的分
```

## 2. 每层独立选 top_k + gather（per-head 还是 per-layer？）

```
# 训练路径 aggregate=None (:402-403): token_retrievers=q_len, head_retrievers=num_heads
mem_selected_idx = mem_attn_weights.topk(k=top_k, dim=-1).indices    # :408
   # [bsz, num_heads, q_len, top_k]  —— ★per-head + per-query-token 各自选 top_k 个 block
selected_keys = past_key_mem.gather(dim=3, mem_selected_idx_expanded) # :419-421 沿 num_mems gather
   # 每 (head, token) 取自己选中的 block 的完整 (mem_freq+1) 个 KV
```
**结论：Landmark 选择是 per-layer × per-head × per-query-token 三重独立**：
- 每**层**用自己 q/k_proj → 不同层选不同 block（不同层 attend 不同语义）。
- 每**head**独立 topk → 同层不同 head 可选不同 block（多视角投票）。
- 每**query token** 独立选 → 长 query 里不同位置可检索不同历史（passkey 场景 query 短，差异小）。
- **无跨层/跨head聚合**：不是"选一次广播"，是 32×num_heads 次独立选择，各自只 gather 自己选中的块拼进自己的 attention。
- offload(推理)路径 :397-401 才聚合成 per-layer 单 kept-set（max_over_tokens），训练路径不聚合。

## 3. 映射到 mem_space + 改动量

### 现状
`GistReadout.retrieve()`（rawkv_readout.py:193+）= **全局选一次**：
- 用共享 gist_query_proj/gist_key_proj 打分 → 选一个 kept-set（per-batch 单一）→ 同一份 sel_hidden 喂所有注入层（16/20/24）。
- top1 命中仅 22.5% = 单点全局打分的方差大、无 per-layer/per-head 投票。

### 复刻 per-layer 选择的改动
1. **retrieve 从"全局一次"改成"每注入层各调一次"**：
   - 把 `retrieve(query_hidden=...)` 的 `query_hidden` 换成**该注入层自己的 hidden**（layer.py 里每层已有），每层各算 score → 各自 topk → 各自 gather。
   - 改动点：`layer.py:2378` 的 retrieve 调用已在每层 forward 内（已 per-layer 上下文）；retrieve 内部把"用共享 gist_proj"改成可选"用本层 q_proj over chunk-key"。
2. **chunk key 选项**：
   - (A) 沿用 per-chunk gist_src（block mean/landmark），但**打分 query 用本层 hidden**（最小改动，先验这个）。
   - (B) 完整复刻：给每 block 存一个 landmark-style key（需在 store 写入时多存一个"block 概括 token"的表示），各层用本层 k_proj 投影它。改动大。
   - **建议先 (A)**：per-layer query × 共享 block-key，已能拿到"不同层选不同块"的多样性，验证 per-layer 是否提升 precision；若不够再上 (B) 的 per-block landmark key。
3. **per-head 是否要**：mem_space 现 gist 打分是 pooled（无 head 维）。加 per-head 投票改动中等（score 保留 head 维、topk per-head、union/vote 选块）。**建议第二步**：先 per-layer（无 head），不够再 per-head。
4. **gather 不变**：已给的 select_and_gather_blocks 片段就是 per-layer 调用 → 每层 sel_hidden 独立。多层各自 gather 各自 sel_hidden 喂 build_retrieved_kv。

### 改动量评估
- **(A) per-layer query 打分**：~中等。retrieve() 加一个 `query_hidden_override` 入参 + 调用处传本层 hidden；score 用本层 hidden（可选过一个小 proj 或直接 d 空间）。store 不变。~30-50 行。
- **per-block landmark key (B)**：大。store 写入要多存 block-summary 表示，训练时学它 → 需重训。先不做。
- **per-head 投票**：中。score/topk 保留 head 维 + 选块投票聚合。第二步。

## 4. per-layer 为什么可能 >> 全局（机制假设）

1. **方差降低（投票）**：全局单点打分 top1=22.5%，单次 miss 就错。per-layer × per-head = 32×nh 个独立选择器，对同一 needle block 多次投票 → 即使单个选择器命中率低，"至少一层/一头选中"的概率随独立选择器数指数上升（类 ensemble）。Landmark 每层各 gather 自己选中的，只要**该层**选对该层就能读出，不需要全部层都对。
2. **层间语义互补**：不同深度层 attend 不同抽象层级（浅层词形/位置、深层语义）。needle(passkey 数字)在某些层的表示空间里更显著 → 那些层选得准，把信息读进自己 residual，后续层接力。全局单一打分空间只能用一种语义视角。
3. **per-query-token 局部性**：长 query 里不同 token 可检索不同历史块（mem_space last_chunk 的 target chunk 内多 token 各自检索）。全局 pooled query 丢掉这个。
4. **和我们 S5 结论自洽**：S5 证"读出必须多层分布式";per-layer 选择是其前提——每层要先各自选对块才能各自读出。全局选一次喂多层 = 多层读但都读同一个(可能错的)kept-set，没有 per-layer 纠错。

## 5. 实施顺序（待 methodA-eval chunk-64 (a) 确认粒度后）
1. 先验 (A) per-layer query 打分（共享 block-key，最小改动 ~30-50 行）：retrieve 每注入层用本层 hidden 打分各自选。判据 W0 oracle 关掉、用 per-layer reader-attn 选，看 top-k recall 是否从全局 22.5% 提升 + W0 是否爬向 97%。
2. 不够再加 per-head 投票。
3. 仍不够再上 (B) per-block landmark key（需重训）。
