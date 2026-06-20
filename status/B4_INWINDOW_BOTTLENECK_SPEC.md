# B4 接入 spec — in-window summary bottleneck (选择侧, 梯度源2)

**作者**: methodA-eval (接入 spec) + landmark-repro (review + store/launcher)
**日期**: 2026-06-20
**状态**: SPEC — 等纯T2 step100 W0 gate。纯T2 floor → 实装;升 → 缓(锦上添花)。
**前置**: landmark-repro 已 scaffold config flag `rawkv_inwindow_summary` + layer.py install wiring + launcher INWINDOW=1 env-gate (默认 off byte-identical, committed)。

---

## 0. 目标 (一句话)

让 `summary_proj`(per-sub-block 可训练 key 投影)从**带梯度的 target chunk 自己的 intra-chunk attention** 拿到 Landmark 式的**密集 in-window 概括梯度**(梯度源2),而不是只靠 readout-time 的弱选择梯度(梯度源1)。推理时用训好的 summary_proj 算 context chunk 的 summary key 做跨块选择。

## 1. 梯度路径 (B4 核心, 为什么成立)

```
dolmino target chunk (512 token 自然文本, grad-bearing, full-target NTP loss labels=target_input)
  → intra-chunk self-attn 施 in-window bottleneck:
      target 切 8 个 64-token sub-block (sub_size=64)
      later sub-block 的 token 不能直连 attend earlier sub-block 的个体 token,
      只能经 earlier sub-block 的 summary key 中转 (grouped-softmax bottleneck)
      summary_key_j = summary_proj( pool(target hidden[sub-block j]) )   # target hidden 带梯度
  → target LM loss (每位监督) 反传 → summary_proj 拿密集 in-window 梯度 (梯度源2)
```

**为什么不碰 no_grad (解 landmark-repro 的死结)**: in-window 训练发生在 **target chunk**(带梯度),不在 context chunks(:2070 no_grad)。不需要让 context pass 带梯度(不爆显存 = 非 B1),不需要 aux loss(就是 target 自己的 NTP loss = 非 B3),target 是自然文本有内部结构可概括(非纯T2 的 Q+A = 非 B2 无意义)。

**为什么不 leak (一石二鸟)**: bottleneck 让 target 的 later sub-block **不能直连抄 earlier sub-block 的 raw token**,只能经 summary 概括 → 结构上破了 A 的 adjacent-copy leak。in-window 是 LOCAL(同 chunk 内 later 读 earlier),非跨块 readout。

## 2. 接入点 (inattn_kv.py forward)

inattn_kv wrapper forward (:109) 当前流程:
- `injected = self._inattn_kv` (readout 层 target forward 时由 layer.py 设, 含 K_raw/V_raw[/col_bias])。
- 算 q/k/v from hidden_states (:142), native 部分 + 拼接 retrieved KV → attention。

**B4 改动 (仅当 `self._rawkv_inwindow_summary` and not eval-cross-chunk)**:
在算 **native self-attn 部分**(当前 chunk 的 q·k over 自己的 token)时,把标准 causal softmax 换成 **sub-block grouped bottleneck**:

```python
# 伪码, 加在 inattn_kv forward 算 native attention 处 (retrieved concat 之前/并列)
if getattr(self, "_rawkv_inwindow_summary", False) and self.training and Lq == cur_chunk_len:
    gs = int(getattr(self, "_rawkv_subblock_size", 64))
    # Lq = target chunk len (512), 切 n_sub = Lq//gs 个 sub-block
    # 1. 每 sub-block summary key = summary_proj(pool(hidden[sub-block]))  [B, n_sub, hd]
    #    pool = mean over sub-block tokens (或 max, 和 readout 一致)
    #    ★summary_proj 在 self_attn 上 (store 挂载由 landmark-repro 接), hidden 带梯度
    # 2. in-window bottleneck attention (用 landmark-repro 的 _inwindow_bottleneck_ref.py):
    #    later sub-block token 的 attention:
    #      - 同 sub-block 内: 正常 causal (token 看本 sub-block earlier token)
    #      - earlier sub-block: 只能 attend earlier sub-block 的 SUMMARY key (非个体 token)
    #      - 即 query token t (在 sub-block s) 的 key 集 = [本 sub-block s 的 earlier token] +
    #        [sub-block 0..s-1 各自的 summary key] (每 earlier sub-block 1 个 summary, 非 64 token)
    #    grouped-softmax over 这个 key 集 → bottleneck (earlier 个体 token 零直连 mass)
    attn_out = inwindow_bottleneck_attention(q, k, v, summary_keys, summary_vals, sub_id, gs)
else:
    # 原 native causal attention (byte-identical)
```

**关键**: earlier sub-block 的个体 token 对 later query 的 attention mass = **0**(只经 summary)。这是 landmark-repro `_inwindow_bottleneck_ref.py` smoke 验的"earlier-sub 零 mass" assert。

## 3. summary key/value 构造 (landmark-repro 接 store, 这里定义接口)

- `summary_proj`: 新可训练 `nn.Linear(hd, hd)`(或 d→d), 挂在 self_attn (随 L16-31 unfreeze 训)。
- summary_key_j = self_attn.k_proj 之外的独立投影? **建议**: summary_key = summary_proj(pool_j), summary_val = pool_j (或 v_proj(pool_j))。pool_j = mean(target_hidden[sub-block j 的 token])。
- ⚠️ summary 必须 **causal**: sub-block j 的 summary 只能被 sub-block >j 的 token attend(j 在它们之前)。同 sub-block 内不经 summary(正常 causal)。

## 4. 推理期 (跨块选择, 梯度源2 训好后用)

- eval 跨块: 对每个 context chunk 算 summary_key = summary_proj(pool(chunk hidden)) → query·summary_key 选 top-k chunk (替 reader native q·k / gist)。
- 这条复用现有 `_reader_attn_keep_set`(layer.py:1135)逻辑, score 换成 query·summary_key。

## 5. 回归 / smoke (必做, landmark-repro review 闸)

1. **默认 off byte-identical**: `_rawkv_inwindow_summary=False` → 原 native attention, 数值 bit-identical。
2. **earlier-sub 零 mass**: later sub-block query 对 earlier sub-block 个体 token 的 attention weight == 0 (assert, landmark-repro standalone smoke 已 PASS, 搬进集成)。
3. **summary_proj grad 非零**: 训练一步后 summary_proj.weight.grad norm > 0 (梯度源2 通)。
4. **★短程不崩 ablation (landmark-repro 提的)**: bottleneck 后 target 自己的 LM loss 略升可忍(later 不能直连 earlier),但不能崩。跑 base 模型 in-window on vs off 的 target NTP loss, 确认 on 不显著高于 off(sub_size=64 和 Landmark mem_freq64 同量级, 预期可忍)。崩了 → sub_size 调大或 bottleneck 软化。

## 6. 节奏 / 配方

- B4 = dolmino in-window bottleneck (训 summary, 梯度源2) + T2 跨块 recall (用 summary 选择)。**dolmino-mix 回来**(自然文本 target 是 in-window 载体), 和纯T2 (dolmino_mix=0) 正交。
- gate: 纯T2 step100 W0。floor → B4 实装(梯度源2 必需);升 → B4 缓(消费侧已够, B4 锦上添花提选择)。
- launcher: landmark-repro 的 INWINDOW=1 env-gate + dolmino-mix; 或合进 self-study(自然文本 target 一致, 见我给 team-lead 的统一建议)。
