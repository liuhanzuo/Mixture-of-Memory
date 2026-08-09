# Paper D 可行性评估（R3：我们自己能不能做）

> 范围：只回答「**我们的模型资产 / 代码 / 算力能不能支撑 Paper D**」。文献 novelty 与 benchmark 设计由另外两个 agent 负责，本文不涉及。
> 所有数字均为盘上实测（2026-08-05/06，LOCAL 8×L20A，`/opt/conda/envs/torch-base/bin/python`，transformers 5.13.1 / torch 2.13.0）。
> 复现脚本：`paperD_research/smoke_stitch_cpu.py`，原始输出 JSON：`paperD_research/smoke_out/`。
> **未跑任何 GPU 训练**；只用 GPU 做 forward 提 hidden states（全部 smoke test 合计 < 25 分钟 GPU 时间）。

---

## 0. 一句话结论（先给最重要的）

**这条路"物理上能拼、语言上拼完就废"。**

跨家族拼接可以 forward、无 shape error、无 NaN；但即使给它一个**理论最优的 affine readout adapter（oracle ridge，在真实到达的激活上拟合）**，最好的一档 CE 仍然是 **6.39 vs 原模型 2.93**（ppl 596 vs 18.8）——**掉 3.5 nat**。而同一 harness 下**自拼自（A[0:k]+A 自己的尾巴）只掉 0.64 nat**（CE 3.57），证明**不是我们的测量/管线有问题，是跨家族的 residual stream 真的不兼容**。

关键的量化事实：跨家族 hidden state 之间 **z-scored 线性 CKA 在中层只有 0.35-0.47**（同模型相邻层是 0.90-0.99，随机初始化模型是 0.13）。也就是说跨家族表示**远好于随机、但远差于"同一条计算链上的相邻层"**——这正好落在"1-2 层桥不动"的危险区间。

**建议：不要把"1-2 层 stitching 冻结其余部分"当作 Paper D 主线。** 若仍要做，唯一有胜算的形态是「**同家族不同 size / 不同数据配方**」拼接，或「**stitch 层数 ≥ 4 且允许 B 的前若干层解冻**」——但那样就退化成已被大量占坑的 model-merging / depth-upscaling，novelty 归另一个 agent 判断。

---

## 1. 核实后的模型参数表

用户给的表**有 3 处需要修正**（下表 ⚠️ 标注）。数据来自逐个 `config.json` + `safetensors` 头部实测。

| model | type | L | hidden | head_dim | heads | kv_heads | ffn | vocab | rope_theta | max_pos | rms_eps | tie_emb | norm 位置 | QK-norm | attn bias | 权重 dtype | 磁盘 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OLMo-2-1124-7B | olmo2 | 32 | 4096 | 128 | 32 | 32 | 11008 | 100352 | 500000 | 4096 | 1e-6 | False | **POST** | **有**（per-head 全维 4096） | False | **fp32** | 27.19 GiB |
| Llama--Llama3-8b | llama | 32 | 4096 | 128 | 32 | 8 | 14336 | 128256 | 500000 | 8192 | 1e-5 | False | PRE | 无 | False | bf16 | 14.96 GiB |
| Llama--Llama2-7b | llama | 32 | 4096 | 128 | 32 | 32 | 11008 | 32000 | **无(默认1e4)** | 4096 | 1e-5 | False | PRE | 无 | False | fp16(+部分fp32) | 12.55 GiB |
| Qwen3-8B-Base | qwen3 | 36 | 4096 | 128 | 32 | 8 | 12288 | 151936 | 1000000 | 32768 | 1e-6 | False | PRE | **有**（per-head 128） | False | bf16 | 15.26 GiB |
| Qwen--Qwen3-8b | qwen3 | 36 | 4096 | 128 | 32 | 8 | 12288 | 151936 | 1000000 | **40960** | 1e-6 | False | PRE | 有(128) | False | bf16 | 15.26 GiB |
| Hunyuan-A13B-Pretrain | hunyuan | 32 | 4096 | 128 | 32 | 8 | **3072(MoE,64专家 topk8)** | 128167 | **10000** | 32768 | 1e-5 | **True** | PRE | 有 | False | bf16 | **149.76 GiB** |
| OLMo-2-0425-1B | olmo2 | 16 | 2048 | 128 | 16 | 16 | 8192 | 100352 | 500000 | 4096 | 1e-6 | False | **POST** | **有**(2048) | False | **fp32** | 5.53 GiB |
| Llama-3.2-1B | llama | 16 | 2048 | **⚠️64** | **⚠️32** | 8 | 8192 | 128256 | 500000 | 131072 | 1e-5 | **True** | PRE | 无 | False | bf16 | 2.30 GiB |
| Qwen--Qwen3-1.7b | qwen3 | 28 | 2048 | 128 | 16 | 8 | 6144 | 151936 | 1000000 | 40960 | 1e-6 | **True** | PRE | 有(128) | False | bf16 | 3.78 GiB |
| Qwen3-4B（附加） | qwen3 | 36 | **2560** | 128 | 32 | 8 | 9728 | 151936 | 1000000 | 40960 | 1e-6 | True | PRE | 有 | False | bf16 | 7.49 GiB |

### 对用户初始表的 3 处修正

1. **⚠️ Llama-3.2-1B 不是 `heads=32, head_dim=64` 之外的默认推断**：它 `num_attention_heads=32` 但显式 `head_dim=64`，所以 `32×64 = 2048 = hidden`（不是 `hidden/heads`＝64 的巧合，而是 config 显式声明）。而 OLMo-2-0425-1B 是 `16 heads × 128 = 2048`。**两者 head_dim 不同（64 vs 128）**——但这只影响层**内部**，不影响拼接（见 §1.2）。
2. **⚠️ `Qwen--Qwen3-1.7b` / `Llama-3.2-1B` / `Qwen3-4B` / `Hunyuan` 是 `tie_word_embeddings=True`**，OLMo-2 两个和 Llama-2/3-8B、Qwen3-8B 是 False。tie=True 意味着 `lm_head.weight` 不是独立张量（`_tied_weights_keys` 指向 embed），**做 transplant / state_dict 断言时张量计数会不同**，现有 `N_NONLAYER_KEYS = 3` 的硬编码会失效。
3. **⚠️ Llama-2-7b 的 `config.json` 里没有 `rope_theta`**（走 transformers 默认 1e4），且它的 safetensors 里还有历史遗留的 `self_attn.rotary_emb.inv_freq`（现代 transformers 会当 unexpected key）。**Llama-2 不适合当 pilot。**
4. 补充：**`models--openai-community--gpt2` / `models--Qwen--Qwen3-0.6B` / `models--Qwen--Qwen3-1.7B` 三个 HF-cache 目录是空壳**，只有 `refs/main`，**没有权重**（实测 `ls` 只有一个 40 字节的 ref 文件）。想用 gpt2 得先下载。

### 1.2 结构差异里哪些真的挡住拼接

我实际把层拼起来跑通之后，可以把"看起来像障碍"的因素分成三类：

| 差异 | 是否阻挡拼接 | 实测依据 |
|---|---|---|
| **hidden_size 不同** | **硬阻挡**（必须投影） | `StitchedLM.__init__` 里 `assert B.hidden == A.hidden`；4096 vs 2560 无法直连 |
| **head_dim / heads / kv_heads 不同** | **不阻挡** | OLMo-2-1B(16h×128, MHA) + Llama-3.2-1B(32h×64, GQA kv=8) 实拼 forward 成功。head_dim 只在层内 `q_proj → reshape → attn → o_proj` 闭环，出层就回到 hidden 维 |
| **ffn intermediate_size 不同** | 不阻挡 | 同上，层内闭环（8192 vs 8192 这里恰好同，但 11008 vs 14336 也一样不影响） |
| **PRE-norm(Llama/Qwen) vs POST-norm(OLMo-2)** | **不阻挡但严重伤性能** | 见 §3 的 residual RMS：OLMo-2 层 8 输出 RMS≈0.25，Llama-3.2 层 8 输出 RMS≈2.07，**差 8.2×**。POST-norm 的 OLMo-2 每层都被 norm 压回小尺度；PRE-norm 的 Llama 残差流自由增长 |
| **QK-norm 有/无** | 不阻挡 | QK-norm 也在 attn 内部（作用在 q/k 上），拼接看不到。⚠️ 但形状语义不同：OLMo-2 的 `q_norm` 是 **[hidden]=4096/2048**（对整个 flat q 做 norm），Qwen3 的是 **[head_dim]=128**（per-head）。**同 family 内 transplant 没事，跨 family 复制 q_norm 张量会 shape mismatch** |
| **rope_theta / rope_scaling 不同** | **不阻挡，但必须各家用各家的 rotary** | 实测必须给 A 段和 B 段**各自算 cos/sin**（`self.A_rotary` / `self.B_rotary`）。Llama-3.2-1B 还有 `rope_scaling.rope_type="llama3"`（factor 32），theta 表完全不能混用 |
| **tokenizer / vocab 不同** | **不阻挡**（丢弃 B 的 embed+head） | 按用户建议只用 A 的 tokenizer + A 的 embed/lm_head，只取 B 的中间层。这是唯一干净起点，已实现 |
| **dtype 不同（fp32 vs bf16）** | 不阻挡 | 统一 `.to(torch.float32)` 加载即可 |
| **MoE（Hunyuan）** | **硬阻挡（工程上）** | `trust_remote_code` 的自定义 `hunyuan.py`，`auto_map` 走本地文件；且 149.76 GiB。不适合 pilot |

**核心可拼条件只有一条：`hidden_size` 相等。** 其余差异都不产生 shape error，只产生**表示不兼容**（这才是真问题）。

### 1.3 配对可行性矩阵（按难度排序）

盘上 `hidden=4096` 有 5 个（OLMo-2-7B / Llama3-8B / Llama2-7B / Qwen3-8B-Base / Qwen3-8b / Hunyuan），`hidden=2048` 有 3 个（OLMo-2-1B / Llama-3.2-1B / Qwen3-1.7B）。

| 等级 | 配对 | hidden | 可否直连 | 障碍 |
|---|---|---|---|---|
| **A：直接可拼（无投影）** | Qwen3-8B-Base ↔ Qwen--Qwen3-8b | 4096 | ✅ | 几乎同一模型（base vs instruct），36L 全同构。**最容易但最没信息量** |
| **A** | OLMo-2-7B ↔ Llama3-8B | 4096 | ✅ | POST vs PRE norm；vocab 100352 vs 128256；32L=32L 对齐好 |
| **A** | OLMo-2-7B ↔ Qwen3-8B-Base | 4096 | ✅ | POST vs PRE；32L vs 36L 深度不等；q_norm 形状语义不同 |
| **A** | Llama3-8B ↔ Qwen3-8B-Base | 4096 | ✅ | 都 PRE-norm、都 GQA kv=8、都 head_dim=128 → **结构最接近的 7B 级跨家族对** |
| **A** | **OLMo-2-1B ↔ Llama-3.2-1B** | 2048 | ✅ | 16L=16L，最便宜。**← pilot 选它** |
| **A** | Llama-3.2-1B ↔ Qwen3-1.7B | 2048 | ✅ | 16L vs 28L；两边都 tie_emb=True |
| **A** | OLMo-2-1B ↔ Qwen3-1.7B | 2048 | ✅ | POST vs PRE + 16L vs 28L 双重不匹配 |
| **A'（同家族跨 size，作上界参照）** | OLMo-2-7B ↔ OLMo-2-1B | 4096 vs 2048 | ❌ 需投影 | hidden 不等，但**同 family 同 tokenizer**，是最干净的"表示应该最像"参照 |
| **B：需投影** | 任意 4096 ↔ 任意 2048 | — | 需 2048↔4096 线性投影 | 投影本身就是 stitch 的一部分，增加混淆 |
| **B** | 任意 ↔ Qwen3-4B (2560) | — | 需投影 | 同上 |
| **C：不建议/不可行** | 任何 ↔ Llama-2-7B | 4096 | 技术上可 | 无 rope_theta、有遗留 `inv_freq` key、vocab 只 32000、模型太旧 |
| **C** | 任何 ↔ Hunyuan-A13B | 4096 | 技术上可 | MoE + remote_code + 149.76 GiB + `tie=True`；工程成本高到不该做 pilot |

### 1.4 推荐 pilot 对：**OLMo-2-0425-1B (A) + Llama-3.2-1B (B)，k=8**

具体理由（不是"因为它小"）：

1. **完全同构的宏观形状**：都 16 层、都 hidden=2048、都 ffn=8192 → 层索引可 1:1 对齐，不需要处理"深度不等怎么映射"这个额外自由度。
2. **迭代成本极低**：两个模型加起来 7.8 GiB，单卡 fp32 全放得下；一次完整 smoke（含 ridge 拟合 + 4 个 arm 的 CE）**实测 87 秒**。
3. **它是"最难但最有代表性"的差异组合**：POST-norm(OLMo-2) vs PRE-norm(Llama)、MHA vs GQA、有 QK-norm vs 无、tie vs untie。如果 Paper D 想主张"跨家族通用"，这对不通基本就是判死刑；这对通了再上 7B 才有意义。
4. **有直接的同家族上界参照**：`OLMo-2-1B ↔ OLMo-2-7B` 同 tokenizer 同 family，可以当"表示相似度天花板"（已实测，见 §3.2）。
5. **和项目已有资产对齐**：OLMo-2 是 Paper B/C 的主力模型，`data/ood_ppl/*.npy` 已经是 OLMo-2 tokenizer 切好的，PPL eval 口径可直接复用。

⚠️ 但注意：Llama-3.2-1B 的 `tie_word_embeddings=True`，如果反向拼（Llama 当 A、提供 embed/head）要处理 tied weight。**建议固定 OLMo-2-1B 当 A（提供 tokenizer + embed + lm_head，tie=False 更干净）。**

---

## 2. 现有代码能改多少就支持

### 2.1 现状盘点（实测）

- `scripts/train_olmo2_arch_probe2.py`（1006 行）：**完全锁死在单一 family 内部**。
  - `build_olmo2_minimal()`（L359-394）：写死 `Olmo2Config.from_pretrained` + `Olmo2ForCausalLM(cfg)`，只有一个 `base_path`。
  - `transplant_front()`（L170-258）：写死 `Olmo2ForCausalLM.from_pretrained`；**4 条 sanity assert 全部基于"单一 base 的 state_dict 键集"**，其中 assert 3 是 `len(keep_keys) == N_NONLAYER_KEYS + N_TENSORS_PER_LAYER * keep_front`，用了模块级常量 `N_TENSORS_PER_LAYER = 11`（L104，OLMo-2 每层 11 个张量）和 `N_NONLAYER_KEYS = 3`（L105）。**Llama 每层只有 9 个张量**（无 q_norm/k_norm），Qwen3 是 11，Llama-3.2-1B 因 tie 只有 2 个非层张量 → **这两个常量必须变成"per-family 查表"**。
  - `_copied_keys()`（L122-137）：按 `lid < keep_front_layers` 过滤，**只会"保留前段丢弃后段"，没有"从第二个模型取 layers[k:] 并重编号"的概念**。跨模型拼接需要把 B 的 `model.layers.{k+i}` 重映射到新壳的 `model.layers.{k+1+i}`（因为中间插了 stitch 层）——**现有代码完全没有 layer-index 重映射逻辑**。
  - `_assert_fresh_init()`（L140-167）：断言 `post_attention_layernorm` / `q_norm` 全 1 —— 这两个键名是 OLMo-2 专有（Llama 是 `input_layernorm`、且无 `q_norm`），**跨 family 直接 KeyError**。
  - `_classify_param()`（L420-450）：只有 `inherited`/`fresh` 二分，对应"前段 vs 尾部"。Paper D 需要 **三分：`A_inherited` / `stitch_fresh` / `B_inherited`**（因为要冻结 A 和 B、只训 stitch）。
  - `apply_freeze_front()`（L397-417）：只按 `lid < keep_front` 冻结。Paper D 要冻结的是 `lid != stitch_index`（除 stitch 层外全冻），语义相反。
- `scripts/eval_olmo2_probe2_ppl.py::build_pruned_shell()`（L56-68）：也是写死 `Olmo2Config`/`Olmo2ForCausalLM`，只做 `cfg.num_hidden_layers = keep+fresh`。**它靠"trained state_dict 是完整的、strict-load"绕过了 arch 细节** —— 这一点对 Paper D 是好消息（见 §2.3）。
- **跨模型加载的现成代码：没有。** 全仓 `scripts/*.py` 325 个文件、185 个用到 `load_state_dict`/`from_pretrained`，但实测**没有任何一个把两个不同 family 的 layer 混装进一个模型**。同一文件里出现两个 family 类名的 14 个脚本（如 `eval_qwen3_probe2_ppl.py` 同时 import `Olmo2ForCausalLM` 和 `Qwen3ForCausalLM`）都只是**逐 family 分支 dispatch**，不是混装。
  - 最接近的是 `scripts/train_qwen3_arch_probe2.py`（把 OLMo-2 版逐字 port 到 Qwen3）——说明这个项目的做法是**复制一份改 family**，而不是抽象出 family-agnostic 层。
  - 唯一可直接复用的是 `scripts/probe_paperC_adaptation_onset.py::linear_cka()`（L173-188）—— 我的 smoke test 就用了同一个 Kornblith 闭式（并加了 fp64 校验）。

### 2.2 关键判断：改造成本

**结论：不要改 `train_olmo2_arch_probe2.py`，写新文件。**

理由：那 1006 行里，与 Paper D 冲突的是**架构假设本身**（单 base、单 family、层数常量、前/后二分冻结），而不是几个可参数化的点。硬改会同时破坏 Paper B/C 已跑完 200k step 的 arm 的可复现性（`arch_meta.json` 里记着 `model_family: "olmo2"` 和 sanity 计数），风险远大于收益。

**新写 MVP 的工作量估计（诚实版）：**

| 组件 | 内容 | 行数 | 时间 |
|---|---|---|---|
| `build_stitched_model()` | 新壳 = A 的 config（层数改 `k_A + n_stitch + (L_B - k_B)`），逐层用 `type(B.layers[0])(B_cfg, idx)` 构造 B 段（因为 B 段必须用 B 家族的层类 + B 的 config），A 段用 A 的层类。**注意 HF 的 `XForCausalLM(cfg)` 只会造同一种层**，所以必须手工 `nn.ModuleList` 混装 + 覆盖 `forward` | ~150 | — |
| 双 rotary 管理 | A 段用 A 的 `rotary_emb`，B 段用 B 的（实测**必须**，theta/scaling 不同）。要处理 `position_embeddings` 在两段间切换 | ~30 | — |
| `transplant_two()` + sanity | per-family 的 `n_tensors_per_layer` 查表（olmo2=11 / llama=9 / qwen3=11）、tie_emb 处理、layer-index 重映射（B 的 `layers.j` → 新壳 `layers.k+n_stitch+(j-k)`）、逐张量 max-diff==0 断言 | ~180 | — |
| `_classify_param` 三分 + 冻结 | `A_inherited` / `stitch` / `B_inherited`，`--freeze_inherited` 冻结前两类之外的全部 | ~50 | — |
| gradient checkpointing / DDP | 混装 ModuleList 不是 `PreTrainedModel`，`gradient_checkpointing_enable()` 不能直接用；DDP 下 `find_unused_parameters` 要注意冻结参数 | ~40 | — |
| eval 侧 | **几乎零成本**：`build_pruned_shell` 的思路（造壳 + strict-load 完整 state_dict）可直接搬，只要 ckpt 里记了 `(A_path, B_path, k_A, k_B, n_stitch)` 就能重建壳 | ~60 | — |
| checkpoint / arch_meta | 沿用 `_save()` 格式，meta 加 A/B 路径与切点 | ~30 | — |
| **合计** | | **~540 行新代码** | **1.5-2.5 人天**（含调试双 rotary、混装 ModuleList 的 gradient checkpointing 这两个已知坑） |

**MVP 应该长什么样**（最小可行、不要一次做全）：

1. 固定 A=OLMo-2-1B、B=Llama-3.2-1B、`hidden` 必须相等（`assert`，不做投影）。
2. 一个 `StitchedLM(nn.Module)`：`A.embed → A.layers[0:k] → stitch(1-2 层，B family) → B.layers[k:] → A.norm → A.lm_head`。**我的 smoke test 里这个类已经写好并跑通了**（`paperD_research/smoke_stitch_cpu.py::StitchedLM`，~70 行），可直接当 MVP 骨架。
3. 只训 stitch + `A.norm` + `lm_head`（+ 可选 B 的第一层），其余 `requires_grad_(False)`。
4. 数据/PPL eval 直接用 `data/dolmino_now_val.npy` + `data/ood_ppl/*.npy`（已是 OLMo-2 tokenizer）。
5. **先只跑 1B pilot，看 PPL 能不能压回接近 A_full；压不回就停，不要上 7B。**

### 2.3 一个对我们有利的发现

`eval_olmo2_probe2_ppl.py` 的设计（造壳 + strict-load 完整 state_dict）**天然支持 Paper D**：只要训练时把拼好的模型整体存成一个 flat state_dict，eval 侧不需要知道"哪些层来自 A、哪些来自 B"，只要能重建同形状的壳。所以 **eval 基础设施基本不用重写**，这是省下的最大一块。

---

## 3. Smoke test 实测结果（本节是核心交付）

脚本：`paperD_research/smoke_stitch_cpu.py`（三个子命令 `--test splice / repr / oracle`）
语料：`data/ood_ppl/wikitext103_test.npy` 解码回文本，切成 60-word 片段（跨 tokenizer 唯一公平做法）
对齐：不同 tokenizer 用 **fast-tokenizer offset mapping 做 whitespace-word 级 mean-pool**，得到行对齐的 `[N_words, D]`；脚本内 `assert keys_a == keys_b` 保证对齐没错位

### 3.0 先说两个方法论修正（否则数字会骗人）

**(i) 管线自检（必须先过）。** 我加了 `SELF_SPLICE`：把 B 换成 A 自己、stitch=Identity，走完全同一条拼接代码路径。实测 **CE = 2.933477266588427，与 `A_full` 的 2.933477266588427 逐位相同（delta = +0.00e+00）**。说明双 rotary、mask=None、层调用签名等管线细节都对，后面的坏数字不是 bug。

**(ii) 原始 CKA / R² 会被 massive activation 污染，必须 z-score。** 实测残差流里**前 8 个维度占了目标总方差的 98.7%-99.9%**（`target_var_share_top8`）。所以原始（未标准化）指标几乎只在描述那几个巨大维度：

| 对 | 原始 R²（骗人） | z-scored R²（真实） | 原始 CKA | z-scored CKA |
|---|---|---|---|---|
| OLMo-2-1B l8 → Llama-3.2-1B l8 | **0.9780** | **0.7028** | 0.4246 | 0.4155 |
| OLMo-2-1B l8 → Qwen3-1.7B l14 | **0.9942** | **0.5124** | 0.4269 | 0.4438 |
| Llama-3.2-1B l8 → Qwen3-1.7B l8 | **0.9619** | 0.4972 | **0.9693** | 0.9191 |

一个极端例子：`llama32_1b : qwen3_1p7b` 的**原始 CKA 矩阵在整个中层区几乎恒等于 0.9693**（a=2..12 对 b=4..21 全是 0.969），这是典型的"两个模型都有一个巨大维度"造成的假高相似。**下文所有引用的数字都是 z-scored 版本。**

### 3.1 (a) 能不能物理拼起来并前向不崩 → **能，但语言能力归零**

`splice_olmo2_1b_llama32_1b_k8.json`（A=OLMo-2-1B，B=Llama-3.2-1B，k=8，50 段文本）

| 变体 | forward | shape 正确 | 全部 finite | CE | ppl |
|---|---|---|---|---|---|
| `A_full`（参照） | ✅ | ✅ | ✅ | **2.9335** | **18.79** |
| `B_full`（B 自己的 tokenizer，不可直接比） | ✅ | ✅ | ✅ | 2.9969 | 20.02 |
| `SELF_SPLICE A+A`（管线自检） | ✅ | ✅ | ✅ | **2.9335** | 18.79（delta=0，管线 OK） |
| `A_earlyexit_k8`（A 前 8 层直接接 A 的 readout） | ✅ | ✅ | ✅ | 14.04 | 1.25e6 |
| stitch=none（直接对接） | ✅ | ✅ | ✅ | 16.21 | 1.10e7 |
| stitch=scale（RMS 尺度匹配，×8.22） | ✅ | ✅ | ✅ | 13.84 | 1.03e6 |
| stitch=linear_rand（随机线性） | ✅ | ✅ | ✅ | 14.71 | 2.45e6 |
| stitch=xfmr（**未训练**的新 Llama 层） | ✅ | ✅ | ✅ | 16.33 | 1.24e7 |
| stitch=ridge（word-aligned 拟合的线性映射） | ✅ | ✅ | ✅ | 18.99 | 1.77e8 |
| stitch=ridge + unstitch=ridge | ✅ | ✅ | ✅ | 14.41 | 1.81e6 |

**读法**：
- **`forward` 100% 成功，无一例 shape error，无一例 NaN/Inf**（`logits_all_finite=True` 全绿）。物理拼装完全没问题。
- 但**所有变体 CE 都在 13.8-19.0**，而随机猜 100352 词表是 `ln(100352)=11.5`。**拼完的模型比随机猜还差** —— 因为它不是"不知道"，而是被喂了完全不在分布里的激活，输出高置信度的错答案。
- 注意 `A_earlyexit_k8` = 14.04 也很差：说明**"A 自己的 lm_head 读不懂 A 自己第 8 层"** 本来就是个大问题（logit lens 的已知事实）。所以上表的绝对值混杂了两个因素。**这就是为什么必须做 §3.3 的 oracle 测试来解耦。**
- **残差流尺度实测（这是最有解释力的单一数字）**：

```
OLMo-2-1B  各层输出 RMS: [0.15, 0.16, 0.18, 0.19, 0.20, 0.22, 0.23, 0.24, 0.25, 0.27, 0.28, 0.31, 0.37, 0.46, 0.59, 0.76, 2.37]
Llama-3.2-1B 各层输出 RMS: [0.02, 0.14, 2.06, 2.06, 2.06, 2.06, 2.06, 2.07, 2.07, 2.07, 2.08, 2.08, 2.08, 2.09, 2.10, 2.09, 2.30]
```
POST-norm 的 OLMo-2 残差流一路被压在 0.15→0.76；PRE-norm 的 Llama 从第 2 层就冲到 2.06 然后**几乎不变**。在 k=8 处两者**差 8.22×**。这不只是"缩放一下就好"（`stitch=scale` 只把 CE 从 16.21 降到 13.84，仍然废）——方向结构也不匹配。

### 3.2 (b) 层间 CKA 矩阵（z-scored）→ 跨家族中层只有 0.35-0.47

`repr_1b_triple.json`（4000 个对齐 word）、`repr_7b8b.json`（3000 个）。fp32-GPU 与 fp64-CPU 交叉校验 **abs diff ≤ 5.7e-07**，数值可信。

**四条基准线（这是解读一切的尺子）：**

| 参照 | z-scored CKA | 含义 |
|---|---|---|
| **同模型相邻层**（OLMo-2-1B l8↔l9 / l12↔l13） | **0.953 / 0.903** | "一层能桥接的距离"的经验值 |
| 同模型相邻层（OLMo-2-7B l16↔l17 / l24↔l25） | **0.991 / 0.976** | 同上，7B 级 |
| **随机初始化模型**（OLMo-2-1B vs 随机 OLMo-2-1B，中层带均值） | **0.126** | 绝对下界 |
| 同家族跨 size（OLMo-2-7B vs OLMo-2-1B，中层带均值） | **0.346** | ← 注意，**这个上界参照本身就很低** |

**跨家族实测：**

| 对 | 中层带 z-CKA 均值（25%-75% 深度） | 相对深度对角线（z-CKA）最低点 | 最佳层对 |
|---|---|---|---|
| **OLMo-2-1B ↔ Llama-3.2-1B** | **0.467** | 0.198（l6↔l6） | l15↔l16 = 0.776 |
| OLMo-2-1B ↔ Qwen3-1.7B | 0.517 | 0.171（l1↔l2） | l15↔l27 = 0.825 |
| Llama-3.2-1B ↔ Qwen3-1.7B | 0.606 | 0.335（l12↔l21） | l2↔l3 = 0.966 |
| **OLMo-2-7B ↔ Llama3-8B** | **0.383** | 0.186（l12↔l12） | l31↔l29 = 0.886 |
| OLMo-2-7B ↔ OLMo-2-1B（同家族参照） | 0.346 | 0.207（l14↔l7） | l0↔l0 = 0.945 |

**关键观察（三条，都反直觉且重要）：**

1. **跨家族 CKA 不是"极低"，是"中等"** —— 0.35-0.61 远高于随机的 0.13。所以不能说"表示完全无关"。但**它也远低于同模型相邻层的 0.90-0.99**。用户在任务里问"如果 <0.3 就很可疑"——实测**没有低到 0.3 以下，但也没有高到接近 0.9**。它落在**最尴尬的中间带**。
2. **CKA 的深度曲线是 U 型：两端高、中间低。** 所有配对都是 l0 附近高（0.63-0.89，因为都在编码 token identity）、中层塌到 0.17-0.27、末层又回升到 0.78-0.89（都在准备预测下一个 token）。**这意味着"中层拼接"（最有价值的位置，因为那里才有抽象语义）恰好是 CKA 最低的位置。** k=4/8/12 的 oracle 实测（§3.3）确实印证了：k=12（靠后、CKA 回升到 0.65）比 k=4（CKA 0.44）表现好。
3. **同家族跨 size（OLMo-2 7B vs 1B）的中层 CKA 只有 0.346，比跨家族的 OLMo-2-1B↔Llama-3.2-1B（0.467）还低！** 这打破了"同家族一定更像"的直觉——**深度不同（32L vs 16L）比家族不同更能破坏层对齐**。对 Paper D 的启示：如果要找"最像"的配对，应该优先**同深度**而不是同家族。

### 3.3 (c)+(d) 线性映射能不能对齐 → R² 看着不错，但**装回去仍然废**

这是本次评估最重要的一节。分两层：

**第一层：word-aligned ridge（离线拟合，A 的第 k 层 → B 的第 k 层）**

| 对 / 切点 | z-scored 测试 R² | per-dim R² 中位数 | 原始 R²（污染版） |
|---|---|---|---|
| OLMo-2-1B l4 → Llama-3.2-1B l4 | 0.702 | 0.809 | 0.980 |
| OLMo-2-1B l8 → Llama-3.2-1B l8 | **0.703** | 0.800 | 0.978 |
| OLMo-2-1B l12 → Llama-3.2-1B l12 | 0.677 | 0.780 | 0.975 |
| OLMo-2-1B l8 → Qwen3-1.7B l14 | 0.512 | 0.658 | 0.994 |
| **对照：OLMo-2-1B l8 → 自己的 l9** | **0.790** | 0.909 | 0.850 |
| **对照：OLMo-2-1B l8 → 随机模型 l8** | **0.170** | 0.464 | 0.184 |

**读法**：跨家族线性 R²≈0.70，只比"同模型下一层" 0.79 低一点，远高于随机 0.17。**单看这个数字会得出"很有希望"的错误结论。**

**第二层：oracle 测试（本次评估的决定性实验）**

问题在于 R²=0.70 是在**离线、word-pooled、独立拟合**的条件下测的，不代表"装进去能用"。所以我做了更强的测试：**给拼好的模型一个理论最优的 affine readout（在真实到达 readout 的 token 级激活上用 ridge 拟合，train/test split）**。逻辑是：一层 transformer 至少能做一个 affine 映射能做的事，所以**这是"1 层 stitch"能力的下界**。如果连 oracle affine 都救不回来，1 层线性 stitch 就是证明不行。

`oracle_*.json`，A=OLMo-2-1B（`A_full` CE=2.9335），50 段 eval 文本：

| 配置 | k | stitch | oracle unstitch 拟合 R² | **CE** | ppl | ΔCE vs A_full |
|---|---|---|---|---|---|---|
| **上界：自拼自 + oracle readout** | 8 | none (B=A) | 0.959 | **3.5697** | 35.5 | **+0.636** |
| OLMo-2-1B + Llama-3.2-1B | 4 | scale | 0.466 | 7.9554 | 2851 | +5.022 |
| OLMo-2-1B + Llama-3.2-1B | 4 | ridge | 0.509 | 8.3245 | 4124 | +5.391 |
| OLMo-2-1B + Llama-3.2-1B | 8 | none | 0.465 | 8.6052 | 5460 | +5.672 |
| OLMo-2-1B + Llama-3.2-1B | 8 | scale | 0.551 | 7.4680 | 1751 | +4.535 |
| OLMo-2-1B + Llama-3.2-1B | 8 | ridge | 0.592 | 7.3444 | 1548 | +4.411 |
| OLMo-2-1B + Llama-3.2-1B | 8 | xfmr（未训练） | 0.491 | 7.8749 | 2630 | +4.941 |
| **OLMo-2-1B + Llama-3.2-1B（最好一档）** | **12** | **scale** | 0.660 | **6.3904** | **596** | **+3.457** |
| OLMo-2-1B + Llama-3.2-1B | 12 | ridge | 0.653 | 6.7973 | 895 | +3.864 |
| **下界：A 前 k 层 + 随机 A 尾巴 + oracle readout** | 8 | none | 0.354 | **10.3036** | 2.98e4 | +7.370 |
| 下界（k=12） | 12 | none | 0.482 | 8.6012 | 5438 | +5.668 |
| 另一对：Llama-3.2-1B + Qwen3-1.7B（最高 CKA 那对） | 8 | ridge | 0.454 | 6.5119 | 673 | +3.515 |
| 同上 | 8 | none | -0.050 | 11.0495 | 6.29e4 | +8.053 |
| 同上（上界：自拼自） | 8 | none (B=A) | 0.973 | **3.2133** | 24.9 | +0.216 |

**这张表说了三件事：**

1. **拼接确实传递了信息**（不是完全无用）：k=12 + scale 的 CE 6.39 明显好于"随机尾巴"下界 8.60，也好于 stitch=none 的 7.26。**所以 B 的层在做有用的计算。**
2. **但离可用差得极远**：最好一档 CE=6.39 vs `A_full` 2.93 = **+3.46 nat**（ppl 596 vs 18.8，差 **32 倍**）。而同一 oracle harness 下**自拼自只掉 0.64 nat**（3.57 vs 2.93）。**所以那 3.46 nat 里，只有 0.64 是"harness 的代价"，剩下 ~2.8 nat 是纯粹的跨家族不兼容。**
3. **oracle affine readout 的拟合 R² 本身就只有 0.47-0.66**（自拼自是 0.96）。也就是说，**从"拼接后的流"到"A 的正常终层流"，连最优线性映射都只能解释 47-66% 的方差**。这与 §3.3 第一层的 R²=0.70（离线、word-pooled、只跨一个切点）形成对比——**误差在 B 的 8 层里被逐层放大了**。

**k 的趋势很清楚：k 越深越好**（k=4 → CE 7.96 / k=8 → 7.47 / k=12 → 6.39）。这和 §3.2 的 U 型 CKA 曲线一致（深层 CKA 回升）。但"k 越深"意味着**从 B 拿到的层越少**，也就越接近"根本没用 B"——k=12 时只用了 Llama 的最后 4 层。**这是个致命的张力：能拼得动的地方，恰好是拼了也拿不到 B 多少能力的地方。**

---

## 4. GPU·h 估算

### 4.1 基准：从盘上真实日志读吞吐

`logs/olmo2_7B_keep14fresh2.log`（8×L20A，16L/4.06B 参数、全参可训、fp32 master + bf16 autocast、seq_len 2048、bs=16、eff_bs=128）：**1.56 s/step**，200k step 跑了约 4.3 天。
`logs/olmo2_7B_keep14fresh2_freezefront.log`（同上但冻结前 14 层，trainable 1.23B）：**1.32 s/step**，maxmem 87.8 GB（vs 全参 122.3 GB）。
`logs/olmo2_7B_full32_dolmino.log`（32L/7.30B 全参，bs=4×gaccum4）：**3.16 s/step**，maxmem 176.9 GB。

**重要经验值**：冻结 70% 参数只把 step time 从 1.56 降到 1.32（**-15%**）。因为**前向仍要全做、反向仍要传梯度穿过冻结层**，只省了 optimizer step 和部分 grad 存储。**Paper D 的"只训 1-2 层"不会带来数量级加速**，最多 ~25-30%。

### 4.2 Paper D 各阶段估算

单位：GPU·h（1 节点 8 卡跑 1 小时 = 8 GPU·h）

| 阶段 | 配置 | step time 估计 | steps | 墙钟 | GPU·h |
|---|---|---|---|---|---|
| **P0. 1B pilot（先做这个）** | OLMo-2-1B + Llama-3.2-1B，~2.3B 总参数、只训 1-2 层，8 卡 seq2048 | ~0.5 s/step | 5k | ~0.7 h | **~6** |
| P0 的 4 个 arm（k=4/8/12 × stitch=1/2 层） | 同上 ×6 组合 | | 5k each | 4.2 h | **~34** |
| **P1. 7B×7B 主实验（单 arm）** | OLMo-2-7B(前 k 层) + Llama3-8B(后段) ≈ 7.3B 级壳、只训 stitch。参照 full32 的 3.16 s/step，冻结省 ~20% → **~2.5 s/step**，bs 4×gaccum4 | 2.5 s/step | 20k | 13.9 h | **111** |
| P1 同上但 50k step（若 20k 不收敛） | | 2.5 s/step | 50k | 34.7 h | **278** |
| **P1 完整消融**（3 个切点 k × 2 个 stitch 深度 × 2 个方向 A↔B = 12 arm，各 20k） | | | | | **~1330** |
| P2. 加一个第三家族对（Qwen3-8B）验证通用性，4 arm | | | 20k | | **~444** |
| eval（PPL + downstream，沿用现有 harness，8 卡分片） | | | | ~2 h/arm | ~16/arm |

### 4.3 对照我们的实际算力

可调度：**LOCAL 8×L20A + .252 8×B200（wzc1 盘）+ .73 + .82 各 8×H20（zwfy6 盘）= 32 卡**。
（.104 已交还；wzc1 与 zwfy6 **不可跨盘合 DDP**，跨盘搬运需 `scp -O`）

- **P0（1B pilot，~34 GPU·h）**：**半天内在单节点跑完**，甚至一张卡就够。**零风险，应该先做。**
- **P1 单 arm（111 GPU·h）**：单节点 ~14 小时。**可接受。**
- **P1 完整消融（~1330 GPU·h）**：4 节点全投 ≈ **1.7 天纯算力**，但实际要排队/共享，现实是 **4-7 天**。
- **P1+P2（~1774 GPU·h）**：**~1 周 4 节点独占**。

**⚠️ 跨盘资产（已实测，2026-08-06 于 .73 上 `ls`）**：

| 模型 | wzc1 (LOCAL/.252) | zwfy6 (.73/.82) |
|---|---|---|
| OLMo-2-1124-7B | ✅ 27.19 GiB | ✅ 28G |
| Llama--Llama3-8b | ✅ 14.96 GiB | ✅ 15G |
| OLMo-2-0425-1B | ✅ 5.53 GiB | ✅ 5.6G |
| Llama-3.2-1B | ✅ 2.30 GiB | ✅ 2.4G |
| Qwen--Qwen3-1.7b | ✅ 3.78 GiB | ✅ 3.8G |
| **Qwen3-8B-Base** | ✅ 15.26 GiB | **❌ 0 个 safetensors（缺）** |

→ **pilot 对（OLMo-2-1B + Llama-3.2-1B）和主推的 7B 对（OLMo-2-7B + Llama3-8B）两盘都有，4 个节点都能跑，无需 `scp -O`。** 只有 Qwen3-8B-Base 若要在 H20 上用需先跨盘搬 15 GiB。

**算力不是瓶颈。瓶颈是 §3 显示的这个方向可能根本不 work。**

---

## 5. 结论：值不值得做，最大风险是什么

### 5.1 我的判断：**按用户描述的原始形态（1-2 层 stitch + 冻结继承部分），不值得做。**

证据链（全部实测）：

1. **物理可拼 ✅** —— forward 成功率 100%，无 shape error、无 NaN。这部分完全没有障碍，代码约 540 行、1.5-2.5 人天。
2. **表示不兼容 ❌** —— 跨家族中层 z-scored CKA 只有 **0.35-0.47**，而"一层能桥接的距离"的经验值（同模型相邻层）是 **0.90-0.99**。差距不是一点点。
3. **最优线性映射也救不回来 ❌** —— 给一个 oracle affine readout（1 层 stitch 能力的**下界**），最好一档仍然 **CE 6.39 vs 原模型 2.93（ppl 596 vs 18.8，差 32×）**。同 harness 下自拼自只掉 0.64 nat，所以坏结果不是测量误差。
4. **最致命的结构性矛盾**：CKA 的深度曲线是 U 型（两端高、中间低）。**能拼得动的位置（深层，k=12，CKA 0.65）恰好是从 B 那里拿不到多少能力的位置（只用到 Llama 最后 4 层）；真正有价值的中层（抽象语义所在）恰好是 CKA 最低的位置（0.19-0.27）。** 这不是调参能绕过的，是表示几何决定的。

### 5.2 最大的技术风险（按严重性排序）

1. **【最大】"1-2 层 + 冻结"的容量假设是错的。** 实测显示需要跨越的不是"一个线性变换的距离"，而是"整个 residual basis + 尺度 + 逐层放大误差"。oracle affine 只能解释 47-66% 方差。**风险后果**：跑完 1330 GPU·h 的完整消融，得到"所有 arm 都比单模型 baseline 差"的结果——一篇负结果论文。而 §3.2 的第 3 条观察（同家族跨 size 的 CKA 0.346 比跨家族同深度的 0.467 更低）说明连"选对配对"都不能救。
2. **【次大】"同时拿到 A 和 B 的强项"这个目标可能自相矛盾。** 我们用 A 的 tokenizer + A 的 embed + A 的 lm_head（这是唯一干净起点）。那么 **B 的"强项"如果部分寄存在 B 的 tokenizer / embedding / 输出头里（这对多语言、代码、数学能力尤其成立），我们从设计上就拿不到。** 剩下能拿的只有 B 中间层的"通用计算"，而那恰好是各家最像、最没有差异化的部分。**Paper D 想要的"A 的强项 + B 的强项"和技术上能拼的"两家的通用中层"，可能是不相交的两件事。**
3. **【中】评估口径风险。** 项目已有铁律：`chat_template=False`（memory: paper-eval-chat-false-mandatory）、OLMo-2 只能 base 口径（memory: paperb-olmo2-base-not-chat）。拼接模型的 tokenizer 来自 A、能力来自 A+B，**"和谁比"这个 baseline 定义会被 reviewer 攻击**：比 A 强多少？比 B 强多少？比 A+B 的 ensemble 呢？比同等 FLOPs 下继续训 A 呢（这个最难赢，因为我们 Paper B/C 已经证明 prune-then-heal 续训很有效）。
4. **【中】工程坑（已识别，可控）**：混装 `nn.ModuleList` 不是 `PreTrainedModel` → `gradient_checkpointing_enable()` 不能直接用；双 rotary 必须分段（实测必需）；`tie_word_embeddings` 在 Llama-3.2-1B/Qwen3-1.7B 是 True 会破坏 `N_NONLAYER_KEYS=3` 假设；per-family 每层张量数不同（olmo2/qwen3=11、llama=9）会破坏现有全部 sanity assert。这些都是**已知且我已在 smoke test 里踩过并绕过**的，不是主要风险。
5. **【小】跨盘资产风险**：**已实测排除** —— pilot 对与主推 7B 对在 wzc1/zwfy6 两盘都有（见 §4.3 表），只有 Qwen3-8B-Base 在 zwfy6 缺失。

### 5.3 如果还是要做，我建议的最小赌注

**先只花 ~34 GPU·h（半天、单节点）做一个 go/no-go 门槛实验**，而不是直接铺 1330 GPU·h：

- 配置：OLMo-2-1B(前 k) + Llama-3.2-1B(后段)，k ∈ {8, 12}，stitch ∈ {1 层, 2 层}，训 5k step（`data/dolmino_now15b.npy` 已就绪）。
- **明确的 kill 条件（事先写下来，别事后找理由）**：训 5k step 后，val PPL 若不能进到 **`A_full` 的 2 倍以内**（即 ppl ≲ 38，对应从实测 596 降下来 15 倍），就停。理由：oracle affine 已经给了 596；如果真训 1-2 层还进不到 38，那"1-2 层够用"的核心假设就被证伪了。
- 同时跑对照：**同等可训参数量下，直接继续训 A 自己**（这是 reviewer 一定会问的、也是最难赢的 baseline）。
- 只有 1B 门槛过了，才动 7B。

**另外两个更可能成立的变体**（若要转向，需先让文献 agent 确认没被占坑）：
- **同深度优先而非同家族优先**：§3.2 第 3 条观察是本次最有信息量的意外发现（同家族跨 size CKA 0.346 < 跨家族同深度 0.467）。"层对齐由深度而非家族决定"本身可能就是一个可发表的 measurement claim，而且**几乎零额外成本**（我的 `--test repr` 已经能出这个矩阵，扩展到 6-8 个模型只要几小时 GPU forward）。
- **放宽到 stitch ≥ 4 层 + 解冻 B 的前几层**：但这就滑向 depth-upscaling / model-merging 的既有领域。

---

## 附：复现命令

```bash
PY=/opt/conda/envs/torch-base/bin/python

# (a) 物理拼接 + 管线自检 + 各 stitch 变体的 CE
$PY paperD_research/smoke_stitch_cpu.py --test splice \
    --model_a olmo2_1b --model_b llama32_1b --k 8 \
    --n_ce_texts 50 --fit_ridge --n_fit_texts 120 --device cuda:0

# (b)(c) 层间 z-scored CKA 矩阵 + ridge R²（含随机初始化下界）
$PY paperD_research/smoke_stitch_cpu.py --test repr \
    --models olmo2_1b llama32_1b qwen3_1p7b --random_baseline \
    --pairs olmo2_1b:llama32_1b olmo2_1b:qwen3_1p7b llama32_1b:qwen3_1p7b \
            olmo2_1b:RANDOM_olmo2_1b \
    --ridge_layers 4 8 12 --n_texts 300 --max_words 4000 \
    --device cuda:0 --out repr_1b_triple.json

# 7B/8B 级 + 同家族跨 size 上界参照
$PY paperD_research/smoke_stitch_cpu.py --test repr \
    --models olmo2_7b olmo2_1b llama3_8b \
    --pairs olmo2_7b:olmo2_1b olmo2_7b:llama3_8b \
    --ridge_layers 8 16 24 --n_texts 300 --max_words 3000 \
    --device cuda:1 --out repr_7b8b.json

# (d) oracle affine readout（1 层 stitch 的能力下界）+ 上下界参照
for k in 4 8 12; do
  $PY paperD_research/smoke_stitch_cpu.py --test oracle \
      --model_a olmo2_1b --model_b llama32_1b --k $k \
      --n_ce_texts 50 --n_fit_texts 120 --max_tokens 8000 \
      --include_xfmr --device cuda:0
done
```

输出 JSON：`paperD_research/smoke_out/{splice,repr,oracle}_*.json`（含完整 CKA 矩阵、每层最佳配对、per-dim R² 诊断、residual RMS 曲线、fp32-vs-fp64 校验）。
