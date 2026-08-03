# Paper B P2.2 — 因果层恢复 / activation patching harness NOTES

**脚本**：`scripts/eval_olmo2_activation_patching.py`（analysis harness）+ `scripts/_run_olmo2_actpatch_8gpu.sh`（8-GPU launcher）
**类型**：**纯前向推理，绝对无训练**。所有模型要么是预训练 base、要么是 healed keep14 ckpt、要么是二者权重逐张 copy 拼装的 hybrid；没有任何一个参数被优化。
**口径**：BASE 协议——`add_special_tokens=False`（`--add_bos 0`，无 BOS）、`chat_template=False`、zero-shot、无检索、greedy、MMLU = likelihood-based MC、MMLU 全量 `cais/mmlu` test n=14042。PPL = held-out Dolmino `data/dolmino_now_val.npy`，teacher-forced NTP CE，token-weighted。
**指标**：**每个干预点同时报告 PPL 和 MMLU**（TODOList 硬要求，禁止只看一个）。逐 shard 原始 PPL（`sum_nll/n_tokens`）+ 逐题 MMLU（`per_example_mmlu.jsonl`）全部落盘。

---

## 背景与要回答的问题

keep14 = 继承 vanilla OLMo-2-7B（32L base）前 14 层 + 2 个 **fresh** tail block（共 16 层 shell），healed 到 200k。它的 MMLU 低。两种竞争解释：

- **(a) readout-limited**：前 14 层已有可用信息，但 fresh tail 读不出来；
- **(b) computation-deleted**：所需计算在被删的上层（base layer 14–31）里，已经没了。

本 harness 用两个非训练干预区分 (a) / (b)。

---

## 干预 1 — 边界 hidden-state grafting（`--mode graft`）

**注入点定义**：对同一个 token batch，
1. 跑 **base 32L** 前向，用 `base.model.layers[L]` 上的 forward hook 捕获**第 L 层输出**的残差流（OLMo-2 decoder layer `forward` 直接 `return hidden_states`，是一个普通 tensor，已核对 transformers 5.13 `modeling_olmo2.py` L334）。
2. 跑 **keep14** 前向，用 `keep14.model.layers[J]` 上的 forward **pre-hook** 把第 J 层的输入 hidden（positional arg 0）**替换**成捕获的 base hidden；`J` 默认 = `keep_front_layers` = 14 = fresh-tail 的输入位置。
3. keep14 的 tail（layers 14,15）+ final norm + lm_head 完成 readout。

**扫描** `L ∈ {13,16,20,24,28,31}`（可配 `GRAFT_LAYERS`）：
- `L = keep_front-1 = 13`：base 前 14 层的表示 → keep14 tail。近似 in-distribution 控制（keep14 healed 前段 ≈ base 前段）。
- `L > 13`：交给 tail 一个**已经包含被删上层计算**的表示。
  - 若 MMLU 随 L 增大而**显著回升** → 缺的是上层计算，且 keep14 的 tail 在**给到**好表示时**能**读出（支持 (b)，且 tail 本身不弱）。
  - 若**基本持平** → 信息在边界已经有了，瓶颈在 readout（支持 (a)）。

**位置/RoPE 对齐**：两模型 tokenization 完全一致；OLMo-2 的 `position_embeddings`（cos/sin，θ=5e5）由 `Olmo2Model.forward` 从 `position_ids` **一次性**算出（`modeling_olmo2.py` L410），与层数无关，逐层共享 → 注入的残差流在 keep14 layer J 的 attention 里应用同一套 RoPE，positions 完全对齐。dtype：注入前 pre-hook 把捕获 hidden `.to(cur.dtype)`，autocast 下一致。

**控制**：`--graft_layer -1` 关闭注入 = 纯 keep14 走 wrapper，应复现 plain keep14 数字（in-harness sanity）。

---

## 干预 2 — 逐块 upper-layer restoration（`--mode restore`）

**通过 copy 权重拼装 hybrid Olmo2ForCausalLM（无训练）**：

- `tail_keep14`（默认，"恢复接回 tail 之前"，对齐 TODOList 措辞）：
  ```
  layers[0..13]          <- keep14 healed 前 14 层
  layers[14..14+k-1]     <- base 原始 upper layers 14..14+k-1（restore）
  layers[14+k, 15+k]     <- keep14 的 2 个 FRESH tail 层（留在最顶）
  embed / norm / lm_head <- keep14
  ```
  `k=0` → 与 keep14 **逐字节等价**（内建 identity 检查）；`k=18` → keep14 前段 + 全部 18 层 restored upper + fresh tail（34L）。
- `base_head`（去混淆变体，`--restore_readout base_head`）：**丢掉 fresh tail**，用 base 原始 `norm+lm_head` readout：
  ```
  layers[0..13]          <- keep14 healed 前 14 层
  layers[14..14+k-1]     <- base upper 14..14+k-1
  embed <- keep14 ; norm+lm_head <- base
  ```
  `k=n_deleted=18` → keep14 healed 前段 + 完整 base upper stack + base 自己的 readout（隔离假设 (b)，无 fresh-tail readout 混淆）。

**扫描** `k ∈ {0,2,4,6,9,12,18}`（可配 `RESTORE_KS`）：PPL/MMLU 随恢复深度上升 ⇒ 被删计算是必要的（支持 (b)）；持平 ⇒ 冗余（支持 (a)）。

---

## 精确 8-GPU launch 命令

**运行节点 = .104（diskB H20），`PY=/opt/conda/envs/torch-base/bin/python`，`add_bos=0`。**
⚠️ keep14 ckpt + base 目前在 **LOCAL wzc1 盘**，.104 是 diskB（不同 ceph）→ **main 需先 rsync ckpt + base 到 diskB**（见下"依赖 ckpt 路径"），rsync 后相对路径 `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` 保持一致，launcher 默认即可直接用。

```bash
# 全套（graft 扫 + restore 扫，各自 PPL+MMLU 8-shard merge）
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PY=/opt/conda/envs/torch-base/bin/python \
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
KEEP14=outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt \
setsid nohup bash scripts/_run_olmo2_actpatch_8gpu.sh \
  > logs/actpatch_sched.out 2>&1 &

# 只跑 graft / 只跑 restore：DO_RESTORE=0 或 DO_GRAFT=0
# base_head 变体：RESTORE_READOUT=base_head DO_GRAFT=0 bash scripts/_run_olmo2_actpatch_8gpu.sh
```

**先跑 32-item sanity（main 在真机上先做）**：先单点、小样本确认能 model-load + forward：
```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/torch-base/bin/python scripts/eval_olmo2_activation_patching.py \
  --mode restore --task mmlu --restore_k 0 --restore_readout tail_keep14 \
  --base_model <BASE> --keep14_ckpt <KEEP14> \
  --num_shards 8 --shard_index 0 --limit 32 --batch_size 8 \
  --results_root olmo2_actpatch_mmlu_results --output_name sanity_restore_k0
# restore k=0 的 MMLU/PPL 必须 == plain keep14 的已发布数字（identity 检查）。
CUDA_VISIBLE_DEVICES=0 ... --mode graft --task ppl --graft_layer 13 --limit 8 --batch_size 2 ...
```

**手动单点**（绕过 launcher）：
```bash
CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_activation_patching.py \
  --mode graft --task {ppl|mmlu} --graft_layer L [--inject_at_layer 14] \
  --base_model <BASE> --keep14_ckpt <KEEP14> \
  --num_shards 8 --shard_index $g --batch_size {2|4} \
  --results_root olmo2_actpatch_{ppl|mmlu}_results --output_name graft_baseL${L}_injTail
# merge:
$PY scripts/eval_olmo2_activation_patching.py --merge --task {ppl|mmlu} \
  --results_root olmo2_actpatch_{ppl|mmlu}_results --output_name <NAME> [--n_boot 10000]
```

---

## 依赖 ckpt 绝对路径

| 用途 | LOCAL wzc1（当前所在） | diskB（.104/.73 运行节点，rsync 后） |
|---|---|---|
| keep14@200k healed ckpt | `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt`（48.7GB；`final.pt` 同内容亦可） | `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` |
| base OLMo-2-7B（32L）原始权重 | `/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B` | `/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B` |
| held-out PPL val | `data/dolmino_now_val.npy`（相对 PROJECT_ROOT，两盘都有） | 同 |

ckpt arch meta（已核对 `outputs/olmo2_probe2_7B_keep14fresh2/arch_meta.json`）：`keep_front_layers=14, n_fresh_layers=2, num_hidden_layers=16, hidden_size=4096, vocab=100352, tie_word_embeddings=false`。keep_front/n_fresh 由 ckpt meta 自动读取，无需手传（传了会做一致性断言）。

**显存**：graft 每卡同时载 base(32L≈7B fp32≈28GB) + keep14(16L≈4B fp32≈16GB) ≈ 44GB weights，H20 97.8GB 够（故 `GRAFT_BS_PPL=2/GRAFT_BS_MMLU=4` 保守）；restore 单模型（tail_keep14 k=18=34L≈8.6B≈34GB / base_head k=18=32L≈28GB），`RESTORE_BS_PPL=2/RESTORE_BS_MMLU=8`。OOM 就调小对应 BS 环境变量。

---

## 静态验证结果（本机 wzc1，torch-base py3.14 / torch 2.13 / transformers 5.13.1，无 GPU、未载 7B）

- `python -m ...--help`：argparse OK；AST parse OK。
- **CPU selftest**（`--selftest`，tiny OLMo-2 base8L + keep6L 随机权重）全过：
  - graft capture hook 形状 = `[2,9,64]`；**front-equal no-op identity**：当 base 前段权重 == keep 前段时，把 base layer(KF-1) 输出注入 keep layer KF 是数学恒等 → grafted logits 与 plain keep 完全一致（`allclose atol=1e-4`）——证明 hook 捕获/注入接线正确。
  - deep graft（base 顶层输出，base≠keep）改变 logits → inject 确实生效。
  - restore `tail_keep14 k=0` 与 keep sd **逐张 `torch.equal`**（identity）；`k=2` → 8L strict-load，逐层 provenance 正确（front←keep / restored←base / fresh tail←keep）；`base_head k=n_deleted` → norm+lm_head←base、embed←keep，forward OK。
- **PPL driver CPU dry-run**：synthetic npy 12 窗，2-shard(6+6) → `score_windows` 复用 → shard json → `merge_shards` token-weighted 合并（n_windows=12, n_tokens=180）✅。
- **MMLU driver CPU dry-run**：monkeypatch 4 条 synthetic MMLU 例，2-shard(2+2) → `score_examples` 复用 → `per_example_mmlu_shard*of*.jsonl` + shard json（文件名与 `eval_olmo2_mmlu_content.merge` 读取的完全一致）→ merge 重算 aggregate + 配对统计 ✅。（准确率因 tiny 随机模型无意义，仅验管线。）

---

## 架构对齐的坑 / 已知边界

1. **decoder layer 返回 plain tensor（非 tuple）**：OLMo-2 `Olmo2DecoderLayer.forward -> torch.Tensor`（L334）。capture hook 里做了 `output[0] if isinstance(output, tuple) else output` 兼容两种，但当前版本走 tensor 分支。若升级 transformers 使其返回 tuple，代码自动兼容。
2. **hidden_states 是 positional arg 0**：`Olmo2Model.forward` 里 `decoder_layer(hidden_states, attention_mask=..., position_embeddings=..., ...)`（L412-416）。inject pre-hook 用 `with_kwargs=True` 只替换 `args[0]`，其余 kwargs 原样透传 → 对 attention_mask/position_embeddings/cache 位置健壮。
3. **GraftedModel 的 keep 子模块被注册了 pre-hook**，故**不能**再单独直接调用它（会触发"inject before capture"）；`GraftedModel.forward` 保证先跑 base 捕获、再跑 keep。（selftest 里所有 "plain" 基线都用另建的 hook-free 模型。）
4. **restore 拼装依赖"层形状全同"**：keep14 前段、base upper、fresh tail 的每层张量 shape 完全一致（同 hidden/heads/GQA），故可自由错位拼装并 `strict=True` 加载。`layer_idx`（attention 内）只用于 KV-cache 索引；eval `use_cache=False` 时 restored base 层放到不同 target 索引不影响数值。
5. **干预 2 的两个 readout 变体各有其混淆边界，须在论文里讲清**：
   - `tail_keep14`：在 restored upper 与 readout 之间**插了 keep14 的 fresh tail**，fresh tail 原本训练来读 keep14 layer-13 输出，现在读的是 base layer-(13+k) 输出，存在分布偏移 → 若曲线平，不能完全排除"tail 无法消化 restored 表示"。
   - `base_head`：去掉 fresh tail、用 base 原始 readout，去掉了 readout 混淆，但 base upper 层此时读的是 **keep14 healed（漂移过）的前段输出**（非 base 自己的前段），也存在轻微 OOD（healing inherited-LR=2e-5 很低，前段漂移小）。两变体互补，建议都跑，交叉印证。
6. **graft L>13 的分布偏移**：base layer-L(L>13) 输出的残差流对 keep14 fresh tail 而言是 OOD（tail 只见过 keep14 layer-13 尺度的输入）；若 grafted MMLU 不回升，需谨慎——可能是"tail 无法读 OOD 表示"而非"信息本就没用"。`L=13` 控制点（近 in-distribution）用于校准这一偏移。这是本干预的固有解释边界，已在结论中标注。
7. **未做 cross-L base 复用优化**：graft 扫每个 L 各起一个 job，各自重跑一遍 base 前向（多算几遍 base）。offline 分析可接受；若要提速可改成一次 base 前向 capture 多个 L（未实现，留作 TODO）。
8. **注入点当前限于 decoder 层输入**（默认 layer 14 = fresh-tail 输入；`--inject_at_layer` 可设 15）。"注入到 final norm 之前、完全跳过 tail" 需在 `model.norm` 上挂 hook，当前未实现；如需该变体可用 `base_head` restore（k 任意）近似达到"绕过 fresh tail"的效果。
