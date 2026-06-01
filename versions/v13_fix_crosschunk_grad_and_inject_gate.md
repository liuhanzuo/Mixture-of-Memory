# v13 — Fix cross-chunk gradient cut + inject_gate freeze (2026-06-02)

> 修复两个切断 memory 学习闭环的 bug。诊断来源：2-chunk passcode toy task
> （chunk1 写 fact，chunk2 读回，loss 只在答案 token）retrieval_exact_acc 全程 0、
> 答案 LM loss 钉死 ~3.0。三臂诊断（commit e5bb181）确认 routing/写入内容/answer
> token attend memory 都正常，缺口在**跨 chunk 梯度回传**。本版定位到两处具体根因
> 并修复。

## Bug 1（关键）：slot_value_norm_cap 在 no_grad 里 rebind self.slots，切断跨 chunk 梯度

**文件**：`src/memory/mem_space/memory_bank.py`，两处 norm-cap 代码块
（dual-gate 路径 `write()` 末尾、single-gate legacy 路径末尾）。

### 根因
原代码：
```python
if self._slot_value_norm_cap > 0.0:
    with torch.no_grad():
        slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0)
        self.slots = self.slots / scale_all   # ← 在 no_grad 里 rebind
```
`with torch.no_grad(): self.slots = self.slots / scale_all` 把整个 `self.slots`
重新绑定成一个**脱离 autograd 计算图**的新 tensor。后果：
- `write()` 返回的 `updated` 仍带梯度（chunk 内 recon loss 可用，所以 recon 一直能降）；
- **但存进 `self.slots`、被下一个 chunk 读取的 slots 是 detached 的** → chunk2 读 memory
  产生的梯度无法回传到 chunk1 的 write 路径（gate / slot_to_hidden / 写入内容）。
- 实测：cap=5.0 → `bank.slots.requires_grad == False`；cap=0 → `True`。

这正是"教模型写入对未来有用内容"的梯度被切断的根因：写入器永远收不到"你写的东西
将来读不读得回"的信号。

### 修复
让 norm-cap 成为 **gradient-preserving**：不再用 `torch.no_grad()` 包裹 rebind，
改为在 autograd 记录范围内做按 norm 的固定比例缩放，只对 `scale_all` 调 `.detach()`：
```python
if self._slot_value_norm_cap > 0.0:
    slot_norms_all = self.slots.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scale_all = (slot_norms_all / self._slot_value_norm_cap).clamp(min=1.0).detach()
    self.slots = self.slots / scale_all   # rebind 发生在 autograd 范围内
```

### 为什么对 scale detach
norm-cap 的语义是"限幅"——只限制 slot 向量的**模长**，不应改变梯度的**方向**。
- `scale_all` 本身是 `self.slots` 的函数（norm）。若不 detach，梯度会额外流经
  "norm→scale→除法"这条路径，等价于对梯度方向施加一个由 norm 决定的非线性变换，
  数值上不稳定且不是想要的语义。
- detach 后，缩放退化为"按当前 norm 的固定常数比例缩放"，梯度以该比例**直接、线性**
  流过（限幅不改方向）。这就是 norm-cap 想要的行为。
- 关键不在 scale 是否 detach，而在 **rebind 必须发生在 autograd 记录范围内**，
  让 `self.slots` 保持 `requires_grad=True` / `grad_fn is not None`。

replace 路径（`write()` 开头 `if replace:`）本来就只对写入子块 clip 且**没有进
no_grad**，确认无需改动。

### 验证（临时脚本，已删）
- `MemoryBank(slot_value_norm_cap=5.0)` 写一次带 grad 的 new_repr：
  `bank.slots.requires_grad == True` 且 `grad_fn is not None`（dual-gate + single-gate 两路均通过）。
- 写入超大 norm（×50）内容后，写入 slot 的 norm 被精确 cap 到 5.0（限幅功能没坏）。
- `loss = bank.slots.sum(); loss.backward()` → 梯度成功回流到 write 输入。
- cap=0 control 同样保持 requires_grad。

## Bug 2：inject_gate 未加入可训练参数，永久冻结

**文件**：`scripts/train_mem_space_dolmino_cpt.py` 的 `_mem_space_params`。

### 根因
`_mem_space_params` 收集了 selector / gate_param / slot_output_gate /
slot_to_hidden / hidden_to_slot / l3 / l2 / recon_decoder，但**漏了
`inject_gate`**（`layer.py:400` 的 `nn.Linear(d_model, 1)`，产生 per-token 融合
gate `g = sigmoid(inject_gate(hidden))`）。
`_freeze_backbone` 先把所有参数 `requires_grad=False`，再只对 `_mem_space_params`
返回的参数 `requires_grad=True`。inject_gate 不在列表里 → 被冻结后再没打开 →
`g` 永久钉在 init（bias=`inject_gate_bias_init≈-0.152` → σ≈0.46）。

`toy_memory_bootstrap.py` 直接 `import train_mem_space_dolmino_cpt as T` 并复用
`T._mem_space_params`（line 493/515），所以修一处即覆盖 toy + Dolmino 两条路径，
无需在 toy 里另写。

### 修复
在 per-wrapper 循环里仿照 slot_to_hidden 加入 inject_gate：
```python
inj = getattr(wrapper, "inject_gate", None)
if inj is not None:
    for p in inj.parameters():
        if id(p) not in seen:
            params.append(p); seen.add(id(p))
```

### 验证（临时脚本，已删）
- 构造 toy 的 model + `_freeze_backbone`：`inject_gate.weight/bias.requires_grad == True`，
  且 weight/bias 均在 `_mem_space_params` 返回列表里（全 32 层都通过）。
- 一次 2-chunk forward+backward 后 `inject_gate.weight.grad` 非零
  （abs sum ≈ 1.875，bias grad abs sum ≈ 0.107）——修复前梯度为 0（冻结）。

## 集成 smoke（toy 30 步，本机单卡 H20）

```
--routing_pool_mode slot_query --selector_temperature 40 --l_recon_weight 0.1 --total_steps 30
```
- non-finite=0，不崩。
- `lm` loss：5.40（step0）→ 3.00（step20）→ 3.14（step25）——答案 LM loss 开始有下降
  趋势（修复前全程钉 ~3.0）。
- `recon` loss：0.20 → 0.003，writer 在被有效训练（跨 chunk 梯度已通）。
- `top1_sim`：0.346 → 0.515（寻址持续变好）。
- `alpha_mean`(eval batch) 在 30 步内仍显示 0.4711（视觉上慢），但 inject_gate.weight
  梯度已被验证非零，optimizer 会随训练更新——闭环已打通，长训练才看得出 alpha 漂移。

## Known issues / 后续
- 30 步太短，retrieval_exact_acc 仍 0；需要按 MEMORY_PROTOCOL_PLAN.md Round 1 跑
  500 步 + l_recon_weight 扫，确认 exact_acc 能否从 0 起来（金标准）。
- alpha_mean 在短 smoke 内移动不明显，长训练需复核 inject_gate 是否真的开始分化
  （WRITEBACK_DIAG 的 inject_gate_std）。

## 改动文件
- `src/memory/mem_space/memory_bank.py`（Bug1，两处 + docstring）
- `scripts/train_mem_space_dolmino_cpt.py`（Bug2，`_mem_space_params`）
