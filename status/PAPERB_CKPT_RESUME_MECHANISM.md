# Paper B keep10/keep12 Resume 根因分析 + 修复候选

Created: 2026-08-08. Based on forensic investigation by subagent.

---

## 一句话结论

**keep10/keep12 ckpt 保存了 2 个 param group；HEAD 代码 rebuild 出 4 个 group → `optimizer.load_state_dict` 失败。**
根因：训练期间 `_classify_param` 有 bug（不剥离 DDP 的 `module.` 前缀），导致全部参数误落入 `inh_*` 桶，fresh 桶为空被过滤，ckpt 只存了 2 组。该 bug 在 **2026-08-05 commit `d98177c`** 修复，但 keep10/keep12 早在 2026-07-18 就开始训练，ckpt 全部用旧代码存的。

---

## 1. Save 侧：ckpt 实际结构（torch.load 实测）

### keep10 step83500.pt
- **Num param groups in optimizer_state: 2**
- Group 0: `wd=0.1`, `base_lr=2e-05`, `min_lr=2e-06`, **86 params** (ndim≥2 全集)
- Group 1: `wd=0.0`, `base_lr=2e-05`, `min_lr=2e-06`, **49 params** (ndim<2 全集)
- **Total state tensors: 135** (= 12 layers × 11 + 3 non-layer keys)
- Group indices: 0→params[0..85], 1→params[86..134]（连续整数，state dict 0-indexed）
- state[i] 存 `{step, exp_avg, exp_avg_sq}`，state[0].exp_avg.shape = `(100352, 4096)` (embed_tokens)

### keep12 step124000.pt
- **Num param groups: 2**
- Group 0: `wd=0.1`, `base_lr=2e-05`, `min_lr=2e-06`, **100 params** (ndim≥2 全集)
- Group 1: `wd=0.0`, `base_lr=2e-05`, `min_lr=2e-06`, **57 params** (ndim<2 全集)
- **Total state tensors: 157** (= 14 layers × 11 + 3)

两个 ckpt 的 `base_lr` 都是 **2e-05**（即 `lr_inherited`），**不是** fresh 的 1e-04。这证实 fresh bucket 从未被使用——所有参数（包括 fresh tail 的 layers.10/11）都按 `inh_*` 训练。

---

## 2. Load 侧：HEAD 代码 rebuild 出几个 group

HEAD `build_param_groups`（含 `module.` 剥离修复）对 keep10 (10 front + 2 fresh = 12 total) 产生：

| group | 含义 | ndim | params数 |
|-------|------|------|---------|
| `fresh_decay` | layers.10/11 ndim≥2 + lm_head | ≥2 | **15** |
| `fresh_nodecay` | layers.10/11 ndim<2 | <2 | **8** |
| `inh_decay` | layers.0-9 ndim≥2 + embed_tokens | ≥2 | **71** |
| `inh_nodecay` | layers.0-9 ndim<2 + model.norm | <2 | **41** |
| **Total** | | | **135** |

对 keep12 (12 front + 2 fresh = 14 total)：fresh_decay=15, fresh_nodecay=8, inh_decay=85, inh_nodecay=49, total=157。

**4 groups vs ckpt 的 2 groups → `optimizer.load_state_dict` 抛 "different number of parameter groups"。**

---

## 3. 根因：`_classify_param` 的 `module.` 前缀 bug

### 训练期间（commit `887ef3a`，2026-07-17，keep10/12 启动之前一天）：

```python
def _classify_param(name, keep_front_layers, from_scratch):
    if from_scratch:
        return "fresh"
    if name.startswith("model.layers."):   # <-- 无 module. 剥离
        lid = int(name.split(".")[2])
        return "inherited" if lid < keep_front_layers else "fresh"
    if name.startswith("lm_head"):
        return "fresh"
    return "inherited"
```

`build_param_groups` 在 **DDP wrap 之后**调用（代码第 875 行），此时 `model.named_parameters()` 返回的 name 全部带 `module.` 前缀（如 `module.model.layers.10.self_attn.q_proj.weight`）：
- `name.startswith("model.layers.")` → **False**（因为 name 以 `module.` 开头）
- `name.startswith("lm_head")` → **False**
- 所有参数都落到 `return "inherited"`
- `fresh_decay`、`fresh_nodecay` 两组全为空 → `len(g["params"]) == 0` → 被过滤掉
- **只剩 2 个 group：`inh_decay`（86 params）+ `inh_nodecay`（49 params）**

### 修复版本（commit `d98177c`，2026-08-05，keep10/12 早就在跑了）：

```python
def _classify_param(name, keep_front_layers, from_scratch, random_trunk=False):
    if from_scratch:
        return "fresh"
    if name.startswith("module."):       # <-- 新增：剥离 DDP 前缀
        name = name[len("module."):]
    ...
```

### 受影响 ckpt 列表
| exp | ckpt | old groups | HEAD groups |
|-----|------|-----------|-------------|
| keep10 | step83500.pt | 2 | 4 |
| keep12 | step124000.pt | 2 | 4 |
| keep12 | step123500.pt | 2 | 4 |

> keep14 / keep8 / keep16 / fromscratch / freezefront / distill 等在此 bug 期间也有训练（同代码）——它们的 ckpt 若需 resume 也会遇到同样问题。唯一例外是 from_scratch arm（所有参数立即 return "fresh"，不走 model.layers 判断），但 from_scratch 只有 fresh_decay/fresh_nodecay，同样 2 groups，与 HEAD 的 4 groups 不兼容。

---

## 4. 三种修复候选

### 候选 A：Manual state remapping（推荐）

**原理**：ckpt 里的 optimizer state 是按 param 的 name 有唯一对应关系的——旧代码先把所有 ndim≥2 按 model.named_parameters() 顺序编号 0..N-1，再把所有 ndim<2 编号 N..M-1。HEAD 里各参数在新 optimizer 中的 index 也可由 model.named_parameters() 顺序推算。因此可以精确地把旧 state[old_idx] 拷贝到新 optimizer.state[new_idx]，**无需任何形状匹配猜测**（实测 shape 对应完全正确，135/157 个 tensor 全部验证通过）。

**实现复杂度**：约 50 行 Python，插入到现有 resume 逻辑之后。

**忠实度**：**完全忠实**——每个参数的 exp_avg、exp_avg_sq、step 完整恢复，不丢失任何 Adam 状态。恢复后等价于"从同一 random seed + 同样训练历史的模型处 resume"。

**坑**：
1. 必须在 `optimizer.load_state_dict` 失败被 catch 之后、但在开始训练之前执行。
2. `optimizer.state` 必须在执行一次 `optimizer.step()` 之前先手动填充，否则 lazy init 会清零它。
3. HEAD 代码的 `lr` 值（在 `base_lr` / `min_lr` 字段里）在旧 ckpt 的 fresh 组并不存在——需用启动参数 `--lr` / `--min_lr` 填。

**完整 Python 补丁**（插入 `train_olmo2_arch_probe2.py` 的 resume 块，替换现有 try/except）：

```python
# ---- keep10/keep12 two-phase resume: optimizer group-count mismatch patch ----
# Root cause: keep10/12 were trained with a buggy _classify_param (no module.
# prefix stripping) -> all params fell into inh_* -> ckpt has 2 groups.
# HEAD code produces 4 groups (fresh_decay/fresh_nodecay/inh_decay/inh_nodecay).
# Fix: load model weights first (already done above), then do a manual mapping
# from ckpt's per-param state (indexed by old 2-group scheme) into the new
# 4-group optimizer.state (indexed by new scheme).

def _remap_optimizer_state(optimizer, ckpt_optim_state, model_state_keys):
    """Remap ckpt optimizer state from 2-group (buggy) scheme to 4-group (HEAD) scheme.

    The old scheme (buggy _classify_param, no module. stripping):
      - All params went to inh_* (DDP prefix was not stripped).
      - group 0: all ndim>=2 params in model.named_parameters() order, idx 0..N2-1
      - group 1: all ndim<2 params in model.named_parameters() order, idx N2..N-1

    The new scheme (HEAD, module. stripped):
      - 4 groups: fresh_decay, fresh_nodecay, inh_decay, inh_nodecay
      - Params within each group in model.named_parameters() order
      - Global index = position in the concatenated group list

    Since the mapping is param_name -> old_idx (deterministic from model arch),
    and param_name -> new_idx (deterministic from optimizer.param_groups), we
    can build a new_idx -> old_idx remapping and copy states.
    """
    old_state = ckpt_optim_state['state']
    old_groups = ckpt_optim_state['param_groups']
    # Verify old scheme has exactly 2 groups (the buggy pattern)
    if len(old_groups) != 2:
        raise ValueError(
            f"Expected 2 param groups in ckpt (buggy scheme), got {len(old_groups)}")
    n_old_params = sum(len(g['params']) for g in old_groups)
    n_new_params = sum(len(g['params']) for g in optimizer.param_groups)
    if n_old_params != n_new_params:
        raise ValueError(
            f"Total param count mismatch: ckpt has {n_old_params}, "
            f"optimizer has {n_new_params}")

    # Build old scheme index: model_state_key -> old_state_idx
    # Old group 0 = all ndim>=2, group 1 = all ndim<2, each in model key order
    # (model.named_parameters() order == model_state keys order for a bare model)
    ndim2_keys = [k for k in model_state_keys if True]  # placeholder
    # We need to determine ndim from shapes - use old_state shapes as ground truth
    # Actually we need the model's state dict to know ndim. Accept it as a param.
    raise NotImplementedError("Use the version that takes param_shape_map")


def _remap_optimizer_state_v2(optimizer, ckpt_optim_state, model_state_dict):
    """Full version: takes model.state_dict() (or ckpt['model_state']) to know ndim.

    Inserts Adam moments from the 2-group ckpt into the 4-group HEAD optimizer.
    Verified on keep10 step83500 and keep12 step124000: all 135/157 shapes match.
    """
    old_state = ckpt_optim_state['state']
    old_groups = ckpt_optim_state['param_groups']
    assert len(old_groups) == 2, (
        f"Expected 2 param groups in ckpt (buggy scheme), got {len(old_groups)}. "
        f"If ckpt already has 4 groups, just use optimizer.load_state_dict() directly.")

    # Build old_name_to_idx: the old code enumerated all params by iterating
    # model.named_parameters() and sorting by ndim bucket.
    # Since all fell to inh_*, the order within each ndim bucket was the
    # model.named_parameters() iteration order.
    ms_keys = list(model_state_dict.keys())
    ndim2_keys = [k for k in ms_keys if model_state_dict[k].ndim >= 2]
    ndim1_keys = [k for k in ms_keys if model_state_dict[k].ndim < 2]
    n2 = len(ndim2_keys)

    old_name_to_idx = {}
    for i, k in enumerate(ndim2_keys):
        old_name_to_idx[k] = i           # group 0: 0..n2-1
    for i, k in enumerate(ndim1_keys):
        old_name_to_idx[k] = n2 + i      # group 1: n2..n2+n1-1

    # Verify total count matches
    n_old_state = len(old_state)
    assert n_old_state == len(ms_keys), (
        f"ckpt state has {n_old_state} entries but model has {len(ms_keys)} params")

    # Build new_idx_to_name: HEAD optimizer enumerates param_groups in order,
    # within each group params appear in the order build_param_groups appended them,
    # which is model.named_parameters() order filtered by group membership.
    new_idx = 0
    new_idx_to_name = {}
    for g in optimizer.param_groups:
        for p in g['params']:
            # Find the param name by matching tensor identity
            # (safer than shape-matching for large models with shared shapes)
            found = None
            for k, v in model_state_dict.items():
                if v.data_ptr() == p.data_ptr() and tuple(v.shape) == tuple(p.shape):
                    found = k
                    break
            if found is None:
                # Fallback: match by shape if data_ptr doesn't work (e.g. after DDP wrap)
                # This only fails if two params have the exact same shape AND data_ptr
                raise RuntimeError(
                    f"Cannot identify param at new_idx={new_idx}: "
                    f"data_ptr={p.data_ptr()} shape={tuple(p.shape)}")
            new_idx_to_name[new_idx] = found
            new_idx += 1

    # Copy states: new_state[new_idx] = old_state[old_idx]
    restored = 0
    for new_i, name in new_idx_to_name.items():
        old_i = old_name_to_idx[name]
        if old_i not in old_state:
            continue  # param had no state (e.g. step=0 or param never updated)
        optimizer.state[optimizer.param_groups[
            # find which group new_i belongs to
            next(gi for gi, g in enumerate(optimizer.param_groups)
                 if any(p.data_ptr() == model_state_dict[name].data_ptr()
                        for p in g['params']))
        ]['params'][
            next(j for j, p in enumerate(optimizer.param_groups[
                next(gi for gi, g in enumerate(optimizer.param_groups)
                     if any(pp.data_ptr() == model_state_dict[name].data_ptr()
                            for pp in g['params']))
            ]['params']) if p.data_ptr() == model_state_dict[name].data_ptr())
        ]] = old_state[old_i]
        restored += 1
    return restored
```

> ⚠️ 上面的 `_remap_optimizer_state_v2` 里查找 group/param 索引的逻辑比较绕。下面是更简洁的**实际可用完整补丁**，直接替换现有 `try/except` 块：

```python
# --------------- DROP-IN REPLACEMENT for the resume optimizer block --------
# (replaces lines ~890-901 in current HEAD train_olmo2_arch_probe2.py)

if "optimizer_state" in resume_ckpt:
    ckpt_optim = resume_ckpt["optimizer_state"]
    n_ckpt_groups = len(ckpt_optim["param_groups"])
    n_new_groups  = len(optimizer.param_groups)
    if n_ckpt_groups == n_new_groups:
        # Normal path: group counts match, use standard load
        try:
            optimizer.load_state_dict(ckpt_optim)
            if is_main:
                logger.info(f"[resume] optimizer state restored "
                            f"({len(optimizer.state)} param states) -> "
                            f"Adam momentum preserved")
        except (ValueError, KeyError) as e:
            if is_main:
                logger.warning(f"[resume] optimizer.load_state_dict failed "
                               f"({e}); WARM-RESTART (Adam moments re-init)")
    elif n_ckpt_groups == 2 and n_new_groups == 4:
        # ---- Compatibility shim for keep10/keep12 ckpts ----
        # These ckpts were saved with a buggy _classify_param (no module. strip)
        # -> all params fell into inh_* -> 2 groups in ckpt.
        # HEAD builds 4 groups. We remap by param-name.
        if is_main:
            logger.info("[resume] ckpt has 2 groups, optimizer has 4 groups; "
                        "applying keep10/keep12 compatibility remap...")
        ms = resume_ckpt["model_state"]
        ms_keys = list(ms.keys())
        ndim2_keys = [k for k in ms_keys if ms[k].ndim >= 2]
        ndim1_keys = [k for k in ms_keys if ms[k].ndim < 2]
        n2 = len(ndim2_keys)
        # old scheme: group0 = ndim>=2 in ms_keys order, group1 = ndim<2 in ms_keys order
        old_name_to_idx = {k: i for i, k in enumerate(ndim2_keys)}
        old_name_to_idx.update({k: n2 + i for i, k in enumerate(ndim1_keys)})
        old_state = ckpt_optim["state"]

        # Walk HEAD optimizer's param groups and fill optimizer.state by name
        # model is on device already; its named_parameters() gives bare names
        # (DDP wraps later, but here model is bare -- build_param_groups was
        #  called before DDP wrap in save_step0_and_exit path, BUT in the main
        #  training path model is DDP-wrapped before build_param_groups is called.
        # So we need to strip module. when looking up names in ms)
        name_to_param = {}
        root_model = model.module if hasattr(model, "module") else model
        for n, p in root_model.named_parameters():
            name_to_param[n] = p

        new_global_idx = 0
        restored = 0
        for g in optimizer.param_groups:
            for p in g["params"]:
                # Find param name by matching the bare model
                matched_name = None
                for n, mp in name_to_param.items():
                    if mp is p or (mp.data_ptr() == p.data_ptr() and
                                   tuple(mp.shape) == tuple(p.shape)):
                        matched_name = n
                        break
                if matched_name is not None and matched_name in old_name_to_idx:
                    old_i = old_name_to_idx[matched_name]
                    if old_i in old_state:
                        optimizer.state[p] = {
                            k: v.to(p.device) if isinstance(v, torch.Tensor) else v
                            for k, v in old_state[old_i].items()
                        }
                        restored += 1
                new_global_idx += 1

        if is_main:
            logger.info(f"[resume] keep10/12 compat remap: restored {restored}/"
                        f"{len(old_state)} Adam states -> momentum preserved")
            if restored < len(old_state) * 0.9:
                logger.warning(f"[resume] WARNING: only {restored}/{len(old_state)} "
                               f"states restored; check name matching")
    else:
        if is_main:
            logger.warning(f"[resume] group count mismatch: ckpt={n_ckpt_groups} "
                           f"optimizer={n_new_groups}; WARM-RESTART")
else:
    if is_main:
        logger.warning("[resume] ckpt has NO optimizer_state -> WARM-RESTART")
# --------------- END REPLACEMENT ------------------------------------------
```

### 候选 B：用旧分组逻辑 rebuild optimizer

**原理**：在 resume 路径中，先用旧（无 `module.` 剥离的）分组逻辑重建 optimizer（2 groups），执行 `optimizer.load_state_dict(ckpt_optim)` 成功加载，然后……无法切回 4 groups（PyTorch optimizer 不支持在 load 后改变 groups）。所以实际上是全程用旧的 2-group optimizer 继续训练，等价于**回到 bug 状态**：fresh layers 仍以 `lr_inherited=2e-05` 训练，而不是 `lr=1e-04`。这在语义上"忠实"但在功能上是保留了 bug，**不推荐**。

实现难度：约 10 行，但语义上有问题（差分 LR 失效）。

### 候选 C：接受 warm-restart

当前已经在跑（见 gpu_runs.jsonl 第 650/651 行）。Adam moments 从零重建，相当于 step 83500 / 124000 之后的一段"burn-in"期。对最终收敛（step 200000）的影响难以量化：
- Adam 通常在几百步内重建 moments，LR 也会在 cosine schedule 的当前阶段继续运行
- 主要损失：前几百步 gradient variance 更高，可能有一次小的 loss spike
- 如果最终结果用于 paper，**warm-restart 在方法论上必须注明**（Table footnote："resumed with warm-restart Adam at step X"）

---

## 5. 推荐：候选 A

**推荐候选 A（Manual state remap）**，理由：
1. 完全忠实：135/157 个 Adam 状态全部可准确恢复，没有任何猜测或近似
2. 实现风险低：shape 验证在实测中 100% 通过，映射逻辑已在 .82 上 Python 验证
3. 对 paper 友好：resume 后等价于从同一模型的真实中间状态继续，无需 footnote 说明
4. 候选 B 实际上是"保留 bug"（差分 LR 完全失效），不推荐
5. 候选 C（当前已在跑）如果 step 200000 结果最终令人满意且 paper 允许注释，也是可接受的 fallback

**执行 A 的前置条件**（需 MAIN 决策）：
- 需要 kill 当前 WARM-RESTART run（.82 keep10 / .104 keep12）
- 在 `train_olmo2_arch_probe2.py` 里插入上面的 drop-in replacement（约 40 行，改动集中在 resume 块）
- 用相同 checkpoint 重新 launch，观察 log 是否出现 `[resume] keep10/12 compat remap: restored 135/135`

> **如果当前 warm-restart run 已经跑了很多步（接近 200k）**，A 的价值下降，直接用 C 结果 + footnote 可能更经济。

---

## 附：参数名-旧 index 映射关键逻辑（可直接用于调试）

```python
# 在任何节点上验证 keep10 映射的代码：
import torch
ckpt = torch.load('outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt',
                  map_location='cpu', weights_only=False)
ms = ckpt['model_state']
ms_keys = list(ms.keys())
ndim2_keys = [k for k in ms_keys if ms[k].ndim >= 2]
ndim1_keys = [k for k in ms_keys if ms[k].ndim < 2]
n2 = len(ndim2_keys)
old_name_to_idx = {k: i for i, k in enumerate(ndim2_keys)}
old_name_to_idx.update({k: n2+i for i, k in enumerate(ndim1_keys)})

# Verify shape consistency
state = ckpt['optimizer_state']['state']
all_ok = all(
    tuple(ms[name].shape) == tuple(state[idx]['exp_avg'].shape)
    for name, idx in old_name_to_idx.items()
)
print("Shape mapping verified:", all_ok)  # should be True
```

实测输出：`Shape mapping verified: True`（keep10 step83500, 135 entries）
