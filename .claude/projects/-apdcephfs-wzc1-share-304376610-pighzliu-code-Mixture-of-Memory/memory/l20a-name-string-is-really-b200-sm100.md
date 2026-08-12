---
name: l20a-name-string-is-really-b200-sm100
description: ★★「NVIDIA L20A」只是 name 字符串显示问题，LOCAL 和 .21 的真实硬件就是 B200(sm_100 / 148 SM / 178GB HBM)；不要再按 name 断言它不是 B200
metadata: 
  node_type: memory
  type: reference
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**`nvidia-smi` 报的 `NVIDIA L20A` 是 name 字符串的显示问题。LOCAL 和 `.21` 的真实硬件就是 B200。**
用户已多次指出这一点，我又按 name 字符串"实测纠正"过一次，是错的。

## 判据：看 compute capability 和硬件规格，不看 name

2026-08-12 实测（`torch.cuda.get_device_properties(0)`，torch 2.13.0）：

| 项 | LOCAL / .21 | .73/.82/.104 (H20) |
|---|---|---|
| `name` | `NVIDIA L20A` ← **不可信** | `NVIDIA H20` |
| **capability** | **`sm_100`** | `sm_90` |
| **SM count** | **148** | 78 |
| **total_mem** | **178.4 GB** | 95.0 GB |

`sm_100` = **Blackwell** = B200。`sm_90` = Hopper = H100/H20。
`torch.cuda.get_arch_list()` 在这些机器上含 `sm_100`，torch 2.10+/2.13 才支持。
**148 SM + 178GB HBM + sm_100 就是 B200 的规格**，L20A（Ada，sm_89，48GB）根本不长这样。

## 不要再做的事

- ❌ 用 `nvidia-smi --query-gpu=name` 的输出去断言"这不是 B200"
- ❌ 在文档里写"旧记录说 .21 是 B200 是错的" —— **旧记录是对的，是我按 name 误纠了**
- ❌ 因为 name 显示 L20A 就说"LOCAL 和 .21 只是同型 L20A，测不到跨架构差异"
  （真实情况：B200(sm_100) vs H20(sm_90) 本身就是跨架构，且 B200 明显更强）

## 该做的事

- 判断硬件代际一律用 `capability` / `multi_processor_count` / `total_memory`，或直接
  `python -c "import torch;p=torch.cuda.get_device_properties(0);print(p.major,p.minor,p.multi_processor_count)"`
- `configs/password_b200_19021.txt` 这个文件名是**正确**的命名，不是历史遗留
- 谈算力迁移时：**B200(LOCAL/.21) 明显强于 H20**，既有 sm_100 的架构优势也有 178GB vs 95GB
  的显存优势（1.87×）。把重型训练从 H20 迁到这两台是合理的。

## 迁移时仍然成立的真实约束（与硬件无关）

1. **跨盘**：LOCAL/.21 在 **wzc1**，.73/.82/.104 在 **zwfy6**，见
   [[cluster-two-disks-not-shared]]。搬 ckpt 要 `scp -O` 且实测只有 12MB/s 单流 /
   37MB/s 四流 → 大 ckpt 不划算，更适合"在 B200 上起新 run"而不是搬正在跑的。
2. **plain DDP 不是 FSDP**，见 [[ddp-not-fsdp-per-card-mem]]：加卡不减每卡内存，
   B200 的收益来自**单卡能开更大 batch**，不是分片。
3. 16 卡多机 DDP 只能**同盘内**合并（LOCAL+.21 合法）。

相关：[[dllm-h20-node]]（集群台账）、[[b200-prefer-paperB-when-free]]（B200 空出时的优先级）。
