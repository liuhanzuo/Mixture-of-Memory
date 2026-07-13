# Long-Context Baseline Eval — Collaborator Task Package

**目标**：在 RULER 风格合成长上下文任务上，跑几组 **baseline 方法**，产出 recall 数字，供我们和主方法做 head-to-head 对照。**完全自包含**——只依赖 `torch` + `transformers`，不需要我们的私有代码或数据。

**你的资源**：2 × 8-GPU A100 (40GB)。两个节点各自独立跑同一条命令即可（互不依赖）。

---

## 1. 一次性环境准备（每个节点）
```bash
pip install "torch>=2.1" "transformers>=4.44" accelerate
# 可选（更快，非必需）：pip install flash-attn --no-build-isolation
# 下模型（Llama 需 HF token 同意协议）：
huggingface-cli download Qwen/Qwen3-8B
huggingface-cli download meta-llama/Meta-Llama-3-8B
```

## 2. 跑（每个节点执行；重复跑安全，自动跳过已完成的 cell）
```bash
# 节点各跑一个模型，分工：
#   节点A:
MODEL=Qwen/Qwen3-8B bash run.sh
#   节点B:
MODEL=meta-llama/Meta-Llama-3-8B bash run.sh
```
默认会跑 `{full, streaming} × {niah_single, niah_multikey, vt} × {1k..32k}`，每 cell 100 样本，8 卡并行。

## 3. ⚠️ 40GB A100 注意（重要）
- **`streaming` 方法任意长度都能跑**（KV 预算固定 = `--window_tokens`，默认 4096，显存恒定）。
- **`full` 方法（全上下文）在 8B 模型上，长档会 OOM**：40GB 大概能到 **16k~32k**，**64k/128k 很可能 OOM**。
  - 若 `full` 在某长度 OOM：**把该长度从 `full` 里去掉**，只保留能跑的（`streaming` 那档照常跑）。这本身就是有用信息（"full-context 在 40GB 上跑不动 64k+" 是我们要的结论之一）。
  - 想多跑长档：`MODEL=... LENGTHS="1k 2k 4k 8k 16k" bash run.sh` 先跑短的确认没问题，再单独试长的。
- 每个 cell 独立进程、独立 GPU，**某个 OOM 不影响别的**（log 里会看到 CUDA OOM，跳过即可）。

## 4. 方法说明（你不用改代码，知道在跑啥即可）
| method | 含义 |
|---|---|
| `full` | 全上下文，无压缩。**上界参考**（能跑多长受显存限）。|
| `streaming` | **StreamingLLM**：模型只看 `[前 sink_tokens] ++ [后 window_tokens]`（注意力汇聚点 + 滑动窗口，KV 预算固定）。检索任务里窗口外的 needle 找不到——这正是要测的。|

`--window_tokens`（默认 4096）= StreamingLLM 的 KV 预算。**保持 4096 不要改**（我们的主方法读预算也是这个量级，要可比）。

## 5. 判分（已内置官方口径，你别改）
用 **RULER 官方 `string_match` recall**：output 里出现几个 gold 答案串 / 总 gold 数，再对样本取平均。脚本已实现，**不要改成正则/模糊匹配**——和我们数字的可比性全靠这个口径一致。

## 6. 输出 & 回传
- 结果在 `results/<model>/`：每 cell 一个 `<method>_<task>_<length>.json`（含 recall）+ `.csv`（per-sample）+ `log_*.txt`。
- **跑完把整个 `results/` 文件夹发回给我们**（`.json` 就够，`.csv`/log 可选）。
- 每个节点跑完，终端会打印一份 `RESULT ... recall=XX` 汇总，可直接截图发我们先看。

## 7. 我们会做的验证（你不用管）
我们会本地复核关键几格（尤其 `streaming` 同预算那档），确认口径一致，防错。

---

## 8. 如果这批顺利，第二批（先别做，等我们确认）
- 加 **H2O / SnapKV**（KV 驱逐基线，40G 能跑）——会再给你 clone+run 的 recipe。
- 加 **babilong**（qa1/qa2/qa5）——需要额外 `pip install datasets` + babilong 包，我们再给脚本。

有任何 OOM / 报错 / 跑不起来，把 `log_*.txt` 发我们，我们帮你调。
