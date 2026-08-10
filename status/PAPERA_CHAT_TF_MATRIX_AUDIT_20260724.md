# Paper A 全套「方法 × benchmark × chat 状态」矩阵审计（2026-07-24，SSH 实地清点 diskB .73）

> 用户 2026-07-24 指令：Paper A 要**全套数据**——所有 baseline + CoMem，**chat=True 和 chat=False 都要有**（否则 baseline 用 chat=True、我们 CoMem 用 chat=False = 不公平）。
> 本审计 = SSH 进 diskB `.73`（只读，不碰 GPU）实地 `ls` 各 `*_results/` 目录，按方法 + chat 后缀分类得出的**真实**覆盖状态（非从文档推断）。
> 目录命名：chat=True = `*_chatnothink` / `*_chatnothink_ad` / 无后缀旧 dir；chat=False = `*_chatFALSE`。范围 = Qwen3-8B flagship（MemoryLLM=Llama-3、LLoCO=LLaMA-2 异基座）。

## ★ 定论矩阵（Qwen3-8B，T=chat=True / F=chat=False）

| 方法 | RULER | LongBench | LongEval | LoCoMo | BABILong |
|---|---|---|---|---|---|
| **CoMem（本文）** | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ |
| KV-Direct（full-ctx 上界） | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ |
| HCache | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ |
| StreamingLLM | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ | T✅ F✅ |
| MemoryLLM | T✅ F✅ | T✅ F✅ | T✅ **F❌** | T✅ F✅ | T✅ F✅ |
| **InfLLM（旗舰 baseline）** | T✅ **F❌** | T✅ **F❌** | T✅ **F❌** | T✅ F✅ | T✅ **F❌** |

## ★ 缺口（要补齐"全套"只差这些）

1. **InfLLM chat=False × {RULER, LongBench, LongEval, BABILong}（4 benchmark）** — 主缺口。InfLLM chat=False **LoCoMo 已 done**（`locomo_results/infllm_8b_chatFALSE` 18 files，errata §9b），其余 4 个只有 chat=True。InfLLM 是 thunlp paper-faithful 旗舰 baseline，会进主对照表 → chat=False 必须补。
2. **MemoryLLM chat=False × LongEval（1 格）** — 次要，可能属 #50 标记的 non-paper 格（LongEval/LongBench MemoryLLM 曾被标非论文）。需确认是否进表；进则补。
3. **一致性 errata（非缺数据）**：KV-Direct/HCache 的 RULER chat=False 用了 `sel=bm25` 而非 flagship `iter_bm25`（errata §8c，task#10 待统一）。数据在，但 selector 口径与 CoMem 不一致，需重跑或标注。

## ★ 已完整（无需动）

- **CoMem / KV-Direct / HCache / StreamingLLM**：5 benchmark × T/F **全齐**。
- **MemoryLLM**：4 benchmark T/F 齐（仅缺 LongEval F）。
- **InfLLM chat=True**：5 benchmark 全齐（RULER 10 files / LongBench 105 / LongEval 57 / LoCoMo 17 / BABILong 1 聚合 json）。**InfLLM chat=False：仅 LoCoMo 齐。**

## ★ 补齐所需资源 + 约束

- **补 InfLLM chat=False 需 GPU + Qwen3-8B + eval 数据**，三者都在 **diskB**（H20）。**wzc1（LOCAL+.252）没有 Qwen3-8B**（只有 Llama-3-8B）→ 在 .252 上补需先 rsync Qwen3-8B+harness+数据，且打断 Paper B keep12。
- **2026-07-24 11:xx 实测 diskB .73 八卡全忙**（GPU0-7 各 1 进程 = 用户/dllm 在用）→ 现无法在 diskB 起 InfLLM eval，除非用户腾一个 H20 节点。
- 现成 runner：`scripts/_chatFALSE_baselines_driver.sh` + `_run_baselines_locomo_chatFALSE_8gpu.sh`（LoCoMo 版，已把 InfLLM LoCoMo chatFALSE 跑出）；RULER/LongBench/LongEval/BABILong 的 InfLLM chatFALSE 需按同法把现有 `infllm_8b`(chat=True) runner 去掉 `--use_chat_template` 写到 `infllm_8b_chatFALSE` 并行 dir。
- **时间估**（8×H20 pooled）：InfLLM RULER(niah×2+vt × 8k-128k) ~3h + LongBench(6ds) ~1.5h + LongEval(4k-128k) ~1.5h + BABILong(qa1/2/5×0k-32k n=100) ~1.5h ≈ **6-8h 单节点跑完**。

## ★ 决策待用户

- InfLLM chat=False 补齐需 8 GPU + Qwen3-8B（在 diskB）。**建议在 diskB H20 就地补**（模型/数据都在，无需 rsync）——需用户腾一个 H20 节点（现全忙）。**不建议暂停 .252**（缺 Qwen3-8B，需 rsync + 打断 Paper B）。
- task#10 论文表集成对**已 chat=False 的部分（矩阵 95%）现在就能起**（纯 CPU），InfLLM 那几格等补完再填行。
