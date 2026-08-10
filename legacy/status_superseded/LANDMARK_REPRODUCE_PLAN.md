# Landmark 复现 → 迁移 主计划（2026-06-19 用户批准）

## 方法论（用户 2026-06-19 指令，覆盖旧的"patch v2"路线）
不再 patch 我们自己的 broken SFT setup + 猜哪里错。改为 **diff-based 调试**：
**复现 Landmark Attention（已知能破长程墙）→ 列主要差异点 → 在其代码基础上逐步向 mem_space 迁移，一次只改一个差异，每步用 passkey/NIAH 守门，passkey 准确率断崖处 = 该差异即长程杀手。**

## 为什么走这条（背景）
- 我们 ~30 实验证伪：固定 slot 压缩 + frozen reader 无法长程精确读出；unfreeze v1 全量 SFT 反损伤 backbone（OFF 22→11）。
- 根因（damage-investigator + landmark-researcher 双证）：v1 = 窄数据(纯dolmino单源)×last_chunk_loss×小batch(16 vs 128)×零replay×单层注入污染，token 量比 Landmark 少 ~30-120×（32.8M vs 0.98B）、单源 vs 7源。v1 与 Landmark working recipe 同时差 ~5 个维度 → 无法定位。
- Landmark working recipe（出处 epfml/landmark-attention/llama/train.py）：RedPajama-1T-Sample（1B token,7源,1 epoch）、lr 2e-5 cosine+3%warmup、wd0.1、eff-batch128、15k步、全序列LM loss、full-FT、grouped-softmax 全层 + 真实 landmark token + 推理 top-k 真 raw KV。

## ★执行阶段（heartbeat 按序推进，每阶段完成后落账再进下一阶段）

### Phase 1 — 复现 Landmark working baseline 【当前阶段】
目标：在我们 infra 复现 Landmark 论文的 passkey 32k≈100% 正面结果，建立可信锚点。
- **环境**：建独立 venv（transformers==4.28.1 + torch 2.1/cu118，H20 sm_90 兼容），non-triton 路径（use_flash=False），**不污染主 .venv**。
- **S0（先做，最低成本）**：用官方 released weight-diff（`epfml/landmark-attention-llama7b-wdiff` + weight_diff.py recover）得 tuned LLaMA-2-7B ckpt → 改 run_test.py 硬编码路径 → 跑 passkey（0→38k chars，top_k5，50 tests/长度）→ **必须复现 32k≈100%**（否则 infra/eval 口径有问题，先修）。
- 守门：passkey 32k 准确率复现论文正面结果。
- 产物：可复现的 Landmark eval harness + 锚点数字，落 RUN_REGISTRY。

### Phase 2 — 差异表（working Landmark → mem_space）
7 维差异（已由 landmark-researcher 列出，见 TRAINER_ACTIVITY 2026-06-19 09:15）：解冻范围 / 数据量+源 / ctx-block / 检索 / 注入-读出 / 记忆单元 / base model。每维标改动成本 + 长程影响 + 迁移序。Phase 1 完成后固化成 docs/。

### Phase 3 — 逐步迁移（一次一个差异，passkey/qa1 守门）
- S1 换 base：LLaMA-2-7B → Llama-3-8B（重训或短训），验证机制对新 base 仍 work。
- S2 换数据：RedPajama → dolmino（守门：passkey 是否仍破墙 = 验证"单源/短档"是否杀手）。
- S3 换 ctx 结构：landmark-每50 → chunk512/n_ctx3。
- S4 换检索：grouped-softmax 软选 → top-k selector routing（疑 selector 0% precision 元凶）。
- S5 换读出：grouped-softmax 全层 → in-attn KV concat 单层。
- S6 换记忆单元：in-context landmark token → 128 固定 slot + adapter（最后一跃到完整 mem_space）。
- 每步 passkey 准确率断崖处 = 该差异即长程杀手。最可疑序：S4 检索 > S5 单层读出 > S2 数据。

## 当前状态
- Phase 1 启动中（coder 建 venv + recover wdiff + passkey eval）。
- 旧 v2 SFT（partial unfreeze）HOLD —— 被本复现路线取代/重定向，不盲目跑。
- 节点：H20 全空（B200 offline 凭证轮换待修）。
- 完整负结果文档 docs/32K_WALL_FINDINGS.md 已存（frozen + unfreeze 两框架证伪记录）。

## 节点分组与并行约定（2026-06-19 用户指令）
**4 个 H20 节点按盘分两组,每组一个实验槽,同时并行两个实验,最大化吞吐。**

| 组 | 节点 | 盘 | 组内共享FS |
|----|------|----|-----------|
| Group-A | 本机(29.162.227.178) + .196(28.59.80.196) | diskA share_303098609 | ✅ 代码/数据/ckpt互见,无需rsync |
| Group-B | .76(28.49.57.76) + .249(28.59.33.249) | diskB share_304376610 | ✅ 同上 |

- 同一时刻 Group-A / Group-B 各跑一个实验(各自 8 卡或组内 2 节点)。
- ⚠️ 跨盘(A↔B)需 rsync 同步代码/ckpt;组内不用。B200 offline 暂不计入。
- **Phase 3 迁移并行化**: S1→S6 中相互独立的 diff 步两组并发(如 Group-A 跑换数据、Group-B 跑换检索),守门 eval 统一 passkey/qa1,一轮拿两个归因点;同一 diff 的两变体也可两组对比(如单层 vs 多层读出)。有因果依赖的步(如 S6 依赖 S5 结论)仍串行。
- heartbeat 职责: 两组哪组空了就按 plan 补下一个独立迁移步,不留空转。

## 守门 benchmark（2026-06-19 landmark-researcher 核实，带出处）
Landmark **只有两种评测**,无任何下游QA/summ/LongBench:
- **passkey retrieval**(论文§4.2+Fig3b+AppG, 代码run_test.py)= 唯一长程证明,**仅用于LLaMA-7B fine-tune线**。32k≈98%(50 prompts),base>2k崩0%。设置: garbage档[0..38000]chars(最长≈32k+ tok),top_k=5,num_tests=50/档,passkey=randint(1,50000)藏随机位,问法"What is the pass key? The pass key is",max_new_tokens=10正则抽数字。
- **PPL(PG19/arXiv-Math)**= 仅from-scratch GPT-2(§4.1 Table1),★fine-tune线不报PPL → 不纳入迁移守门(口径不一致)。
- 下游任务: 无。

**Phase 3 守门清单(锁定)**:
- 主守门 = **passkey(run_test.py原生)**: S0复现必达32k≈98%; 每迁移步跑passkey,准确率断崖处=该diff即长程杀手。
- 对照守门 = **我们 BABILong qa1(NIAH,与passkey同语义单针检索)**, 确保迁移到我们infra口径可连续比较。
- qa2/qa5 仅进阶观察(Landmark无对应基线),不作diff判定。
- PPL不纳入。
