# ONBOARDING — 新协作者 30 分钟上手

> 目标读者：第一次接触本仓库的协作者。读完这一篇 + 跑通一个 eval，就能开始干活。
> 最后更新：2026-07-01。**权威实时状态永远以 `status/HEARTBEAT_LATEST.md` 为准**（本文件是稳定的地图，那个文件是当前的天气）。

---

## 0. 一句话项目是什么

让冻结的 **Llama-3-8B** 用一个**有界大小的记忆**处理超长上下文（不让 KV cache 随长度线性膨胀），
在 **BABILong**（qa1/qa2/qa5 × 上下文 2k–32k，n=100）上评测。核心代码是 `src/memory/mem_space/` 这个 adapter。

---

## 1. 先读这三样（按顺序，10 分钟）

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | **`status/HEARTBEAT_LATEST.md`** | **当前**在跑什么、五条线各自状态、最新判据。**这是最新真相。** |
| 2 | `status/PROJECT_TRUE_STATE_20260701.md` | 校准过的基线：三种读出机制、两堵墙、A 模型官方分数、哪些旧结论已撤回。 |
| 3 | 本文件剩余部分 | 目录地图 + 怎么训 / 怎么评 / 三条红线。 |

> ⚠️ 老的 `README.md` 顶部描述的是**更早的 slot 架构**（N 个 memory slot + delta-rule）。那条线仍在，但**当前主攻方向已转向 hidden 级路线**（见 §3）。以 `HEARTBEAT_LATEST.md` 为准，不要以 README 顶部为准。

---

## 2. 环境（2 分钟）

```bash
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
.venv/bin/python --version        # -> Python 3.11.13 (torch 2.10 + transformers)
```

- **一律用 `.venv/bin/python`**，不要用系统 python。
- 远程节点：`WANDB_MODE=offline`，HF cache / 代理 export 见 `CODEBUDDY.md`。
- **集群节点表（IP / SSH / 每盘 python / rsync 配方）在 `CODEBUDDY.md`**，经常变，本文件不复制。跑远程作业前先读它 + `status/SESSION_HANDOFF.md`。

---

## 3. 现在的研究方向（必读，避免走回头路）

我们把"记忆读出"拆成**三种互不相同的机制**，别混为一谈：

1. **纯 hidden FIFO**（`--use_fifo_memory`）：buffer 存最近 ~25 个 chunk 的原始 hidden（detached），
   拼进 attention 做前缀，标量注入门 g≈0.12。**开 FIFO 时所有 slot 路由被旁路（互斥）。**
2. **token reforward**：把选中 chunk 的**原始 token 重新过一遍整个模型**（≈RAG，昂贵）。
   **⛔ 这条线已停** —— 用户明确判定"跟 RAG 类似，没有实际用处"。别再往这上面投入。
3. **选择器**：flat reader-attn（冻结）/ HNST 树（冻结）/ 重训的 reader-attn（`t2_select_loss`）。

**当前主攻（hidden 级三线）**：
- **训练信号 / 数据迭代**（最押）：合成 qa5 的噪声结构做得更接近真 babilong，破"读出墙"的根因。
- **HNST v2 选择器树**：可训练 summary 树 + select_loss（v1 用 max-pool 毁 needle 信号，已知失败，v2 迭代中）。
- **想法3 多尺度 beacon**：beacon 金字塔 + reader 训练来消费。

**两堵墙**（问题的核心框架）：
- **读出墙**：选对了 chunk，但它的 hidden 读出很弱。（A 模型 16k：hidden FIFO=9 vs oracle 定位重生成=45 vs token reforward=52）
- **选择墙**：选择器选错 chunk。

**已定的负结论（别重做）**：解冻更多层**不能**破读出墙（16k：5层45→8层38→full 22，越解冻越过拟合/灾难遗忘）。根因是**训练信号**，不是容量 / 架构 / 解冻层数。

---

## 4. 目录地图

```
src/memory/mem_space/     现役 adapter
    layer.py              核心：_forward_fifo / _fifo_select_keep_set_tree(树) / prefix 注入
    config.py             所有开关
    niah_chunked_dataset.py  合成训练数据(qa5 give-event 等)
    beacon_pyramid.py / tree_summary.py   想法3 / v2 树的新模块
scripts/
    train_mem_space_dolmino_cpt.py   ★ 训练器（唯一）
    run_babilong_mem_space.py        ★ BABILong 推理
    score_nested_babilong.py         ★ 官方判分聚合器
    launch_*.sh   训练启动器   |   eval_*.sh / _*.sh   评测启动器(_前缀多为临时)
status/           实时状态流（见 §1）
third_party/babilong-pkg/babilong/metrics.py   官方判分 compare_answers（唯一合法判分）
CODEBUDDY.md      完整操作手册 + 权威集群/GPU 配置（CLAUDE.md 是它的软链）
HEARTBEAT.md      自主运维 / 监控 playbook
legacy/           已废弃方向（进去看它自己的 README）
```

---

## 5. 怎么训

```bash
# 现役 chunk512 hidden-FIFO 启动器（示例）
bash scripts/launch_mem_space_fifo_chunk512_196.sh
```

启动器包一层 `torch.distributed.run --nproc_per_node=8 scripts/train_mem_space_dolmino_cpt.py`。要点：
- `--chunk_size 512`，早停（step500–1000 ≫ step5000，过训伤 BABILong），`--eval_interval 0`（DDP 内联 eval 会 NCCL 挂）。
- hidden 路线开关：`--use_fifo_memory`（FIFO）、`--use_tree_summary` + `--t2_tree_branch/beam`（v2 树）。
- **多机加速**：若要跑的训练 ≤3 个，把同盘节点合成一个 16 卡节点跑多机 DDP（`--nnodes 2 --rdzv_backend c10d`）。同盘分组见 `CODEBUDDY.md`。

---

## 6. 怎么评（务必用官方判分）

```bash
# 1) 推理：某 ckpt -> 每 (task,length) 的 CSV
.venv/bin/python scripts/run_babilong_mem_space.py --help

# 2) 聚合判分：qa1/qa2/qa5 × 0k–32k，n=100，官方 compare_answers
.venv/bin/python scripts/score_nested_babilong.py <results_dir>
```

`score_nested_babilong.py` 用的是 `third_party/babilong-pkg/babilong/metrics.py:compare_answers`（首句 + 排除问题标签 + 要求 target 唯一）。
**⛔ 严禁用 `re.search(\b target \b)` 判分** —— 那是"送分"判据，会虚高分数，历史上污染过大量端到端结论。

---

## 7. 三条红线（违反会毁实验或误导结论）

1. **绝不在 BABILong 上训练。** `--babilong_mix_fraction` 已在代码里**永久硬禁**
   （`scripts/train_mem_space_dolmino_cpt.py`：`>0` 直接 `raise SystemExit`）。BABILong 是评测集，训了就是泄漏。
   历史上 `b50` 泄漏 ckpt（mix=0.15）虚高过分数，相关目录已隔离到 `*_LEAKED_VOID*`。**别用任何 b50 ckpt。**
2. **判分只用官方 `compare_answers`；** 报告分数必须全档 + n=100 + 同设定。合成 loss=0 / 小 n 不算数，只看真 BABILong 满 n。
3. **下结论前查全证据。** 别凭 log 一眼就断言"崩了/卡死/无用"。多个旧结论就是这么误判又撤回的（见 `PROJECT_TRUE_STATE` 的撤回清单）。

---

## 8. 有问题问谁 / 看哪

- 当前跑什么：`status/HEARTBEAT_LATEST.md`
- 集群怎么连：`CODEBUDDY.md`
- 校准过的基线数字与撤回清单：`status/PROJECT_TRUE_STATE_20260701.md`
- 历史发现：`status/FINAL_FINDINGS_20260630.md`、`ops/research_notes/`
