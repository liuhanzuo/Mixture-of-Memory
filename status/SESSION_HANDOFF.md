# SESSION_HANDOFF.md — compact / 新会话交接文档

> **本文件是 compact 后或新会话启动时的第一手交接。** 读完这份 + `status/RUN_REGISTRY.md` §3/§4 + `status/TRAINER_ACTIVITY.jsonl` 尾部，就能接上当前研究状态。
> 维护规则：main agent 每当方向/结论/在跑实验有重大变化时，**覆盖更新本文件的「当前快照」区**（保持精简，旧结论沉淀到 RUN_REGISTRY）。
> 最后更新：2026-06-25 13:35 GMT+8

> **🆕 全新 agent 接手?先读 `status/HANDOFF_NEW_AGENT_20260625.md`(零上下文完整交接 + 决策树 + 运维)。**


---

## 0. 一句话现状（结果存疑，正在验证）

⚠️ **2026-06-25 14:55 — 方案B FIFO chunk512/b25 step3000 W0 出了异常高分，但已查实训练含 BABILong 数据泄漏，0k-4k 高分作废，8k-32k 待 memory-disabled 对照验证。**

**.7.53 chunk512/b25 step3000 W0（n=100）：**
```
task       0k      1k      2k      4k      8k     16k     32k
qa1       96     99     99     93     40     34     30
qa2       99    100    100     95     23     32     32
qa5      100    100     97     87     65     76     68
```

**★★ 数据泄漏已坐实（代码核实，非旧 agent 口述）：**
- `train_mem_space_dolmino_cpt.py` 默认 `--babilong_mix_fraction=0.15`：15% 训练步是 BABILong SFT。
- 训练 `--babilong_tasks=qa1,qa2,qa5`（**= eval 任务**）、`--babilong_lengths=0k,1k,2k,4k`（**= eval 的 0k-4k**）、`--babilong_dataset=RMT-team/babilong`（**= eval 同一数据集**）。
- `BABILongTrainDataset`（babilong_dataset.py:79）用 `load_dataset(name,length)[task]`——该 HF 数据集**无 train/test 隔离 → 训练与 eval 样本池完全重叠**。`max_seq_len=2048`：0k/1k 整故事完整泄漏，2k/4k 部分泄漏。
- b25 launch 未 override 任何 babilong 参数 = 全 default。
- **∴ qa5 0k=100/1k=100/2k=97/4k=87 = 背过测试答案，非记忆能力。** 史无前例满分（P11 仅 74）正是泄漏的信号。

**★★★ 8k/16k/32k=65/76/68 也几乎确定主要来自泄漏（2026-06-25 21:30 历史干净 run 锁定）：**
- **历史所有 babilong=0 干净 run，qa5 W0 8k 从没超 ~19、32k 从没超 ~9**（grep 训练日志 "Training complete...babilong=0"，跨蒸馏/self-study/T2/HARDOBJ ~15+ run）：
  - pg19 nctx7（干净 SOTA）= 75/73/51/29/**19/16/9**
  - nctx63（干净）= 61/73/47/27/**14/12/9**
  - **HARDOBJ_lastchunk（干净 + `--last_chunk_loss_only` = 几乎 FIFO 同机制）= 8k 12-15 / 16k 11-14 / 32k 8-9**（N96-N256 全程持平）
- **同机制铁证**：HARDOBJ 用和 b25 几乎一样的 last-chunk memory readout，干净训练 → 8k=12-15/32k=8-9；b25 同机制掺 babilong → 8k=65/32k=68。**泄漏贡献 8k +46(3.4×)/16k +60(5×)/32k +60(7.5×)。**
- ∴ **b25"破墙"约 ~85% 是泄漏幻觉，不是 FIFO 架构能力。真实干净天花板仍是 pg19 的 16k=16/32k=9（项目真 SOTA，一直干净）。**
- **同架构最终归因实验在跑**：NOLEAK b25（.7.53，babilong_mix=0，同 FIFO 配置，ETA 明早 ~07:30 step3000）。预期 W0 落 8k≈15-25/32k≈8-12（干净簇）。task #7。

**∴ "破墙/超越 MemoryLLM/Plan C 过时" 裁决全部撤销。** b25 不是 SOTA；H2 buffer 单调关系（b25>b50>b100）仍成立但都在泄漏放大的尺度上，需 NOLEAK 重测确认 H2 在干净尺度是否还成立。真正的干净 SOTA = pg19 nctx7（16k=16）。Plan C / 树形 / SnapKV-on-chunks 等读出方向仍有效（因为读出鸿沟在干净 run 里从没解决：HARDOBJ W0 8-15 vs swa6 50-60 同款 gap）。

---

## 0.1 在跑 eval / 等出炉的对照实验

| 节点 | run | eval 状态 | 关键变量 |
|---|---|---|---|
| 本机 | b50/c512 W0+W6 | **进行中**（qa2 32k） | 隔离 buffer_length=50（vs b25=25）：测 H2 假说 |
| .245.174 | b100/c512 W0+W6 | **进行中**（qa2 8k） | 隔离 buffer_length=100：测 H2 dilution 剂量曲线 |
| .7.53 | b25/c512 W6 | 排队（W0 已完成） | 测 W6/W0 gap 在破墙后是否消失 |
| B200.53 | b50/c1024 W0+W6 | W0 done(qa5 32k=8) / W6 done(qa5≥8k=0 崩) | 隔离 chunk_size=1024（vs c512）：测 H1 假说 |

**4 臂消融的科学意义已变**：原本是"对标 MemoryLLM"，现在是"哪个变量决定破墙"。

## 0.2 待启动（下一波诊断）
1. **b25 中间 ckpt 早评（step500/1000/1500/2000/2500）on .7.53** → 测过训退化铁律，确认 step3000 是否峰值（历史普遍 step500 是甜区）
2. **b25 LongBench / LongMemEval / LongEval** → 验证 BABILong 破墙是否迁移到真实长文档（pg19 nctx7 案例：BABILong+3 但 LongBench 仅 6.5）
3. （等 eval 全部出齐后）汇总 4 臂 W0 + W6 横向表，定位 chunk_size×buffer_length 平面上的破墙等高线

---

## 1. 关键结论（这几周用 ~50 个实验换来的，别重走）

### 1.1 读出鸿沟已坐实（★最重要）
- **SWA 对比证明**：learnmass_s250 `W0=12 < W1=36 < W2=43 < W4=60 < W6=64 < W8=68`，单调未饱和。
- memory bank 里**存了长程信息**，W0 读不出来，读出效率是真正瓶颈。
- **MECH三臂裁决（step250有效判据）**：learnmass（写入机制完美，dead_slot_read_mass 0.84→0.005），但 W0 长程 ≤ SOTA；combined（W2甜点回升但窗口敏感）；sharedaddr（崩）。
- **下游transfer全军覆没**：combined W0/W2/W4 RULER+LongEval全部~0（comma-spam）。SWA 增益不 transfer 到下游任务。

### 1.2 MemoryLLM baseline（方案A结果，已有168个CSV）
- qa1: 53/42/32/23/14/9/7（0k→32k）
- qa2: 36/35/19/16/15/16/16
- qa5: 47/50/45/39/39/38/**34**
- vs mem_space nctx63 SOTA：qa5 32k = **9** vs MemoryLLM **34**，差 3.8×
- 关键对比：qa2/qa5 MemoryLLM 碾压我们；qa1 基本打平
- **方案A eval 不需要再跑**（168 csv 已在 diskA）

### 1.3 MemoryLLM架构（从源码确认）
- `memory = nn.Parameter([L=32, 12800, 4096])`（num_blocks=50，num_tokens=256，~1B额外参数）
- 写入：inject_memory → forward → 取各层 hidden states → FIFO concat（drop_memory 随机丢1/num_blocks块）
- 读取：cat_memory_and_hiddens → **全量12800 tokens full attention**（无routing，无检索）
- position：memory tokens 用连续编号 0..12799，当前 chunk 接续

### 1.4 旧结论速查
- P11 chunk512 step500 = SOTA（qa5 0k-32k = 74/89/81/60/48/45/44），迄今无配置超过
- 过训单调退化铁律；路由集中是症状不是病根（ROUTE-A 四臂全 REJECTED）
- L3 summary 是长程主力；读出侧 v20 读基生命周期已证伪
- delta-rule写规则不是长程关键；真瓶颈=读出效率（读机制问题）

---

## 2. 当前在跑的实验（2026-06-25 12:10 更新）

| 节点 | 实验 | 状态 |
|---|---|---|
| B200.53 (8卡) | **方案B FIFO 训练 chunk1024/b50** | **✅ DONE step3000**，lm_final=3.849，297.1min |
| .245.174 (8卡) | **方案B FIFO 训练 chunk512/b100** | **✅ DONE step3000**，621.3min，Jun25 07:03 |
| 本机 (8卡) | **方案B FIFO 训练 chunk512/b50** | **✅ DONE step3000**，624.1min，Jun25 07:12 |
| .7.53 (8卡) | **方案B FIFO 训练 chunk512/b25** | **✅ DONE step3000**，622.0min，Jun25 07:07 |
| B200.53 (8卡) | **chunk1024/b50 W0 eval** | **✅ DONE**（结果见§0） |
| B200.53 (8卡) | **chunk1024/b50 W6 eval** | **进行中**（qa5 4k/32k 阶段，约还剩~1-2h） |
| 本机 (8卡) | **chunk512/b50 W0 eval** | **进行中**（qa1 32k 阶段，08:29 启动） |
| .245.174 (8卡) | **chunk512/b100 W0 eval** | **进行中**（qa1 32k+qa2 4k 并行，09:31 启动） |
| .7.53 (8卡) | **chunk512/b25 W0 eval** | **进行中**（qa5 32k 阶段，10:43 启动） |

**消融设计**：chunk_size × buffer_length 双轴
- chunk轴：c1024（B200）vs c512（其余3节点）
- buffer轴（@c512）：b25（.7.53）vs b50（.196）vs b100（.245.174）

---

## 3. 下一步待办（按优先级）

1. **【进行中】全部 W0 eval 完成后聚合结果**
   - 等 .245.174（b100）、本机（b50）、.7.53（b25）W0 eval 完成（预计 Jun25 14:00-15:00 GMT+8）
   - 用 `score_nested_babilong.py` 聚合 4 shards
   - 对比 MemoryLLM baseline，得出 chunk_size / buffer_length 对长程的影响

2. **【进行中】B200 W6 eval 完成后聚合**
   - W6 eval 在跑（预计 Jun25 13:00 GMT+8 完成）
   - 聚合后对比 W0 vs W6 差距（读出鸿沟是否依然存在于方案B）

3. **【等W0完成后自动启动】W6 eval for H20 三臂**
   - 三臂均已设置 driver script，W0 完成后自动串行启动 W6 eval
   - 日志：`logs/fifo_b{25,50,100}_c512_eval_W6.out`

4. **【低优先级】结果分析**
   - 汇总 4 臂 W0/W6 结果后，与 MemoryLLM baseline 和方案A结果对比
   - 分析：FIFO buffer 长度、chunk 大小 对 BABILong 0k-32k 的影响曲线
   - 如果 W0 表现仍远低于 MemoryLLM → 确认读出鸿沟在方案B依然存在 → 转向方案C（蒸馏）

5. **【低优先级】git push**
6. **【低优先级】方案C（蒸馏）**：MemoryLLM teacher + mem_space student

---

## 4. OOM 修复日志（2026-06-24，重要历史）

| 臂 | 节点 | 问题 | 修复方案 | 状态 |
|---|---|---|---|---|
| b50/chunk1024 | B200.53 | lm高方差（1.9~6.3） | 非OOM，正常现象，不干预 | ✅ 训练完成 |
| b50/chunk512 | .196/本机 | OOM@step30（optimizer.step, 94GB, unfreeze_from=16）+ HF dataset lock | 改 unfreeze_from=16→24 + 清除 .hf_cache/*.lock | ✅ 训练完成 |
| b25/chunk512 | .7.53 | 脚本已是 unfreeze_from=24（之前改好），无OOM | — | ✅ 训练完成 |
| b100/chunk512 | .245.174 | OOM@step30（optimizer.step, 94GB, unfreeze_from=16） | 改 unfreeze_from=16→24 重启 | ✅ 训练完成 |

---

## 5. 集群 & 运维（关键，否则踩坑）

- **节点清单**：本机(盘A) + .196(盘A共享) + .7.53/.245.174(回归H20,盘B) + .76/.249(盘B) + B200 .53/.18/.188(wzc1盘)
- **密码文件**：`configs/password_diskA.txt`(.196) / `password_h20_returned.txt`(.245.174 **和** .7.53) / `password_h20_new2.txt`(.76/.249) / `password_b200_53.txt`(B200 .53) / `password_b200_new.txt`(B200 .18:36000) / `password_b200_188.txt`(B200 .188)
- **.7.53 和 .245.174 共用同一密码**：`configs/password_h20_returned.txt`（2026-06-24 用户确认）
- **B200.53 SSH 端口 36000**：`sshpass -f configs/password_b200_53.txt ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -p 36000 root@28.88.184.53`
- **PYBIN**：本机/.76/.249/B200全部 `.venv/bin/python`；.196 **必须用 `.venv/bin/python`**
- **盘B项目根**：`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`（注意304376610）
- **wzc1盘项目根**：`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`
- **eval脚本**：标准eval用 `scripts/_eval_taskpool_2group.sh`（2-组 task-pool 动态调度）
- **训练红线**：`eval_interval=0`（内联eval会NCCL崩）
- **H20 OOM 教训**：unfreeze_layers_from=16 在 H20 95GB 上 optimizer.step() OOM@step30；H20 上必须 unfreeze_from≥24
- **⚠️ HF dataset lock 陷阱**：OOM 崩溃后 `.hf_cache/datasets/*.lock` 残留，再启进程静默退出，必须先清

---

## 6. 关键文件

- `status/RUN_REGISTRY.md` §3/§4 — 所有实验配置+结果+裁决总账（最权威）
- `status/BENCHMARK_RESULTS.md` — 结果汇总（含外部论文参考数字）
- `status/TRAINER_ACTIVITY.jsonl` 尾部 — 每轮巡检流水
- `/apdcephfs_zwfy6/share_303098609/pighzliu_code/MemoryLLM-source/modeling_memoryllm.py` — MemoryLLM 源码
- `src/memory/mem_space/layer.py` — mem_space 核心层（27627 tokens）
- `src/memory/mem_space/config.py` — MemorySpaceConfig dataclass
- `src/memory/mem_space/patch.py` — patch 进 Llama 的入口
