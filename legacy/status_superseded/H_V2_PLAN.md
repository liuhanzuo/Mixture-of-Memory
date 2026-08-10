# H_V2_PLAN.md — 当前执行面主计划（FIFO memory / W0-W6 gap 阶段）

> **本文件 = heartbeat 每轮必读的 live 主计划。** 由 main agent / heartbeat 持续覆盖维护。
> 旧的 "H-series v2" 内容已废弃（那是几个月前的方向）。当前阶段 = FIFO hidden memory + 数据泄漏纠正 + W0/W6 gap 攻坚。
> 最后更新：2026-06-25 23:35 GMT+8（fresh-agent 接手 + 用户 5 点指令）

---

## 0. 用户 5 点指令（2026-06-25，最高优先级，冲突时压倒一切旧文本）

1. **泄漏 run（babilong_mix>0）= 完全不碰**（2026-06-26 用户强调收紧）：不在泄漏 ckpt(b50/b100/P2/c1024 等)上跑任何新 probe/eval，不引用其分数，**不拿它做任何"对照/上界"** —— 泄漏数字不是参照系，是噪音。所有判据只在干净（mix=0）ckpt 上做。（教训：在 b50 上跑 oracle 得 78、又花一轮"修正"，全是自找；干净 NOLEAK oracle 才 20-24。）
2. **之后所有训练必须 `--babilong_mix_fraction 0`**，绝不掺任何 babilong 数据。违反 = red line。
3. **梳理干净 babilong 结果**：memory-slots 历史最佳 vs 当前 FIFO buffer 到了多少 → 定方向。（researcher a9b7e4ee 进行中，产出 `status/CLEAN_SOTA_SURVEY_20260625.md`）
4. **heartbeat 每轮**：读本文件，有空闲卡按计划自主推进（auto_launch）。
5. **full autonomy**：遇问题自主调研论文 / 改代码 / 起实验 / eval / 派 subagent，不问用户，形成闭环。

---

## 1. 一句话现状
### ★★★ 最新结论 (2026-06-27 04:00) — raw-hidden 到顶, 活token重forward是出路
**完整排除链(全部干净 mix=0 ckpt, 泄漏数字已弃):**
- **raw-hidden FIFO 到顶 ~20**: 干净 NOLEAK oracle(完美隔离 needle 的 hidden 快照) qa1=21/20/24/22, qa5=24/18/17/10, 仅比基线(qa1 16/12/8/2)温和正向。hidden-recall logit-lens: needle hidden 对答案 L31 rank=21(random 44)=**"存了但弱/不sharp"**。
- **位置证伪**: ArmC(训练时 real 位置)长档 ≈ 基线 → 位置坍缩(layer.py:1327)是次要, 非主因。
- **rawkv 撞同墙**: 历史干净 rawkv 长档全崩(DILUTION + 也存 hidden 快照)。
- **★活 token 重 forward = 出路**: oracle-token-SWA(选中 chunk 的**原始 token 重 forward**, 非 hidden) qa1 8k=50/16k=28(hidden-oracle 仅 20/24), qa5 8k=20/16k=12 → 单跳大突破, 多跳温和正向。
- **机制根因**: hidden 是"没 attend query 的冻结快照"; token 重 forward 让 needle 重新 attend query + 跨层重算 = 真正解法。
- **新方向 = token-reforward + selection**(researcher a106e5e 设计可部署版中)。瓶颈转移到 chunk 选择精度(oracle 完美选 50 vs reader-attn 近似选低)。
- ArmB(packed)/keepset 训练已停(raw-hidden 路线降级)。ArmA(格式对齐基线)保留跑完。



FIFO 方案B（per-layer hidden FIFO buffer，full-attention readout，RoPE 坍缩到 pos-0）。b25"破墙"已查实 ~85% 是 BABILong 数据泄漏。**真实干净长程天花板 = pg19 nctx7 qa5 16k=16/32k=9**（researcher 已复核确认）。核心未解 = **W0/W6 gap**（纯 memory 读出远差于给原始 token，证明 FIFO hidden 表示有损；头号嫌疑 = pos-0 坍缩 layer.py:1327）。

### ★ CLEAN_SOTA_SURVEY 关键结论（researcher a9b7e4ee, 2026-06-25, 见 status/CLEAN_SOTA_SURVEY_20260625.md）
- **全项目 ZERO 干净 FIFO 数据** —— 所有 FIFO 数字(b25/b50/b100/c1024)全泄漏。本地普查 43 干净 vs 84 泄漏 run。**NOLEAK b25 W0 = 项目史上第一个诚实的 FIFO 测量。**
- **干净 SOTA 锚点确认**：slots-family 最佳 = pg19 nctx7 qa5 W0 = 75/73/51/29/**19/16/9**；MemoryLLM teacher qa5 = 47/50/45/39/39/38/**34**。
- **泄漏量级铁证**：HARDOBJ last-chunk(干净,机制≈b25) qa5 8k=13-15/16k=11-14/32k=8-9；泄漏 b25 同机制=65/76/68 → 泄漏贡献 +50/+62/+59(4.5-7.5×)。**预测 NOLEAK b25 落入 HARDOBJ/pg19 簇**。
- **方向排序**：①reader-attn keep-set / SnapKV-on-chunks(中-高,唯一有正证据:needle 隔离时 dilution 0%→97.5%,reader q·k precision 55%=8.8×随机) ②position-fix 重训(中) ③prediction-CE 压缩(低) ④HNST tree(低-中,#1 的升级)。
- **W0/W6 gap 量级**：干净 run 稳定 3-4×(W0 8-19 vs swa6 50-60 @8k-16k)。FIFO 本不该有此 gap(buffer 含 W6 chunk)→ hidden 有损,pos-0 坍缩(layer.py:1244/1308-1326)是头号嫌疑。
- **推荐下一实验**：Step A = NOLEAK ckpt 落地即跑 5-probe(零训练,判位置 vs dilution)；Step B = 赢家方向 mix=0 重训。成功线 = 干净 W0 qa5 16k>16 且 32k>9。

---

## 2. 节点占用表（每轮 heartbeat 实测刷新，勿假设）

| 节点 | 地址 | 盘/root | 当前任务 | 状态(23:35) |
|---|---|---|---|---|
| 本机 | localhost | diskA `/apdcephfs_zwfy6/share_303098609/...` | b50/c512 W6 eval | 跑(7-8/8) |
| .196 | 28.59.80.196 | diskA(共享FS) | b50 ckpt 早评 W0 | 跑(8/8) |
| .7.53 | 28.48.7.53 | diskB `/apdcephfs_zwfy6/share_304376610/...` | **NOLEAK b25 step3000 训练** | step~780/3000 babi=0, ETA~07:00 |
| .245.174 | 28.58.245.174 | diskB(共享FS) | **P1 packed probe(GPU0,3,4,5) + NOLEAK-500 W0(GPU1,2,6,7)** | 跑(8/8) |
| B200.53 | 28.88.184.53 | wzc1 `/apdcephfs_wzc1/share_304376610/...` | c1024 5ckpt 过训曲线 eval | 跑 |

SSH: `sshpass -f <pw> ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o PreferredAuthentications=password root@<IP>`
密码: diskA=`configs/password_diskA.txt`, diskB=`configs/password_h20_returned.txt`, B200.53=`configs/password_b200_53.txt`。各节点用自己 root 下 `.venv/bin/python`。

---

## 3. 当前状态（2026-06-26 12:15 更新）+ 完成后自动动作

### 3.0 ★★ NOLEAK b25 step3000 训练已完成（06:58）→ 第一个诚实 FIFO 结果已出
**NOLEAK3000_W0（干净，史上第一个诚实 FIFO）qa5 W0 = 21/64/48/~12/~12/(跑)/(跑)，qa1=0/37/17/16/12/8/2，qa2=0/24/7/10/8/3/2。**
- **结论：泄漏假说彻底坐实。** 干净 qa5 8k≈12-25 vs 脏 b25 8k=65 → b25"破墙"~95%泄漏，干净 FIFO≈pg19 SOTA(8k:~20 vs 19)，未超 slots。
- **★发现格式 bug（非纯能力）**：干净模型在 babilong 上复读 PG19 语料`(Score:5,Insightful)by Anonymous Coward`、0k 复读 few-shot example → qa1/qa2 0k=0。根因：mix=0 训练没见过 babilong QA 格式。**NOLEAK 低分混了"记忆差"+"格式没对齐"两部分 → 需合法 task-alignment(T2)。**

### 3.1 在跑 eval（diskB，2 节点合写 noleak_b25_step3000_W0 目录）
- .7.53 原始 W0 driver(pid796534)跑 qa5 8k+32k 长尾；.245.174 resume 跑 qa5 16k + **NOLEAK3000_packed**(GPU4-7,干净 H_POS 判据)。无重复。
- ⚠️ packed 干净 ckpt qa1 4k=2 << plain W0 16 → **eval-time packed 与 pos-0 训练权重失配，H_POS 只能靠训练时用 packed 测**（见 §4 训练）。

### 3.2 其余节点
本机 b50 W6 eval；.196 b50 早评；B200.53 c1024 过训(剩~72)。

---

## 4. PENDING 实验队列（auto_launch 规则）

**节点空出时按此队列推进。所有训练强制 `--babilong_mix_fraction 0`。**

### ★ 给独立 daemon heartbeat 的自包含启动方法（fresh session 必读，不依赖 /tmp）
staging 脚本持久存于 diskA repo：`scripts/_stage_fifo_probes.sh`。diskB 节点用前先 scp 过去（diskA/diskB 不共享 FS）：
```bash
# 从本机(diskA)推送 staging 脚本到 diskB 节点，再远程执行某个 probe：
PW=configs/password_h20_returned.txt; IP=28.58.245.174   # 或 .7.53=28.48.7.53
sshpass -f $PW scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
  scripts/_stage_fifo_probes.sh root@$IP:/tmp/stage_probes.sh
sshpass -f $PW ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@$IP \
  'bash /tmp/stage_probes.sh <PROBE> "<GPU列表>"'
```
PROBE ∈ {P2,P3,P4,P5,NOLEAK3000_W0,NOLEAK3000_W6,NOLEAK3000_packed,NOLEAK3000_keepset}。
脚本内已硬编码 ckpt 路径/flag/lengths，只需传 PROBE 名 + 空闲 GPU 列表（NSHARD 自动=GPU数,≤4）。
（B200.53/wzc1 无 b25/b100 ckpt，probe 只能在 diskB 节点跑。）

| 优先级 | 任务 | 触发条件 | auto_launch | PROBE 名 |
|---|---|---|---|---|
| **P0-train** | **★T2-align 训练(2 臂)**：解决格式没对齐 + 测训练时 position-fix。**任一 diskB 节点空出 8 卡即发**。脚本已部署+code已同步diskB(md5校验). | 任一 diskB 节点 8 卡空闲 | **true** | 见下方训练启动块 |
| P0 | **NOLEAK3000_packed**（干净 H_POS）| 在跑(.245.174 GPU4-7) | — | running |
| P1 | **P2 reader-attn keep-set@b100**（survey #1 方向，零训练）| diskB 节点空 ≥4 GPU 但 <8（不够训练）| true | `P2` |
| P1 | **P3 packed+keepset@b25** | 节点空 | true | `P3` |
| P2 | P4 real-pos@b25 / P5 keepset-top10@b100 | 节点空 | true | `P4`/`P5` |

### ★ T2-align 训练启动块（diskB，需整 8 卡，babilong_mix=0 + 合成 needle）
脚本 `scripts/_launch_t2_posfix.sh`（已部署 diskB），训练脚本已加 `--fifo_pos_mode` 支持(commit 5a2449b, md5校验diskA=diskB)。
**两臂设计（隔离格式对齐 vs 位置修复）**：
- **Arm A** = T2-align only（pos-0 readout）：`bash scripts/_launch_t2_posfix.sh none 29803` → run=`mem_space_fifo_b25_chunk512_t2align`
- **Arm B** = T2-align + 训练时 packed 位置（测 H_POS）：`bash scripts/_launch_t2_posfix.sh packed 29804` → run=`..._t2align_packed`
启动方法（在空出的 diskB 节点上,IP=.7.53 或 .245.174）：
```bash
PW=configs/password_h20_returned.txt; IP=<空节点IP>
sshpass -f $PW ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o PreferredAuthentications=password root@$IP \
  'cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory && bash scripts/_launch_t2_posfix.sh none 29803'
```
**先发 Arm A**（最稳，纯格式对齐效果）；若有第二个 8 卡节点空出，发 Arm B。ETA ~10h/run(0.08 steps/s × 3000)。脚本内置 pgrep 防重复启动。完成后自动 W0+packed eval（同 NOLEAK 流程）。成功线：干净 W0 qa5 8k 显著>25 且格式 bug 消失(0k 正常出分)。

详细 probe 背景见 `status/PENDING_TASKS.md` 末尾 + `status/CLEAN_SOTA_SURVEY_20260625.md`。

---

## 5. 决策树（heartbeat 据此自主走）

```
每轮: 实测5节点 → 查 GPU 抢占(per-proc CUDA_VISIBLE_DEVICES,撞了kill低优先级) → 打分新落地 cell → 记流水
  │
  ├─ NOLEAK step3000 出了? → 立即 W0 eval → 出分后立即 NOLEAK ckpt 上跑 5-probe(干净判据)
  ├─ P1/NOLEAK-500 出 qa5 8k-32k? → 派 researcher 判 H_POS + 泄漏占比 → 写结论入本文件
  ├─ 任一节点空 + auto_launch:true 任务? → 启动(probe 零训练直接起;训练必须 mix=0)
  ├─ 节点空 + 无任务? → 派 researcher 产出下一实验
  └─ researcher/coder 出高置信结论? → 同轮推进到启动,不停在"已分析"
```

---

## 6. Red Lines（本阶段强化）

- ❌ 任何训练带 babilong mix（必须 `--babilong_mix_fraction 0`）。
- ❌ 把泄漏 run 的 0k-4k 当能力结论 / 当干净 SOTA 锚点（P11/b25 都泄漏）。
- ❌ kill 进程不先查 per-proc CUDA_VISIBLE_DEVICES 确认是抢占/孤儿。
- ❌ 报告 BABILong 不标注泄漏/干净。
- ✅ 允许：自主 kill GPU 抢占的低优先级 job、自主起 probe/eval、自主派 researcher/coder、自主改代码修 bug。

---

## 7. 关键文件

- 代码：`src/memory/mem_space/layer.py`（_forward_fifo :1214-1512, pos-0 坍缩 :1327, probe 助手 :1518-1622）；`scripts/run_babilong_mem_space.py`（probe flag 解析 :758/826-858, 应用 :990-1008）；`scripts/_eval_taskpool_2group.sh`（GROUP/NSHARD override）；`scripts/score_nested_babilong.py`。
- 设计：`versions/v_prediction_not_reconstruction_2026-06-25.md`、`versions/vN_HNST_tree_hidden_memory_2026-06-25.md`、`ops/research_notes/fifo_dilution_eviction_litreview_20260625.md`。
- 状态：`status/RUN_REGISTRY.md`(总账)、`status/PENDING_TASKS.md`(probe 命令)、`status/HARDOBJ_FINAL_REPORT.md`(干净对照)、`status/CLEAN_SOTA_SURVEY_20260625.md`(researcher 产出中)。
