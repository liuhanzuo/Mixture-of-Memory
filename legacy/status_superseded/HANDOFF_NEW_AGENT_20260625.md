# 新 Agent 交接文档 — Mixture-of-Memory(2026-06-25 23:00 GMT+8)

> **读者**:一个对本项目零上下文的全新 agent。
> **你的任务**:读完本文档 → 按"第 0 步"去验证当前实验状态 → 按"待办优先级"接着推进。
> **第一原则**:本项目今天(2026-06-25)发生了一次**重大方法论纠错**(数据泄漏),很多历史"成果"已作废。**不要相信任何含 BABILong 训练数据的 run 的 0k-4k 分数。** 详见第 2 节。

---

## 第 0 步:启动必做(按顺序)

1. **读这几个文件**(权威,且比本文档更新得勤):
   - `status/SESSION_HANDOFF.md` §0 —— 一句话现状(我每次重大变化都覆盖更新它)
   - `status/TRAINER_ACTIVITY.jsonl` **尾部 40 行** —— 时间序列流水,有所有 ★ 标记的关键事件
   - `status/RUN_REGISTRY.md` 顶部 —— 最新结果总账
   - `status/PENDING_TASKS.md` **末尾** —— 待办任务(尤其 probe 矩阵,带启动命令)
   - 本项目根 `CLAUDE.md` —— 集群/SSH/规则(权威运维信息)

2. **验证 5 个节点当前状态**(命令见第 5 节)。本文档写于 2026-06-25 23:00,你接手时实验已推进,**必须实测当前状态,不要假设**。

3. **检查两个关键实验是否出结果**(见第 4 节):NOLEAK b25 训练 + 5-probe 矩阵。如果出了,先做第 3 节的"裁决分析"。

---

## 第 1 节:项目是什么

研究 **LLM 长程记忆**:让 Llama-3-8B(32 层,hidden 4096,vocab 128256)在处理几万 token 超长文档时,仍能准确回忆开头的信息。

**评测**:BABILong(`RMT-team/babilong`),任务 qa1/qa2/qa5 × 长度 0k/1k/2k/4k/8k/16k/32k,n=100/cell,babilong.metrics(`compare_answers`)。
- **W0** = 纯 memory 读出(`--swa_eval_chunks 0`):前 N-1 个 chunk 流式进 memory,生成只喂最后一个 chunk(含 question)。**这是主交付指标。**
- **W6** = eval 时额外给最后 6 个 chunk 的原始 token 做 SWA(`--swa_eval_chunks 6`)。是"开卷拐杖",不是纯 memory,用来诊断 memory 里存了多少信息。

**当前主架构 = 方案B FIFO memory**(对标 MemoryLLM):
- 每个 chunk forward 后,把各层 hidden states 存进 FIFO buffer(per-layer list,先进先出)
- buffer 满(`fifo_buffer_chunks` 个 chunk)后淘汰最老的
- 读出 = 当前 chunk 对整个 buffer 做 full attention(`src/memory/mem_space/layer.py` 的 `_forward_fifo`,约 :1193-1362)
- **b25/b50/b100** = buffer 25/50/100 个 chunk;**chunk512/chunk1024** = chunk_size
- ⚠️ **关键架构事实**:FIFO 存的是**原始 hidden(1:1 无压缩)**,且把 buffer 所有 token 的 RoPE 位置**压到 pos-0**(`layer.py:1244`)——这是个大问题,见第 3 节。

**对照基线 MemoryLLM**(我们想超越的目标):qa5 0k-32k = 47/50/45/39/39/38/**34**。

---

## 第 2 节:★★★ 今天最重要的发现 —— 全项目数据泄漏(必读)

### 2.1 泄漏机制(代码 + 训练日志双重坐实,非推测)

`scripts/train_mem_space_dolmino_cpt.py` 的 `--babilong_mix_fraction` **默认 = 0.15**:15% 训练步混入 BABILong SFT。
- 训练默认 `--babilong_tasks=qa1,qa2,qa5`(= eval 任务)、`--babilong_lengths=0k,1k,2k,4k`(= eval 的 0k-4k)、`--babilong_dataset=RMT-team/babilong`(= eval 同一数据集)。
- `src/memory/mem_space/babilong_dataset.py:79` `BABILongTrainDataset` 用 `load_dataset(name,length)[task]` —— **该 HF 数据集每个 length 只有 task split,无 train/test 隔离 → 训练样本池与 eval 完全重叠**。
- `max_seq_len = chunk_size*4 = 2048`:0k/1k 整故事完整泄漏,2k/4k 部分泄漏。
- **几乎所有用这个训练脚本、且没显式传 `--babilong_mix_fraction 0` 的 run 都被污染**,包括历史 SOTA **P11**(训练日志 `logs/mem_space_p11_chunk512_deltarule_normreadout.log` line 255 `babilong_mix=0.15`,line 6178 完成统计 `babilong=3034`)。

### 2.2 哪些可信、哪些作废

| 范围 | 可信度 |
|------|--------|
| 任何泄漏 run 的 **0k-4k** BABILong 分数 | ❌ 作废(背了测试答案) |
| 泄漏 run 的 **8k-32k** | ⚠️ 训练 max_seq_len=2048 不覆盖,是 OOD 相对干净,但 babilong SFT 教的"QA 格式/答案提取技能"可能 transfer 到长档 → **也存疑,正在用 NOLEAK 验证** |
| **babilong=0 的干净 run** | ✅ 可信 |

### 2.3 怎么判断一个 run 是否泄漏

查它的训练日志完成行:`grep "Training complete" logs/<run>.log` → 看 `babilong=N`。`babilong=0` 才干净。
或查 launch 脚本是否有 `--babilong_mix_fraction 0`。

### 2.4 b25 "破墙"是泄漏幻觉(已基本坐实)

今天早些时候 b25/c512 W0 跑出 qa5=100/100/97/87/65/76/68(32k=68,看似 2× 超越 MemoryLLM 的 34),一度以为破墙。**但**:
- 0k=100 史无前例(历史最高 ~93,P11 才 74)= 泄漏指纹。
- **历史所有 babilong=0 干净 run,qa5 8k 从没超 ~19、32k 从没超 ~9**:
  - pg19 nctx7(干净 SOTA)= 75/73/51/29/**19/16/9**
  - **HARDOBJ_lastchunk**(干净 + `--last_chunk_loss_only`,**几乎和 b25 一样的 last-chunk memory 机制**)= qa5 8k 12-15/16k 11-14/32k 8-9
- 同机制下,干净 = 8-15,泄漏 b25 = 65-68 → **b25 长档高分约 85% 来自泄漏,不是 FIFO 架构能力**。
- **真实干净 SOTA 仍是 pg19 nctx7 的 16k=16, 32k=9。**

### 2.5 方法论铁律(写给你)

- **新训练一律 `--babilong_mix_fraction 0`**(除非明确要复现旧泄漏 run 做对照)。
- 报告 BABILong 结果**必须标注是否泄漏**,长档(8k-32k)比 0k-4k 更可信。
- 建立任何"基线"用干净 run(NOLEAK / pg19 / HARDOBJ),**不要再拿 P11 当干净 SOTA 锚点**。

---

## 第 3 节:★★ 核心科研问题(用户今天 sharpen 的)+ 当前假说

### 3.1 W0/W6 gap = FIFO hidden 表示有损(头号问题)

**现象**:W6(加 SWA 原始 token)远好于 W0(纯 memory)。干净 run 里 gap 巨大:HARDOBJ W0 8-15 vs swa6 50-60(3-4×);self-study qa5 swa0 16/11/7 → swa2 ?/26/18。

**用户的关键 reframe(比项目旧框架更准)**:FIFO buffer **本来就存了**最后那几个 chunk 的 hidden(b25 在 32k 留最后 25 chunk,W6 窗口的最后 6-7 个全在 buffer 里)。**既然 buffer 就是那些 chunk,理论上 W0 该 = W6,不该有 gap。gap 的存在 = 我们的 hidden 表示是坏的/有损的**,不是 SWA 占了"开卷"便宜。

**头号嫌疑 = 位置坍缩**:`layer.py:1244` 把 buffer 所有 hidden 压到 RoPE pos-0 → 无序无位置的一袋向量;而 W6 的原始 token 有正常相对位置。模型当然偏好有位置的版本。
**次因**:hidden 是 chunk 当"当前 chunk"时算的,**没见过 query**(staleness)。

**研究目标重定义**:**关闭 W0/W6 gap = 让 FIFO hidden 和原始 token 一样好用。**

### 3.2 位置方案(用户提的难点:稀疏选择时 hidden 不连续)

- 位置是**已知的**:chunk i = 原始 token 位置 [i·512,(i+1)·512)。buffer 记下 chunk index 就知道真实位置。RoPE 是**相对**编码,位置稀疏/有 gap 没问题。
- 两种方案(已在代码里实现成 flag,见第 4 节 probe):
  - **packed(重打包保序)**:选中 chunk 按在 keep-set 里的序号 0,1,2… 重编位置。保**顺序**,丢绝对距离,in-distribution(StreamingLLM 式,arXiv:2309.17453)。
  - **real(真实稀疏)**:用原始 chunk_idx 位置,保绝对距离,但 OOD(32k 超训练 8192,靠 θ=500000 外推)。
- qa5 是时序推理(谁先给谁),**顺序必须保**,绝对距离不一定。预期 packed 够用且更稳。

### 3.3 prediction ≠ reconstruction(用户洞察,改写压缩范式)

memory 目标是 **prediction(下 token 预测)**,不是 reconstruction(重建原文)。所以:
- hidden **可以激进压缩**(大部分维度对 prediction 无贡献),不必保 1:1。
- 压缩训练目标应是 **LM CE**,不是 MSE reconstruction。
- 历史 L3 token-recon aux 全 REJECTED 可能就是**用错了 loss**(recon 与 prediction 冲突)——值得用 prediction-CE 重试。
- 详见 `versions/v_prediction_not_reconstruction_2026-06-25.md`。

### 3.4 抗 dilution(文献 + 树形设计收敛的方向)

- **H2 dilution 发现**:FIFO buffer 越小,长档越好(b25>b50>b100 在 8k-32k)。机制:full attention over buffer,needle 的 25 token 被海量 distractor 在 softmax 里淹没;buffer 小 = distractor 少 = 抗稀释("隐式 isolation")。⚠️ 但这是泄漏尺度上的,需 NOLEAK 确认在干净尺度是否还成立。
- **缺陷**:b25 隐式 isolation 是"丢最老",若 needle 在文档前段就丢了。
- **解药(下一方向)**:用 **reader-native q·k attention** 选 chunk(不是丢最老,是丢"注意力分最低的"),保留全部 chunk 在 buffer 但只 attend 选中的少数。reader-native q·k 是项目里最好的**无训练**选择器(needle precision 55%,8.8× 随机;所有 trained selector 都崩到随机)。
- 文献定位:这是 **H2O(2306.14048)/SnapKV(2404.14469)/ChunkKV(2502.00299)** 在 "训练 hidden FIFO 的 chunk 级" 的应用,文献无此精确组合。完整文献调研在 `ops/research_notes/fifo_dilution_eviction_litreview_20260625.md`。
- **树形升级(HNST)**:把 chunk 组织成 B 叉树,reader-attn 分层 1-of-B 导航(避开 1-of-64 flat selector 死路),叶子 raw hidden(回答)、内部节点 slot 压缩(导航)。设计在 `versions/vN_HNST_tree_hidden_memory_2026-06-25.md`。

---

## 第 4 节:当前在跑 + 待推进的实验

### 4.1 写于 2026-06-25 22:00 的在跑实验(你必须实测当前状态!)

| 节点 | 实验 | 当时状态 |
|---|---|---|
| .7.53 | **NOLEAK b25 训练**(babilong_mix=0,同 b25 配置) | step ~215/3000,babi=0,ETA 明早 ~07:30。output_dir=`outputs/mem_space_fifo_b25_chunk512_noleak`(在盘B) |
| 本机 | b50/c512 final W0 eval | qa5 32k 长尾收尾 |
| .196 | b50/c512 step500+1000 中间ckpt W0 eval | 进行中 |
| .245.174 | b100/c512 final W0 eval | qa5 32k 收尾 |
| B200.53 | c1024 5中间ckpt W0 eval(`fifo_b50_c1024_overtrain_FIXED`) | 进行中 |

### 4.2 ★最高优先级:NOLEAK b25 W0 eval(task #7)

NOLEAK 训练完成后(明早),**立即 W0 eval** 对比脏 b25。这是判定"8k-32k 是不是 babilong SFT 撑起来的"的决定性实验。
- 用户假说(我也认同):NOLEAK 8k-32k 会**掉很多**(掉进干净簇 8k≈15-25/32k≈8-12),证明 b25 长档高分主要靠 babilong。
- 启动:在 .7.53(盘B,持 ckpt 零 rsync)`_eval_taskpool_2group.sh`,CHUNK_SIZE=512,W0,n=100,ckpt=`outputs/mem_space_fifo_b25_chunk512_noleak/full_model.pt`。

### 4.3 ★★★ 5-probe 矩阵(commit eddb4f1,零训练,现有 ckpt)

代码已实现并部署到 3 个盘(盘A/盘B/wzc1)。新 eval flag:
- `--fifo_pos_mode {none,packed,real}`:位置方案
- `--fifo_keep_set_mode {none,flat_readerattn}` + `--fifo_keep_topk 25 --fifo_keep_recency 2`:reader-attn 选 chunk
- `--fifo_keep_all_buffer`:eval 时不淘汰 buffer

5 个 probe(详细启动命令在 `status/PENDING_TASKS.md` 末尾):

| # | ckpt | flag | 测什么 | 预期 |
|---|------|------|--------|------|
| P1 | b25 | `pos_mode=packed` | H_POS:位置是否 gap 主因 | 8k-32k W0 跳升 → 位置是主因 |
| P2 | b100 | `keep_set=flat_readerattn keep_topk=25 keep_all_buffer` | H_DIL:reader-attn 选 chunk | 32k 从 5→30+ → dilution+选择有效 |
| P3 | b25 | `pos=packed keep_set=flat_readerattn keep_all_buffer` | 叠加 | ≈ W6 → 彻底关闭 gap = 顶级突破 |
| P4 | b25 | `pos_mode=real` | real vs packed | ≈packed→只需保序;<packed→real OOD |
| P5 | b100 | `keep_set top-10` | top-k 敏感度 | 找甜区 |

**⚠️ probe 是在泄漏 ckpt 上跑的**,所以看的是"加了 flag 后相对同 ckpt baseline 的**变化量**",不是绝对分数。绝对分数仍含泄漏。等 NOLEAK ckpt 出来后,在干净 ckpt 上重跑 probe 才是最终判据。

### 4.4 其他就绪待发射(脚本已 commit)

- **T2-align 训练**(`scripts/launch_mem_space_fifo_b25_chunk512_T2_diskB.sh`,commit d03db24):babilong_mix=0 + 合成 needle(T2)task-alignment。验证"用独立 needle QA 做 post-training 能否合法提升 held-out BABILong"。
- **b10/b5 H2 外推**(commit 5285f8d):更小 buffer,测 dilution 假说外推。有 leaked 版(同口径对比)和 NOLEAK 版。

---

## 第 5 节:运维(集群 / SSH / eval 命令)

### 5.1 节点(权威见 CLAUDE.md 顶部集群表)

| 节点 | 地址 | 密码文件 | 盘 / 项目根 |
|---|---|---|---|
| 本机 | localhost | — | 盘A `/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory` |
| .196 | 28.59.80.196 | `configs/password_diskA.txt` | 盘A(与本机共享 FS) |
| .7.53 | 28.48.7.53 | `configs/password_h20_returned.txt` | 盘B `/apdcephfs_zwfy6/share_304376610/...` |
| .245.174 | 28.58.245.174 | `configs/password_h20_returned.txt` | 盘B(与 .7.53 共享 FS) |
| B200.53 | 28.88.184.53 (端口 22) | `configs/password_b200_53.txt` | wzc1 `/apdcephfs_wzc1/share_304376610/...` |

- Python:全部用各自项目根的 `.venv/bin/python`。
- SSH 模板:`sshpass -f <pwfile> ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o PreferredAuthentications=password root@<IP> "<cmd>"`
- **盘间不共享**:盘A 改代码后,要在盘B/wzc1 跑得 rsync 过去(代码已同步过 commit eddb4f1)。

### 5.2 查节点状态

```bash
# GPU 占用 + eval/train 进程数
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps -eo cmd | grep -E "run_babilong|train_mem|torchrun" | grep -v grep | wc -l
# eval 进度
tail -5 logs/<run>.out          # 看 GROUP0/1 -> ck0 qaX 进度;SCHED_DONE = 完成
# 训练进度
grep "\[step" logs/<run>.log | tail -2     # 看 step/total, lm, babi(=0 才干净), nf(=0 才健康)
```
- ⚠️ 长档(qa5 32k)单 cell 要跑 1-4h,**GPU 部分卡空闲 + 日志几小时没新行 ≠ stall**,通常是 task-pool 长尾。判 stall 看进程 STAT(R=running)+ pcpu。

### 5.3 标准离线 eval(必用这个脚本)

`scripts/_eval_taskpool_2group.sh`(2-组 task-pool 动态调度),通过环境变量传参:
```bash
RUN_PREFIX=xxx CKPT_FILES="path/full_model.pt" CK_NAMES="xxx" \
ADAPTER_CONFIG=path/adapter_config.json CHUNK_SIZE=512 \
EXTRA_ARGS="--swa_eval_chunks 0" \           # W0; W6 用 6; probe 加 --fifo_pos_mode 等
PROJECT_ROOT=<节点root> PYTHON_BIN=<节点root>/.venv/bin/python \
setsid nohup bash scripts/_eval_taskpool_2group.sh > logs/xxx.out 2>&1 &
```
聚合分数:`<root>/.venv/bin/python scripts/score_nested_babilong.py babilong_results/<run> --expect 100`(每 cell 4 shard × 25 = 100,看完整性)。

### 5.4 GPU 监控前端(本机 8088)

`curl -s -m 8 -o /dev/null -w "%{http_code}" http://127.0.0.1:8088/api/data` → 200 OK。挂了用 Bash `run_in_background=true` 重启(**不要 setsid/nohup**,会被 sandbox 隔离到独立 netns):`cd <root> && .venv/bin/python -u monitor/gpu_monitor_server.py --port 8088 --interval 5`。

### 5.5 落账(每次操作后)

- `status/TRAINER_ACTIVITY.jsonl` append 一行(时间序列流水,append-only,写错追加 correction 不要 edit)
- 重大状态变化覆盖更新 `status/SESSION_HANDOFF.md` §0
- 新 eval 结果 append `status/RUN_REGISTRY.md`
- git commit:committer=LiuHanzuo,**不加 AI 署名**,`git add <具体文件>` 不用 `git add .`(防止提交 *.pt / 密码)。WANDB key 明文在脚本里是既有现状,不重要,可忽略。

---

## 第 6 节:死路(别重走,会浪费算力)

| 死路 | 结论 |
|------|------|
| trained gist/slot selector(flat 1-of-64) | 在 512-chunk/32k scale 崩到随机精度。**用 reader-native q·k(无训练,55%)代替** |
| MemoryLLM 蒸馏 | teacher 自己 qa5 32k 才 34,是 floor 不是 ceiling,蒸它没用 |
| raw-KV grafting / evidence-injection | 给对证据也只 +1~2.5pt(frozen reader 用不上) |
| 单层 readout(只在 L16 注入) | 长程 = 0%。**读出必须多层分布式** |
| 加训练窗口(n_ctx 7→63)/ 加容量(N 128→768)/ 训练侧 mass / 路由旋钮 | 全部对 32k 墙免疫,别再调 |
| L3 token-reconstruction aux | REJECTED,但**可能是用错 loss(recon 而非 prediction-CE)**,值得用 prediction-CE 重试 |
| eval_interval≠0(训练时内联 eval) | NCCL 崩,训练必须 `--eval_interval 0` |
| H20 unfreeze_from < 24 | step30 OOM(optimizer states),必须 ≥24 |

---

## 第 7 节:决策树(你接手后怎么走)

```
读完文档 + 验证当前状态
        │
        ├─ NOLEAK b25 W0 出了吗?
        │    ├─ 8k-32k 大跌(掉进干净簇 8-15/8-12) → 用户假说成立,b25 破墙=泄漏。
        │    │     干净 FIFO 真实力 ≈ HARDOBJ/pg19 水平。聚焦"关闭 W0/W6 gap"。
        │    └─ 8k-32k 持平(仍 50+) → 意外!FIFO 架构真有长档能力,深挖为什么。
        │
        ├─ 5-probe 出了吗?(看相对同 ckpt baseline 的变化)
        │    ├─ P1(pos packed)抬升 W0 → 位置是 gap 主因 → 做 position-fix 重训(packed 位置 train+eval)
        │    ├─ P2(keep-set)抬升 b100 → dilution+选择有效 → 做 reader-attn FIFO 重训
        │    ├─ P3(叠加)≈ W6 → 两者结合关闭 gap → 顶级突破,优先推进
        │    └─ 都不动 → 位置/dilution 都不是主因 → gap 来自 staleness 或 hidden 本身有损 → 换假说
        │
        └─ 都没出 → 按节点空出顺序发 probe(PENDING_TASKS.md 有命令)+ 等 NOLEAK,期间可:
             - 在干净 NOLEAK ckpt 上重跑 5-probe(最终判据,不含泄漏)
             - 起 T2-align 训练(d03db24)建合法 task-alignment 基线
             - 实现 prediction-CE 压缩 aux 重试 L3-recon(第 3.3 节)
```

**当前最高价值的便宜实验 = 5-probe(零训练判定理论假说)+ NOLEAK eval(判定泄漏占比)。这两个出来,下一阶段方向基本就定了。**

---

## 第 8 节:关键文件索引

```
状态(机器+人读):
  status/SESSION_HANDOFF.md          §0 一句话现状(最常读)
  status/TRAINER_ACTIVITY.jsonl      时间序列流水(★事件)
  status/RUN_REGISTRY.md             结果总账
  status/PENDING_TASKS.md            待办 + probe 启动命令
  status/HARDOBJ_FINAL_REPORT.md     干净 last-chunk 基线(对照 b25 的关键)

设计/洞察:
  versions/v_prediction_not_reconstruction_2026-06-25.md   prediction≠recon 范式
  versions/vN_HNST_tree_hidden_memory_2026-06-25.md        树形 hidden memory 设计
  ops/research_notes/fifo_dilution_eviction_litreview_20260625.md  文献调研(H2O/SnapKV/ChunkKV 定位)

核心代码:
  src/memory/mem_space/layer.py      _forward_fifo(:1193-1362) FIFO 读出 + 位置坍缩(:1244) + probe(eddb4f1)
  src/memory/mem_space/config.py     MemorySpaceConfig
  src/memory/mem_space/babilong_dataset.py:79   BABILongTrainDataset(泄漏源头)
  scripts/train_mem_space_dolmino_cpt.py        训练入口(babilong_mix default 0.15 在此)
  scripts/run_babilong_mem_space.py             BABILong eval(probe flag 在此)
  scripts/_eval_taskpool_2group.sh              标准 eval 调度
  scripts/score_nested_babilong.py              聚合 4 shard

关键 commit:
  eddb4f1  FIFO position-fix + keep-set probes(最新,probe 代码)
  5285f8d  b10/b5 + NOLEAK launch 脚本
  d03db24  NOLEAK + T2-align launch 脚本
  c20284e  --memory_disabled flag
  f6b893e  b25 破墙记录(★已降级为泄漏,见 RUN_REGISTRY 顶部修正)
```

---

## 附:一句话总结当前认知

> **b25"破墙"是数据泄漏幻觉(P11 等历史基线同样泄漏)。真实干净长程天花板仍是 pg19 的 16k=16/32k=9。核心未解问题 = W0/W6 gap(FIFO hidden 表示有损,头号嫌疑是位置坍缩 layer.py:1244)。下一步 = 用 5-probe(零训练)判定 gap 是位置还是 dilution 主导,用 NOLEAK eval 判定泄漏占比,然后据此做 position-fix 或 reader-attn-keep-set 的正式重训。所有新训练必须 `--babilong_mix_fraction 0`。**
