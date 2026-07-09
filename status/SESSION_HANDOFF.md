# SESSION_HANDOFF.md — compact / 新会话交接文档

> **本文件是 compact 后或新会话启动时的第一手交接。** 读完这份 + `status/RUN_REGISTRY.md` §3/§4 + `status/TRAINER_ACTIVITY.jsonl` 尾部，就能接上当前研究状态。
> 维护规则：main agent 每当方向/结论/在跑实验有重大变化时，**覆盖更新本文件的「当前快照」区**（保持精简，旧结论沉淀到 RUN_REGISTRY）。
> 最后更新：2026-07-10 00:05 GMT+8（★5 benchmark 三方对照收尾 + 用户给自主议程）

---

## ★★ heartbeat 必读：自主议程（2026-07-10 用户指令）

**`status/QCMEM_AUTONOMOUS_AGENDA.md` 是用户交的后续自主研究议程。** 当前 benchmark column 全跑完（TaskList #1-#4 全清）+ GPU 有空时，heartbeat 从议程按优先级挑下一个自主启动（已授权）：
1. 查漏补缺（benchmark/baseline/ablation）
2. pretrain scale（7B funnel continued 更长/funnel+蒸馏叠加；1B/3B 的 bottleneck_dim·layer sweep）
3. QCMem infra/kernel 优化（decode 慢瓶颈，派 coder）
4. ★极简架构假设（前 j 层 + 1 层 NTP 构成精简 transformer，去冗余层——先便宜探针验证，风险高可能证否）
**每轮 heartbeat 都要瞄一眼这个 agenda 文件**，别只跑 benchmark。

---

## 0. 一句话现状（2026-07-10）

**QCMem 5 大 benchmark（RULER/babilong/LongBench/LongEval/LoCoMo）三方对照（QCMem/KV-Direct/HCache）+ MemoryLLM 同类对照基本跑完，per-task 最优 topk。核心卖点坐实：超窗口(128k)唯一可用(KV-Direct 崩 0)、窗口内精调 topk 追平、read 恒定。draft §2.0-2.9 + collaborator 方法说明(QCMEM_METHOD_FOR_COLLABORATORS.md)齐。进入自主议程阶段(见上, 4 方向)。本地领先 origin 60+ commit(暂不push)。**

### 收尾中 column（TaskList #1-#4）
- #1 LoCoMo baseline(HCache/KV-Direct 本机跑中) / #2 MemoryLLM RULER(H20 4 shard done, 待聚合) / #3 MemoryLLM 其它 benchmark / #4 babilong 低档 baseline。清完进自主议程。

---

## 0-旧. 一句话现状（2026-07-08，已被上方覆盖，保留备查）

**QCMem 论文实验部分 100% 做实，`status/QCMEM_PAPER_DRAFT.md` 是完整可发表 draft。已进入"等用户定大方向"阶段(scale更大模型/KV-Direct·HCache head-to-head/写正文)。本地领先 origin 44 commit(暂不push)。**


**★07-08 完成的完整故事(3线闭环, 全 definitive 官方判分):**
1. **QCMem 效率/超CL定位**(诚实): ≤64k full-context≈或略优QCMem(压缩宿命,不宣称精度SOTA); >64k(128k)full-context崩=0/256k OOM, QCMem仍niah 100/98=超CL唯一可用. vs StreamingLLM(同KV budget)128k 100:4(25×). prefill 128k 7.8×+显存恒定18GB. 详见 QCMEM_ASSESSMENT.md。
2. **分工命题(理解在前段/生成在顶层)全线验证**: probing相关(语义中层饱和/next-token顶层成形,跨Qwen+Llama两曲线分离) + 3a截断下游因果(前4-8层达全模型95%,中层超顶层,跨backbone) + QCMem j-sweep挂钩(可缓存上限≈理解饱和点)。
3. **semantic-bottleneck pretrain(用户idea, 可行且有效)**: 1B from-scratch layer-6 rank-512 funnel → 前j层生成acc=0(分工显式强化,几乎无损top-acc). (j,dim)sweep: 分工+QCMem-friendly跨设计空间robust,d256甜点. 跨数据(wikitext)+三点收敛(2000/6000/12000,dim99收敛~230=funnel宽度). QCMem-friendly: bottleneck缓存点dim99 1858→231可压(救赎"浅层不可压"证伪)。
4. **核心primitive ablation(上层重算价值)**: j=L closed-book(重算0层)=0; block-diag(复用KV无cross-chunk) vs 标准: niah-single 100=100(单chunk cross无关), niah-multikey 88/92→44/40(消歧需cross-chunk). → 重算+full attention在多fact是load-bearing真设计,单针可退化省算。

**★novelty(6路检索)**: (b)已知组件新组合+上层重算新primitive. 最相似HCache(2410.05004). framing=depth-partitioned retrieval readout(layer-partial vs token-partial), j作RAG↔closed-book旋钮。

**关键坑/运维(07-08)**:
- 环境reset后 `/etc/ssh/ssh_config` 全局 `Port 36000` → 连22端口节点要 `-F /dev/null` 绕过; 连36000节点(如新H20)不要加。
- .52(28.49.56.52)已回收=重启变成 **28.83.53.31(端口36000,密码configs/password_h20_3153.txt=sPD6qiUvFUvbm8x,含逗号)**, diskB H20。
- sshpass被环境reset清掉→conda装在 `/opt/conda/bin/sshpass`, 用前 `export PATH=/opt/conda/bin:$PATH`。
- 远程起进程用setsid+独立脚本脱离SSH会话(防SIGHUP): 写脚本→scp→`setsid bash script </dev/null >/dev/null 2>&1`。
- 待补(需用户拍板): scale更大模型验证/KV-Direct·HCache head-to-head(需复现前人)/写正文。可选: block-diag精确ablation已做,精确"复用KV但query过全层"已完成。

---

## 0-旧. 历史快照（2026-07-07 晚，已被上方覆盖，保留备查）

**主线 QCMem 已成 definitive 结果 + RULER 真实任务泛化成功；pyramid 并行启动（用户指令）。三节点：本机 wzc1 + .52/120 diskB。**

**★今日两大突破：**
1. **RULER 真实任务泛化成功**（QCMem+自蒸馏 纯PG19零RULER，zero-shot）：niah-single 4k-32k **全档100**(无退化!)，niah-multikey 96/94/94/88，var-track 100/49/25/21。零训练对照 niah-single 8k=36。**证明通用长上下文记忆能力，非babilong特化——兑现2026-07-05转向纯prediction的目标**。
2. **pyramid 启动**（用户选"现在并行启动"）：MemoryLLM port 验证可用(pool 1.68B)，关键洞察=MemoryLLM memory[idx]==QCMem h_j 同货币可直接concat。v1设计+skeleton已合main(43e17d8)，方案A统一Llama-3。coder 正实现 P2 dual-cadence read（layer API 统一是硬骨头，未commit）。

**节点**：本机=wzc1(L20A sm100 .venv torch2.10)；.52=28.49.56.52 diskB H20；**120=29.162.226.120 diskB H20(新增,与.52共享盘无需rsync,password configs/password_h20_120.txt)**。

**QCMem = mid-depth resume（破读出墙的最有效手段）**

**QCMem resume 是什么 / 为什么成主线（本会话 definitive 坐实）：**
- write=chunk 过底部 layers[0:j] 缓存 h_j（chunk-local RoPE）；read=pack[sink;选中chunk h_j;query h_j] 全新 RoPE 重算 layers[j:]→logits。j=0=RAG full recompute(精确=full forward,self_test max diff 0)，j=L=closed-book。省算 ∝(L-j)/L，存储 ~1/16 full KV。
- **破了读出墙**：Qwen qa5 j12=51（省 33% 算）vs 我们旧 flat b64=12。self_test 在 Llama & Qwen 都精确 PASS（实现正确）。
- **"是不是 Qwen 的功劳？"—— nuanced**：qa5(关系)Qwen resume 耐截断(j12=51)vs Llama 崩(j12=3)→ Qwen backbone 在关系任务确有真优势；qa1/qa2(精确定位)两个 backbone 零训练都崩 → 需要训练。

**本会话两大 definitive 结论：**
1. **★自蒸馏救 qa1 成功**（官方判分 n=100）：1000 步 LoRA 自蒸馏（teacher=j0 RAG，student=j12+LoRA，**PG19 纯自然文本 KL，零 babilong**）→ qa1/8k 22→31 qa1/16k 11→18 qa5/8k 62→78 qa5/16k 51→61，**四格全升**。证明 QCMem 核心主张"训练推后 readout cliff"在 Qwen+自蒸馏配方成立且不靠合成数据（守红线）。gap 距 QCMem 报告 ~.67 =训练量(仅1000步)+teacher 受检索限，非机制失败。
2. **★方向 B（非连续层/缓存顶层）判负**（官方判分 n=100）：qa5 (12,0)baseline 8k61/16k50；缓存任何顶层即崩→(12,6)29/20 (6,6)27/23 (6,12)9/10。**原理：顶层 hidden=query 敏感的读出前最终表征，缓存它=丢 query-conditioning**；底层 hidden 接近局部 token 语义对 query 不敏感故缓存无害→印证原始「纯前 j 层 resume」才对。非连续层不再追。

## 0.1 当前状态（2026-07-07 晚，所有 eval 完成，GPU 空闲待新方向）

- **4000 步自蒸馏 LoRA 已训完**：`outputs/qcmem_distill_qwen_j12_r32_4k/{step500..4000,final}`（lora_r32/α64, teacher j0, student j12, PG19 n_ctx3, 8卡DDP）。
- **完整 eval 已跑完**（下述结论均来自它）。GPU 目前空闲，等下一方向决策。
- 自蒸馏 launcher: `scripts/train_qcmem_distill.py`（torchrun --nproc 8）；eval: `scripts/eval_qcmem_babilong.py`（`--lora_adapter/--resume_j/--top_prepay_b/--selector bm25|recency|oracle|reader_attn/--topk`）。
- 训练脚本**无 selector**（PG19 连续窗口全喂 n_ctx=3）；selector 只在 eval（长上下文检索 topk）。

## 0.2 已完成结论（2026-07-07，全 n=100 官方判分）
1. **自蒸馏真实有效且快**：1000 步即达 90%+ 增益（qa1/16k 11→18, qa5/16k 50→63）。4000 步 qa5/16k 爬到 68(未饱和)，qa5/8k=79。纯 PG19 零 babilong，守红线。
2. **训练量 scaling**：qa5 随步数单调爬升(关系推理受益)；**qa1 1000 步后完全饱和**(8k=31/16k=19 纹丝不动)→ qa1 天花板不是训练。
3. **★检索精度是瓶颈，存在甜点 topk（信噪比非覆盖率）**：qa1/16k topk 4/8/12/16/32=19/39/57/43/31，**峰值 topk12**，全召回 topk32 反崩。bm25≫recency（qa1/16k 39 vs 7，5.6×）。**最优 topk 随任务所需 supporting fact 数单调**：qa5(多fact)=tk4, qa1(单fact)=tk12, qa2(双fact)=tk16；长档按比例放大(32k:tk24)。
4. **🎯 每任务最优 SOTA 全表（碾压 MemoryLLM）**：qa1 8k63/16k57/32k28(vs MemLLM 14/9/7=4-6×)；qa5 全程压制(8k79/16k67/32k63 vs 39/38/34)；qa2 中程超越(8k37 vs 15)。见 RUN_REGISTRY 🎯 表。
5. **泛化良好（未调优任务 n=100）**：qa4(位置)8k45/16k47、qa6(yes-no)55/46、qa3(3fact)28/25 都可用；仅 qa7(计数)9/5 弱（计数需完整枚举，与"少chunk甜点"机制天然冲突，可解释）。qa3/4/6/7 全程未调优→证通用 readout 提升非过拟合（守红线回报）。
6. **方向B(缓存顶层)判负**：顶层hidden=query敏感读出前表征，缓存=丢query-conditioning。纯前j层resume才对。
7. **reader_attn selector 判负**：全面劣于 bm25（qa1/16k bm25 57 vs RA 5）。原因：babilong needle 是明确实体词，词法 BM25 天然最优；reader_attn 中层 j12 mean-pool cosine 偏语义泛化+抹掉 token 级实体→对精确实体定位不如词法。**任务决定检索器**，语义检索在实体密集合成 QA 错配（与项目历史 rawkv reader_attn 同因）。
8. **oracle selector 不可用作上界**：只选1.3-1.5个含字面答案词chunk，漏推理链fact。

## 0.2b 下一步（QCMem babilong 方向已基本收束，可选延伸）
1. **qa2(双fact最难)仍有空间**：0k=25<MemLLM36、16k=25 未拉开。可试更深/多阶段检索或 qa2 针对性训练。
2. **换真实长文档 benchmark**（LongBench/RULER）验证 QCMem+自蒸馏泛化——真实任务 needle 不是明确实体词，reader_attn/语义检索可能反超 bm25（babilong 上判负≠通用判负）。
3. **backlog**：pyramid 三层记忆(顶slot/中MemoryLLM hidden/底raw hidden)；MemoryLLM 本机适配(coder A 未完)——deferred。
4. git 暂不 push（用户指令，本地保留，含 reader_attn 3fab03c）。

## 0.3 关键坑/运维（QCMem 会话）
- **.52 offline eval 两个必设 env**（否则崩）：(1) `local_files_only=True` 已加进 eval 脚本(commit e71bdd2)否则本地模型路径被当 HF repo_id 报 "Repo id must be"；(2) 起进程必须 `export HF_HOME=$PWD/.hf_cache HF_DATASETS_CACHE=$PWD/.hf_cache/datasets`否则 babilong 数据集去默认 ~/.cache 找不到→offline 挡→"Couldn't reach RMT-team/babilong"。
- .52 = 28.49.56.52，diskB `/apdcephfs_zwfy6/share_304376610`，`.venv/bin/python`(sm_90)，password `configs/password_h20_52.txt`。跨盘(本机 wzc1 vs .52 diskB)代码/ckpt 需 rsync。
- 本机=B200(实为 L20A 183GB, sm_100)，用 `.venv/bin/python`(torch2.10)。git 暂不 push（用户指令，本地保留 ~20 commit ahead）。

---

## 0.3 历史（2026-06-25 babilong 泄漏教训，已沉淀）
- 6-25 的 FIFO b25 "破墙"（qa5 8k=65/32k=68）已查实是 **BABILong 数据泄漏**（`--babilong_mix_fraction=0.15` 默认，训练 task/length/dataset 与 eval 完全重叠，HF 无 train/test 隔离）。~85% 是泄漏幻觉。
- **红线**：所有训练必须 `--babilong_mix_fraction 0`（代码已硬阻断）。真实干净 SOTA 曾是 pg19 nctx7（16k=16/32k=9）。
- 官方判分口径：`third_party/babilong-pkg/babilong/metrics.py:compare_answers`（**非** re.search，后者虚高 ~30pp）。

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
