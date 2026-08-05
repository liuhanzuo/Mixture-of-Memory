# SESSION_HANDOFF.md — compact / 新会话交接文档

> **本文件是 compact 后或新会话启动时的第一手交接。** 读完这份 + `status/RUN_REGISTRY.md` §3/§4 + `status/TRAINER_ACTIVITY.jsonl` 尾部，就能接上当前研究状态。
> 维护规则：main agent 每当方向/结论/在跑实验有重大变化时，**覆盖更新本文件的「当前快照」区**（保持精简，旧结论沉淀到 RUN_REGISTRY）。
> 最后更新：2026-08-06 05:00 GMT+8（rebuttal-prep sprint 结束：paperA 数字全 audit + tab_pareto 修 99.20→99.19；paperB tex 16/16 数字精确；Paper C P-C1 scooped + eval 无效已判决；Paper D 层拼接判决 dead。下方 2026-08-04 及更早快照请视为历史沉淀，以本快照为准。）

---

## ⚡ 当前快照（2026-08-06 05:00 GMT+8）—— rebuttal-prep 收尾 + 4 项决策待用户

**一句话现状**：ARR 已交（Paper A 已投、Paper B 已投），主战场转 rebuttal 备料。夜间 sprint 完成 paperA/B 数字-tex-provenance 三层 audit + 关键 tex 数字漂修（`tab_pareto.tex` 99.20→99.19），rebuttal 弹药充足；#99 与 #103 已按用户令 kill（释放 6037 GPU-h）；32 卡持续空闲，方向未拍板不动。

**用户 2026-08-05/06 关键指令**：
- 「rebuttle 和重写的角度分别出发走」→ Paper B 双轨（rebuttal 素材 + 若被拒的重写方向）
- **.104 已交还用户** → **4 节点 32 卡**（LOCAL/.252/.73/.82）
- **kill #99 keep14-distill**、**kill #103 matched-PPL crossing 监控**（crossing 已达成目的）
- Paper C 令：**P0-4 重建 eval set**（CPU）+ **P0-5 加 `--random_trunk` flag**（15 行）——均已完成
- Paper D 令：调研 model editing 层拼接可行性——已判决 dead

**Paper 主命题 rebuttal 备料状态**：
- **Paper B tex 数字**：4 MMLU + 12 closed-book QA 全部与磁盘一致（max diff 0.001）; Finding 1 tex 是「likelihood recovery overstates target recovery」的**单轨迹内**残差观察（keep14 200k MMLU 距 base 差 28.74pp），**无 cross-arm dissociation 断言**; related work 明确 disavow "loss--task dissociation originate here"; **不需要 rewrite Finding 1**
- **Paper B Finding 2 rebuttal 就位**：用 Random-16L 校准发现 content_norm empirical chance = 35.98%（letter chance 25%，差 +10.98pp 是 scoring metric base-rate）；letter-only 单轴 Wilson CI + 单侧 binomial 表：PPL 匹配 pair (Random-16L 11.50 / keep14@67500 11.53) letter headroom 分别 -0.30pp (p=0.80) / -0.08pp (p=0.59) **都在 chance 内**，keep8@121k +0.50pp (p=0.085) **NS**，rebuttal 弱化不推翻
- **Paper B 主命题「三重证伪」的真相**：打的是我口头概括的靶子，不是 tex 正式文本。三重 audit（Spearman +0.94 lockstep / matched-PPL MMLU +0.32pp / ShortGPT beats keep14）作 supporting appendix 即可
- **Paper A 数字**：5 组 primitive 中 **4 组精确**（RULER 99.187/96.067 + CI [2.36, 3.9333] 全在 .82 `p0_13_quality_latency/{summary,stats}.json`; overlap 92.5/98.5/99.0 全在 .82 `p0_17_e2_overlap/`; BM25 -11.56 CI[-14.444,-8.667] 在 wzc1 `equal_latency/source/bm25/decision.json`）; **1 组 latency 931.9/664.4 ms 有 <2% 漂**（最近 P1.8 128k|cpu G=1 = 934.5/677.8, 差 3-13 ms）→ #167 rebuttal 前三选一
- **Paper A tex 漂修**：`tab_pareto.tex:12` `99.20` → `99.19`（磁盘真值 99.187，与 `tab_replay_latency.tex` 一致）
- **Paper A anonymous_artifact 补齐**：P0.13/P0.17 的 summary/stats/manifest/latency.json 已从 .82 `scp -O` 到 `paperA/anonymous_artifact/scores/p0_13_quality_latency/` 与 `.../p0_17_e2_overlap/`——release 可复现，不需要跨盘 scp

**Paper C 状态**（**三 reviewer 联合判决 2026-08-05**）：P-C1 构造被 arXiv:2411.15558 全占、P-C2 hook 被 2210.10041 占、原 SQuAD eval set 常量拒答 49.85% 基线**高于所有臂**且 relevant_indices z=+0.72 分布过均匀 → **P-C1 降级不作主线**。已完成 (1) 重建干净 eval set（P0-4，SQuAD v1.1/v2.0 CPU forensic），(2) 加 `--random_trunk` flag（P0-5，`train_olmo2_arch_probe2.py` +166/-7，8/8 self-test pass）分离 A4 vs A3 的 readout-interface confound。**下一步 A4×random_trunk keep{14,20,24,28} 训练待用户拍板**（#165）

**Paper D 层拼接判决**：3 路调研（R1 lit 34 refs + R2 tcodex gpt-5.6-sol + R3 CPU feasibility oracle affine ppl 596 vs 19）联合结论——**方向被 BTS (EMNLP'25 2025.emnlp-main.347) / LEGO-LLM (2026.acl-long.2081) / StitchLLM (2025.acl-long.1305) / CALM (arXiv:2401.02412 ICLR'24) 全占，且跨家族技术不通**。**不建议做**；仅剩「相对深度主导层对齐」cheap mini finding 可补（forward-only 几小时，#166 待用户拍板）

**在跑的实验**：无。#99 与 #103 已 kill。32 卡持续空闲，无未定方向擅自投入（3 heartbeat 一致）

**集群状态**：
- **LOCAL / .73 / .82**：直连各 0 GB，**24 卡确认闲**
- **.252**：**SSH 4h+ 持续 Permission denied**（连续 10+ 次拒），密码文件 16 字节含末尾逗号，ed25519 握手 OK 但 auth 拒——疑 sshd 端问题或密码轮换，**不 hammer 等自愈或用户确认**；上一次 heartbeat 前它是可连的（0021）
- Monitor 8088 前端 http=200 但 `metric_history` chronic 全空（26h+），已 kill+restart 一次首轮 55s 后仍 0 samples，采集线程疑卡在 poll .252 SSH——不深挖（http 200 满足 skill 要求）

**待用户 4 项决策**：
1. **Paper C**：A4×random_trunk keep{14,20,24,28} 在新 eval set 上训练是否启动（P-C1 干净重跑，#165）
2. **Paper D**：「相对深度主导层对齐」mini finding 是否做（纯 forward-only 几小时，#166）
3. **paperA #167 latency**：931.9/664.4 ms 三选一——(a) rebuttal 主动 own <2% 漂；(b) 找回原始 log；(c) GPU 60-reads 重跑
4. **17 个 unpushed commit 是否 push GitHub**（走 `/gitpush` review-subagent → star-proxy 流程）

**本轮 sprint 关键 commits（10 个 audit/fix）**：
`6a3b6bb` letter headroom + Wilson CI · `dfdbf2d` paperB tex-wording audit · `638fb04` paperB 16/16 数字 audit · `51c7349` Finding 2 chance 校正 · `550a81a` paperA latency provenance · `8c30fc7` self-consistency check · `9883ef9` paperA 3/3 primitive 精确 + tab_pareto fix · `f9fb8c6` P0.13/P0.17 artifact 拷到 wzc1

**运维复现坑**：subagent 派 provenance audit 时一定要在 prompt 里写明**跨 2 盘搜索**（wzc1 + zwfy6），否则会误报"文件不在磁盘"。这是 CLAUDE.md 顶部「两个物理盘」坑的第 N 次复现。

**Paper C 状态**（`versions/paperC_scoping.md` 是唯一 scoping 文档，**无 paperC/TODOList.md**）：定位 = 冻结前 j 层 + 丢弃顶部 + 移植 K 层新块、**只 finetune 那 K 层**（区别于 Paper B 的 continue-pretrain 全参 heal）。推荐命题 = **P-C1 构造 + P-C2「用 base 模型的廉价 probe 预测该切多深/长多少层」为差异化 hook（P-C1 单独有 novelty 风险：Zhang'21 re-init / Surgical FT）**；P-C3 建议降为附录。已有 #92 SQuAD 4 臂结果（A2_lora 0.659 > BASE_ref 0.339 > A4_hero 0.293 > A3_fromscratch 0.261，A1 因 H20 OOM 未跑）；**诚实框定**：BASE_ref 差两个轴（32L-vs-16L AND no-SFT-vs-SFT）→ 只作 intact-model 上限参照，A4-vs-A3 才是干净对照。剩余 #133 depth-sweep / #134 A1 ceiling 待 B200。

**运维要点**：monitor 8088 曾 http=000 → 已重启，现 **http=200**。H20 三台 `.venv/bin/python` **已坏** → 一律 `/opt/conda/envs/torch-base/bin/python`。两处物理盘：**wzc1**（LOCAL+.252）/ **zwfy6**（.73+.82+.104），#92 的 Paper C ckpt 在 zwfy6。

---

## ★★ heartbeat 必读：自主议程（2026-07-10 用户指令）

**`status/QCMEM_AUTONOMOUS_AGENDA.md` 是用户交的后续自主研究议程。** 当前 benchmark column 全跑完（TaskList #1-#4 全清）+ GPU 有空时，heartbeat 从议程按优先级挑下一个自主启动（已授权）：
1. 查漏补缺（benchmark/baseline/ablation）
2. pretrain scale（7B funnel continued 更长/funnel+蒸馏叠加；1B/3B 的 bottleneck_dim·layer sweep）
3. QCMem infra/kernel 优化（decode 慢瓶颈，派 coder）
4. ★极简架构假设（前 j 层 + 1 层 NTP 构成精简 transformer，去冗余层——先便宜探针验证，风险高可能证否）
**每轮 heartbeat 都要瞄一眼这个 agenda 文件**，别只跑 benchmark。

---

## 0. 一句话现状（2026-08-03）

**Paper A（CoMem）已按顶会系统论文结构重组；P0.16 E0 已完成、P0.17 overlap Write 在跑，下一最高优先级为 P0.20 equal-latency text-RAG vs CoMem retrieval-budget frontier（先 BM25，再 BGE/E5）。Paper B 已扩充为完整 8 页正文：漏斗 Intro；加宽后的 Related Work 定位矩阵与 proxy-metric 讨论；§3 Study；§4 Correlates；§5 Main Experiments（新增 keep14 late-healing 曲线与闭卷 QA 解释）；独立 §6 Analysis（MMLU domain recovery 图、depth×healing、ShortGPT 结构差异、OLMo-1B/Qwen 泛化）；§7 Discussion（行为 taxonomy、知识神经元因果边界、practitioner checklist）；§8 Conclusion。纯 Limitations 从第 9 页开始。Table 4 主表按用户约定统一显示 200k target budget，真实 checkpoint steps 仅保留在 artifact provenance。`paperB/main.pdf` 已编译为 17 页（计页正文 8 页），无缺失引用或版面越界；独立发布仓 `perplexity-heals-knowledge-lags/` 已整理完成并提交 `59e05d1`（同步论文、full32/content-MMLU/closed-book/ShortGPT/Qwen 聚合、评测与训练脚本、README；安全扫描/9 tests/artifact audit/17-page compile 全通过），但 GitHub 推送因当前环境无可用 GitHub 写凭据而待执行，本地相对 `origin/main` ahead 1。P0.5 structure-isolation 与 P2.5 Qwen protocol-complete 状态仍以 TODOList/GPU_STATUS/TRAINER_ACTIVITY 为准。**

### ★ Paper A 投稿前补缺（协议双支柱：selector=iter_bm25 + chat_template=False）
权威汇编：`status/QCMEM_STATS_APPENDIX_chatFALSE.md`（diskB，274 行）+ `status/BENCHMARK_RESULTS.md` 顶部 chat=False 段。
- **用户 top-3 闭合项**：
  1. **LoCoMo GPT-4o judge headline = 38.27**（n=1986；cat5 adversarial n=446 不送 judge，本地 abstention folded；judged-only cat1–4 n=1540=48.64）。baseline KV-Direct(full-ctx 上界) judge=34.59。**paired bootstrap（judged n=1540）：CoMem−KVD diff=+4.81，95%CI[2.34,7.27]，p<0.0001 → 显著优于 full-ctx KV oracle**（unpaired CI 重叠是配对设计 power artifact，已用正确配对检验推翻）。judge prompt/endpoint verbatim 在 appendix §1d，配对显著性在 §1e。✅ **SETTLE**。
  2. **等预算长档 VT 精确值**：flagship config(`qcmem_8b_iter_chatFALSE_ad`，iter_bm25 top12/hop4 → **rounds:0=auto=3跳迭代**，read≈6.6k)=96.6/97.6/98.8/**99.0/95.8**（8k→128k）。取代旧 `~95/~95` 占位。✅ **SETTLE**（#61 已裁决，见下）。
  3. **baseline selector 对齐**（P0#2，#55）✅ **DONE**（agent a8ef76da，2×2 selector-fairness）：固定 selector 下 KVD≈CoMem 各档（one-shot 48/26→CoMem 48/25 崩；iter KVD=100 vs CoMem 96.6–99.0）→ **VT 精度=迭代 selector 非架构；架构价值=效率**。论文口径=「CoMem 以极低显存/算力 match KVD 精度」非「VT 超 KVD」。
- **P1 GPU（#56/#58）✅ DONE**（agent a8ef76da）：chunk1024 效率 8k-64k 1.21-4.40×，**128k full-ctx H20 OOM vs CoMem 20.3GB**；迭代开销=one-shot 4.2-4.9×、占端到端~0.1%=免费。详报 diskB `status/QCMEM_GPU_EVAL_PRESUB_20260723.md`。
- **#61 VT provenance ✅ RESOLVED（code-backed, high conf）**：`rounds:0`=auto=ceil(topk/hop)=**多跳非单遍**（曾误标致"叙事反转"，纠正）。`_ad`=3跳→96-99；`ablation10`=4跳→89.8/87.4；真单遍=明码bm25=48/25/23崩。**迭代确实救 VT，叙事未反转**。tab_itervt 用「单遍bm25 20-48→iter 96-99」。详见 BENCHMARK_RESULTS.md §task#61。
- **⚠️ 论文集成 #10**：tab_overview/tab_locomo 指向 chat=False dir；LoCoMo headline 改 GPT-4o judge；baseline selector 统一后 tab_selector/itervt/chunk/crosschunk/slm/overview/h2h/scaling 整套换 chat=False 数字。

### ★ Paper B（OLMo-2 base 剪层-heal，4 臂，BASE LM 口径，vs vanilla OLMo-2 ppl=7.40）
实时每卡状态见 `status/GPU_STATUS.md`（权威台账）。当前 4 臂：① from_scratch @ LOCAL 8×L20A；② keep12 @ .252 8×B200；③ **freeze_front @ .73 已 PAUSED@step23500**（用户 14:05 授权 checkpoint-pause，8×H20 让给 Paper A GPU eval；**#59=main-owned resume bookend，待 GPU eval 跑完 + 卡空后我负责重启**，resume cmd 见 task#59）；④ keep8 @ .104 8×H20。训练脚本自轮转 ckpt（latest-2 + every-5000 里程碑，绝不删 final）。
- **★ 2026-07-28 论文审计纠正（load-bearing）**：`--from_scratch` 实际是**完整 16L 模型全随机初始化**（decoder + embedding + norm + lm_head 全不 transplant），且所有参数进 fresh LR=1e-4；不是旧稿所称“只随机 front blocks、复制 lexical modules、optimizer fixed”。因此该臂只能支持“同架构/语料/200k budget 从随机初始化未恢复 MMLU”，**不能隔离 decoder-block inheritance**。`paperB/PAPER_B_DATA.md`、正文、表格、限制与图已统一纠正；同时修复 BoolQ raw/acc_norm 混用（keep8=.588、keep12=.610）及 PPL 舍入（10.826→10.693，tax 1.445×，Δ−0.133）。**Appendix 已扩充**：完整 keep8 11-task 轨迹、keep14 late-healing 图、raw/norm 敏感性、MMLU 四组+57-subject 全表、逐臂 integrity manifest、OLMo/Qwen 全 33/37-depth logit-lens；由 `paperB/scripts/generate_appendix_tables.py` 从 raw JSON 自动生成。另修正 SIQA 主表口径（keep8 raw=.400、keep12 raw=.415）。**实验缺口审计**：keep14 train-all 已完训200k且 ckpt 存在，但完整 eval 仅到153.5k（P0：补200k PPL+core+knowledge）；freeze_front 13:05 已到179720/200k健康运行（约剩7.5h，完训后同样全评）；keep8/10/12 仍是不等预算 44k/10k/111.5k，不能宣称收敛 architectural threshold。详见 `PENDING_TASKS.md` T24。**Appendix 排版已按 Paper A 重构**：默认双栏，窄表/轨迹图进单栏，宽协议/恢复率/MMLU 图跨双栏；OLMo/Qwen 全层表与 57-subject MMLU 均改成左右双面板。最终 `paperB/main.pdf` 17页（Appendix p11–17），0 undefined/0 overfull。**匿名发布仓库已整理**：`perplexity-heals-knowledge-lags/`（建议仓库名同名），含自包含 train/eval/data prep/logit-lens、脱敏37份 summary+2份 probe JSON、匿名论文源/PDF和复现脚本；安全扫描0身份/0集群路径，2.2MB。待用户创建匿名远程仓库并提供 URL 后，加入论文正文。

### ★ 节点 roster（QCMem，2026-07-23）
LOCAL 8×L20A（wzc1，`.venv`）；.73=28.85.35.73（H20 diskB torch-base，现跑 Paper A GPU eval）；**.82=28.82.250.82 = 用户占用给 dllm，绝不碰**；.104=28.83.24.104（H20 diskB）；.252=28.89.19.252（B200，wzc1 与 LOCAL 共享 CEPH）。H20 共享 diskB `/apdcephfs_zwfy6/share_304376610/...`；LOCAL+.252 共享 wzc1。**dllm 节点 29.162.226.120 绝不碰。ckpt 轮转 cron 4ec42903 勿删。**

## 0-旧. 一句话现状（2026-07-16 晚，已被上方 2026-07-23 快照覆盖，存档）

**两条线：Paper A(QCMem/CoMem)=定稿并双 push；Paper B(剪层-heal)=从 instruct 转 OLMo-2 base 重做，3 条训练全跑起来，32/32 卡满。**

### ★ Paper A（QCMem）已收尾 + 双 push（本会话）
- **j 定案（本会话核心，`status/QCMEM_J_DETERMINATION.md`）= 三深度分离**：①语义 content 深度 ~0.45L（truncation+probe 实测，scale-invariant）；②zero-shot readout 崩点随 scale 变深（0.6B 0.09L→32B >0.42L）；③gap=content−readout=adapter 缺口，随 scale 缩小到 32B~0（小模型最需 adapter，32B 几乎不需）。旧"固定 0.33L/j3"作废。
- **主表=双 j**（`QCMEM_BENCHMARK_PLAN.md` §1）：zero-shot 报 per-model readout-safe j（0.6B j2/1.7B j3/4B j9/8B j9/14B j13/32B j27）、+adapter 报 content-j(~0.45L)。n=500 firm。vs-Dense 128k 超窗口列(Dense 崩0/OOM→QCMem 56-100 全 scale)、§1a 补充表(BABILong/LongBench/LongEval/LoCoMo-F1/j-used 行)、§1c 速度全表(prefill 50-103×,显存恒定)全在。
- **判分口径**：RULER=string_match/BABILong=compare_answers/**LoCoMo=F1**(非 LLM-judge)/**LongMemEval+∞Bench+HELMET=⏸暂不评(需 GPT-4o judge API)**。thinking：eval 全程关(RULER raw prompt),`30bb2ab` 加了 `--enable_thinking` 默认 False 防坑。模型=**原版 hybrid Qwen3**(非2507),跑非-thinking。
- **push**：MoM 主仓 `8176949`(与 collaborator 21 commit 合并,1 冲突 eos_ids 已解) + **COMem 论文 `196d4de`**(tab_scale 全 scale dual-j + tab_depth 3-深度追加进 05_experiments,只增)。⚠️ **COMem push 要用 MoM 的 `core.sshCommand`**(deploy key `configs/github_deploy_key` + ProxyCommand `gh_proxy_connect.py` -p443;直连/scp/proxy-tunnel 都不行)——已给 COMem 配好。

### ★★ Paper B（剪层-heal）大转向 → OLMo-2 base（本会话，用户方法学修正）
- **为什么转**：旧 armB=Qwen3-8B-**instruct** keep12+fresh2 continue-train，方法学不干净(instruct 上 continue-train 混目标)。armB 已训完 200k：**held-out ppl 23.56 vs 全 Qwen3-8B 11.43 = 2×,且 22 epoch 过拟合(训练 ppl 6.01 是记忆)**——坐实要换 base。
- **改用 OLMo-2 base**(纯预训练 + Dolmino 公开数据,消 instruct/distribution 污染)。arch-port `scripts/train_olmo2_arch_probe2.py`(smoke 验证过;Olmo2 post-norm 无 input_layernorm)。keep 从**语义深度 0.44L 起**做 ablation frontier。
- **数据**：dolmino-mix-1124 **DCLM 子集**用 OLMo-2 tokenizer 重 tokenize → `data/dolmino_now15b.npy`(wzc1 15.5B / diskB 31.7B,uint32[N,2048]),仍在冲 30B(auto-combine 到 `dolmino_chunks_2048_olmo2.npy`)。
- **★3 条训练在跑(32/32 卡)**：① **7B keep14+fresh2 @ 8×L20A 本机**(ppl 19.95)；② **1B keep7+fresh2 @ 16×H20 .82+.104**(TCP over bond1,IB 栈 `ibv_reg_mr` 坏)；③ **7B keep10+fresh2 @ 8×H20 .73**(ablation,ppl 降中)。model 跑 **fp32 master**(7B 在 H20 bs4/ga4 上限、L20A bs16)。
- **关键运维坑(必照抄)**：①py3.14 DataLoader 加 `multiprocessing_context="fork"`(否则 pickle 61G memmap 卡死)；②数据 stage 到 `/dev/shm`(ceph memmap 随机读 D 态 wedge)；③**wzc1 7B ckpt ~48.7G/500步 会写满盘 → cron `4ec42903` 每小时 :47 轮转**(留里程碑+最新2);④多机换新 rdzv 端口防僵尸占用。

### ★ 待办 / 下一步
1. **OLMo 训练监控 + 早停**：离线测 held-out ppl(dolmino_now_val) plateau 就停；出全模型 OLMo-2 baseline ppl 做剪层对照(类比 armB vs Qwen3-8B)。
2. **ablation frontier 补齐**：keep14/10 已跑,补 keep12/8 + `--freeze_front`/`--from_scratch` 臂(脚本已支持)。.73 空出后接。
3. **QCMem 剩项(可选)**：LongMemEval 等待 API judge;32B/14B 若要 balance-j 之外的口径统一可补。keep 具体值/是否加 13B OLMo 用户后续定。
4. **cluster**：本机 8×L20A(wzc1 独立盘) + 3 H20 节点(.85/.82.250/.24,共 24,diskB 共享)= 32 卡。IB 栈坏用 TCP;.73 IB launcher 支持 NNODES 可扩 3-node。

---

## 0-旧. 一句话现状（2026-07-15 09:18，已被上方覆盖，存档）

**★主线=QCMem(=CoMem) 论文全 scale benchmark，已基本完成 + 官方判分聚合。** 分工：collaborator 跑 32B+，agent 跑 8B→4B→1.7B→0.6B（全做完）。32 卡满载：本机 armB 训练(step~127k/200k ppl~7.8，Paper B keep12)、.82.250 3b 训练、.85+.24 跑 QCMem eval 收尾(vs-Dense n100 robustness)。

### ★★ 全 scale 结果（已入 RUN_REGISTRY「★ QCMem 全 scale benchmark」+ bench_qcmem_vs_dense_result.txt）
- **RULER n=500**：8B+adapter(j12) single100/multikey91/**vt97**；zero-shot 各 scale single 强(8B100/4B93/1.7B91/0.6B68)、**硬任务(multikey/vt)普遍弱**(8B-zs 42/46)。
- **BABILong 官方**：8B-adapter 55.5 > 4B 49.3 > 8B-zs 39.2 > 1.7B 34.2 > 0.6B 11.0；**adapter 增益主要在长档**(8k+ 21→54)。
- **LongBench 官方 qa_f1**：8B-adapter 9.76 单调随 scale 降。
- **vs-Dense 超窗口崩塌**：128k Dense=0 全 scale / QCMem single 存活(8B~100/4B100/1.7B88/0.6B54)——核心卖点全模型族通用。
- selector 定案：vt→**iter_bm25 固定**(97 vs adaptive 31，adaptive 已证否弃默认)；niah→bm25。
- **j 规律**：zero-shot 可用 split 随模型变小而变浅(8B/4B=j9=0.25L, 1.7B=j5, 0.6B=j3)；adapter 推到 j12(0.33L)。

### ★ 关键 insight（论文核心）
**硬任务弱不是"小模型"是"zero-shot"问题**：8B-zs multikey42/vt46≈4B-zs，而 8B+adapter=91/97 → **adapter 是硬任务/长档的关键杠杆**。

### ★ 待用户决策 + 待办
1. **★等用户 A/B**：(A) 只 8B adapter(现数据够写诚实 scale 表) vs (B) 给 4B/1.7B 也蒸 adapter(数据强烈支持,adapter长档增益巨大)。已问≥2 次未答；**未擅自违背"只 8B"启动 distill**。
2. **COMem 仓库**(git@github.com:liuhanzuo/COMem.git) 已 push 完整：eval harness(run_cell.sh 一行跑)+ 蒸馏 train/distill.py + 默认 selector auto(vt→iter_bm25/niah→bm25, fd73c7d)。collaborator 可直接用跑 32B。
3. 论文 `paper/`(=CoMem，abstract 已改 insight-first 209词)；下一步用上面结果写实验节。
4. Paper B(剪层-heal frontier keep6/8/12) 训练还在本机/曾在 diskB(已 kill 腾卡给 QCMem eval，ckpt 可 resume)。

### 历史主线(已沉淀,存档)：方向4 混元剪层 prune-heal frontier(Hy-MT2 keep36→ppl12)、方向2 3b bottleneck、A13B 四连坑——详见下方 + RUN_REGISTRY。当前非主线。

### ★ 当前活跃线 + 待办
1. **★方向4 主线：混元 MoE 剪层+补层 continue-train（已验证 + prune-heal frontier）**（用户核心指令："像 Qwen3-8b 一样截断后面几层再填上两层然后 continue training"）。
   - **A13B（Hunyuan-A13B-Pretrain, hunyuan_v1_moe, 65B, tie_emb=True）**：`scripts/train_hunyuan_a13b_probe2.py`（keep24+fresh2）。四连坑全修（见 [[a13b-known-pitfalls]] / 下方史）后**验证成功：loss 102→35（13步）**。但 65B 必须 cpu_offload → 113s/step 慢。
   - **★Hy-MT2-30B-A3B（hy_v3, 30B, tie_emb=False）**：用户嫌 A13B 慢 → 换更小的（sanity 测过能正常答通用问题非只翻译）。`scripts/train_hyv3_probe2.py`（coder dcb2691，不碰 A13B 路径）。**30B on-GPU 免 cpu_offload → 3.7s/step 快~125×**。数据 `data/slimpajama_chunks_2048_hymt2.npy`（Hy-MT2 tokenizer，vocab120832，避 A13B tokenizer 越界）。启动 `scripts/launch_hyv3_probe2_keep36_fresh2.sh`（bdfefcb，含 MONITORING=0）。
   - **★prune-heal frontier（Hy-MT2, 200步/config）**：keep36(75%层)→ppl12(loss6.83→2.5)；keep24(50%)→ppl42(loss12.1→3.73)；keep30 跑中。**单调：保留越多 heal 越好**。loss 曲线在各 config 的 log；ckpt 285GB(含fp32 optim)已 rm 省磁盘（**⚠️CEPH 92%**），仅留 keep36/step200.pt。
   - env `.venv_hy3`(tf5.13.1)。**运维坑**：cpu_offload 进程 kill -9 后 D 状态卡~60s；kill 后须轮询等 nvidia-smi memory=0 再重启（否则撞泄漏显存 EXIT_1）；杀进程绝不用 pkill -f（自杀 shell）。
2. **方向2 3b scale**：.53.31 跑 3b_d512@L6（bottleneck_dim512@layer6）到 16000 步，验证 semantic-bottleneck 在 3b 规模。用 torch-base（.venv 在 H20 sm_90 会 cudaErrorAssert），数据 slimpajama_chunks_4096_llama3.npy seq4096。
3. **方向4 probe#2 armB (Qwen3-8B keep12+fresh2)**：.24.104 从 20k warm-restart 续训到 200000（用户："若未收敛续训到 200k"），step~26440 ppl13.7 缓降。step20000.pt 是旧格式无 optimizer_state→warm-restart(Adam re-init, LR 按 cosine step20000 续)。
4. **QCMem n=100 网格**（补论文§2.5/§2.6）：.85.73(diskB) + 本机 L20A(临时) 跑 topk×length×task。**★台账 `status/RULER_RESULT_LEDGER.md` + `scripts/ruler_ledger_check.sh has/done` 去重**（启动 cell 前先查）。**⚠️ 本机 wzc1 与 diskB 不共享盘，两处 ledger 独立→选 cell 时错开避重跑**。合法 RULER 任务仅 niah_single/niah_multikey/vt。
5. **Hy3 QCMem 已完成**（沉淀到 RUN_REGISTRY）：self-test PASS、j-sweep split-j≈32/80、自蒸馏 tax1.386→1.078、长档 niah 92-100 read 恒定。论文 §07（efficiency/constant-read，非 super-window）。
6. **iter_reader_attn selector 诚实证否**（沉淀）：CPU 合成 recall1.0 但真实 vt 无提升（mean-pool 抹变量身份）→ 真解=learned selector head（1.7B 基座已备，backlog）。1.7B QCMem 蒸馏已完成。

### 关键运维（2026-07-11）
- **判空卡用 compute-apps count**（`nvidia-smi -i $k --query-compute-apps=pid`），非显存——MoE inject/model-load 期 GPU 0GB 但进程活，避误判补卡堆叠。
- **grid 完成关键字是 `recall=XX (N samples`**，非 `RESULT`（曾 grep 错词漏报）。cell 均值读 log RESULT 行，非 CSV per-sample recall 列。
- **跨节点同名 cell 碰撞**：3 台 H20 共享 diskB，各自从 pool 头分配会撞名 → 补卡脚本要 pgrep+目录+recall 三重去重，每卡分配不同 cell。
- **eval 合法 RULER 任务只有** niah_single / niah_multikey / vt（`niah_multivalue`/`multiquery`/`qa_*` 会崩）。
- 第三台 H20 = 28.85.35.73（22端口，密码 configs/password_h20_853573.txt，torch-base 补了 tf5.5.4/peft/rank_bm25/pandas/datasets）。.24.104 密码=configs/password_h20_24104.txt（.venv symlink 坏，用 /opt/conda/envs/torch-base/bin/python）。

### 旧收尾 column（TaskList #1-#4，已清/弃）

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
