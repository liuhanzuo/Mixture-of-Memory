# 晨间总结 — 一夜探索成果 (2026-06-30 凌晨)

> 给回来的你: 一夜从"mean-pool以为是突破"到证伪、再到读出墙诊断的完整脉络。**待你拍板的决策在最后。**
> ⚠️ 06-30 07:30 修订: 原"读出端W4金矿+17"已撤回 — 那是 needle 位置 confound(babilong needle 偏末尾0.83, W4 窗口恰框住70%样本needle), 非真读出能力。真读出基准是 oracle 端到端 59。详下。

## 一句话(修订版)
slot注入是死路、零训练选择器追不上BM25、读出端即使把needle强制喂进window(oracle)端到端也只59(41%真损失)。多个独立负结果指向: **选择用BM25兜底, 真瓶颈在读出端(oracle 59<满分)**。

## 这一夜的坚实结论(全部满n100 / 同probe / 已查confound, 可信)

### 1. slot记忆注入净负(三档n100, +机制坐实)
8k/16k/32k全W, slot ON一致 < slot OFF(-2~-18), W越大拖越重。
机制(读代码): inject_gate是标量常数(g≈0.12无差别注入)+ slot有损压缩 + position-0 → 注入的是噪声。
**含义**: slot存了信息(slot+SWA×2-3)但"注入压缩内容"路径错; 应"用slot选chunk重forward"。

### 2. 学习选择器 vs BM25(reader-attn recall, n100同批)
| selector | recall@4 |
|---|---|
| last-token(原始) | 0.13 |
| mean-pool | 0.40 |
| IDF-mean | 0.39 |
| mean+bm25 fusion | 0.36 |
| **BM25** | **0.52** |
| oracle | 0.72 |
- mean-pool >> last(0.40 vs 0.13), 但所有零训练变体都 < BM25。
- reader-attn r@1强(mean 0.28>bm25 0.19, 选top1准)但覆盖弱=结构性。
- full-query-attn证伪(0.05, 被停用词噪声主导)。

### 3. 读出墙: oracle 59才是真上限(W4金矿已撤回=位置confound)
- ⚠️ W-sweep(W2/W4/W16...)受 **needle位置confound**: swa_eval_chunks=W喂最后W+1个chunk, 而babilong needle中位相对位置0.83(偏末尾), 16k 70%样本needle在最后5 chunk → W4(喂末尾5)高=框住needle的巧合, 非读出能力。原"W4=48金矿+17"撤回。
- **真读出墙(无confound)**: oracle_token强制把needle chunk放进window(不管位置), 端到端=59 → 即使needle在场, 读出端仍有41%损失。这才是真瓶颈。
- needle_excluded W4对照跑中(拆confound)。

## 我犯过的错(诚实)
- mean-pool: n7吹"翻倍"→n50误判"证伪"→n100才看清"真强(0.40)但不及BM25"。
- W4"金矿": 没查needle位置就吹+17, 实为位置confound。
- slot W2: n60噪声看成"+2微正", B200满100纠正。
- **铁律(本夜3次教训): 任何信号先查(a)有无confound (b)n够不够。** 已固化。

## ★待你拍板的决策
**选择器走哪条**:
- **(a) 认账BM25做selector**: r@4=0.52最强, 免费零训练, 立即可部署。学习选择暂不如启发式(诚实结论)。
- **(b) mean-pool重训**: 纯推理已0.40, 训练推理一致(MEM_SALIENCE_QPOOL=mean)可能补上覆盖、超BM25。代码已就绪可启。
- **(c) 攻读出端**: oracle 59<满分(41%真损失)。但需先搞清41%损失来源 — 很可能oracle只定位target字面chunk、漏qa5多跳推理链(=oracle定义缺陷, 已坐实结论3)。即full-chain oracle(含全部supporting facts)真上限可能>59。

**我的建议**: (c)+(a) 组合 —— selector用BM25(够用), 主攻读出端headroom(W2-4最优已知, 下一步看怎么把oracle端到端从59往上推)。但等你定。

## 其他
- full-chain oracle(C方向): agent探查中(oracle当前只定位target字面chunk, 可能漏qa5多跳推理链→低估)。结果未定。
- 5节点: 读出墙W曲线满n50收尾中(reforward慢)。heartbeat正常。
- 红线全程守(mix=0, 不碰泄漏ckpt, 不写论文/不碰babilong监督)。
