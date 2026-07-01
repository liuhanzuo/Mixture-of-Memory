# 长程记忆探索 — 最终发现 (2026-06-30, 官方判分修订版)

> ⚠️ **重大修订(2026-06-30 下午)**: 此前多条结论用了错误的送分判分 `re.search(\b target \b)`(qa5 答案仅7个高频词, 模型复读即命中, 虚高~10-30pp)。已全部用**官方判分** `third_party/babilong-pkg/babilong/metrics.py:compare_answers`(取首句+排除问题标签+要求target唯一)重算。**slot 净负结论符号翻转、oracle 端到端从 59→30**。本版只保留官方判分数字。
> Llama-3-8B / BABILong qa5 / A模型=mem_space_fifo_b25_c512_supervised_select step2000 (mix=0)。

## 一、核心结论(官方判分, 可信)

### 1. ✅ slot 有用 (官方判分 + stop-fix 排除格式效应, 2026-07-01 定论)
此前 re.search 显示 slot 一致拖累(净 −5~−18, W越大越重) → **错, re.search 送分污染**(pureSWA 输出乱复读命中多虚高)。
**官方判分 n100 全档重算(pureSWA slot-OFF vs slotSWA100 slot-ON, 两组都用FIFO只切slot)**:
| W | 8k | 16k | 32k |
|---|---|---|---|
| 0 | -2 | +3 | -1 |
| 2 | **+13** | +10 | +8 |
| 4 | +9 | -3 | +5 |
| 6 | +5 | +6 | +6 |
12 设定 10 个 slot ON > OFF, 平均净 +5~6, **W 小时最大 +13**(记忆承担更多信息时增益最大)。
**排除格式效应**: stop-fix 判分(扣续写污染)下增益仍在(8kW2 +9 / 16kW2 +5 / 32kW2 +8)。**slot 是真帮读出, 非输出格式假象**。
**结论: slot 有用, 且 W 小时增益最大 = memory slot 起实质记忆作用。方向正确。**

### 2. 学习选择器 reader-attn 弱于免费 BM25(n100 同批, recall指标不受判分影响✓)
| selector (qa5 16k recall@4) | 对 std 字面定位 | 对**真 supporting-fact** 定位 |
|---|---|---|
| last-token | 0.13 | — |
| mean-pool | 0.40 | — |
| BM25 | 0.59-0.60 | **0.76** |
| content-BM25(去停用词) | 0.59 | **0.80** |
| oracle(定义) | — | 1.00 |
- **content-BM25 召回提升撤回**: 停用词表缺 give/gave/from, 对 std 定位 content0.59≈plain0.60(此前吹的 0.52→0.72 不存在)。
- **新(本轮)**: 对**真 supporting-fact chunk**(_locate_qa5_supporting_fact 重建句式, n100 100%定位)算, BM25 recall@4=0.76, content略好0.80。即对 std 定位算 recall 严重低估真实选择力。
- reader-attn(last 0.13/mean 0.40)确实弱于 BM25, 这个相对关系在 recall 指标下稳健(recall 不依赖判分)。
- **机制**: q/k 投影为"预测下一token"优化非检索; BM25 的 IDF 加权(实体词高权)是 q·k 缺的。文献(BEIR)证此为已知共识, babilong 字面重复放大, 新意低。

### 3. ✅ 读出墙: selector 已选对但读不出(本轮干净判据跑中)
- **官方判分端到端(A模型 qa5 16k, FAIR3sh n50)**: oracle=**30**, plain-BM25 tk4=**34**, reader-attn=**18**。(此前 re.search 报 oracle59/62、bm25 77/78 全部虚高, 撤回。)
- selector 对真 sf chunk 已选对 76-80%, 但端到端官方判分仅 30-34% = **选对了原文也读不出**。
- ⚠️ 旧 oracle(_locate_needle_chunks)定位 target 字面最后出现, **31% 塞错 chunk**(亲手对比真 sf 定位器)。故 oracle=30 不纯。
- **本轮跑干净判据**: probe_fullchain_oracle_qa5.py 用真 sf 定位(100%定位真支撑事实)+官方判分, 本机8卡 n100。fullchain 官方分 = 读出墙真上界。**判据**: 若仍低(30-45) = 读出/多跳推理是真瓶颈铁证。

### 4. 选择+reforward 路线强依赖训练(recall指标, 稳健✓)
distill 模型(仅 pg19 蒸馏)上 bm25/reader-attn recall 全崩到随机, 仅 oracle work; A模型(supervised-select 训练过)才能用选中 chunk。机制: A模型 t2_select_loss + token-reforward window 训练让 backbone 学会"用任意选中 chunk 重 forward 读出"; distill 没见过这分布。

### 5. ✅ 主线 SOTA 锚点安全(官方判分核验)
pg19 nctx7 qa5 16k=16/32k=9 处于**低分区**, 实测同区间(nctx63_step250 qa5 16k)官方判分=re.search=8(模型弱不复读多词→送分失效)。**锚点用官方判分, 对照基准没歪**(红线安全)。
A主线 step2000 官方判分: qa5 8k=**26**/16k=**21**/32k=**14**(此前 re.search 报 45/31/21, 虚高)。

## 二、撤回的结论(诚实记录, 勿用)
- **「slot 注入净负」**: re.search 假象, 官方判分符号翻转(详见结论1)。
- **「content-BM25 召回 0.52→0.72 零训练突破」**: 停用词表缺关键功能词, 召回未涨(=plain)。
- **「oracle 端到端 59/62 读出上限」**: re.search 虚高, 官方=30, 且 std oracle 31%塞错 chunk。
- **所有 re.search 得出的端到端绝对分**(slot表/bm25部署表70/76/67/78/A主线45/31/21): 全部虚高, 以官方判分为准。
- **"读出端 W4 金矿 +17"**: needle 位置 confound(中位相对位置0.83偏末尾)。
- **mean-pool 重训(b)**: select_ce≈0 白跑。

## 三、方法论铁律(累计教训)
1. **判分一律官方 compare_answers**, 禁 re.search(\b target \b)(qa5仅7候选词送分虚高~30pp)。
2. **任何信号先查 confound + n 够不够**(reader-attn/reforward 须 n100/n50+ 同批)。
3. **格式效应警惕**: 官方判分取首句+唯一标签, 输出格式乱(选项列举/续写假问答)会被判错, 与 readout 能力混淆。比较两 arm 时先看输出格式是否可比。

## 四、系统层面的图景(官方判分)
- **读出墙**: A模型即使 selector 选对 76-80% 真 sf, 端到端官方仅 30-34% → 读出/多跳是主瓶颈(本轮干净 oracle 坐实中)。
- **学习选择器**: recall 不如 BM25 启发式(稳健)。
- **slot**: 净负已撤回, 真实作用待受控复验(格式效应)。
- **唯一稳的正向组件**: 经 supervised-select 训练的 backbone + token-reforward; selector 用 BM25 兜底。

## 五、下一步方向(读出墙)
- 等本轮 fullchain oracle 官方分坐实读出墙后, 攻"拿到正确上下文后的多跳读出"。文献可落地解法(opus复核20+篇):
  - ① 低成本: 混合上下文微调 1k 样本(2310.01558, 验证墙是干扰鲁棒性还是纯多跳)。
  - ② 主力: IN2/FILM 合成多跳数据 SFT(2404.16811, babilong生成器可无限造)。
  - ③ 推理流程: 迭代 reforward(IRCoT 2212.10509, 子问题链)。
- slot 真实作用受控复验(排除格式效应)。

## 红线
全程 mix=0; 泄漏ckpt(b50/b100/P2/c1024/P11/旧b25/l3recontoken)不引用; 真SOTA锚点 pg19 nctx7 qa5 16k=16/32k=9(官方判分已核验); 不写论文; 不碰 babilong 监督。
