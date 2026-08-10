# 中间结论 — reader-attn 选择路线收口 (2026-06-28)

## 已确认收口的结论(全部干净 mix=0, 满/近满100)

### 1. 读出墙: 已解(token-reforward)
- 纯 memory hidden 读出(W0)受限: qa5 8k 训练前12, oracle完美隔离needle的hidden快照也只~20-24。
- token-reforward(选中chunk的原始token重新forward, query在场)→ oracle qa5 8k=66。
- 机制: 冻结hidden快照query-blind, 重读原token让needle在query在场下重新contextualize才破墙。
- W0也被训练改善: 12→28→34(破20墙), 但仍远低于token-reforward = 廉价但弱的部署路径。

### 2. 选择(reader-attn): 短档可训, 长档收口(墙)
- 8k(候选集16 chunk小): reader-attn选得准, 监督训练28→46(逼近oracle 66), step1000后平台。qa1不退化。
- ★长档(16k/32k, 候选集25=buffer cap): reader-attn small-K选择 ≈ 随机:
  - recall@4=0.15-0.17(chance 0.16), recall@8=0.34-0.44(chance 0.32), recall@16=0.62-0.73(chance 0.64)
  - 中位rank 7-10 / 25候选(中游) → 有微弱信号但信噪比不够, 精排能力弱
- ★扣floor真相: qa5答案空间仅7个词, 不读needle蒙对floor≈13。部署净分=38-13=+25, oracle净分=63-13=+50 → reader-attn只拿到完美选择reforward收益的一半。

### 3. 长档瓶颈定位(证伪了2个假说)
- eviction假说【证伪】: keep_all(装全64 chunk)qa5 32k=15 < evict(cap25)=32, 加大buffer有害。
- 瓶颈不是召回/eviction, 是【大候选集下选择精度↓】(distractor稀释打分信号)。

### 4. SOTA成果(干净, 远超锚点)
- K4部署 qa5 16k=38/32k=32 vs pg19真锚点16k=16/32k=9 → 2.4-3.6×, 达MemoryLLM teacher水平, 无泄漏。

### 5. 与MemoryLLM(已联网核实)
- 我们=select-then-reforward(无损原token+query-conditioned+训练selector) vs MemoryLLM=compress-then-inject(token数压缩+全维latent+KV注入+随机遗忘), 相反范式。
- 我们实测inject式(Method A raw-KV)给对证据只+1-2.5(frozen reader用不上)才转reforward。

## ★下一方向候选: slot 全局表示 + reforward 结合(重新框定)
- 昨天slot+reforward否决理由(slot选needle 22% < reader-attn 55%)基于8k; ★长档reader-attn已掉到≈随机, 否决理由在长档不成立。
- 重新框定: 不是"slot当检索器", 而是"用slot的【训练时持续更新+跨chunk聚合的全局表示】做更强粗筛/检索, 再reforward选中chunk原token"。slot有reader-attn没有的全局信息(reader-attn只是单层q·k局部打分)。
- 卡点(昨天读码): slot_kv存hidden非token-id、无document-chunk-id、整chunk复制非精确定位 → 要真reforward需加chunk-id写入通道(~120-160 LOC)。
- 待验证: slot全局表示选chunk是否比reader-attn单层打分在长档更准? (新的Gate Probe)
