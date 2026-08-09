# B07 — Mutable and Concurrent CoMem Service

## 状态

**BACKLOG SYSTEMS。现有是组件 benchmark，不是生产端到端系统。**

## 目标

把 depth-residual object 升级为：

- 可并发；
- 可版本化；
- 可增量编辑；
- 可跨 HBM/CPU/NVMe/network 分层；
- 可按预期 reuse 决定 Write 或 raw replay。

## 两个核心实验

### 1. 并发 serving

- document 32k/128k
- generation 128
- concurrency 1/8/32
- CoMem vs matched `j=0`

指标：TTFT p50/p95/p99、QPS、queue、OOM、HBM/host、真实 `Q*`。

### 2. 增量编辑

随机修改 1%/5% chunks，比较：

- full rewrite
- edited chunk only
- chunk + overlap neighbor
- lazy rewrite + raw fallback

指标：update latency、rewritten bytes、stale read、unaffected-query equivalence、
quality near edit、invalidation fan-out。

## 关键设计

- chunk/version/content hash
- backbone/tokenizer/split/adapter compatibility hash
- fail-closed stale object
- dependency graph for overlap
- mixed-version read only with explicit fallback

