---
name: h20-papera-over-paperb-priority
description: "★用户2026-08-01:三个QCMem H20节点(.73/.82/.104)长期先跑PaperA再PaperB,任何节点空出优先补PaperA待跑项,排空后才resume暂停的PaperB"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
  modified: 2026-08-04T08:04:12.070Z
---

用户 2026-08-01 指令（两条，后一条是长期化）：
1. "这三个H20你记得要先跑paperA" —— 三个 QCMem H20 节点 .73(28.85.35.73)/.82(28.82.250.82)/.104(28.83.24.104) 收回给 Paper A。
2. "我的意思后面也是先跑paperA再B" —— **这是长期优先级策略，不只当前这批。**

**How to apply（heartbeat 补卡决策铁律）：**
- 任何 H20 节点空出 → 先查 Paper A 有没有 pending 实验（`paperA/TODOList.md` 的 P0.2/P1.1/P1.2/P1.5，之后 P0.8/P2.1/P2.2、#10 论文集成等）→ 有就**先起 Paper A**。
- **⚠️ 2026-08-04 更新：那三个"暂停的 depth-ladder heal"（keep8/keep10/keep12）已全部跑到 200k 收尾**——paperB/TODOList.md 行 248-250 显示 keep8(10L)/keep10(12L)/keep12(14L) 均 `[DONE]` 200k（对应 task #95/#96/#114 completed）。**所以"resume 暂停 heal"这条已无对象可 resume，别再据旧 memory 去找 keep8@45000 之类的断点起训。**
- **只有 Paper A 待跑项全部排空后**，才考虑 Paper B 训练；但当前（2026-08-04）Paper B 剩余训练均被用户明确 defer/park：#84 contamination-ablation(deferred)、#99 keep14-distill-heal(PARKED per user)、#123 general-SFT-pipeline(用户 cost-benefit 判太贵)。**这些不属于 heartbeat 可自主 auto-launch 的范围**（会违背用户明确的 defer 指令）→ H20 空出时若 Paper A 也排空，宁可 idle 也不擅自起被 defer 的训练。
- LOCAL B200 ShortGPT16(#98/#145) 已跑完 closed-book QA 收尾。

**Why:** Paper A（CoMem/QCMem，ARR 在投）是当前最高优先级交付；Paper B（OLMo-2 剪层-heal depth-ladder）可暂停+resume，机会成本低。用户把这条从"当前这批"升级为"长期"，所以要写进持久调度规则而非一次性。

相关：[[paper-eval-chat-false-mandatory]]、[[paperb-olmo2-base-not-chat]]、[[qcmem-eval-selector-iterbm25]]
