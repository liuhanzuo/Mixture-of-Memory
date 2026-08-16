---
name: qcmem-eval-selector-iterbm25
description: "用户指令(2026-07-17):所有 QCMem eval 统一用 selector=iter_bm25,覆盖旧的 bm25/per-task 混选"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 788acf89-c9f4-4bff-a535-3c7be5d412ee
---

**2026-07-17 用户指令:所有 QCMem eval(RULER / BABILong / LongBench / LoCoMo / vs-Dense,所有 scale / 所有 j / 所有 task)统一用 `selector=iter_bm25`。** 不再用 bm25 单遍或 per-task 混选(qa 用 bm25、vt 用 iter_bm25 的旧口径作废)。

**Why:** 用户要统一 selector 口径,消除跨 cell 的 selector 不一致(audit 发现的协议混乱之一)。

**How to apply:**
- taskpool 启动传 `SELECTOR=iter_bm25`;单 cell 传 `--selector iter_bm25`。
- 之前用 bm25 跑的结果(如 8B-adapter BABILong chat+nothink 得 62.2;iter_bm25 重跑得 57.1)**作废/以 iter_bm25 为准**。
- chat+no-think 是另一条标配(`--use_chat_template`,enable_thinking 默认 False);QCMem 生成边界 no-think 前缀由 commit `c056a6d` 修好。
- **2026-07-17 加强:所有 benchmark + 所有 baseline(含 MemoryLLM/HCache/KV-Direct/Dense)都用同配置测,保证可比。**
  - HCache/KV-Direct/Dense 走 `eval_ruler_qcmem.py --baseline`/`eval_qcmem_babilong.py`(有 chat 旗标)→ 传 `--use_chat_template` 重跑。
  - ⚠️ **MemoryLLM=`YuWangX/memoryllm-8b-chat`=Llama-based**:无 thinking(enable_thinking 仅 Qwen3,no-think 天然满足)、无 bm25 selector(内部 stateful memory,iter_bm25 不适用)。同配置对它=用 chat template。需专用 env(`../MemoryLLM-source`+`external/memoryllm_venv`),单独 track。
- 已同步写入 `HEARTBEAT.md` 顶部「EVAL 统一协议」块(每轮 heartbeat 必读)。
