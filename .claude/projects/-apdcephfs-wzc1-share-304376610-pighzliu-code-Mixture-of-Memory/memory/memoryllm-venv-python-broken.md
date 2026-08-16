---
name: memoryllm-venv-python-broken
description: "MemoryLLM env on diskB .73 — venv bin/python 被 reset 换成 py3.14,包在 py3.11;必须用 /usr/bin/python3.11 + PYTHONPATH=venv-site-packages"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 788acf89-c9f4-4bff-a535-3c7be5d412ee
---

**MemoryLLM (`YuWangX/memoryllm-8b-chat`, Llama-based) 在 diskB 上跑的 env 坑（2026-07-17 踩）：**

`external/memoryllm_venv/bin/python`(python/python3/python3.11 三个符号)在 2026-07-13 机器 reset 后全部被换成**系统 Python 3.14.6**,但 venv 的包(pandas 3.0.3 + transformers 4.43.4 + modeling_memoryllm 依赖)装在 `lib/python3.11/site-packages`。→ venv 自带 python 全都 `import pandas` 失败(No module named pandas),脚本 `run_babilong_memoryllm.py` line23 `import pandas` 秒崩。

**正确跑法(验证可用)：**
```
SP=<DB>/external/memoryllm_venv/lib/python3.11/site-packages
M=<DB>/../MemoryLLM-source
PYTHONPATH=$SP:$M:<DB>:<DB>/third_party/babilong-pkg /usr/bin/python3.11 scripts/run_babilong_memoryllm.py ...
```
`/usr/bin/python3.11`(真 3.11.6)+ 把 venv site-packages 放 PYTHONPATH → pandas 3.0.3 + transformers 4.43.4 都能 import,MemoryLLM 正常加载(Memory Pool Parameters 打印=OK)。

**Why:** MemoryLLM 需 pinned transformers 4.43(它的 modeling_memoryllm),不能用 torch-base 的 transformers 5.x;而 venv python 被 reset 打断。
**教训:** 别用 `external/memoryllm_venv/bin/python`;别 kill 正在正常跑的 MemoryLLM run 去重排序(我为了 0k-first kill 了正常 run,重启才踩这个坑)。FA2 not installed → 自动 fallback sdpa(正常,不影响)。
