---
name: paper-eval-chat-false-mandatory
description: "★用户2026-07-22:全论文所有结果统一chat_template=False,并明确写明我们模型无SFT/RL故用chat template不公平;旧chat=True+no-think数字全作废需重跑"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

用户 2026-07-22 指令：**论文（Paper A QCMem + Paper B OLMo-2）的所有 eval 结果统一按 `chat_template=False` 报告**，并在论文里**明确指出：我们的模型没有 SFT/RL（是 continue-train 的 BASE LM），因此使用 chat template 不公平**（chat 格式的特殊 token 是 instruction-tuned 模型才见过的，对 base 模型 OOD）。

**Why:** CoMem 是在 base backbone 上加 adapter 续训（无指令微调），OLMo-2 剪层-heal 也是 Dolmino NTP 续训无 SFT。chat_template 注入 `<|im_start|>user...` 等格式 → base 模型从没见过 → 不公平且 OOD。base 模型的原生 eval 协议 = raw text completion (chat=False)。这与 [[paperb-olmo2-base-not-chat]] 一脉相承（Paper B 早已 base-only），现扩展到整个 Paper A。

**被 Task#44 验证:** LoCoMo chat T/F 消融显示——chat=True 时 token-F1 KVD 40 ≫ CoMem 19.5（假象），但换 chat=False 后 F1 打平（9.02≈9.15），GPT-4o judge 两协议都打平（CoMem 37.76~38.27 ≈ KVD 34.59~38.22）。→ chat=True 的差距是 chat-template 伪影；chat=False 才是公平口径。

**How to apply:**
- 论文所有 benchmark 表（tab_locomo / tab_overview / RULER / LongEval / LongBench / BABILong）的 CoMem + 所有 baseline 数字，一律用 chat=False 重跑口径；现有 `*_chatnothink`（=chat=True+no-think）dir 的数字**作废**，需 chat=False 重跑。
- 与 [[qcmem-eval-selector-iterbm25]] 叠加：QCMem eval 的两根协议支柱 = **selector=iter_bm25 + chat_template=False**（no-think 仍保留，因为 chat=False 本就没有 thinking）。
- 已有 chat=False 数据（2026-07-22）：LoCoMo 的 CoMem(iter 9.15/iter-judge 38.27)、KV-Direct(9.02/judge 34.59)、HCache(4.67)、CoMem-bm25(8.76)；baseline 三方(InfLLM/Streaming/MemoryLLM) chat=False 由 coder abaed0fc 在 .82 跑中。其余 4 benchmark（RULER/LongEval/LongBench/BABILong）仍全是 chat=True → 待 chat=False 重跑（GPU 排队在 Paper B 训练之后）。
- 论文正文要加一句方法学说明：baseline 与 CoMem 共用同一 base backbone、同一 chat=False 协议，公平对照。
