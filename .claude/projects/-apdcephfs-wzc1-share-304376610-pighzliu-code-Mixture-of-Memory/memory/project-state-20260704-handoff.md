---
name: project-state-20260704-handoff
description: 2026-07-04 主力机 offline 后接力状态——最新突破、待跑清单、本机 B200 决策
metadata: 
  node_type: memory
  type: project
  originSessionId: 0dac7a11-5048-4ecf-85c3-ff6b9fab88d3
---

2026-07-04 主力机(diskA `_zwfy6/share_303098609`)offline,接力在本机 B200(NVIDIA L20A 183GB, `_wzc1/share_304376610`)。本机 git 曾落后 origin 506 commit,已 reset 到 origin/main(6bbe346)+ 7 个 local-ahead 文件(QCP/tree-readout/reforward-guard,已 commit 5d613ec 本地未 push)。

主力机最新工作(07-01→07-03)不在 git,通过跨 ceph 目录拷贝同步到 `_wzc1/.../MoM_mainserver_20260704/`(工具是整目录一对一拷,清单见 SYNC_MANIFEST_20260704.txt)。

**核心突破(你记忆里"objective 改进效果很好",已确认,官方判分干净 n=100)**:PG19 pretrain + 合成 qa5 SFT 两阶段。qa5: A基线 2k60/4k40/8k17/16k8 → +SFT step500 2k62/4k49/**8k28/16k16**(16k 追平项目 SOTA 锚点)。level8 多模板 qa5(commit db37ecb, 5 模板 T1-T5)是升级,07-03 在训 step10 未出结果。

**两堵墙(qa5 16k 官方判分)**:读出墙(oracle 完美选择端到端仅 30,纯 hidden FIFO 更低)+ 选择墙(重训 selector 24 vs oracle 43)。关键诊断:真瓶颈是读出墙;qa5 needle 是 recency-biased,b25 纯 recency 窗 92-100%,选择墙对 qa5 没咬住 → HNST 树路线裁决"present 形式不值得继续"。

**待跑清单(最高价值,几乎没跑)**:
1. 文献三方共识数据配方修复(`LITERATURE_TRAINING_DATA_20260702.md`):主 loss 从"只答案 NIAH"改成 dense LM + recall ≤20% 辅料 + 同词表 distractor + 随机 needle 位置。过拟合根因就是 NIAH 当主信号(t2longgap 的 t2_needle=0 是活证据)。
2. `HIDDEN_READOUT_EXPERIMENT_DESIGN.md` 实验2:蒸馏 reforward→hidden tree(--distill_hidden+--distill_kl)。
3. 多源 SFT(你 07-03 要求):加非 babilong 通用长文档 QA,证明通用能力非 babilong 定制。

**本机当前**:t2longgap_16k_b64 跑到 step2490/6000(=实验1 纯 hidden 读出监督变体,t2_select_loss_weight=0,buffer64,gap 3584/8192/15872)。ckpt step500/1000/1500/2000 在手。干净 A 模型 ckpt 在 `outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt`。

红线:所有训练 babilong_mix=0(代码硬 guard);判分只用官方 compare_answers(禁 re.search,曾虚高 ~30pp 污染大量结论);泄漏 ckpt(b50/b100/c1024/P11/旧b25)不碰不引用;真 SOTA 锚点 pg19 nctx7 qa5 16k=16/32k=9。
