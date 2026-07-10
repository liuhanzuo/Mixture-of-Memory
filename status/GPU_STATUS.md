# GPU_STATUS.md — 两节点 GPU 实时台账

> **每次启动/kill GPU 任务时更新**（CODEBUDDY 规则）。heartbeat 先读此文件→对照 nvidia-smi→台账说跑但实际空=补卡。
> ★29.162.226.120=dllm专用不碰。最后更新：2026-07-10 10:20 GMT+8

## 本机 B200/L20A (wzc1, 8卡) — 满载
- LoCoMo QCMem 多 topk 对照(tk4/tk6/tk12 各 shard, 补 LoCoMo 最优 topk) + LoCoMo kvdirect baseline(Task#1) + babilong 低档补档(Task#4, 快, 完即补)
- 全部耐跑(LoCoMo 对话长). 教训: 别用 babilong 低档(秒完)填, 已改 LoCoMo

## H20 (28.83.53.31, diskB, 8卡) — 满载(空4)
- MemoryLLM RULER 64k/128k 超长档(inject 极慢, 数h) + MemoryLLM LongEval 8k/16k/32k(Task#3)

## 待跑队列(见 QCMEM_AUTONOMOUS_AGENDA.md)
1. Task#1 LoCoMo baseline 聚合→draft §2.9  2. Task#3 MemoryLLM LongBench  3. Task#4 babilong 补全
4. 全清→自主议程(方向1查漏/2 pretrain scale/3 infra/4极简架构探针)

## 备注
- MemoryLLM inject 极慢 util 常低但在跑, 查 log 进度别误判卡死
- durable cron a065ebb2(7/27/47分)每20min自动heartbeat; tmux tclaude常驻
