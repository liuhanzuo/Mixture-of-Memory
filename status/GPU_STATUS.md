# GPU_STATUS.md — 两节点 GPU 实时台账

> **每次启动/kill GPU 任务时更新此文件**（CLAUDE.md 规定）。heartbeat 每轮先读此文件 → 对照 nvidia-smi 实测 → 发现"台账说在跑但实际空"= 任务完成/崩溃，需补卡。
> 节点：本机 wzc1 B200/L20A(8卡) + H20 28.83.53.31 diskB(8卡)。★29.162.226.120=dllm专用不碰。
> 最后更新：2026-07-10 10:20 GMT+8

---

## 本机 B200/L20A（wzc1, 8卡）
| GPU | 在跑 | 起始 | 预计 |
|--|--|--|--|
| 0 | LoCoMo kvdirect shard0 (Task#1) | 10:03 | ~1-2h |
| 1 | LoCoMo kvdirect shard1 (Task#1) | 10:03 | ~1-2h |
| 2 | babilong kvdirect qa5 2k | 10:12 | 快 |
| 3 | babilong kvdirect qa2 4k | 10:20 | 快 |
| 4 | babilong kvdirect qa2 8k | 10:12 | 快 |
| 5 | babilong hcache qa2 8k | 10:12 | 快 |
| 6 | babilong kvdirect qa5 4k | 10:12 | 快 |
| 7 | babilong kvdirect qa2 4k | 10:12 | 快 |

## H20（28.83.53.31, diskB, 8卡）
| GPU | 在跑 | 起始 | 预计 |
|--|--|--|--|
| 0-3 | MemoryLLM RULER 64k/128k (超长档, inject极慢) | 09:xx | 慢, 数h |
| 4 | MemoryLLM LongEval 32k shard0 (Task#3) | 10:20 | 慢 |
| 5 | MemoryLLM LongEval 32k shard1 (Task#3) | 10:20 | 慢 |
| 6-7 | MemoryLLM LongEval recheck (Task#3) | 10:20 | 慢 |

---

## 待跑队列（GPU 空出后按此接, 见 QCMEM_AUTONOMOUS_AGENDA.md）
1. TaskList #1 LoCoMo baseline 三方 → 聚合补 draft §2.9
2. #3 MemoryLLM LongBench/LoCoMo
3. #4 babilong 三方补全档（进行中）
4. 全清 → 自主议程方向1-4（查漏/pretrain scale/infra/极简架构探针）

## 备注
- ⚠️ MemoryLLM eval 极慢（inject_memory 串行，util 常显示低但在跑，别误判卡死；查 log 进度确认）。
- ⚠️ babilong 低档(0k-8k)快，几分钟完，易空转 → heartbeat 勤补。
