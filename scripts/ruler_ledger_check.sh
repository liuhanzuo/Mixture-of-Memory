#!/bin/bash
# ruler_ledger_check.sh — QCMem RULER 结果台账 / 去重助手
# 用法:
#   bash scripts/ruler_ledger_check.sh scan          # 重新扫描两盘, 刷新 status/RULER_RESULT_LEDGER.md
#   bash scripts/ruler_ledger_check.sh done           # 打印所有已完成 (task,length,topk) 一行一个
#   bash scripts/ruler_ledger_check.sh has TASK LEN TOPK  # 查某 cell 是否测过, 测过 exit 0 并打印
#
# 目的: 启动任何 QCMem RULER n=100 cell 前先 `has` 一下, 避免重测 (用户 2026-07-11 指令).
# 已完成来源: logs/qcw_qcmem_n100_*.log 里的 "recall=XX.XX (N samples" 行 (diskB, 权威).
# 历史 n=50 来源: ruler_results/*/_summary.json (本机 wzc1).
set -euo pipefail
R="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$R"

cmd="${1:-done}"

# 已完成 (task,length,topk) 从 n=100 log 抓 (跨盘: 本命令应在挂 diskB 的节点上跑, 或本机看本机 logs)
list_done_n100() {
  for f in logs/qcw_qcmem_n100_*.log; do
    [ -f "$f" ] || continue
    # 只算真完成 (有 recall=XX (N samples)
    grep -qE "recall=[0-9.]+ \([0-9]+ samples" "$f" 2>/dev/null || continue
    nm=$(basename "$f" | sed 's/qcw_qcmem_n100_//;s/.log//')
    # nm 形如 niah_single_tk8_16k -> task=niah_single topk=8 len=16k
    task=$(echo "$nm" | sed -E 's/_tk[0-9]+_[0-9]+k$//')
    topk=$(echo "$nm" | grep -oE 'tk[0-9]+' | tr -d 'tk')
    len=$(echo "$nm" | grep -oE '[0-9]+k$')
    echo "$task $len tk$topk"
  done | sort -u
}

case "$cmd" in
  scan)
    echo "[ledger] 扫描请用 main agent 的 /tmp/build_ledger2.py (需两盘 json). 本 helper 只做 done/has 快查." ;;
  done)
    list_done_n100 ;;
  has)
    task="${2:?task}"; len="${3:?len}"; topk="${4:?topk}"
    key="$task $len tk${topk#tk}"
    if list_done_n100 | grep -qxF "$key"; then
      echo "ALREADY_DONE: $key (n=100, 见 status/RULER_RESULT_LEDGER.md) — 不要重测"
      exit 0
    else
      echo "NOT_DONE: $key — 可以跑"
      exit 1
    fi ;;
  *)
    echo "usage: $0 {scan|done|has TASK LEN TOPK}"; exit 2 ;;
esac
