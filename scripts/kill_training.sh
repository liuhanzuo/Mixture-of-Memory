#!/bin/bash
# Kill training processes on this node, excluding SSH session
TARGET=$1
# Only kill python processes with train_mem_space in cmdline, not bash/ssh
for pid in $(ps -eo pid,cmd | grep 'python.*train_mem_space' | grep -v grep | awk '{print $1}'); do
  kill -TERM $pid 2>/dev/null
  echo "Killed python worker $pid"
done
sleep 2
# Kill remaining torchrun
for pid in $(ps -eo pid,cmd | grep 'torchrun' | grep -v grep | awk '{print $1}'); do
  kill -TERM $pid 2>/dev/null
  echo "Killed torchrun $pid"
done
sleep 3
echo "Remaining train_mem_space processes: $(ps -eo cmd | grep 'train_mem_space' | grep -v grep | wc -l)"
