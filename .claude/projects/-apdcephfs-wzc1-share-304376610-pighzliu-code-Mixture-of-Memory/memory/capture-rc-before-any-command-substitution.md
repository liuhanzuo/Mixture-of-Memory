---
name: capture-rc-before-any-command-substitution
description: "`printf \"%s rc=%s\" \"$(basename $f)\" \"$?\"` 里 $(...) 先跑并把 $? 重置为 0 → 整轮 gate sweep 的退出码全是伪造的; rc 必须在任何替换之前存进变量"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**任何命令替换 `$(...)` 都会重置 `$?`。** 所以 `printf "%-40s rc=%s\n" "$(basename "$g")" "$?"` 打印的**永远**是 `basename` 的退出码 0，不是你真正关心的那条命令的。

**Why:** 2026-08-16 我用这一行跑 paperC 的 5 个 gate sweep，得到 **5/5 rc=0**，据此准备提交。但 `gate_exact_floor_tail.py` 是确定性 rc=1（它自己 docstring 写明 1 = informational）。连跑三次都是 rc=1，说明**不是 gate 抖动，是我的循环在编数字**。最小复现：

```bash
false; printf "rc=%s\n" "$(basename /a/b)" "$?"   # 打印 rc=0，真实是 1
```

同一轮里这个 bug 还**掩盖了一次真 rc=2**：`gate_build_record_matches_pdf.py` 在第一次（写法正确的）sweep 里报 rc=2，我修好后重跑用了上面的坏写法 —— 如果修没生效，我不会知道。

这与 [[a-pipe-makes-a-failing-command-report-success]] 是**同一个错误的第二套外衣**：都是「我的测量方式把失败读成成功」。管道版我已经写过独立条目，但我只记住了「不要管道」，没抽象出**「rc 必须紧贴命令取，中间不能有任何东西」**这条更一般的规则，于是换个语法就又中招。

**How to apply:**
- **循环里报退出码，先把替换算完再跑命令**：
  ```bash
  for g in code/gate_*.py; do
    name=$(basename "$g")     # 替换在前
    python "$g" > /tmp/o.txt 2>&1
    rc=$?                     # 紧贴命令，中间零语句
    printf "%-40s rc=%s\n" "$name" "$rc"
  done
  ```
- **判据：`$?` 与它所指的命令之间不能有管道、不能有 `$(...)`、不能有别的命令。** 有就先 `rc=$?`。
- **自查信号：「全部通过」比「部分通过」更该怀疑。** 一个确定性 rc≠0 的 gate 突然报 0，先怀疑测量而不是庆祝。我这次是因为**记得那个 gate 应该 rc=1** 才发现；如果都是新 gate，我会直接带着假的「全绿」去提交。
- 通用形式：**凡是「我的工具报告成功」而不是「产物本身显示成功」，都要问一句这份 rc 是怎么拿到的。**

见 [[a-pipe-makes-a-failing-command-report-success]]、[[read-what-the-consumer-reads-not-the-bare-key]]。
