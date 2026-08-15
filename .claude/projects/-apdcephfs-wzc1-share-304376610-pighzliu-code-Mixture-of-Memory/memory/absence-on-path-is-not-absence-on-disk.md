---
name: absence-on-path-is-not-absence-on-disk
description: "★2026-08-15 我因 `command -v pdflatex` 空 + /usr/local/texlive 不存在, 就告诉 agent「latex 编译不可能, 改静态校验」; 实际 TeX Live 2026 在仓库内 ./.texlive/2026/bin/x86_64-linux/, 正因在项目盘才活过重启 —— 与我的推论完全相反"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Rule**：`command -v X` 为空 **不等于** X 不在盘上。判「工具不存在」之前，必须查
**仓库内的自带 toolchain 目录**（`./.texlive/`、`./venv*/`、`./.venv*/`、`./vendor/`、`./node_modules/.bin/`），
以及**产物自己记录的版本证据**。

**Why**（2026-08-15 实测）：paperC 的 E3 要求「latex 编译通过、0 error」。我实测：

```
command -v pdflatex latexmk xelatex tectonic  -> 全部 NOT-FOUND
/usr/local/texlive, /opt/texlive, /root/texlive, /usr/share/texlive -> 全不存在
```

我据此告诉 subagent：**「texlive 被节点重启抹掉了，编译不可满足，改成静态校验，并且不得声称编译通过」**。

**这个结论是错的。** TeX Live 2026 在 **`./.texlive/2026/bin/x86_64-linux/`** —— **仓库内、项目盘上**，
只是不在 `$PATH`。agent 实测 `pdflatex --version` = `3.141592653-2.6-1.40.29 (TeX Live 2026)`，
与 `main.log` 完全一致，并真的跑出 `latexmk rc=0 / 19 pages / 0 errors / 0 undefined`。

**我的推理方向刚好反了**：我把「节点重启会抹掉 conda env / sshpass / tcodex」这条真实教训
（见 [[persist-artifacts-on-wzc1-or-diskb]]）**套用成了「所以 texlive 也被抹了」**。
但那条教训的**结论正相反** —— 放在项目盘上的东西**才不会**被重启清掉。
`.texlive/` 在 wzc1 项目盘，所以它**恰恰是活下来的那类**。

我手上当时就有反证却没用：`main.pdf` 的 mtime 是当天凌晨、`main.log` 明写 TeX Live 2026。
「今天还编译过」+「现在查不到」的正确解释是**没在 PATH 上**，而不是**被删了**。

**How to apply**：
1. 判工具缺失前，按此顺序查：`command -v` → **仓库内 toolchain 目录** → 有界系统路径。
   **绝不 `find / -maxdepth N`** —— CephFS 全盘遍历会挂几十小时（同日发现一个 `bfs /` 跑了 18.8 h）。
2. **产物即证据**：若 artifact 是近期生成的（pdf/log/wheel），它的 log 里通常写着用的哪个版本，
   顺着它找 binary，别从「不存在」出发。
3. 给 subagent 降低验收标准前**先自证不可满足**。我这次把一个**能做的真编译**降级成静态校验，
   还额外命令它「不得声称编译通过」—— 幸好 agent 自己去核了 audit 行点名的路径，
   否则我会让一个可验证的结论永久停留在「未验证」。
4. 反过来，agent 说「你的前提错了、而且对你有利」时，**先复核再高兴**：我复核了
   `latexmk rc=0` + `main.aux` 里表格 label 已解析 + checker 81/81，才接受。

**Related**：[[persist-artifacts-on-wzc1-or-diskb]]（被我用反了的那条）、
[[two-disk-rule-applies-to-main-too]]（「不存在」的举证门槛同型教训）、
[[one-sample-is-not-a-trend-or-state]]（单次探测不足以定状态）、
[[read-env-not-source-defaults-for-running-procs]]（该查运行时真相而非自己的假设）。
