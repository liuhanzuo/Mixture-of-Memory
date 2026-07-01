# RESEARCH_ROADMAP.md — 当前研究路线图

> 用户 2026-06-06 口头总结 + 下一步方向，作为权威路线图。每个方向推进时回填结果。

最后更新：2026-06-06 16:50 GMT+8

---

## A. 已确立的发现（findings）

### F1. 长 chunk size 训练显著优于短 chunk size（高置信）
- 证据（RUN_REGISTRY §3，同口径 BABILong n=100）：
  - chunk1024 基线 qa1_0k=89 / qa5_2k=93；chunk128 p7p9 qa1_0k=55 / qa5_2k=36。
  - chunk256_p8_nullsink step500 qa1_0k=98、qa5_0k=78（接近 chunk1024）。
- 机理：缩小 chunk 削弱每步局部窗口（SWA），长句掉分；同时注入频率= seq_len/chunk_size 升高，放大不稳定。
- **chunk size 是目前最大的杠杆。**

### F2. 训练存在不稳定（TF-loss spike），但「是否专属 chunk=128」存疑（中置信）
- 现象：null-sink P8 各臂 step500 健康、step1000 崩；chunk128 崩成 token 重复死循环乱码，chunk256 崩成连贯续写 haystack。
- 根因（gp-35 报告，high）：chunk128 的 TF lm 在 step895-1010 飙到 PPL~3000，step1000 ckpt 恰好撞在 spike 顶；chunk 越小注入越频繁（128=256 的 2×）放大崩坏。
- **存疑点**：spike 是否所有 chunk 都会出现、只是 chunk128 因注入频率把它放大成乱码？还是 chunk128 独有？→ 需对照各 chunk 的 TF-loss 全程曲线 + 固定 seed 复现，确认 spike 普遍性。
- 操作结论（当前）：按 TF-loss-min 选 ckpt，不要固定步数。

---

## B. 下一步方向（directions，按用户 2026-06-06 列举）

### D1. slot_dim scale-up 4096 → 16384 是否有用
- 现状：唯一的 16384 run（wbmode_lowrank）启动即崩、无 ckpt 无 eval（RUN_REGISTRY §4.2）。
- 待办：修 wbmode 启动失败 → 跑一个干净的 16384 对照（其余 verbatim chunk512 p8_nullsink）→ 同口径 BABILong vs 4096。
- 前置依赖：无（可独立起）。

### D2. previous-chunk SWA + slot 是否有用（核心方向）
- 当前 eval **无 cross-chunk SWA**：前 N-1 chunk 只喂 memory，仅最后 chunk 正常 forward（train_mem_space_dolmino_cpt.py:1326）。可能系统性低估真实能力。
- 三个子问题（用户明确）：
  - (a) 训练时是否更换 objective（加上 SWA）？
  - (b) 一开始就换 / 后期再加 / 只在 eval 时开？
  - (c) 三种策略的对照。
- 待办：先做「只在 eval 开 SWA」的零训练对照（最便宜，验证 eval 是否低估）→ 再决定是否改训练 objective。

### D3. 不同长度 chunk 配不同 slot 容量（依赖 D1 结果）
- 想法：chunk128 的 slot 轻量、chunk1024 的 slot_dim 大。
- 前置依赖：**必须先有 D1（slot_dim scale-up 是否有用）的结论**，否则无法定容量。
- 状态：BLOCKED by D1。

### D4. 逐步提升 chunk size 的 curriculum 训练（H800 在训）
- 现状：H800 在跑渐进式 chunk size。⚠️ 注意 CODEBUDDY.md 记录「H800 旧 IP 全失效」——需先确认 H800 是否还活着、当前 run 真实状态（gp-1 正在查 H20/H800 效率，可能有交集）。
- 待办：确认 H800 run 状态 → 出 ckpt 后同口径 eval vs 固定 chunk。

### D5. 是否所有 slot 方案都无法解决 LongEval
- 假说（RUN_REGISTRY §4.4）：slot 装 token 级 hidden 而非语义摘要 → BABILong（NIAH 事实定位）相对行、LongEval（需全局总结）弱。
- 待办：跑现有最佳臂（chunk512 step500 / chunk256_nullsink）在 LongEval 上的系统对照，确认是否结构性短板。

### D6. xattn 是否真的有用
- 背景：old-P8 的 xattn 读路径全 FROZEN（q/k/v/out_proj + 门控 + zero-init null_value），注噪声导致 0k 也塌；null-sink 版把 xattn 改可训练+存盘后 0k 从 11→98。
- 但这混淆了「xattn 有用」与「null-sink 有用」两个变量。
- 待办：单变量对照 —— (xattn ON, null-sink ON) vs (xattn OFF, null-sink ON) vs (xattn ON, null-sink OFF)，隔离 xattn 的真实贡献。

---

## C. 优先级 / 依赖图

- 可立即并行（无依赖）：D1（修 wbmode + 16384 run）、D2a（eval-only SWA 零训练对照）、D6（xattn 消融）。
- 依赖 D1：D3。
- 需先确认资源：D4（H800 是否活）。
- 较重、可稍后：D5（LongEval 系统对照）。
