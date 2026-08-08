# Resume-to-200k 无法忠实续跑：param-group 不匹配让三个 arm 全部退化成 WARM-RESTART

**日期**: 2026-08-08 ~12:00 CST。**发现者**: MAIN，实际启动 resume 时从 trainer 日志读到。
**这条直接改变 #192（Table 4 budget defect）的答案空间。**

## 实测事实

启动 keep10@83500 → 200k（.82）与 keep12@124000 → 200k（.104），两个 ckpt **都有
`optimizer_state` + `rng_state`**（我事先验过），但 trainer 日志两边都打出：

```
[resume] loading ckpt .../step83500.pt (saved at step 83500, has_optimizer=True)
[resume] restored 135 model tensors (strict, fp32 master weights)
[resume] optimizer.load_state_dict failed (loaded state dict has a
         different number of parameter groups); WARM-RESTART (Adam moments re-init)
[resume] continue @ step=83500 epoch=0 warmup=150 max_steps=200000
         lr_fresh(now)=6.659e-05 lr_inh(now)=1.332e-05
```

keep12 同样（157 tensors，`epoch=1`，`lr_fresh(now)=3.847e-05`）。

**所以「keep10/keep12 能忠实 resume、只有 keep8 不能」是错的 —— 三个 arm 都不能。**
我在 heartbeat 里跟用户说过这个不对称，现在纠正：不对称不存在，缺陷是一致的。

## 为什么 param group 数会变

存 ckpt 时的 optimizer 有 N 个 param group，现在重建的有 M≠N 个。最可能的原因是
**差分 LR 的 `_classify_param` 分组逻辑在 ckpt 存下之后被改过**（CLAUDE.md 记过一条相关的：
`train_olmo2_arch_probe2.py:316` 加过 `module.` 前缀剥离的修复，而 distill 版没有）。
分组数一变，`load_state_dict` 就拒绝加载。**我没有进一步确认是哪个 commit 改的**——要确认得
diff 出 ckpt 存盘时的 trainer 版本，而那需要 git 考古；先记录现象。

## 对 #192 的后果（重要）

原来的选项空间是「A: 如实披露真实 steps / B: resume 到 200k / C: 混合」。现在：

- **选项 B 的成色下降了**：resume 得到的 200k **不是**原 schedule 的忠实延续，而是
  「83.5k 处 Adam moments 清零后再跑 116.5k」。这本身是一次 optimizer 状态扰动，
  和原始的 keep14/ShortGPT-16（一口气跑到 200k，从未清零）**不同源**。
  如果用它去补 Table 4，会引入一个新的、和 depth 混在一起的变量：
  **「中途重启过一次」vs「一气跑完」**。
- 也就是说 B 并不能真正消掉 budget confound，只是把 budget 不齐**换成** optimizer-restart 不齐。
- **这让选项 A（如实披露）相对更有吸引力了**，而且 A 有个之前我低估的优势：
  keep8（10 层，121k 步）预算比 keep10（12 层，83.5k 步）多 45%，**却仍然 core6 更低**
  （.52328 vs .52999）。也就是说 ladder 的单调性是在「更浅的那个还多拿了 45% 预算」的
  条件下成立的 —— 这比等预算 ladder **更强**，不是更弱。如实披露反而能这样论证。

## 当前状态（不 kill，理由）

两个 resume 都在 100% 利用率跑（.82 keep10 89 GB/卡、.104 keep12 97 GB/卡）。
**我不 kill 它们**，因为：
1. 即使是 warm-restart 的 200k 点，也是**有用的补充证据**（可以回答「给足预算会不会追上」
   这个 reviewer 一定会问的问题），只要在写作时如实标注 "resumed with optimizer re-init"。
2. 卡本来空着。
3. 但 **它不能当作 Table 4 主表的等预算修复**——主表口径必须由用户在 #192 里定。

预计到 200k：keep10 还差 116500 步、keep12 还差 76000 步。按历史 s/step 需要重新测
（首批 step 出来后再估），量级是数天。

## 待办

- [ ] 确认 param-group 数变化的根因（哪个 commit 改了分组），并判断能否写个 shim
      让 optimizer state 可加载（若能，则真正的忠实 resume 才可行）
- [ ] #192 决策时把本条作为输入：**选项 B 无法给出忠实的等预算 ladder**
- [ ] 若最终采用这两个 200k 点，写作时必须标注 optimizer warm-restart，不能混进
      "trained to 200k" 的同一口径

## Provenance

- `.82`: `zwfy6:logs/olmo2_7B_keep10fresh2_resume200k.log`（启动 11:58:52）
- `.104`: `zwfy6:logs/olmo2_7B_keep12fresh2_resume200k.log`（启动 11:58:28）
- 相关: `status/PAPERB_TABLE4_BUDGET_DEFECT.md`、`status/PAPERB_SFT_FIT_CONFOUNDED.md`
- 我先前那次误报的记录：本轮 heartbeat 曾说「keep10/keep12 有 optimizer_state → 能忠实
  resume」，该说法**已被本条推翻**（ckpt 里有 ≠ 能加载）。
