# 用户裁定 2026-08-13：`paperD` 归新晋升的 proposal（旧 paperD 已死，名字释放）

用户原话：**「是啊 原来的paperD已经没了 所以新晋升的proposall就是paperD」**

## 后果

- 下一个从 `proposal/` 晋升的方向占用 **`paperD/`**，不跳字母、不叫 `paperE`。
- 旧的 `proposal/archive/paperD-cross-family-stitching/`（跨家族层拼接，DEAD）
  **保留不动** —— 它是那个方向的 provenance 入口，按 CLAUDE.md 「proposal 目录不删」。
  但它**不再拥有 `paperD` 这个名字**。
- 因此任何提到 "Paper D" 的历史文档（`status/*`、`UPDATELOG.md`、tasks #163/#166/#168）
  指的都是**已死的层拼接方向**，不是新的 `paperD/`。引用旧文档时必须消歧。

## 当前的 paperD 候选（尚未晋升）

margin-trajectory instability：在剪层-恢复文献的通用下游口径下，恢复臂相对 null 的
margin 沿训练轨迹**非单调游走**，幅度与判定「已恢复」的 Δ（1.86 pp）同阶 → 单点 NI
accept 不报邻域 checkpoint 就不可解释。

三条独立证据（全部可从盘上 json 复算）：
1. `A04_KEEP14_TRAJECTORY_NI_VERDICT.md` — keep14 popqa 沿 25,500 步确定性变差 −0.6729 pp (p=0.0001)
2. `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` — 单次 500 步移动值 1.1202 pp = popqa 整个 Δ 的 49.9%
3. full32 CPT 轨迹扫描 — mmlu_content margin 全 ACCEPT 但峰值在中段（reading (b)）

**晋升前置**：novelty 核查（agent `ac2645a8` 在跑，0 GPU），需回答它是独立成篇
还是 paperC/A01 的一节。核查结论回来前不建目录。
