# B04 — Evaluation Fragility under Model Damage

## 状态

**INCUBATOR。当前只证明 margin compression，不足以声称广义 fragility。**

## 已成立

正确 `acc_norm` 下，六个 damage rungs：

- median margin：`0.124594 → 0.075801`
- Spearman(core6, median margin)：`+1.00, p=.0028`
- near-tie `<.005`：`2.012% → 4.461%`
- Spearman(core6, frac<.005)：`-.9429, p=.0167`

即：damage 与 decision-margin compression/near-tie density 相关。

## 未成立

- flip rate 随 damage 单调；
- margin 中介 damage→flip；
- 跨 nuisance variable 的复现。

当前 bs8/bs16 仅 2/6 rung：

- base 0.081%
- ShortGPT 0.640%

不能做有效趋势检验。

## 完成 gate

1. 完成 6/6 bs ladder；
2. exact test 成立；
3. LOO margin model 优于 constant-rate null；
4. 第二种 nuisance（torch/GPU architecture）复现。

失败则作为 A01 appendix 的 negative result，不单独成篇。

