# MMLU Interface Case

## Surviving claim

受损模型上，answer-letter scoring 可能退化为 input-blind predictor；在使用其
模型间差异前，必须先证明每个 arm 显著高于 best-constant floor。

### OLMo

- best constant：always-D `0.2689`
- 三个 arm 显著低于该 floor，三个 arm 与其不可区分
- content interface 在十个 arm 上均高于自己的 `.2845` floor

### Qwen

极端受损 arm 的 failure mode 不是大量 tie，而是 letter prior collapse：
近乎总是输出同一个 letter。这支持“instrument failure 的实现机制随家族变化”。

## 已撤回

- “两个有效 interface 会翻转模型排序”。
- “fails below chance” 作为标题。正确对象是 best-constant floor，不是 `.25`。

## 独立成篇 gate

只有同时完成下列三项，才从 A01 拆为独立 paper：

1. full-fp32 forward 消除 ties，并恢复 letter validity；
2. 第三个模型家族复现；
3. 第二个 MC benchmark 复现。

