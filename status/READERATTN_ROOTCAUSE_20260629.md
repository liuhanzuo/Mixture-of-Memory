# reader-attn 选择器弱: 根因 + 改进方案 (2026-06-29)

> 进展快照。background 调研 agent(读代码 + layer sweep)+ 我核实。

## 现象
A 模型 P1 recall(qa5 16k, n50): reader-attn **r@1=0.05 / r@4=0.19** << bm25 0.29/0.62 << oracle上界 0.86。
端到端 token-reforward(A 16k): reader-attn tk1=18/tk4=46 << bm25 tk1=43/tk4=74。
→ reader-attn 选择信号本身极弱, 几乎不能定位 needle。

## 根因(坐实)
### 主因: 训练/eval dilution 严重不匹配
- 训练 `_launch_t2_supervised_select.sh`: `t2_gap_tokens=3584` → **n_ctx=7**(buffer 只 7 chunk), 且**无 `--t2_curriculum`**(n_ctx 整训练固定 7)。
- eval: `fifo_buffer_chunks=25`(16k 文档 ~31 chunk, buffer 留最近 25)。
- 训练 random recall@4 = 4/7 = **57%**(容易), eval = 4/25 = **16%**(难 3.6×)。
- selector 只在 7 候选里练"找1针", 学不到 25 候选的能力 → 迁移失败。

### 次要
- keys 无 RoPE(query 有), chunk 位置匿名化, 降低区分度(训练/推理共有, 非不一致)。
- T2 针格式(MEMORIZE: 名=5位数)vs babilong 自然语言段落, 可能轻度过拟合。

### 训练 vs 推理 salience 算法一致(排除)
逐项核对(query=hs[L16]末位, key=buffer chunk raw hidden 过 pre_norm, pooling=amax_head(amax_token(q·k)), 层号均16): **完全一致**。不是不一致问题。

### layer sweep: 换层不是银弹(排除)
A 模型 qa5 16k limit20, reader-attn recall@4: L8=0.35 / L12=0.24 / **L16=0.35** / L20=0.24 / L24=0.12。**所有层 << BM25 0.53**。L8/L16 并列最佳但 n17 宽CI, 真实≈0.19。深层更弱(hidden 已融合, 失 chunk 区分度)。

## 改进方案(排序)
1. **零训练**: BM25 做 selector(recall 0.53 >> reader-attn 0.19, 免费)。短期直接用。
2. **修 dilution 重训**(攻主因): `t2_gap_tokens=8192`(n_ctx=16)+ `--t2_curriculum 0:16` + `t2_num_keys≥3`, 从 step2000 续训(`--init_checkpoint`), mix=0, 新 RUN 名。预期接近 eval 难度。**但历史 T2→babilong 迁移成功率不高(待验证)。**
3. salience 加 key RoPE / 多层投票(收益待确认)。
4. 换 babilong supporting-fact 位置做监督(解决 T2 格式过拟合, 工程量大)。

## 执行状态
- 训练脚本/参数均支持(`--t2_curriculum` `--t2_gap_tokens` `--init_checkpoint` 已验证)。
- **但当前无干净 8 卡整节点训练槽**(本机4空/.196.7.53各6空但在跑FAIR收尾/.245 1空/B200 8空但缺dolmino+pg19训练数据且跨wzc1盘)。
- 方案2 是数小时 8 卡训练 + 需中断现有 eval = 承诺级决策 → **待用户拍板**。

## 红线
重训严禁泄漏 ckpt 起点(b50/b100/P2/c1024/旧b25/P11/l3recontoken); mix=0; 真SOTA锚点 pg19 nctx7 qa5 16k=16/32k=9。
