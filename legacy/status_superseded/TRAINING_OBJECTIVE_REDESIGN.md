# 训练 Objective 重设计:逼 memory 干活(2026-06-13,主会话探讨)

## 核心诊断:当前 objective 太容易
- **现状(代码核实)**:dolmino 主路径(85%)= windowed-BPTT,每 chunk 都算 next-token loss(train script:1613/1648),梯度在 bptt_window=2 内跨 chunk。BABILong(15%)= 只最后 chunk 算 loss,context 全程 no_grad(:1727)。
- **病根**:next-token 的信息 **绝大部分在当前 chunk 内**(causal self-attn 就够),memory 只需偶尔补一点跨 chunk 信息。模型靠"局部语言建模"就能拿低 loss,**memory 没被逼到极限**。
- 与两个已知现象同源:
  - "训练加 SWA 反而更差"(D2b REJECTED):给"直接看前文"的拐杖→memory 退化。
  - "memory eval 44 vs full-attn/SWA 73":信息在上下文里但 memory 没传递成功。

## ★用户的验证方法(整个设计的支点)
**拿 P11 step500 ckpt,对任意候选任务测:memory-chunked eval 差 + full-attention eval 好 = 这个任务真的逼 memory**(信息在远端,局部解不了)。
- 代码现状:无独立 full-attn 开关,但 **SWA 把窗口 W 开到覆盖全文档 ≈ full attention**(run_babilong_mem_space.py swa_eval_chunks)。已有数据:SWA W6 qa5 32k=73 vs chunked 44 → gap=29 就是"memory 没传递的信息量"。
- **+ 远端层退化检查**(用户补充):看深层(远离 memory 注入层)表征是否随距离退化——若退化说明信息在前向传播中丢失,不只是读出问题。

## 设计方案(按"逼 memory"强度排序)

### 方案 T1 — 跨 chunk 预测(用户想法1的正确版)
当前每 chunk 预测**自己**的 next-token(局部可解)。改成:**target chunk 的 loss 只算在"必须用前文才能预测"的 token 上**。
- T1a. **延迟预测**:target chunk 不预测自己,而是预测**下 N 个 token 但屏蔽掉 target chunk 内的近距离 context**——强制只能靠 memory。
- T1b. **跨 chunk copy**:在 context chunk 里埋一个标记串,target chunk 要求复述它(RMT 的 copy task 思路)。局部 100% 解不了,信息只在 memory。
- 风险:纯合成 copy 可能 overfit 到"找标记"而非真理解。需混真实文本。

### 方案 T2 — 加难的合成任务混入(用户想法2 + 调研)
RMT/记忆模型常用**合成长程任务**逼 memory:
- T2a. **associative recall**:context 给 key-value 对(散在各 chunk),target 给 key 问 value。NIAH 的训练版。
- T2b. **reverse/sort**:context 给序列,target 要求逆序/排序输出——必须记住全部。
- T2c. **multi-hop**:答案需串联多个 chunk 的 fact(BABILong qa2/qa3 就是,但可加难)。
- 做法:不取代 dolmino,而是**提高 BABILong-like 难任务的混入比例**(当前 15%)或加自造合成任务,且**长度课程拉长**(当前 babilong 只到 4k,可拉到 16k/32k)。

### 方案 T3 — 信息瓶颈正则(让 memory 必须承载)
- T3a. **context dropout**:训练时随机**整块丢弃 target 的局部 context**(只留 memory readback),逼 loss 走 memory 通道。这是"训练加 SWA"的反向操作——不是给拐杖,是抽掉腿。
- T3b. **L2/L3 已存的 recon**:ICAE token-recon 已 REJECTED(逼存所有 token 与检索冲突),但**检索式 recon**(让 summary 指向"后文要查的内容")未试。

## ★推荐执行序(投入产出比)
1. **先用 P11 step500 ckpt 做"任务甄别"**(用户方法,零训练成本):造几个候选任务(associative recall / multi-hop / copy),各跑 chunked-eval vs SWA-full-eval。**选 gap 最大的任务**(memory 最差、full-attn 最好)= 最能逼 memory 的训练信号。
2. 选定任务后,**T2(混入难任务)+ 长度课程拉长** 最稳(不改 loss 机制,低风险)。
3. 若 T2 不够,上 **T3a(context dropout)** —— 直接抽掉局部拐杖,机制上最对症。
4. T1b(copy)作为极端对照,验证 memory 容量上限。

## 待办
- [ ] 写任务甄别脚本:P11 step500 ckpt × {associative_recall, multihop, copy, plain} × {chunked, SWA-full} → gap 表。
- [ ] 远端层退化诊断:加 hook 看 memory 注入层后,深层表征随 chunk 距离的衰减曲线。
- [ ] 当前 3 路架构实验(L1 key/erase, L2)继续跑——它们是"读出"侧,与本文"训练 objective"侧正交,可并行。
