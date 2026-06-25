# vN_HNST — Hierarchical Native-Salience Tree(树形 hidden memory,2026-06-25 设计 wiq6fz89m)

## 一句话
叶子保留所有 chunk(不丢早期 needle),query 时用 frozen reader 自己的 q·k 在 B 叉树上分层导航,只让路径上 O(log N) 个 chunk 进 attention = "keep all, attend few"。既要 b25 抗稀释又不要 b25 健忘。

## 为什么绕开死路
- 选择器死路:树每层选 1-of-B(B=8)非 flat 1-of-64 → per-level 精度高 + 用 reader-native q·k(无训练 55% precision)非 trained selector → H2 咬不到
- 单层读出死路:复用现有 rawkv_readout_layers 多层注入(单层 L16=0%)
- 容量/窗口/mass 死路:这改"attend 什么"(结构),非训练旋钮(那些对 32k 墙免疫)

## 结构
- 叶子(L0):每个 chunk,key=gist_src[c](max-pool,H1 抗稀释)
- 内部节点(Lℓ≥1):B 个子节点 pool 成 summary,因果增量构建(只更新 right-spine 祖先,segment-tree 式,无未来泄漏,O(1)/chunk)
- B∈{4,8},64 chunk → B=8 深度2,1M token → 深度4
- detached,文档边界 reset,shared singleton 挂 model root(不进 state_dict)

## 导航(reader-native,零训练)
泛化现有 _reader_attn_keep_set(layer.py:1368)从 flat top-k 到递归 top-b 下降:
q=RoPE(q_proj(norm(h_last))); 逐层 K_child=k_proj(norm(children_summaries)), sal=max_heads(q·K/√hd), beam top-b 下降, 返回到达的叶子 chunk indices。
beam b≥2(非贪心,错的高层转弯可恢复)。b·B≈8-16 chunk ≈ b25 的 25,稀释相当但从全部 chunk 选。
纯 inference no_grad 无 trained scorer → H2 不触发。默认读路径。

## 读出(多层,复用现有)
kept leaf indices → 现有 GistReadout.retrieve(keep_set_override=kept):物理 gather 掉非 kept chunk(硬隔离零稀释),raw leaf token 经各层 native k/v_proj 在真实 RoPE 位置注入。rawkv_readout_layers 多层。

## slot×tree(v2 升级,轻训练)— 用户明确想要
内部节点 key 从 raw pool → L3SummaryPool 压缩(l3_summary.py cross-attn+FFN+diversity loss+recursive mode)。只训 summary projection(冻结 backbone),梯度只流选中路径(STE)。叶子永远 raw hidden。
★ 正好实现"prediction 不需 reconstruction"洞察:导航激进压缩(250:1),回答保精度(1:1)。

## 训练(泄漏干净)
必须 --babilong_mix_fraction 0(default 0.15 泄漏 0k-4k)。只报 8k-32k OOD。v1 零训练,v2 训 L3+gist projection ~3000步 chunk512 冻结 backbone。

## 第一实验(零训练,与 b25 直接可比)
run_babilong_mem_space.py 加 rawkv_keep_set_mode='tree'(已有 ''/'gist'/'reader_attn'/'oracle' 在 line765),调新 _tree_nav_keep_set(layer.py,递归 beam 泛化 _reader_attn_keep_set)。B=8 max-pool tree over rawkv store gist_src,native RoPE(q_proj) beam(b=2)下降,返回叶子 indices 喂 keep_set_override。现有 ckpt(outputs/mem_space_fifo_b50_chunk512/full_model.pt 本地 + b25 .7.53)。eval qa1/qa2/qa5 × 8k/16k/32k W0。对比 3 mode 同 ckpt:(a)flat reader_attn top-k (b)tree-nav (c)full-buffer。零训练一次 eval sweep。
★ 决定性对照:position-stratified needle(early/mid/late)。HNST 必须在 early-needle 16k/32k 赢 b25(b25 结构盲区)。

## kill criteria
- tree-nav 不超 b25 8k-32k qa1/qa2 ≥3pt(matched leaf-count)→ 分层导航无增益,树premise死
- position-stratified early-needle 16k/32k 不超 b25 → 树无机制优势
- 顶层 reader q·k precision <2×随机(beam path 含真 needle <50% @32k)→ pooled summary 毁了 needle 信号
- beam b 需 O(B^depth)才够(=keep-all)→ 树坍缩成 flat
- v2 slot×tree 不超 v1 → 砍 slot intersection 只 ship v1
- 32k nav 比 b100 慢 2× → 需 kernel work

## novelty(诚实)
- vs MemWalker(2024):它是 TEXT 树 + 重 prompt LLM 导航(fine-tuned traces);HNST 在 HIDDEN 空间用 frozen reader 自己 q·k,零训练,无生成文本无导航监督
- vs RAPTOR(2024):它是外挂 embedding RAG;HNST 无外部 embedder,导航 key = backbone native k_proj,leaf 经 in-attention KV 真 RoPE 多层注入,全 in-graph
- vs Compressive Transformer(2020):固定 2 级无选择无导航;HNST 多级 O(log N) query-conditioned 选择
- vs 自己 slot work:旧 trained flat 1-of-64 selector(死路);HNST 1-of-B per level + 无训练 reader-native + 叶子 raw
- 真原创窄claim:(a)reader-native 训练free 分层导航 hidden 树;(b)keep-all-leaves/attend-O(log N) 作为 H2 dilution-vs-amnesia trade-off 的原理解;(c)slot×tree split(内部压缩导航/叶子 raw 回答)
