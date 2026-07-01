# 项目真实状态基线 (2026-07-01, 主控亲自逐个核实)

> 建此文件因: 主控之前对进度理解含糊(HNST零训练v1 vs 重训版没分清/reforward与hidden分数混报/泄漏ckpt漏隔离)。此处从原始CSV+代码+config逐个核实, 作为统一标尺。全官方判分compare_answers, 排OOM。

## 一、核心架构澄清(代码坐实)
Llama-3-8B + FIFO hidden memory(`use_fifo_memory`)。三种"读出机制"必须分清:
- **纯hidden FIFO**: buffer存chunk的raw hidden, 选中的hidden直接拼进注意力读出(`prefix=torch.cat(kept_chunks)` hidden向量, layer.py:1376)。**hidden level, 不重过模型**。=正确方向。
- **token reforward**: 选中chunk的**原始token**重新forward整个模型(`chunks[c]`=token id, probe_fullchain:437)。≈RAG重算, 要停。
- **选择器**(都在hidden level选哪些chunk进注意力): (a)flat reader-attn frozen零训练; (b)HNST tree frozen零训练=v1; (c)重训reader-attn(t2_select_loss)。

## 二、A模型(mem_space_fifo_b25_c512_supervised_select, 干净mix=0, buffer25)三机制 qa5 各档
| 长度 | ①纯hidden FIFO(端到端) | ②token reforward(fullchain oracle=完美选择) |
|---|---|---|
| 2k | 54 | - |
| 4k | 43 | 35 |
| 8k | 18 | 44 |
| 16k | **9** | **52** |
| 32k | - | 58 |
→ **短档hidden不输(2k 54)**; **长档hidden崩(16k=9) vs reforward(16k=52)差43分=读出墙的量化**。

## 三、选择器三方 qa5 16k 端到端(A模型)
| 选择器 | 16k端到端 |
|---|---|
| 重训reader-attn | 24 |
| bm25(零train启发式) | 44 |
| oracle(假设选100%对) | 43 |
→ 重训选择器24 vs oracle43=**选择墙19分**; oracle43本身低=**读出墙**(选对也读不全)。

## 四、两道墙(项目核心矛盾)
- **选择墙**: 从buffer选对含needle chunk。重训reader-attn 24, HNST tree needle-recall 83(选得准但端到端没干净数)。
- **读出墙**: 选对后纯hidden读出弱。oracle完美选择16k才43, 纯FIFO才9。
- **胜利条件**: 让hidden路线16k端到端从9(或选对后43)做到接近满分。

## 五、HNST tree澄清(重要, 之前含糊)
- **v1=零训练**(frozen q·k + max-pool tree, `_fifo_select_keep_set_tree` with no_grad)。needle-recall qa5 16k=83/32k=54(干净, base Llama探针)。但**KILL**: qa1 tree<flat不稳/max-pool毁needle信号(beam1=20%)/无干净端到端(A模型hidden路径吐乱码, 只泄漏b50能生成)。
- **用户要的是**: 重训选择器 + tree结构(v1没重训)=**HNST v2**(可训练summary树+解冻reader), 正在派agent做。

## 六、泄漏ckpt(红线)
b50=mem_space_fifo_b50_chunk512, **babilong_mix_fraction=0.15**(训练混了15% babilong eval数据=泄漏, 分数虚高不可信)。红线: b50/b100/P2/c1024/旧b25完全不碰不引用。A模型mix=0干净。HNST v1两批端到端误用b50已隔离(VOID/VOID2)。

## 七、当前三线(2026-07-01下午)
| 线 | 状态 | 判据 |
|---|---|---|
| HNST v2(可训练树+解冻) | agent派出, 本机+.196待起(GPU还空, 已问进度) | needle-recall>flat + 端到端>A模型 |
| 解冻验证 | .7.53+.245 eval中 | fullchain oracle读出随解冻层升? |
| 混档v2 | 优先级降(16k非容量墙) | - |

## 八、已定型干净结论
1.slot有用(官方+5~13). 2.hidden FIFO长档失效(16k on-off -5). 3.16k=读出墙+选择墙非容量墙. 4.HNST树v1零训练KILL. 5.根因H1过拟合. 6.token reforward≈RAG停.

## 判分铁律
官方compare_answers禁re.search; 全档+n100+同设定; 派agent必分独占IP+明列禁用泄漏ckpt(b50等); 三种读出机制/零训练vs重训 分数不混报。
