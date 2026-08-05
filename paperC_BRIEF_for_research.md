# Paper C 现状简报（给外部调研 agent，2026-08-05）

## 我们的构造（core object）
**"Shallow Regrown Cap on a Frozen Trunk"** — 拿一个预训练 decoder LLM（OLMo-2-7B, 32 层）：
1. 保留前 `j` 层并**冻结**（frozen trunk）
2. **丢弃**顶部 `L-j` 层
3. 嫁接 `K` 个**全新随机初始化**的 transformer 层（K 小，通常 2）
4. **只 finetune 这 K 层** + final norm + lm_head，在下游/instruction 数据上

净效果 = 一个**更浅**的 finetuned 网络（j+K < L），不只是"re-init 顶部若干层"。

## 与我们另两篇的区别（同一批作者）
| | 对象 | 训练机制 | 数据 |
|---|---|---|---|
| Paper A (CoMem) | depth-split + retrieval memory | 自蒸馏 LoRA adapter | long-context |
| Paper B (prune-heal) | keep 前 N 层、丢顶部、嫁接 K 新层、**continue-PRETRAIN 全参** | 续预训练（heal base LM） | Dolmino DCLM |
| **Paper C（本篇）** | keep 前 j（**冻结**）、丢顶部、嫁接 K、**只 finetune 那 K 层** | 监督 finetune，trunk 冻结 | 下游 / instruction |

## 已有的两条命题
- **P-C1「Finetuning 主要发生在顶层」**：(测量) 标准 FT 时 per-layer ‖ΔW‖ 与 CKA drift 集中在顶层，前层几乎不动；(构造) 因此冻结前 j、重长 K 层、只训 K → 追平 full-FT、胜过参数匹配的 LoRA（尤其分布偏移下）、胜过同深度 from-scratch。
- **P-C2「Adaptation depth 可预测」**（我们自认为的差异化 hook）：切点 j 与重长 K 可由 **base 模型的廉价 forward-only probe** 预测 —— 一个"adaptation-onset depth"，复用我们已有的 logit-lens / edge-probe 设施。多个"计算深度"不同的任务 → per-task 最优 K 曲线 → probe 预测深度 vs 实测最优 K 的相关性。
  - 我们已测得：OLMo-2-7B knowledge onset = layer 18 = 0.562L，sat95 = 0.594L；Qwen3-8B sat95 = 0.694L。
- **P-C3「Modular shallow caps」**（多任务 serving：一个共享冻结 trunk + N 个小 cap）—— 我们自己判断这条最被 LoRA/adapter multi-task serving 文献占据，倾向降为附录。

## 已有实证结果（真实数字，别当假设）
### #92 SQuAD 4 臂（keep14+fresh2，1000 steps ≈166 epochs，over-trained/capacity-bound）
| arm | 构造 | EM | F1 |
|---|---|---|---|
| A2_lora_r160 | full 32L + LoRA r=160（参数匹配） | **0.6590** | 0.7139 |
| BASE_ref | 原始 OLMo-2 32L，**无 SFT**（confounded 参照） | 0.3385 | 0.3999 |
| A4_hero | keep14 冻结 + fresh2，只训 fresh+norm+lm_head | 0.2930 | 0.2970 |
| A3_fromscratch | 16L 全随机初始化，全参训练 | 0.2605 | 0.2612 |

- **A4 > A3 显著**：McNemar χ²_cc=11.41, **p=7.3e-4**；paired bootstrap 95% CI on EM diff = [+1.4pp, +5.2pp]（不含 0）。这是干净受控对比（同 16L 深度/数据/步数/eval，唯一差异 = 前段继承+冻结 vs 随机初始化）。
- **BASE_ref 是 confounded 的**（差两个轴：32L-vs-16L AND no-SFT-vs-SFT），只作 intact-model 上限参照。

### #133 深度扫（新，2026-08-05）
A4（freeze-graft）SQuAD EM 随保留深度**单调上升**：keep14=0.2930 → keep20=**0.3440** → keep24=**0.3560** → keep28=**0.4190**。
A3（from-scratch 控制）在 H20 上 keep20/24/28 **全部 OOM_BLOCKED**（7B 全参 fp32-AdamW 装不下 97.8GB），正在 B200 上补。

### #132 第二任务能力评测（新，14 个 benchmark，4 臂，n=78,656 pooled）
**关键且不利的结果**：A4>A3 在能力型 benchmark 上只剩 **+0.39pp**（χ²_cc=10.45, p=1.2e-3，CI[+0.15,+0.63]）—— 比 SQuAD 的 +3.25pp **小约 8 倍**，14 格里 9 格 null，boolq 上 A4 反而输 1.56pp。
**更关键**：两个 16L 臂**几乎处处在随机猜测水平或以下**。A4 MMLU 0.2596 仅比 0.25 猜测下限高 z=+2.6；A3 0.2474 = z=−0.7（就在 chance）。closed-book QA 整体崩塌（EM≈0），A4 在 52.4% PopQA / 53.2% NQ-open 上吐同一句中文拒答，而 BASE 只有 2.2%/1.4%。→ 剪枝臂几乎没有参数化知识残留、decoder 部分崩坏。
**我们自己的诚实结论**：SQuAD 上的 A4>A3 **不是知识/推理恢复**，只是在近-chance 区间里的小优势；keep14 下"对比发生在两个坏模型之间"。这支持窄框定「哪种 init 在激进剪枝下恢复更好」，**反对** general capability retention 的主张。
另注：A2（LoRA）在**每一个**能力 benchmark 上都低于 BASE（MMLU .5935 vs .6056，triviaqa EM .541 vs .636）→ SQuAD-format SFT 牺牲了通用能力。

## 我们已知的 novelty 风险（请重点挑战/深化）
构造这一半**部分被占**：
- Zhang et al. 2021「Revisiting Few-sample BERT Fine-tuning」——FT 前 re-init 顶部 K 个 encoder 层
- Surgical Fine-Tuning (Lee et al., ICLR'23) —— 只调**已有** block 的一个子集
- ULMFiT gradual unfreezing
我们自认的区分点：(a) **丢弃**顶部 L-j 并重长 K（K < L-j）→ 净更浅的网络（有 compute 收益），不只是 re-init；(b) trunk 真冻结（Surgical FT 仍更新所选 block）；(c) decoder LLM @7B 规模，不是 BERT。
P-C2 的 a-priori probe 预测深度是我们认为真正新的地方（Surgical FT 用 post-hoc gradient/Fisher 选 block，需要 FT 信号；我们想用 base 模型的 intrinsic probe，在任何 FT 之前）。

## 现成设施（决定什么实验便宜）
- `scripts/train_olmo2_arch_probe2.py` 已实现全部构造：`--keep_front_layers j` / `--n_fresh_layers K`（断言真随机初始化）/ `--freeze_front` / `--from_scratch`（同深度控制）/ 差分 LR / grad-ckpt / fp32 master。
- probe 设施：`scripts/probe_linguistic_layerwise.py`（POS/DEPREL/CoLA/WiC/SST2/RTE edge-probing + logit-lens + knowledge_logit_lens），forward-only。
- 下游 eval：`scripts/eval_olmo2_probe2_downstream.py`（hellaswag/arc/piqa/winogrande/openbookqa + mmlu/lambada/boolq/csqa/siqa）；`eval_olmo2_closedbook_qa.py`（PopQA/TriviaQA/NQ-open）；`eval_paperC_squad_emf1.py`。
- 盘上模型：OLMo-2-1124-7B（主）、Qwen3-8B-Base、OLMo-2-0425-1B、Qwen3-4B。
- 盘上数据：SQuAD（有）；Dolmino/PG19/slimpajama（预训练式）。**没有**：GLUE、Alpaca/Tulu instruction mix（需下载）。
- 算力：5 节点 40 卡（2×B200-183GB + 3×H20-97.8GB）。**实测显存**：A4 freeze-graft 16L→50.8GB、22L→56.4GB（H20 可跑）；A3 全参 16L→76.8GB，22L+ 在 H20 OOM（必须 B200）。
- 协议铁律：全部 base 口径 **chat_template=False、no BOS、likelihood-based MC**（OLMo-2 是 BASE LM，无 SFT/RL）。
