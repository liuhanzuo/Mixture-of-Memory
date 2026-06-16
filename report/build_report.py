#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 MoM 课程成果报告 PDF（fpdf2 + Noto CJK）。
正文 ≤10 页 + 附录。运行: python report/build_report.py"""
from fpdf import FPDF
import os

FONT_REG = "/usr/share/fonts/google-noto-cjk/NotoSansCJKsc-Regular.otf"
FONT_BLD = "/usr/share/fonts/google-noto-cjk/NotoSansCJKsc-Bold.otf"
FIG = os.path.join(os.path.dirname(__file__), "figs")

class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("noto", "", 8)
        self.set_text_color(150)
        self.cell(0, 6, "Mixture-of-Memory · 大模型驱动的软件开发 课程成果报告", align="C")
        self.ln(8)
        self.set_text_color(0)
    def footer(self):
        self.set_y(-12)
        self.set_font("noto", "", 8)
        self.set_text_color(150)
        self.cell(0, 6, f"- {self.page_no()} -", align="C")
        self.set_text_color(0)

def H(pdf, txt, size=14, top=3):
    pdf.ln(top)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("noto", "B", size)
    pdf.set_text_color(20, 40, 90)
    pdf.multi_cell(0, size*0.62, txt)
    pdf.set_text_color(0)
    pdf.ln(1.2)

def P(pdf, txt, size=10.3, lh=5.0):
    pdf.set_x(pdf.l_margin)
    pdf.set_font("noto", "", size)
    pdf.multi_cell(0, lh, txt, align="L")
    pdf.ln(0.8)

def BUL(pdf, items, size=10.3, lh=4.8):
    pdf.set_font("noto", "", size)
    for it in items:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, lh, "· " + it, align="L")
        pdf.ln(0.3)
    pdf.ln(0.5)

def fig(pdf, name, w=150):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        x = (pdf.w - w) / 2
        pdf.image(p, x=x, w=w)
        pdf.ln(2)

pdf = Report(format="A4")
pdf.set_auto_page_break(True, margin=15)
pdf.add_font("noto", "", FONT_REG)
pdf.add_font("noto", "B", FONT_BLD)
pdf.set_margins(18, 15, 18)

# ============ 封面区（第1页） ============
pdf.add_page()
pdf.ln(8)
pdf.set_font("noto", "B", 21)
pdf.set_text_color(20, 40, 90)
pdf.multi_cell(0, 10, "Mixture-of-Memory：用固定大小记忆缓冲\n压缩长上下文的 LLM 适配器", align="C")
pdf.set_text_color(0)
pdf.ln(2)
pdf.set_font("noto", "", 12)
pdf.multi_cell(0, 6, "—— 一次由大模型 Agent 自主驱动的研发实践", align="C")
pdf.ln(6)
pdf.set_font("noto", "", 10.5)
pdf.multi_cell(0, 5.5,
    "课程：大模型驱动的软件开发（2026 春季学期）成果报告\n"
    "姓名：刘涵祚    班级：计科 41\n"
    "源码：https://github.com/liuhanzuo/Mixture-of-Memory", align="C")
pdf.ln(5)
pdf.set_draw_color(180); pdf.set_line_width(0.3)
pdf.line(40, pdf.get_y(), pdf.w-40, pdf.get_y()); pdf.ln(4)
pdf.set_font("noto", "B", 11); pdf.set_text_color(20,40,90)
pdf.set_x(pdf.l_margin)
pdf.multi_cell(0, 6, "摘要"); pdf.set_text_color(0)
P(pdf,
 "本工作研究如何把超长上下文压缩进一个固定大小的记忆缓冲（memory bank），"
 "使 Llama-3-8B 在有界 KV 预算下处理长序列。我们提出的 mem_space 适配器与混合专家（MoE）"
 "高度同构——可视为 Mixture-of-Memory（MoM）：N 个记忆 slot 即"
 "“专家”，一个轻量 selector 对每个 chunk 做 top-k 路由，写入只更新被选中的 top-k slot（稀疏更新），"
 "读出则对全部 N 个 slot 做独立 softmax。我们诊断出该架构的核心瓶颈是"
 "“写得进、读不准”（纯记忆读出 W0 仅 10–30，而开卷上限 50–60），"
 "并系统比较了四类干预：token-mass 读出加权、self-study 蒸馏、长度课程、以及二者叠加。"
 "关键发现是弱 mass（强度≈0.5）与蒸馏在长程（16k/32k）上首次产生协同，"
 "超过任一单独方法；mass 过强反而干扰。LongBench 真实长文档评测进一步把能力缺失定位到"
 "“超长叙事连贯”与“细粒度抽取”。"
 "尤为重要的是，整个研发过程——集群巡检、实验调度、代码实现、缺陷排查与质量护栏——"
 "由大模型 Agent 闭环自主驱动，本报告同时是这一研发范式的实证。", size=10, lh=4.8)

# ============ 1. Agent 驱动研发 ============
H(pdf, "1  引言：由大模型 Agent 自主驱动的研发", 14)
P(pdf,
 "本课题的特别之处在于：从实验设计、五节点 GPU 集群调度、代码实现、到缺陷排查与结论分析，"
 "几乎全程由大模型 Agent 以闭环方式自主驱动，人类仅在关键方向上给出判断。具体工作流包括：")
BUL(pdf, [
 "Heartbeat 主动巡检闭环：每 30 分钟一次定时巡检，真查 5 节点（本机 + 3×H20 + 1×B200）GPU 占用，"
 "对在跑的训练/评测做健康检查（step 递增、nf=0、无 NCCL/OOM），空闲节点立即补任务、不留空转。",
 "多 Agent 并行协作：研究调研、设计文档、代码实现、根因排查由多个子 Agent 分工并行，主 Agent 综合结论。",
 "双重验证原则：仅看 GPU busy 快照会误判——busy=8/8 可能是 NCCL teardown 僵死、proc=0 但 busy=8/8 是残留显存、"
 "4/8 可能是 eval 切任务的瞬时空隙；必须交叉验证进程状态与产物。",
 "踩坑→护栏沉淀：每个静默失效（数据饥饿、缓存失配、显存竞争）都被转化为代码级 fail-fast 护栏（见第 4 节）。",
])
P(pdf,
 "这种范式的价值在于：它能在长达数十小时的实验中保持高 GPU 利用率与实验连续性，"
 "同时把人类从重复的运维与监控中解放出来，只聚焦科学判断。下文先介绍方法（第 2 节），"
 "再展开核心科学问题与攻坚（第 3 节），最后总结工程护栏（第 4 节）与结论（第 5 节）。")

# ============ 2. 方法 MoM ============
pdf.add_page()
H(pdf, "2  方法：Mixture-of-Memory（MoM）", 14)
H(pdf, "2.1  动机：有界 KV 预算下的长上下文", 11.5, top=1)
P(pdf,
 "标准 Transformer 的 KV cache 随序列长度线性增长，超长上下文下显存不可承受。"
 "我们的思路是把任意长的上下文压缩进一个固定大小（N 个 slot）的记忆缓冲：输入流按 chunk_size=512 "
 "切块，逐块流式写入记忆；回答时只从记忆读出，KV 预算与序列长度无关。")
H(pdf, "2.2  核心类比：MoM ≈ MoE（top-k 路由那一维的并行）", 11.5)
P(pdf,
 "mem_space 适配器与混合专家（MoE）在结构上高度同构，这是我们理解与设计该架构的主线：")
BUL(pdf, [
 "N 个 memory slot ≈ N 个“专家”。一个所有 Transformer 层共享的 MemoryBank（每样本一份）持有这些 slot。",
 "Selector ≈ 门控网络：对每个 chunk 给全部 N 个 slot 打分，选出 top-k=16 个 slot 参与本步——这正是 MoE 的稀疏门控。",
 "写入（Write）= 仅对 top-k 选中的 slot 做门控 delta-rule 更新（稀疏更新，每步至多触碰 k 个 slot），对应 MoE 只激活 top-k 专家。",
 "读出（Read）= 专用 MemoryCrossAttentionRead，对全部 N 个 slot（含一个 null/sink）做独立 softmax。",
 "读写路径刻意解耦：写是 top-k 稀疏，读是 all-N 稠密——这点与标准 MoE“激活即输出”不同，也埋下了后文的核心瓶颈。",
])
H(pdf, "2.3  架构细节", 11.5)
BUL(pdf, [
 "Chunk 流式写入：context chunk 在 no_grad 下逐块写入并 detach，目标 chunk 带梯度——逼记忆通道真正承载长程信息。",
 "Dual-gate delta-rule 写回：输入门 + 遗忘门控制写入，可选残差 delta-rule 更新（按输入门加权）。",
 "L3 summary：共享的 L3 交叉注意力路径产出 K=64 个 summary token，承载长程信息，是长程主力。",
 "代码：src/memory/mem_space/（config.py / layer.py / selector.py / memory_bank.py / l3_summary.py）。架构迭代见 versions/。",
])

# ============ 3. readout 攻坚 ============
H(pdf, "3  核心问题与攻坚：readout 瓶颈", 14)
H(pdf, "3.1  诊断：“写得进，读不准”", 11.5, top=1)
P(pdf,
 "我们用 BABILong（qa1/qa2/qa5，0k–32k）评测，并区分两种读出口径：开卷 SWA（query 段局部 attention 直接看原始 KV）"
 "与闭卷 W0（纯记忆读出，--swa_eval_chunks 0）。关键观察：开卷 SWA 能到 50–60 分，说明信息确实进了记忆；"
 "但纯记忆读出 W0 只有 10–30，长程 32k 跌到约 6。即——信息写得进，却读不准。瓶颈在读出机制，而非存储。")
H(pdf, "3.2  四类杠杆", 11.5)
BUL(pdf, [
 "token-mass 读出加权：让 slot 按其浓缩的真 token 数在 readout softmax 前加 log1p(mass) 偏置（高信息量 slot 更易被读到）。",
 "self-study 蒸馏：teacher = 冻结 backbone 看完整上下文（开卷），student = 记忆读出（闭卷），用 logits-KL + hidden-cosine 让闭卷逼近开卷。",
 "长度课程（curriculum）：训练序列长度 4K→8K→16K→32K 渐进（仅作用于 T2 合成检索流，dolmino 锁定避免数据饥饿）。",
 "叠加：token-mass × 蒸馏 的不同强度组合。",
])
H(pdf, "3.3  关键结果", 11.5)
fig(pdf, "levers_qa5.png", w=140)
P(pdf,
 "（1）mass 与蒸馏各自有效：mass 在中短程把 qa5 翻倍（1k 31→58）且 seed 鲁棒；蒸馏可复现、8k 处最强（15）。"
 "（2）长度课程救回了“直训 32k 退化”，但未突破中程天花板。"
 "（3）★最重要：弱 mass（coef≈0.5）+ 蒸馏在长程产生协同——下图显示随 mass 强度呈倒 U，峰值 coef≈0.5 时 "
 "16k=13、32k=9，首次超过所有单独方法；mass 过强（coef=2.0）则与蒸馏目标冲突、长程崩到 5–6。")
fig(pdf, "coef_invertedU.png", w=130)
P(pdf,
 "（4）长训退化：最优配置在约 500 步即收敛，继续训到 1000 步长程反而下降（32k 9→6）——“早停”是甜点，不应盲目加训练量。")
H(pdf, "3.4  LongBench 真实长文档：能力缺失定位", 11.5)
fig(pdf, "longbench_gap.png", w=145)
P(pdf,
 "我们首次在真实长文档 QA（LongBench 6 数据集）上拿到记忆模型的 W0 分（此前因显存竞争与格式问题屡次静默失败）。"
 "对照 base 模型开卷基线，能力缺失高度集中：narrativeqa（超长叙事）几乎全失效（16.0→2.6，仅保留 16%），"
 "qasper（科学论文细节）保留 38%；而多跳检索类（2wikimqa/hotpotqa）保留 66–75% 相对最好——"
 "恰是合成检索训练强化的“跨片段关联”能力。这把下一步要攻的短板明确指向"
 "“超长单一文档的连贯保持”与“细粒度信息保真”，而非关联检索。")

# ============ 4. 工程护栏 ============
pdf.add_page()
H(pdf, "4  工程实践与质量护栏", 14)
P(pdf,
 "大模型驱动的高强度实验会遇到大量“看着在跑、其实没效果”的静默失效。本工作把每次踩坑都沉淀为代码级护栏，"
 "形成质量闭环——这是“大模型驱动软件开发”能否可靠的关键：")
BUL(pdf, [
 "数据饥饿 → 启动即报错：dolmino 单文档上限 4096 token，当 (n_ctx+1)×chunk_size 超限时 loader 零产出、"
 "DDP 第一步静默死锁。修复：零产出 epoch 直接 raise，并把长度课程解耦到 T2 流（dolmino 锁 n_ctx=3）。",
 "蒸馏缓存失配 → 指纹断言 + 命中率 fail-fast：teacher 缓存按位置索引，跨盘数据集行序不同 / 文件漏同步会导致 "
 "100% cache miss、蒸馏静默失效。修复：缓存记录数据集 fingerprint，训练启动断言一致；step50 命中率为 0 即 raise。",
 "NCCL teardown 僵死：训练逻辑完成却卡在 teardown，busy=8/8 假象占卡数小时。巡检需交叉验证进程状态与 final ckpt。",
 "显存竞争 OOM：启动训练前必须查 nvidia-smi 真实显存（不能只看 proc 数），避免与残留进程争抢。",
 "跨盘一致性：三个独立物理盘（diskA/diskB/B200-wzc1），代码改动需完整 rsync（含 build/launch/src 全部文件）。",
 "LongBench chat_template：base 模型无 chat_template，脚本硬编码 --use_chat_template 致全 worker 首样本崩溃；改为默认关闭。",
])
P(pdf,
 "这些护栏的共同主题是：把静默失效转化为快速、显式的失败，让 Agent 在下一个巡检周期就能发现并自愈，"
 "而不是浪费数小时算力。")

# ============ 5. 结论 ============
H(pdf, "5  结论与展望", 14)
BUL(pdf, [
 "MoM 的 MoE 式 top-k 路由能在有界 KV 下承载长上下文，但读写解耦带来“读不准”的核心瓶颈。",
 "机制侧杠杆（mass、蒸馏）优于训练侧（容量/长度/课程）；弱 mass+蒸馏的组合在长程首次超过单独方法。",
 "长程 32k 仍是共同天花板（6→9），未被根本突破；真实长文档上缺失集中于超长叙事与细粒度抽取。",
 "展望：沿最优弱叠加配置 + 针对“连贯/细粒度”的新机制，配合 LongBench 持续验证。",
])
P(pdf,
 "方法论层面，本工作验证了大模型 Agent 可以自主驱动一个真实的多节点 ML 研究项目——"
 "从调度到排错到结论——并通过护栏沉淀保证工程可靠性。")

# ============ 附录 ============
pdf.add_page()
H(pdf, "附录 A  完整 W0 对照表（BABILong qa5，chunk512，0–32k）", 12.5)
rows = [
 ("配置","0k","1k","2k","4k","8k","16k","32k"),
 ("baseline (T2_chunk512)","70","31","53","22","13","8","6"),
 ("mass coef0.5","73","49","40","25","10","12","9"),
 ("mass coef2.0","78","58","48","28","10","12","7"),
 ("mass coef2 (seed1234)","63","58","46","26","12","10","8"),
 ("蒸馏 A+B","70","59","45","25","15","11","8"),
 ("叠加 coef0.5+蒸馏","70","49","44","25","14","13","9"),
 ("叠加 coef0.7+蒸馏","69","57","36","26","11","10","8"),
 ("叠加 coef2.0+蒸馏","71","50","33","22","10","6","5"),
]
pdf.set_font("noto","",8.6)
cw=[52,17,17,17,17,17,17,17]
for ri,row in enumerate(rows):
    if ri==0: pdf.set_font("noto","B",8.6); pdf.set_fill_color(225,232,245)
    else: pdf.set_font("noto","",8.6); pdf.set_fill_color(255,255,255)
    pdf.set_x(pdf.l_margin)
    for ci,c in enumerate(row):
        pdf.cell(cw[ci],6,c,border=1,align="C",fill=True)
    pdf.ln(6)
pdf.ln(3)

H(pdf, "附录 B  弱叠加 coef 精扫（长程 qa5）", 12.5)
rowsB=[("coef","8k","16k","32k"),("0.3","14","11","8"),("0.5","14","13","9"),
       ("0.7","11","10","8"),("2.0","10","6","5")]
pdf.set_font("noto","",9)
for ri,row in enumerate(rowsB):
    if ri==0: pdf.set_font("noto","B",9); pdf.set_fill_color(225,232,245)
    else: pdf.set_font("noto","",9); pdf.set_fill_color(255,255,255)
    pdf.set_x(pdf.l_margin)
    for c in row: pdf.cell(28,6,c,border=1,align="C",fill=True)
    pdf.ln(6)
pdf.ln(3)

H(pdf, "附录 C  LongBench W0（F1）vs base 开卷基线", 12.5)
rowsC=[("数据集","mem W0","base 开卷","保留率"),
 ("narrativeqa","2.6","16.0","16%"),("qasper","5.2","13.9","38%"),
 ("musique","3.5","7.0","50%"),("multifieldqa_en","12.3","24.9","49%"),
 ("hotpotqa","6.5","9.8","66%"),("2wikimqa","9.2","12.2","75%"),
 ("平均","6.56","14.0","47%")]
pdf.set_font("noto","",9)
cwC=[44,28,30,24]
for ri,row in enumerate(rowsC):
    if ri==0: pdf.set_font("noto","B",9); pdf.set_fill_color(225,232,245)
    elif row[0]=="平均": pdf.set_font("noto","B",9); pdf.set_fill_color(240,240,240)
    else: pdf.set_font("noto","",9); pdf.set_fill_color(255,255,255)
    pdf.set_x(pdf.l_margin)
    for ci,c in enumerate(row): pdf.cell(cwC[ci],6,c,border=1,align="C",fill=True)
    pdf.ln(6)
pdf.ln(3)

H(pdf, "附录 D  关键代码与脚本", 12.5)
BUL(pdf, [
 "架构：src/memory/mem_space/{config,layer,selector,memory_bank,l3_summary}.py",
 "token-mass：--use_readout_mass_bias --readout_mass_coef（commit 79ad265）",
 "self-study 蒸馏：scripts/build_distill_cache.py + scripts/launch_distill_chunk512_AB.sh（commit 19477fb / 8ea2969）",
 "长度课程解耦 + 数据饥饿护栏：--t2_curriculum（commit dadc19f / 2c22d80）",
 "缓存指纹 + 命中率 fail-fast：commit 0cf2c23 / 438b674",
 "设计文档：versions/v21_selfstudy_distillation.md",
 "阶段报告：status/READOUT_ATTACK_STAGING_REPORT.md",
 "源码仓库：https://github.com/liuhanzuo/Mixture-of-Memory",
], size=9.6, lh=4.6)

out = os.path.join(os.path.dirname(__file__), "MoM_report.pdf")
pdf.output(out)
print("PDF written:", out, "pages:", pdf.page_no())
