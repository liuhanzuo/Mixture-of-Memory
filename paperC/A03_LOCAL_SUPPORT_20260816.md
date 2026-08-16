# A-03 局部支撑核实 / Local-Support Verification — paperC 六条经典统计学引用

**日期**: 2026-08-16
**范围**: A-03 的**局部支撑轴**（axis 2）。元数据轴（axis 1）已由 MAIN 关闭（6/6 PASS），本文件不重复。
**机器可读产物**: `paperC/evidence/A03_local_support_verification.json`
**GPU 用量**: 0（纯文献核实）

---

## 0. 先说最重要的：没有 fatal 级错误，但有一条 HIGH 级发现

**⚠️ 没有任何一条引用实质性错了。** 我专门检查了「论文说 A 推荐 X，实际 A 反对 X」这类方向反转 ——
六条全部**方向正确**。不需要「投稿前必须修」的 red-alert。

**但有一条 HIGH 级发现，方向和 A-03 原始表述相反，且它对论文是【加分项】：**

> **Brennan & Prediger (1981) 的摘要里还有第二条建议 —— 论文从未引用它 —— 而它正是 paperC 的
> v1 best-constant floor 的心理测量学前身。而 v1 floor 是论文的招牌统计量，论文【没有】免责声明它。**

详见 §2 finding_1。这是本次审计唯一需要 MAIN 认真处理的一条。

---

## 1. 六条裁定总表

| # | 引用键 | 论文断言 | 裁定 | 证据等级 |
|---|--------|---------|------|---------|
| 1 | `bennett1954communications` | Bennett's $S$ 令 $p_e=1/k$ | **SUPPORTED** | `SECONDARY_REVIEW` |
| 2 | `brennan1981kappa` | 他们**推荐**该 free-marginal 替代 | **SUPPORTED** | `PRIMARY_ABSTRACT_ONLY`（双权威字节一致）|
| 3 | `frary1988formula` | formula scoring 是 **per-item** 选项数的显式函数 | **PARTIALLY_SUPPORTED** | `PRIMARY_ABSTRACT_ONLY` |
| 4 | `brenner1996weightedkappa` | weighted $\kappa$ **systematically drifts** with $k$ | **PARTIALLY_SUPPORTED** | `PRIMARY_ABSTRACT_ONLY` |
| 5 | `devries2008pooledkappa` | 跨**异质** item group 的 pooling 已 established | **PARTIALLY_SUPPORTED** | `PRIMARY_ABSTRACT_ONLY` |
| 6 | `cohen1960kappa` | (a) = κ 分子；(b) constant-emitter 零是标准性质 | **SUPPORTED** | `MATHEMATICAL_DERIVATION_BY_ME` |

**统计**: 3 SUPPORTED / 3 PARTIALLY_SUPPORTED / 0 NOT_SUPPORTED / 0 UNVERIFIABLE / 0 FATAL。

**三条 PARTIALLY 全部是同一种病：论文把地盘让多了（over-disclaiming）。**
修法全是「加精度」而非「削主张」，且四条修改**全部对论文的 novelty boundary 有利**。

### 诚实声明：全文一篇都没拿到

六篇**全部 paywalled，两个盘（wzc1 + zwfy6）都没有本地副本**（有界搜索，未用 `find /`）。
所以：
- 5 篇靠**出版商/作者摘要逐字**（能交叉验证的都做了多权威比对）；
- Bennett 的公式靠**同行评议二级转述**（Artstein & Poesio 2008，ACL 期刊，全文已读）；
- Cohen 两条靠**我自己的推导**（这是此处能达到的最强等级，且不需要原文）。

**任何关于 Frary 1988 / Brenner 1996 正文（超出摘要）的说法仍属未核实。**

---

## 2. HIGH 级发现：finding_1 — Brennan 1981 是 v1 floor 的未引用前身

### 证据（逐字，双权威字节一致）

Brennan & Prediger (1981) 摘要，**作者本人语气**：

> "In validity studies, we suggest considering whether one wants an index of improvement beyond
> \"chance\" or beyond **the best a priori strategy employing base rates**. ... In the latter case,
> it is suggested that **the largest marginal proportion for the criterion measure be used in place
> of the \"chance\" term in kappa**."

来源：Crossref CSL-JSON（`<jats:p>` 包裹）+ OpenAlex `abstract_inverted_index`，**两者字节一致**；
ERIC EJ253083 只有前两句（截断），Semantic Scholar 被出版商 elided。

### 为什么这就是 paperC 的 v1

paperC `03_method.tex:12` 定义

$$f_{\mathrm{const}}=\max_{L\in\mathcal{L}}\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[y_{i}=L]$$

= **最大的 gold label marginal**，用它**取代 chance**。
Brennan & Prediger 推荐 **"the largest marginal proportion for the criterion measure ... in place of
the chance term"**。paperC 里 criterion measure 就是 gold label（每题单一正确答案），
所以 "largest marginal proportion for the criterion measure" **就是** $f_{\mathrm{const}}$。
连 "chance vs best a priori strategy" 的对立框架都和 paperC intro 一致
（"Chance can credit a literal constant emitter as competent"）。

### 论文现状

`grep` 确认 `brennan1981kappa` 全文**只出现 2 处**，**都在 v2 的 1/k 免责段里**
（`03_method.tex:52`、`09a_relocated.tex:8`）。
v1 的 majority-class/best-constant 思想，论文只用 ML 侧的 `balepur2024artifacts` 归因
（`02_related.tex:4`、`01_introduction.tex:7`），**没有任何心理测量学前身被credit**。

### 双向风险

**(i) 引用完整性方向**：论文引了 Brennan 的 1/n 建议，却对**同一篇**的 largest-marginal 建议保持沉默，
同时把 largest-marginal 统计量作为自己的招牌 gate 提出。
一个审稿人打开 Brennan（1502 引、就在论文自己的参考文献里、正是这个领域的经典），
会发现**论文的核心 reference 被一篇它已经引用了的文献预见了**。这是最糟糕的暴露方式。

**这不是抄袭，也不是虚假陈述** —— 论文从未声称 best-constant 统计量是新的
（`01_introduction.tex:7` 明确 credit 了 `balepur2024artifacts`）。
但「已引用文献内部的未引用前身」是真实的归因缺口。

**(ii) 这条同时【帮】论文**（更重要）。处理得当是净收益：
- v1 floor 不再是 ad-hoc ML 启发式，而是**心理测量学对 validity study 推荐的 reference，1981 年就有出处**；
- 正好补上 **X1 说的 "boundary against the closest work is incomplete"** ——
  论文的贡献是**协议**（pre-comparison gate、自由度审计、双 reference 分离），不是统计量；
  加上这条让这个分界**更干净**而非更模糊；
- 提供了 X1 要的那个「最近的前作」。

### 置信度

- 逐字文本：**HIGH**（Crossref + OpenAlex 字节一致 + ERIC 前两句吻合）
- 「这就是 v1」：**MEDIUM-HIGH** —— 依据是摘要的作者自述句，**我没读到正文**。
  criterion-measure → gold-label 的映射是我做的，但在 paperC 的设定下（每题单一 gold）这个映射是被迫的。

### 建议动作：**只加，不减**

在 `02_related.tex` 的 nulls 段、或 `03_method.tex:14` v1 定义之后加：

> The best-constant reference itself has a psychometric precedent: for validity studies
> \citet{brennan1981kappa} recommend replacing the chance term with ``the largest marginal
> proportion for the criterion measure'', which is exactly $f_{\mathrm{const}}$ when the criterion
> is the gold label. We therefore claim neither the statistic nor the idea of referencing against a
> best a priori strategy; what we add is its use as a pre-comparison gate, the audit of the degrees
> of freedom inside it, and the separation of the arm-independent from the arm-conditional question.

⚠️ **给 MAIN 的 caveat**：引文出自**摘要**。若有 co-author 能拿到正文，请补页码。
拿不到也安全 —— 上句逐字引作者自述，未对正文作任何断言。

---

## 3. finding_2（MEDIUM，纯加分）：paperC 的分层 κ **就是** De Vries 的 pooled 估计量

### 我的推导（`MATHEMATICAL_DERIVATION_BY_ME`）

令 $w_s=n_s/n$，由 $\sum_s w_s=1$：

$$\kappa_{\text{paperC}}=\frac{\mathrm{acc}-\widehat{\mathrm{acc}}}{1-\widehat{\mathrm{acc}}}
=\frac{\sum_s w_s\,(p_{o,s}-p_{e,s})}{\sum_s w_s\,(1-p_{e,s})}$$

这个「加权和之比」正是 **pooled kappa** 形式，**不是** $\mathrm{mean}_s\,\kappa_s$
（average kappa —— 恰是 De Vries 等人论证要避免的那个）。

**数值验证**（合成 variable-`n_opt` 数据，8 strata，模拟 MMLU-Pro，seeds 7/42/1234）：
`|κ_paperC − κ_pooled|` = 3.4e-17 / 1.7e-17 / 4.7e-17。
而 average-of-strata κ 三次都**显著不同**（如 seed 7：pooled −0.00148 vs averaged −0.01273），
说明这个 identity **专指 pooled，不是巧合**。

### 为什么重要

论文现在把 `devries2008pooledkappa` 当成**让出去的地盘**（"pooling ... is likewise established"）。
实际上论文的估计量**就是**那个估计量的实例 → 可以**免费继承 De Vries 的效率论证**，
从而正面回答「为什么 pool 而不是 average 各 stratum 的 κ？」——
**论文目前对这个问题没有任何回答。**

---

## 4. 逐条细节与改写建议

### #1 `bennett1954communications` → **SUPPORTED**

原文摘要**不足**（只说 "greater than could be expected on the basis of chance"，无函数形式）——
**MAIN 的判断被确认，未被推翻**。但二级证据把它补齐了：

Artstein & Poesio 2008 §2.4.1（PDF p.6），小节标题就叫 *"All Categories Are Equally Likely: S"*：

> "The simplest way of discounting for chance is the one adopted to compute the coefficient S
> (Bennett, Alpert, and Goldstein 1954) ... the computation of S is based on an interpretation of
> chance as a random choice of category from a uniform distribution — that is, all categories are
> equally likely. If coders classify the items into $k$ categories, then the chance $P(k|c_i)$ of any
> coder assigning an item to category $k$ under the uniformity assumption is $1/k$; hence the total
> agreement expected by chance is $A_e^S=\sum_{k\in K}\frac{1}{k}\cdot\frac{1}{k}=k\cdot(1/k)^2=1/k$"

来源可靠性：ACL 期刊 *Computational Linguistics* 34(4):555-596，DOI `10.1162/coli.07-034-r2`，
自述目的即 "expose the mathematics and underlying assumptions of agreement coefficients"，
参考文献里逐条列了 Bennett 1954 和 Brennan & Prediger 1981。
获取：`https://aclanthology.org/J08-4004.pdf`，http=200，291595 bytes，
md5 `d79331aab8017c87f144a46cface467c`。

**动作：不改。** 论文写 $p_e$、A&P 写 $A_e$，同一对象。

### #2 `brennan1981kappa` → **SUPPORTED**（句子本身）

> "When either or both of the marginals are free to vary, however, **it is suggested that** the
> \"chance\" term in kappa **be replaced by 1/n, where n is the number of categories**."

- **"recommend" 这个强动词站得住**：这是作者自己语气的建议，不是「讨论/列举」。
- **条件从句忠实**：原文条件是 "marginals are free to vary"，论文写 "wherever $\kappa$'s marginal
  dependence misleads" —— 吻合。
- **"free-marginal" 是标准下游命名**（Brennan-Prediger coefficient / Randolph's free-marginal kappa；
  参 Warrens 2010 摘要 "Randolph's kappa generalizes Bennett et al. S to multiple raters"）。
- **方向未反转**，二级证据双向确认：A&P p.7 "It has been argued that uniformity is the best model ...
  (Brennan and Prediger 1981)"；另 A&P §4 "reservations about the use of $\kappa$ have been noted by
  Brennan and Prediger (1981)" —— 既推荐替代、也批评 κ 的 marginal 依赖。

**动作：句子保留不改。** 但 **finding_1 必须另行处理**。

### #3 `frary1988formula` → **PARTIALLY_SUPPORTED**

摘要**支持**的部分：「formula scoring corrects for guessing」完全支持 ——
"a procedure designed to reduce multiple-choice test score irregularities due to guessing";
"a formula score is obtained by subtracting a proportion of the number of wrong responses from the
number correct"。

摘要**不支持**的部分：**"as an explicit function of the PER-ITEM option count"**。
摘要只说 **"a proportion"**，从未点明该比例、从未写出 $S=R-W/(k-1)$、从未说 $k$ 可逐题变化。
经典公式 $S=R-W/(k-1)$ 里 $k$ 依赖选项数几乎必然是正文事实；
但 **"per-item"（$k$ 按题索引）无任何我能触达的权威支持**，而这恰是**承重词**。

**🔧 finding_3：论文自伤**。同一段里论文同时断言
(a) Frary 的修正是 **per-item** 选项数的函数，且
(b) **"The works above assume a single global $k$"**。
字面读**互相矛盾**：若 Frary 的 $k$ 真是 per-item，论文两句后的 novelty 主张
（"what we are not aware of ... is a null in which $k$ varies item to item"）就被自己的归因句削掉了。
**删掉 "per-item" 把这块地盘还给论文，零成本。**

**建议改写**：
- 原：`formula scoring corrects for guessing as an explicit function of the per-item option count`
- 改：`formula scoring corrects for guessing as an explicit function of the number of options`

（若某 co-author 能读到正文并确认 Frary 真的按题索引 $k$，才保留 "per-item"。）

**已穷尽的权威**：unpaywall(is_oa=false)、Semantic Scholar(elided/CLOSED)、OpenAlex(仅摘要)、
ERIC EJ374435(description 截断、无全文)、`files.eric.ed.gov`(404)、Wiley 出版商 PDF(403)、
NCME ITEMS module 两条路径(404)、枚举 64 篇引用它的 OA 文献并抓取 4 篇最相关(全 403)、
针对 $S=R-W/(k-1)$ 的 OpenAlex 全文检索(无相关命中)、两盘本地(无)。

### #4 `brenner1996weightedkappa` → **PARTIALLY_SUPPORTED**

**方向正确且带符号**：drift 存在、是**增加**、且 "expected under a broad variety of conditions"
（这就 license 了 "systematically"）；结尾 "require careful consideration in the interpretation"
也 license 了论文把它当作要规避的 pathology。

**为什么只是 partial**：摘要的结论**按 weighting scheme 分层**，论文无限定的 "a weighted $\kappa$" 抹掉了分层：

> "an increase of **quadratically** weighted kappa coefficients with the number of categories is
> expected under a broad variety of conditions, whereas **linearly** weighted kappa coefficients
> appear to be **less sensitive** to the number of categories."

另：其 scope 是 **ordinal** ratings；paperC 的 letter 是 **nominal**，而对 nominal unweighted κ
这篇什么都没说 —— 而 paperC 的 $\Delta_{\mathrm{perm}}$ 恰恰是 nominal unweighted。

**建议改写**：
- 原：`\citet{brenner1996weightedkappa} show that a weighted $\kappa$ drifts systematically with the number of categories`
- 改：`\citet{brenner1996weightedkappa} show that quadratically weighted $\kappa$ increases systematically with the number of categories (linear weights being far less sensitive)`

若版面允许，更诚实（且 novelty 上略赚）的框法是：$k$-依赖 pathology 是在 **weighted ordinal** κ 上被记录的，
因此它**motivate 而非已覆盖** paperC 的 nominal + variable-$k$ 情形。

### #5 `devries2008pooledkappa` → **PARTIALLY_SUPPORTED**

**支持**：「跨 item group 做 pooling 已 established」—— 是。他们正是提出 pooled kappa 估计量，
在 2,176 个 rated item 上聚合、并 "summarize interrater agreement **by domain**"（即按 item 分组 pool）。
231-256 引，"established" 公允。

**不支持**：**"HETEROGENEOUS"** 这个词。摘要的动机是**统计效率**
（"many items but few subjects"），对照物是 "average kappa"；
**从未**提及各组的 category 数不同，也没把 heterogeneity 当作 pool 的理由。
论文的 "heterogeneous" 把 paperC 自己的动机（strata 按 `n_opt` 不同）悄悄塞进了 De Vries 的论文。
「跨组 pool」= established；「跨**响应类别数不同**的组 pool」= 无权威支持。

**建议改写**（两处修）：
- 原：`Pooling an agreement coefficient across heterogeneous item groups is likewise established \citep{devries2008pooledkappa}`
- 改：`Pooling an agreement coefficient across item groups is likewise established \citep{devries2008pooledkappa}, and our stratified statistic is exactly their pooled (not averaged) estimator with strata defined by option count.`

理由：(1) 删 "heterogeneous"；(2) 加上 finding_2 的 identity（已验到 1e-17）——
把**单纯的让让让**转成**继承来的设计正当性**。

### #6 `cohen1960kappa` → **SUPPORTED**（六条里唯一定案到确定性的）

Cohen 1960 **没有摘要可引**（OpenAlex/Crossref 都无 —— 1960 年的文章本身就没有摘要；
Semantic Scholar elided；unpaywall is_oa=false）。**这条不需要摘要。**

**断言 (a)：$\Delta_{\mathrm{perm}}$ = 分层内 Cohen κ 的分子 → TRUE，精确成立**

论文估计量（`03_method.tex` eq.1；实现在
`paperC/code/heal_readout_v2_permutation_null.py:171` `_acc_hat`）：

$$\widehat{\mathrm{acc}}=\sum_s\frac{1}{n\,n_s}\sum_L c^{\mathrm{pred}}_{s,L}c^{\mathrm{gold}}_{s,L}$$

推导：
1. $\frac{1}{n\,n_s}\sum_L c^{\mathrm{pred}}c^{\mathrm{gold}}
   =\frac{n_s}{n}\cdot\frac{1}{n_s^2}\sum_L c^{\mathrm{pred}}c^{\mathrm{gold}}
   = w_s\sum_L P(\mathrm{pred}{=}L\mid s)P(\mathrm{gold}{=}L\mid s)$，其中 $w_s=n_s/n$
2. $\sum_L P(\mathrm{pred}{=}L\mid s)P(\mathrm{gold}{=}L\mid s)$ 正是 stratum $s$ 内 Cohen 的 $p_e$
   （两 rater 独立、各用自己的 marginal —— 即 A&P §2.4.3 的 $A_e^\kappa$）
3. 故 $\widehat{\mathrm{acc}}=\sum_s w_s p_{e,s}$，而 $\mathrm{acc}=\sum_s w_s p_{o,s}$
4. $\Delta_{\mathrm{perm}}=\sum_s w_s\,(p_{o,s}-p_{e,s})$ = 各分层 Cohen κ **分子**的 $w_s$-加权和 ∎

**数值验证**（合成 variable-`n_opt` 数据，seeds 7/42/1234）：
`|Δ_perm − Σ_s w_s(p_o,s−p_e,s)|` = 2.7e-17 / 1.4e-17 / 3.8e-17；
`|acc_hat − Σ_s w_s p_e,s|` ≤ 2.8e-17。**机器精度。**

论文自己在 `03_method.tex:37-39` 的注解（$\kappa=(p_o-p_e)/(1-p_e)$，
故 $\Delta_{\mathrm{perm}}=\kappa(1-p_e)$ identically）同样成立且自洽。

**断言 (b)：constant-emitter 得零是该文献标准性质 → TRUE**

二级逐字（A&P 2008 p.5，作为整个 S/π/κ 族的定义性性质）：

> "All three coefficients therefore yield values of agreement between $-A_e/1-A_e$ (no observed
> agreement) and 1 (observed agreement = 1), **with the value 0 signifying chance agreement
> (observed agreement = expected agreement)**."

我的推导：常数发射字母 $L$ 时，每个 stratum 内 $p_{o,s}=m_{s,L}$ 且
$p_{e,s}=\sum_{L'}\delta_{L'L}m_{s,L'}=m_{s,L}$ → 每个分层分子归零 →
$\Delta_{\mathrm{perm}}=0$ **精确**，且对**每个合法字母**成立（不只 modal 那个）。
数值：10 个字母各自限制在其合法 item 上，$|\Delta|\le$ 2.8e-17。

**动作：不改。** 论文正确地把它 disclaim 掉了。

---

## 5. 对 `assumptions.md` A-03 的建议

- **axis 1（元数据）**：CLOSED（6/6 PASS，MAIN 已做）
- **axis 2（局部支撑）**：**CLOSED WITH FINDINGS** —— 3 SUPPORTED / 3 PARTIALLY / 0 NOT_SUPPORTED / 0 UNVERIFIABLE / 0 FATAL

**A-03 的表述必须改写（确认 MAIN 的方向纠正）**：
原失败模式「论文是否把 Frary 1988 已说过的当成自己的新贡献」**结构性不存在** ——
该段明文 disclaim 了 Frary 的内容。
**真实的失败模式正是 MAIN 预测的反方向：over-disclaiming**，
6 句里 3 句中招，**外加一条论文【没有】disclaim 的统计量（v1 floor）存在未引用前身**（finding_1）。

**两类缺陷全部靠「加精度」修好，四条修改全部对论文的 novelty boundary 有利。**

**必须诚实保留的残余限制**：六篇**无一拿到全文**。五条依赖摘要，一条依赖二级转述，
两条（Cohen）依赖我的推导。**关于 Frary 1988 / Brenner 1996 正文超出摘要的任何说法仍未核实。**

---

## 6. 交付与边界

- 机器可读：`paperC/evidence/A03_local_support_verification.json`
  （含每条的证据等级、试过的 authority 清单及其返回、逐字引文、可复现的推导与数值）
- **我没有改任何 `.tex`** —— 另有 agent 正在改 paperC/sections（MMLU-Pro null 修正）。
  §2/§4 的改写建议仅为文本，由 MAIN 合并。
