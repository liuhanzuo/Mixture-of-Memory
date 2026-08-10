# Paper Draft — Reference Fact-Check + Limitations & Conclusion

> Draft author: paper-writing agent (read + write + web only; no experiments run, no code changed).
> Companion to `PAPER_DRAFT_INTRO_RELATED_20260628.md` (§1–2) and
> `PAPER_DRAFT_METHOD_EXP_20260628.md` (§3–4). Terminology / notation
> (W0 / oracle / token-reforward / read-out wall / selection wall /
> select-then-reforward vs. compress-then-inject) kept consistent with those drafts.
> **All numbers cited are clean `babilong_mix=0` measurements**; leaked `mix>0` ("wall-break")
> scores are never used.
>
> **Date:** 2026-06-28.
> **Verification method:** system proxy (`hy-proxy.woa.com:3128`) + `curl` of arXiv abstract pages
> (`citation_title` / `citation_author` / `citation_date` / `citation_arxiv_id` meta tags) and the
> LongChat/LongEval GitHub repo (canonical BibTeX). `WebSearch`/`WebFetch` were unavailable this
> session (WebSearch returned a server-side 502; WebFetch is environment-blocked), so all checks were
> done by direct `curl` of the primary source, which is the stronger evidence anyway.
>
> **Status tags per reference:**
> **[web-verified]** = title / authors / date / arXiv-id confirmed against the primary source this
> session. Venue lines marked **(venue: training knowledge)** were *not* re-confirmed online and
> should be double-checked against the proceedings before camera-ready (the arXiv handle is the
> safe fallback).

---

## Part 1 — Reference fact-check (~9 groups, 18 papers)

Every entry below had its **title, author list, submission date, and arXiv id confirmed online this
session** unless noted. Venues are from training knowledge and flagged as such.

### 1. BABILong benchmark — [web-verified]
- **Title:** *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack*
- **Authors:** Yuri Kuratov, Aydar Bulatov, Petr Anokhin, Ivan Rodkin, Dmitry Sorokin, Artyom
  Sorokin, Mikhail Burtsev (the "RMT-team" — same group as RMT below)
- **arXiv:** 2406.10149 (submitted 2024-06-14)
- **Venue (training knowledge):** NeurIPS 2024, Datasets & Benchmarks Track
- Confirmed abstract facts (usable in text): 20 reasoning tasks; models effectively use only
  **10–20%** of context; RAG reaches ~60% on single-fact QA independent of length; recurrent memory
  transformers process up to **50M tokens** after fine-tuning. Our qa1/qa2/qa5 are the
  fact-chaining / two-fact / counting subset.

```bibtex
@inproceedings{kuratov2024babilong,
  title={{BABILong}: Testing the Limits of {LLMs} with Long Context Reasoning-in-a-Haystack},
  author={Kuratov, Yuri and Bulatov, Aydar and Anokhin, Petr and Rodkin, Ivan and Sorokin, Dmitry and Sorokin, Artyom and Burtsev, Mikhail},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year={2024},
  note={arXiv:2406.10149}
}
```

### 2. Llama-3 / Llama-3-8B — [web-verified]
- **Title:** *The Llama 3 Herd of Models*
- **Authors:** Llama Team / Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. (Meta AI; very
  large author list — use "Llama Team, AI @ Meta" or "Grattafiori et al." in text)
- **arXiv:** 2407.21783 (submitted 2024-07-31)
- **Venue (training knowledge):** technical report (no peer-reviewed venue; cite the arXiv tech report).
- Note: this is the canonical citation for the Llama-3 / Llama-3-8B backbone we use.

```bibtex
@article{grattafiori2024llama3,
  title={The {Llama} 3 Herd of Models},
  author={Grattafiori, Aaron and Dubey, Abhimanyu and Jauhri, Abhinav and others (Llama Team, AI @ Meta)},
  journal={arXiv preprint arXiv:2407.21783},
  year={2024}
}
```

### 3. PG19 dataset + Compressive Transformer — [web-verified]
- **Title:** *Compressive Transformers for Long-Range Sequence Modelling*
- **Authors:** Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Timothy P. Lillicrap
- **arXiv:** 1911.05507 (submitted 2019-11-13)
- **Venue (training knowledge):** ICLR 2020
- Note: this paper **introduces both** the PG-19 long-document language-modelling benchmark *and* the
  Compressive Transformer. Our clean dense long-context anchor is trained on pg19; cite this for both
  "pg19 dataset" and "Compressive Transformer" mentions.

```bibtex
@inproceedings{rae2020compressive,
  title={Compressive Transformers for Long-Range Sequence Modelling},
  author={Rae, Jack W. and Potapenko, Anna and Jayakumar, Siddhant M. and Lillicrap, Timothy P.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
  note={arXiv:1911.05507; introduces the PG-19 benchmark}
}
```

### 4. KV-cache compression / eviction — all four [web-verified]

**SnapKV** — title confirms the "attention-salience before generation" framing.
- **Title:** *SnapKV: LLM Knows What You are Looking for Before Generation*
- **Authors:** Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye,
  Tianle Cai, Patrick Lewis, Deming Chen
- **arXiv:** 2404.14469 (2024-04-22) · **Venue (training knowledge):** NeurIPS 2024
- Salience attribution: clusters/pools the model's **own attention scores from an observation window**
  to select which KV positions to keep — i.e. attention-salience-based **selection/eviction**. ✓

**H2O** — heavy-hitter (high-attention) tokens; attention-salience eviction.
- **Title:** *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*
- **Authors:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao
  Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen
- **arXiv:** 2306.14048 (2023-06-24) · **Venue (training knowledge):** NeurIPS 2023
- Salience attribution: keeps "heavy hitters" — tokens with **high accumulated attention scores** —
  and evicts the rest. Canonical attention-salience eviction. ✓

**StreamingLLM** — attention sinks + recent window.
- **Title:** *Efficient Streaming Language Models with Attention Sinks*
- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis
- **arXiv:** 2309.17453 (2023-09-29) · **Venue (training knowledge):** ICLR 2024
- Salience attribution: observes that initial tokens act as **attention sinks** (absorb large
  attention mass); keeps sinks + a sliding recent window, evicts the middle. Attention-pattern-based
  retention (rather than a per-token salience score). ✓ (note: it is the "attention-sink + sliding
  window" variant of the salience-eviction family.)

**FastGen** — adaptive KV compression; the title in the draft ("FastGen") is the method nickname; the
arXiv title is the descriptive one.
- **Title:** *Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs*
- **Authors:** Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, Jianfeng Gao
- **arXiv:** 2310.01801 (2023-10-03) · **Venue (training knowledge):** ICLR 2024
- Salience attribution: profiles **per-head attention structure** ("the model tells you what to
  discard") and adaptively keeps the KV each head actually uses. Attention-salience-based. ✓

```bibtex
@inproceedings{li2024snapkv,
  title={{SnapKV}: {LLM} Knows What You are Looking for Before Generation},
  author={Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and Locatelli, Acyr and Ye, Hanchen and Cai, Tianle and Lewis, Patrick and Chen, Deming},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}, note={arXiv:2404.14469}
}
@inproceedings{zhang2023h2o,
  title={{H2O}: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models},
  author={Zhang, Zhenyu and Sheng, Ying and Zhou, Tianyi and Chen, Tianlong and Zheng, Lianmin and Cai, Ruisi and Song, Zhao and Tian, Yuandong and R\'e, Christopher and Barrett, Clark and Wang, Zhangyang and Chen, Beidi},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}, note={arXiv:2306.14048}
}
@inproceedings{xiao2024streamingllm,
  title={Efficient Streaming Language Models with Attention Sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}, note={arXiv:2309.17453}
}
@inproceedings{ge2024fastgen,
  title={Model Tells You What to Discard: Adaptive {KV} Cache Compression for {LLMs}},
  author={Ge, Suyu and Zhang, Yunan and Liu, Liyuan and Zhang, Minjia and Han, Jiawei and Gao, Jianfeng},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}, note={arXiv:2310.01801}
}
```

### 5. RoPE + positional interpolation — both [web-verified]

**RoPE / RoFormer** — Su et al. confirmed.
- **Title:** *RoFormer: Enhanced Transformer with Rotary Position Embedding*
- **Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- **arXiv:** 2104.09864 (2021-04-20) · **Venue (training knowledge):** *Neurocomputing*, 2024
  (journal publication; arXiv handle is the common citation).

**Positional Interpolation** — Chen et al. confirmed.
- **Title:** *Extending Context Window of Large Language Models via Positional Interpolation*
- **Authors:** Shouyuan Chen, Sherman Wong, Liangjian Chen, Yuandong Tian
- **arXiv:** 2306.15595 (2023-06-27) · **Venue (training knowledge):** technical report / arXiv
  (widely cited as arXiv; no formal proceedings).

```bibtex
@article{su2021roformer,
  title={{RoFormer}: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and Wen, Bo and Liu, Yunfeng},
  journal={Neurocomputing}, year={2024}, note={arXiv:2104.09864}
}
@article{chen2023positional,
  title={Extending Context Window of Large Language Models via Positional Interpolation},
  author={Chen, Shouyuan and Wong, Sherman and Chen, Liangjian and Tian, Yuandong},
  journal={arXiv preprint arXiv:2306.15595}, year={2023}
}
```

### 6. Recurrent / segment-level memory — all [web-verified]

**RMT (Recurrent Memory Transformer)** — Bulatov et al. confirmed (same group as BABILong).
- **Title:** *Recurrent Memory Transformer*
- **Authors:** Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev
- **arXiv:** 2207.06881 (2022-07-14) · **Venue (training knowledge):** NeurIPS 2022

**Transformer-XL** — Dai et al. confirmed.
- **Title:** *Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context*
- **Authors:** Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov
- **arXiv:** 1901.02860 (2019-01-09) · **Venue (training knowledge):** ACL 2019

**Compressive Transformer** — Rae et al.: **same paper as group 3** (1911.05507, ICLR 2020). Cite once;
it covers both the pg19 anchor and the segment-memory family mention.

```bibtex
@inproceedings{bulatov2022rmt,
  title={Recurrent Memory Transformer},
  author={Bulatov, Aydar and Kuratov, Yuri and Burtsev, Mikhail S.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}, note={arXiv:2207.06881}
}
@inproceedings{dai2019transformerxl,
  title={{Transformer-XL}: Attentive Language Models Beyond a Fixed-Length Context},
  author={Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V. and Salakhutdinov, Ruslan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2019}, note={arXiv:1901.02860}
}
```

### 7. Retrieval-augmented generation — all four [web-verified]

**RAG** — Lewis et al. confirmed.
- **Title:** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- **Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
  Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel
- **arXiv:** 2005.11401 (2020) · **Venue (training knowledge):** NeurIPS 2020

**REALM** — Guu et al. confirmed.
- **Title:** *REALM: Retrieval-Augmented Language Model Pre-Training*
- **Authors:** Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang
- **arXiv:** 2002.08909 (2020-02-10) · **Venue (training knowledge):** ICML 2020

**RETRO** — Borgeaud et al. confirmed.
- **Title:** *Improving language models by retrieving from trillions of tokens*
- **Authors:** Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford,
  Katie Millican, et al. (DeepMind)
- **arXiv:** 2112.04426 (2021) · **Venue (training knowledge):** ICML 2022

**kNN-LM** — Khandelwal et al. confirmed.
- **Title:** *Generalization through Memorization: Nearest Neighbor Language Models*
- **Authors:** Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, Mike Lewis
- **arXiv:** 1911.00172 (2019-11-01) · **Venue (training knowledge):** ICLR 2020

```bibtex
@inproceedings{lewis2020rag,
  title={Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}, note={arXiv:2005.11401}
}
@inproceedings{guu2020realm,
  title={{REALM}: Retrieval-Augmented Language Model Pre-Training},
  author={Guu, Kelvin and Lee, Kenton and Tung, Zora and Pasupat, Panupong and Chang, Ming-Wei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}, note={arXiv:2002.08909}
}
@inproceedings{borgeaud2022retro,
  title={Improving Language Models by Retrieving from Trillions of Tokens},
  author={Borgeaud, Sebastian and Mensch, Arthur and Hoffmann, Jordan and Cai, Trevor and Rutherford, Eliza and Millican, Katie and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}, note={arXiv:2112.04426}
}
@inproceedings{khandelwal2020knnlm,
  title={Generalization through Memorization: Nearest Neighbor Language Models},
  author={Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}, note={arXiv:1911.00172}
}
```

### 8. MemoryLLM + M+ — [web-verified, from MEMORYLLM_FACTCHECK_20260628.md]
Already verified in `MEMORYLLM_FACTCHECK_20260628.md`; cited directly. For completeness:
- **MemoryLLM** — Yu Wang et al., *MEMORYLLM: Towards Self-Updatable Large Language Models*,
  arXiv 2402.04624 (2024). Venue (training knowledge): ICML 2024.
- **M+** — Yu Wang et al., *M+: Extending MemoryLLM with Scalable Long-Term Memory*,
  arXiv 2502.00592 (2025). Venue (training knowledge): ICML 2025.

```bibtex
@inproceedings{wang2024memoryllm,
  title={{MEMORYLLM}: Towards Self-Updatable Large Language Models},
  author={Wang, Yu and Gao, Yifan and Chen, Xiusi and Jiang, Haoming and Li, Shiyang and Yang, Jingfeng and Yin, Qingyu and Li, Zheng and Li, Xian and Yin, Bing and Shang, Jingbo and McAuley, Julian},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}, note={arXiv:2402.04624}
}
@inproceedings{wang2025mplus,
  title={{M+}: Extending {MemoryLLM} with Scalable Long-Term Memory},
  author={Wang, Yu and others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}, note={arXiv:2502.00592}
}
```
> Note: MemoryLLM/M+ author lists beyond the lead author were not re-pulled this session (the
> mechanism facts are what `MEMORYLLM_FACTCHECK` verified); confirm the full author list before
> camera-ready. arXiv ids and titles are solid.

### 9. RULER + LongEval (Limitations) — both [web-verified]

**RULER** — Hsieh et al. confirmed.
- **Title:** *RULER: What's the Real Context Size of Your Long-Context Language Models?*
- **Authors:** Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia,
  Yang Zhang, Boris Ginsburg (NVIDIA)
- **arXiv:** 2404.06654 (2024-04-09) · **Venue (training knowledge):** COLM 2024

**LongEval** — verified via the official **LongChat/LongEval** GitHub repo (`DachengLi1/LongChat`),
which hosts the canonical BibTeX. LongEval is the long-context evaluation suite (topic-retrieval +
line-retrieval tasks) released with the LongChat blog post; it is a **blog/tech-report artifact, not an
arXiv paper**, so there is no arXiv id. Cite the LMSYS post (verified title + author list below).
- **Title:** *How Long Can Open-Source LLMs Truly Promise on Context Length?* (LMSYS blog, 2023-06-29)
- **Authors:** Dacheng Li*, Rulin Shao*, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez,
  Ion Stoica, Xuezhe Ma, Hao Zhang
- **URL:** https://lmsys.org/blog/2023-06-29-longchat ; repo: github.com/DachengLi1/LongChat
- (Optional companion for the same "lost-in-the-middle" point: Liu et al., *Lost in the Middle: How
  Language Models Use Long Contexts*, arXiv 2307.03172, TACL 2024 — **[web-verified]** — useful if a
  peer-reviewed citation for "models under-use long contexts" is wanted alongside the LongEval blog.)

```bibtex
@inproceedings{hsieh2024ruler,
  title={{RULER}: What's the Real Context Size of Your Long-Context Language Models?},
  author={Hsieh, Cheng-Ping and Sun, Simeng and Kriman, Samuel and Acharya, Shantanu and Rekesh, Dima and Jia, Fei and Zhang, Yang and Ginsburg, Boris},
  booktitle={Conference on Language Modeling (COLM)},
  year={2024}, note={arXiv:2404.06654}
}
@misc{li2023longchat,
  title={How Long Can Open-Source {LLMs} Truly Promise on Context Length?},
  author={Li, Dacheng and Shao, Rulin and Xie, Anze and Sheng, Ying and Zheng, Lianmin and Gonzalez, Joseph E. and Stoica, Ion and Ma, Xuezhe and Zhang, Hao},
  howpublished={\url{https://lmsys.org/blog/2023-06-29-longchat}},
  year={2023}, note={LongChat / LongEval}
}
@article{liu2024lostmiddle,
  title={Lost in the Middle: How Language Models Use Long Contexts},
  author={Liu, Nelson F. and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal={Transactions of the Association for Computational Linguistics (TACL)},
  year={2024}, note={arXiv:2307.03172}
}
```

### Verification summary table

| # | Reference | Title / authors / date / arXiv | Status |
|---|---|---|---|
| 1 | BABILong (Kuratov/Bulatov et al.) | 2406.10149, 2024-06-14 | **[web-verified]** ✓ |
| 2 | Llama 3 Herd (Meta) | 2407.21783, 2024-07-31 | **[web-verified]** ✓ |
| 3 | Compressive Transformer + PG19 (Rae et al.) | 1911.05507, 2019-11-13 | **[web-verified]** ✓ |
| 4a | SnapKV (Li et al.) | 2404.14469, 2024-04-22 | **[web-verified]** ✓; salience-eviction ✓ |
| 4b | H2O (Zhang et al.) | 2306.14048, 2023-06-24 | **[web-verified]** ✓; heavy-hitter salience ✓ |
| 4c | StreamingLLM / Attention Sinks (Xiao et al.) | 2309.17453, 2023-09-29 | **[web-verified]** ✓; sink+window ✓ |
| 4d | FastGen / "Model Tells You What to Discard" (Ge et al.) | 2310.01801, 2023-10-03 | **[web-verified]** ✓; per-head salience ✓ |
| 5a | RoFormer / RoPE (Su et al.) | 2104.09864, 2021-04-20 | **[web-verified]** ✓ |
| 5b | Positional Interpolation (Chen et al.) | 2306.15595, 2023-06-27 | **[web-verified]** ✓ |
| 6a | RMT (Bulatov et al.) | 2207.06881, 2022-07-14 | **[web-verified]** ✓ |
| 6b | Transformer-XL (Dai et al.) | 1901.02860, 2019-01-09 | **[web-verified]** ✓ |
| 6c | Compressive Transformer | = #3 | **[web-verified]** ✓ |
| 7a | RAG (Lewis et al.) | 2005.11401, 2020 | **[web-verified]** ✓ |
| 7b | REALM (Guu et al.) | 2002.08909, 2020-02-10 | **[web-verified]** ✓ |
| 7c | RETRO (Borgeaud et al.) | 2112.04426, 2021 | **[web-verified]** ✓ |
| 7d | kNN-LM (Khandelwal et al.) | 1911.00172, 2019-11-01 | **[web-verified]** ✓ |
| 8a | MemoryLLM (Wang et al.) | 2402.04624 | prior fact-check ✓ |
| 8b | M+ (Wang et al.) | 2502.00592 | prior fact-check ✓ |
| 9a | RULER (Hsieh et al.) | 2404.06654, 2024-04-09 | **[web-verified]** ✓ |
| 9b | LongEval / LongChat (Li et al.) | LMSYS blog 2023; no arXiv | **[web-verified]** ✓ (blog, not arXiv) |
| (9c) | Lost in the Middle (Liu et al.) | 2307.03172, 2023-07-06 | **[web-verified]** ✓ (optional) |

**Caveats to carry forward:** (a) all *venues* are training-knowledge and unverified online —
confirm against proceedings before camera-ready; the arXiv id is the safe fallback. (b) LongEval has
**no arXiv paper** — cite the LMSYS blog/GitHub repo, not a phantom arXiv id. (c) StreamingLLM is the
attention-*sink*+sliding-window member of the eviction family, not a per-token salience score —
phrase its attribution accordingly. (d) FastGen's arXiv title differs from its nickname; keep both.
No reference came back **[unverified]**; every primary source resolved.

---

## Part 2 — Limitations & Conclusion (paper sections, English)

### 5. Limitations

We are deliberately explicit about the boundaries of our claims, because one of our contributions is a
negative result and a diagnostic, not a universal method.

**The long-context selection wall is unsolved — it is our central limitation, not a footnote.**
On long contexts (a 25-candidate buffer at 16k/32k), the deployable selector cannot reach the
oracle's read-out ceiling. The information is present — the oracle that re-forwards the answer chunk's
original tokens scores 60–70 at 16k/32k — yet both selection signals we tried, the reader's native
$q\!\cdot\!k$ attention salience and slot-retrieval scores, sit at $\approx$ chance on the large
candidate set (recall@8 $\approx0.31$–$0.45$ vs. a chance floor of $0.32$; the needle's median rank is
$\sim7$–$10.5$ of 25). Net of BABILong qa5's $\approx13$ chance floor, the deployable selector
captures only about *half* of the perfect-selection re-forward gain (qa5 32k: deploy $+25$ vs. oracle
$+50$). This is an *information-theoretic* wall on large candidate sets, distinct from the (solved)
read-out wall, and our method does not break it. We do, however, falsify the most obvious
explanation: the bottleneck is **not** the needle being evicted (filling the buffer with all 64 chunks
*lowers* qa5 32k from 32 to 15) — it is selection precision on a larger candidate set. A stronger
selection mechanism than reader-native salience is required, and finding it is open.

**Compute cost of token-reforward.** Breaking the read-out wall is bought with recomputation. Because
the selected chunks are re-forwarded through all 32 layers jointly with the query (with
`use_cache=False`, every decode step re-forwards the whole window), latency scales with the window
size: $K{=}2$ costs $\sim$6$\times$ a pure-hidden W0 read-out at 8k ($\sim$4$\times$ at 16k, where long
streaming dilutes the penalty), $K{=}4$ costs $\sim$18$\times$, and $K{\geq}6$ frequently OOMs on long
documents and is disabled. The two obvious mitigations — a window KV-cache (prefill-once + incremental
decode, $\sim$20$\times$) and shrinking the 32-layer FIFO selection index to a single layer / pooled
($6.25$ GB $\to$ 8–256 MB) — are *purely engineering* and unimplemented here; we report unoptimized
wall-clock and do not claim deployment efficiency. (Storage is *not* a limitation: the re-forward
payload is token-ids, $\approx0.26$ MB at 32k.)

**Evaluation is BABILong-only; downstream transfer is unverified.** All results are on BABILong
qa1/qa2/qa5 with a Llama-3-8B backbone. In preliminary exploration we did **not** observe the read-out
improvement transferring to other long-context formats such as RULER or LongEval. We therefore
position the accuracy gain as a **BABILong-specific read-out gain**, and the two-wall decomposition as
an *architecture-invariant diagnosis*, rather than a general-purpose long-context method. Validating
(or bounding) the transfer of select-then-reforward to RULER/LongEval and to real long-document QA is
necessary future work before any generality is claimed.

**Synthetic-to-natural transfer is real but incomplete.** The selector is trained only on T2 synthetic
needle data (`babilong_mix=0`, no BABILong in training); this *does* transfer to held-out BABILong
(zero-train $28 \to$ trained $46$ on qa5 8k, K2), which we consider genuine synthetic$\to$natural
transfer rather than memorization. But it is *partial*: the trained selector still trails the oracle's
66, and the synthetic curriculum (single random needle, $\geq3$ keys) does not cover the natural
distractor structure of long BABILong, which is part of why the long-context selection wall persists.
We mitigate overfitting (random needle placement, multi-key examples, model selection on held-out
BABILong only), but cannot rule out residual T2$\to$BABILong distribution gap.

**Single-layer selector; multi-layer selection unexplored.** Selection salience is read from a single
layer ($L_{16}$), unfrozen so the $q\!\cdot\!k$ score receives gradient, with the selection layer and
top-$K$ tied between train and eval. We did not explore aggregating salience across multiple layers,
learning the selection layer, or richer (non-attention) selection heads — any of which might raise the
long-context recall that currently caps the method. The choice of $L_{16}$ is empirical, not optimized.

**The slot$+$reforward quadrant is not implemented.** Our architecture-invariance claim rests on three
realized cells of a $2{\times}2$ grid (memory $\in$ {slot-routed, pooled-hidden, FIFO-snapshot}
$\times$ read-out $\in$ {near-window SWA, token-reforward}). The fourth nominal cell — slot-routed
memory with selected-chunk token-reforward — is **not built**: slot channels store hidden states with
in-chunk positions and carry no document-chunk id, so the slot$\to$document-chunk mapping
($\sim$100–130 LOC) needed to fetch original tokens does not yet exist. We flag this as future work,
not as a tested result; the three realized lines already establish the claim.

**Scope of the integrity protocol.** Finally, our numbers are deliberately conservative: we report
only clean `babilong_mix=0` measurements and exclude the historical "wall-break" scores (qa5 8k
$\approx85$) as $\approx85\%$ leakage artifacts from BABILong-SFT train/test overlap. This is the
honest baseline; readers comparing against leaderboard numbers trained with in-distribution BABILong
data should account for this difference.

### 6. Conclusion

We reframed long-range memory away from *capacity* and *forgetting* and toward a **two-wall
decomposition** of the read bottleneck, measurable independently for any bounded streaming memory.
The **read-out wall**: a frozen, query-blind compressed memory cannot be read out well — even an oracle
that perfectly isolates the answer chunk's frozen snapshot tops out at $\approx20$–$24$ on BABILong
qa5, so the bottleneck is the *representation*, not selection. The **selection wall**: with a perfect
read-out, the system must still *locate* the relevant chunk in the streamed candidate set. Oracle-based
probes attribute each lost point to the right wall, making the framework a reusable diagnostic.

Our methodological contribution is the **select-then-reforward** paradigm, in contrast to the
**compress-then-inject** paradigm of latent-memory methods such as MemoryLLM and M+. Rather than
compressing the past into per-layer latent tokens and injecting them as cross-attention prefixes, we
keep the *original token-ids*, train a reader-attention selector to pick the top-$K$ relevant chunks,
and **re-forward those original tokens through the full model jointly with the query**. We also show
empirically that *injection* — not lossy compression — is the binding constraint behind the read-out
wall: a frozen reader handed the correct evidence as injected KV gains only $+1$ to $+2.5$, because it
barely uses representations it did not compute through its own attention.

Three results anchor the framework. (i) **The read-out wall breaks**: re-forwarding the answer chunk's
*original tokens* reaches 66 (qa5 8k) versus $\approx20$–$24$ for the same chunk's frozen snapshot — a
$\sim$3$\times$ jump from the *same stored information*, read differently — and even the cheap
pure-hidden W0 path, once trained, clears the snapshot wall ($12 \to 28 \to 34$). (ii) **Short-context
selection is trainable to a clean SOTA**: supervised training on synthetic needles lifts deployment
from zero-train 28 to 46 (qa5 8k) toward the oracle's 66, and our $K{=}4$ deployment reaches qa5
16k$=$38 / 32k$=$32 — **2.4–3.6$\times$** a clean dense long-context anchor and matching a latent-memory
teacher at 32k, all under a strict leakage-free protocol. (iii) **Long-context selection is an
honest negative result**: the answer is in memory (oracle high) but reader-attention and slot
retrieval both score $\approx$ chance on the 25-candidate buffer, and we falsify the eviction
hypothesis (keeping all 64 chunks *lowers* accuracy). The diagnosis is architecture-invariant: across
a $2{\times}2$ grid of memory architectures and raw-token read-outs, adding a raw-token read-out beats
a pure compressed/frozen read-out by $\sim$2–5$\times$ on the same written memory.

**Future work** follows directly from the limitations. First, **a stronger selection mechanism for
long contexts**: reader-native salience is enough at 8k but $\approx$ chance at 16k/32k; multi-layer
salience aggregation, learned selection heads, or hierarchical/coarse-to-fine selection are the natural
next steps to push deployment toward the high oracle ceiling. Second, **the cheap W0 path**: training
already lifts pure-hidden read-out past the frozen-snapshot wall ($\to 34$) at $\sim$3$\times$ lower
cost than reforward, and closing the W0$\leftrightarrow$reforward gap would yield a memory that is both
cheap and accurate. Third, **the unbuilt slot$+$reforward quadrant**: adding a slot$\to$document-chunk
mapping would let a compressed-memory architecture fetch and re-forward the original tokens of its
selected slots, unifying the storage efficiency of latent memory with the lossless query-conditioned
read-out of token-reforward. We also leave validating transfer to RULER / LongEval and real
long-document QA as the test of whether the read-out gain generalizes beyond BABILong. The two-wall
decomposition itself, we believe, will remain a useful lens regardless of which mechanism eventually
scales the selection wall.
