# `data/knowledge_axes/` — A03 candidate knowledge-axis raw data

Downloaded 2026-08-10 on **wzc1** (LOCAL) via `hy-proxy.woa.com:3128`. CPU + network only;
no GPU was touched. **Not committed to git** (data files); this manifest is the tracked artifact.

Feasibility analysis, null floors, integration cost and the go/no-go recommendation live in
`proposal/active/A03-parametric-vs-external-memory/KNOWLEDGE_AXES_FEASIBILITY.md`.

**Disk**: wzc1 only, 197 MB total. Nothing here existed on either disk before (verified:
`find` over wzc1 `data/` + `.cache/`, and over zwfy6 `data/` + `.cache/` from `.82`).

## Files (all sizes/hashes measured on disk, not copied from dataset cards)

| file | bytes | sha256 (full) |
|---|---:|---|
| `counterfact/test-00000-of-00001-bacb83500fca49a9.parquet` | 1250790 | `c7b37802e6ae381998dadac86e8fe89e241c29c356f67448a1700227dd7ba22c` |
| `counterfact/train-00000-of-00001-05d11247db7abce8.parquet` | 11136400 | `af168d40d0d6ff6caf6263b8c82a0240b39dbe2a7472ebbb5fc635398a4893e0` |
| `counterfact/KnowEdit_wiki_counterfact_test_cf.json` | 2924593 | `efc01ba398dfe04975939f729aab45523addb3c6b3cf41d7429ba5de1d45225b` |
| `counterfact/README.md` | 1119 | `1b8a4158618a70ad414e2d43322c8f0d9b8d3a6bbaec3e57cad06d3ab3aff327` |
| `mquake/MQuAKE-CF-3k.json` | 16055945 | `ce27a39c39f2983512b9b5578fadea5fbe352e5368f49d64f38d37ce304edc80` |
| `mquake/MQuAKE-CF-3k-v2.json` | 15383628 | `f82091cbb668cef8f2537f79f1768af1840184450b347ff8c344336090ddc71e` |
| `mquake/MQuAKE-CF.json` | 44662559 | `fbf1ab9e5243e52da429f7636990096ae0b5f8fbf60f1d4d3a4bf0c9214cd6ea` |
| `mquake/MQuAKE-T.json` | 8889248 | `58500a1d4aaf9e036c23a1d9dd6be2d73d693e35a695614842f211ea5aff59f2` |
| `zsre/zsre_mend_eval.json` | 8091864 | `8a371d512f8a6ab175db4ef672181d5c0e52da23298cb51e5a2eeea2f89aa9cd` |
| `zsre/zsre_mend_train.json` | 69136174 | `4f3ac245e9c0baaf633f9e6cb765ffa8e83d594a74520a6cc3539497328c0722` |
| `zsre/KnowEdit_ZsRE-test-all.json` | 1519289 | `0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c` |
| `hotpotqa/distractor-validation-00000-of-00001.parquet` | 27452575 | `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6` |
| `hotpotqa/README.md` | 9522 | `3cfab003a856275d3198b031c6b2ac46c63178fb462a4123705f652b71b22813` |

## Provenance (exact URLs)

* CounterFact — `https://huggingface.co/datasets/azhx/counterfact/resolve/main/data/{test,train}-*.parquet`
  (HF API 200, no auth). Downloaded sizes match the remote tree byte-for-byte.
* CounterFact (KnowEdit variant) — `https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/benchmark/wiki_counterfact/test_cf.json`
* MQuAKE — `https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/<file>`
  (the HF mirror is gated/401; GitHub raw is open). All four dataset files taken.
* zsRE — `https://rome.baulab.info/data/dsets/zsre_mend_{eval,train}.json` (ROME/MEND canonical
  mirror; **this is the working source** — the two URLs A03's STATUS.json implies are dead do 404).
  Plus the KnowEdit `benchmark/ZsRE/ZsRE-test-all.json` de-duplicated 1301-item variant.
* HotpotQA — `https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet`
  (only the validation split; train is 330 MB and not needed for a closed-book eval).

## Measured contents

| dataset / file | rows | top-level fields |
|---|---:|---|
| `counterfact` test parquet | **2191** | `case_id, pararel_idx, requested_rewrite{prompt,relation_id,subject,target_new{id,str},target_true{id,str}}, paraphrase_prompts[2], neighborhood_prompts[10], attribute_prompts[10], generation_prompts[10]` |
| `counterfact` train parquet | **19728** | same (disjoint `case_id`; test+train = 21919 = canonical CounterFact size) |
| `counterfact` KnowEdit `test_cf.json` | **839** | `subject, prompt, target_new, ground_truth, portability{Reasoning,...}, locality, rephrase` |
| `mquake/MQuAKE-CF-3k.json` | **3000** | `case_id, requested_rewrite[1-4], questions[3], answer, answer_alias, new_answer, new_answer_alias, single_hops[2-4], new_single_hops, orig{triples,triples_labeled,new_triples,...}` |
| `mquake/MQuAKE-CF-3k-v2.json` | 3000 | same schema (v2 fixes) |
| `mquake/MQuAKE-CF.json` | 9218 | same schema (full pool) |
| `mquake/MQuAKE-T.json` | 1868 | same + `answer_extended`; all 1-edit (temporal) |
| `zsre/zsre_mend_eval.json` | **19086** | `subject, src, pred, rephrase, alt, answers[list], loc, loc_ans, cond` |
| `zsre/zsre_mend_train.json` | 163196 | same |
| `zsre/KnowEdit_ZsRE-test-all.json` | 1301 | `subject, target_new, prompt, ground_truth[list], rephrase_prompt, cond, locality, portability` |
| `hotpotqa` distractor validation | **7405** | `id, question, answer, type, level, supporting_facts{title,sent_id}, context{title,sentences}` |

### One sample per dataset (truncated)

**CounterFact** (note: `prompt` is a **cloze template with `{}`**, not a question; `target_*.str` are all single-word):
```
case_id=20952  requested_rewrite={prompt: "{} is located in", relation_id: "P30",
  subject: "Angola", target_true: {id: Q15, str: "Africa"}, target_new: {id: Q51, str: "Antarctica"}}
rendered -> "Angola is located in"   true="Africa"  false="Antarctica"
```

**MQuAKE-CF-3k**:
```
case_id=1  requested_rewrite=[{prompt:"{} is a citizen of", target_true:"United States of America",
  target_new:"Croatia", subject:"Ellie Kemper"}]
questions=["Who is the head of state of the country where Ellie Kemper holds a citizenship?", ...x3]
answer="Donald Trump"  new_answer="Kolinda Grabar-Kitarović"
single_hops=[{question:"What is the country of citizenship of Ellie Kemper?",
  cloze:"Ellie Kemper is a citizen of", answer:"United States of America", answer_alias:[...]}, ...]
```

**zsRE** (`answers` = the TRUE fact, `alt` = the counterfactual, `pred` = a stale model artifact — do not use `pred` as gold):
```
subject="Watts Humphrey"  src="What university did Watts Humphrey attend?"
answers=["Illinois Institute of Technology"]  alt="University of Michigan"  pred="Trinity College"
```

**HotpotQA** (open-book by construction; `context` holds 10 docs / ~1264 OLMo-2 tokens mean):
```
id=5a8b57f25542995d1e6f1371  type=bridge|comparison  level=hard(all)
question="Were Scott Derrickson and Ed Wood of the same nationality?"  answer="yes"
context={title:[10 titles], sentences:[[...]x10]}  supporting_facts={title:[2], sent_id:[2]}
```

## Measured properties that matter (see the feasibility report for the analysis)

* **CounterFact**: 34 relations, 20391 unique subjects, all 21919 targets are **single-word**;
  no alias lists at all (single gold string). 826 distinct target strings overall.
  Rendered cloze prompts are short: mean 7.8 OLMo-2 tokens, **97.2 % are < 13 tokens**.
* **MQuAKE-CF-3k**: 3000 cases × 3 paraphrased multi-hop questions; 9000 single-hop
  sub-facts (3473 unique clozes). Original-answer distribution is extremely skewed —
  `Washington, D.C.` alone is 17.7 % of gold.
* **zsRE mend_eval**: 19085/19086 unique questions, 10720 unique subjects, 53.0 % of true
  answers are multi-word.
* **HotpotQA validation**: 5918 bridge / 1487 comparison, all `level=hard`;
  6.2 % yes-no items; **10.2 % of answers appear verbatim inside the question**
  (43.2 % among comparison items).
