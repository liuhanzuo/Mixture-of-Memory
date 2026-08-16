---
name: verify-the-metric-convention-before-declaring-a-mismatch
description: "★★ 我按 n_correct_acc 重算 keep12 ARC-E 得 0.724, 与 agent 的 0.694 不符, 但差异是 acc vs acc_norm 口径不是数据错; 同一 JSON 同时存 n_correct_acc/n_correct_accnorm, 表格声明的是 character-normalized → 宣布 mismatch 前必须先确认口径"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

Before reporting that a recomputation disagrees with a published number, **confirm you computed
the same statistic the table declares.** A convention mismatch and a data error look identical
from one number.

**2026-08-17, measured.** Verifying a sub-agent's Table 4 repair, I recomputed keep12 ARC-Easy
from the per-shard JSONs and got **0.724** against its reported **0.694**. My first instinct was
that one of us had the wrong files. Neither did:

```
7B_keep12_step124000_v2:  acc = 1720/2376 = 0.723906 (.724)
                     acc_norm = 1648/2376 = 0.693603 (.694)   <- what the table prints
```

The same `tasks.arc_easy` object carries **both** `n_correct_acc` and `n_correct_accnorm`, and
`tab_downstream.tex`'s caption says **character-normalized**. The agent was right; I had summed
the wrong field. All three shallow rows then matched exactly: keep8 `.655`, keep10 `.648`,
keep12 `.694`.

**Two compounding mistakes of mine, both in the same probe:**

1. I guessed the key name. My first pass summed `n_correct`, which **does not exist** in this
   schema — so it printed `sum n_correct 0 sum n 0 acc None` for five of eight shards. That is
   the same failure the sub-agent independently reported making ("a probe reading a nonexistent
   per-shard key"), which is a hint that *guessing field names against an unread schema* is the
   recurring hazard, not any one key. **Print `list(d.keys())` and the actual leaf object first.**
2. Having found a real key, I did not check it was the *reported* key. `n_correct_acc` is a
   perfectly good field; it is just not this column's convention.

**Why:** an eval JSON that stores several conventions side by side makes "recomputed from the
raw records" insufficient on its own. Raw-record provenance answers *which items*; it does not
answer *which statistic*. Relates to [[read-what-the-consumer-reads-not-the-bare-key]] — there
the consumer was a tool, here it is a caption.

**How to apply:**
1. Read the leaf object once (`json.dumps(v)[:220]`) before writing any aggregation.
2. Grep the table/caption for the convention (`acc_norm`, `character-normalized`,
   `token-normalized`, `raw accuracy`) and match it explicitly. If the caption does not say,
   that is itself a defect to report.
3. When two computations of "the same" number differ by a plausible-looking amount (here 3.0 pp),
   check convention **before** checking data. A data error usually produces a wild difference or
   an `n` mismatch; a convention error produces a believable one — which is what makes it
   dangerous.
4. Related good practice the sub-agent used and I confirmed: it recomputed from
   `shard{0..7}of8.json` rather than `summary.json` **because `summary.json`'s `n_shards` counter
   is exactly what hid the original 6/8-merge defect** (the stale file reads `n_shards: 8` while
   `arc_easy n_scored=1782`). A field that was wrong once should not be the field you verify with.
5. Also verified in the same pass, worth keeping: `paperB/sections/app_tab_protocol_controls.tex`
   is **never `\input`** while `app_tab_metric_sensitivity.tex:21` `\ref`s its label, so a 64-line
   fix committed to it cannot reach the PDF. **An edit to an un-included file is a no-op** —
   check inclusion before crediting a fix, cf. [[hand-composed-demo-strings-must-be-executed]].
