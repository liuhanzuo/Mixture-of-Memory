# Disk survey — LANE: WHO OWNS THE SPACE

Surveyor run 2026-08-16 ~20:40-20:55 CST. **READ-ONLY. Nothing was deleted, moved, or modified.
Zero GPU used.**

Verdict up front: **the earlier note claiming "96% belongs to a dozen-colleague share and our
ckpts are ~1% of it" is REFUTED for wzc1 and CONFIRMED-ish for zwfy6.** We are the #2 largest
consumer on wzc1 at **15.6%**, and a small player on zwfy6 at **3.5%**. See section 5.

---

## 0. Method

CephFS exposes recursive byte counts (`rbytes`) as the `st_size` of a directory. This gives exact
totals instantly with no `du` tree walk. Validated three ways:

**(a) small dirs, both disks, byte-exact:**
```
wzc1:  stat -c '%s %n' public_resources tools  ->  142394985 / 24056882
       du -sb            public_resources tools ->  142394985 / 24056882   (DU_RC=0)
zwfy6: stat  public_resources claude_memcalc pzongghzliu_code -> 142394985 / 535472 / 146486
       du -sb same                                            -> 142394985 / 535472 / 146486  (DU_RC=0)
```

**(b) at 585 GB scale, byte-exact** (this is the important one — proves no drift at ckpt scale):
```
stat  outputs/olmo2_probe2_7B_keep10fresh2  ->  585144334902
du -sb same                                 ->  585144334902   (DU_RC=0, ~7 min walk)
```

**(c) in aggregate against `df`:** sum of all wzc1 top-level rbytes = 120228789583683 B, vs
`df --block-size=1` Used = 120609119928320 B. Shortfall = 380330344637 B (0.32%). That shortfall
is almost exactly the rstat propagation lag I measured inside our own tree
(parent rbytes 18744515652616 vs children-sum 19124849755452, delta **380334102836 B**) — i.e.
the missing 0.32% is *our own actively-written data* whose rstat has not yet propagated to the
top-level inode. Two independent quantities agreeing to within 4 MB out of 380 GB.

**Consequence: rbytes is in the same unit as `df` (no replication multiplier), and the numbers
below are exact measured bytes, not estimates.** The only known bias is that actively-written
directories read ~2% low at the parent inode; I report both bounds where it matters.

`df --block-size=1` at survey time:
```
wzc1  : 131511898603520 total  120609119928320 used  10902778675200 avail  92%
zwfy6 : 756773237555200 total  732744900411392 used  24028337143808 avail  97%
```

---

## 1. wzc1 top-level, `/apdcephfs_wzc1/share_304376610` — exact bytes

| entry | bytes | TiB | % of used | ours? |
|---|---|---|---|---|
| eachwang | 19001892521863 | 17.282 | 15.80% | no |
| **pighzliu_code** | **18744515652616** | **17.048** | **15.59%** | **YES** |
| cyanbi | 16423693396035 | 14.937 | 13.66% | no |
| jinfanhe | 11603404323995 | 10.553 | 9.65% | no |
| hunyuan (37 sub-users) | 11473114669458 | 10.435 | 9.54% | shared, 0.0006 TiB ours |
| macroliu | 9333602533526 | 8.489 | 7.76% | no |
| leoxjhuang | 6613185775833 | 6.015 | 5.50% | no |
| mingjihan | 5613469719609 | 5.105 | 4.67% | no |
| datasets | 4664231104085 | 4.242 | 3.88% | no (vlm encrypted data) |
| ptm_resources | 4367305084725 | 3.972 | 3.63% | no (PTM infra) |
| ckpts | 4306040260488 | 3.916 | 3.58% | no (hunyuan/ptm_v2_ci) |
| ddylanwang | 4065758149561 | 3.698 | 3.38% | no |
| kwinsheng | 2743441467109 | 2.495 | 2.28% | no |
| leopoldzeng | 372263195995 | 0.339 | 0.31% | no |
| data | 286877400658 | 0.261 | 0.24% | no (a3b/megatron) |
| eganhong | 231401405176 | 0.210 | 0.19% | no |
| ptm_v2_ci | 171870481875 | 0.156 | 0.14% | no |
| **out_llama** (share root) | **70460690022** | **0.064** | **0.06%** | **probably YES** — see §4 |
| leoylwang | 61760277782 | 0.056 | 0.05% | no |
| ericccao | 49045808945 | 0.045 | 0.04% | no |
| 4b_mid_train_100b_sft_data | 12239463277 | 0.011 | 0.01% | no |
| jeffreyjgao | 10474065656 | 0.010 | 0.01% | no |
| 4b_mid_train_100b | 8449821160 | 0.008 | 0.01% | no |
| public_resources | 142394985 | 0.000 | — | no |
| AngelPTM | 124494694 | 0.000 | — | no |
| tools | 24056882 | 0.000 | — | no |
| chesterdu | 1302837 | 0.000 | — | no |
| `[eganhong]-*` (3 dirs) | ~64091 | 0.000 | — | no |
| rukizheng | 745 | 0.000 | — | no |
| tmp, tmp_dist_ckpt, gangyiwang | 0 | 0 | — | no |
| **SUM of top-level rbytes** | **120228789583683** | **109.347** | | |
| (df Used, incl. our un-propagated 0.32%) | 120609119928320 | 109.693 | 100% | |

Distinct human identities on this disk: 34 top-level entries + 37 more inside `hunyuan/`
(22 of which appear nowhere at top level) = **56 distinct names**. So "a dozen colleagues" is
an undercount by ~4x; the sharing is much wider than the old note assumed.

## 2. zwfy6 top-level, `/apdcephfs_zwfy6/share_304376610` — exact bytes (via ssh .73)

| entry | bytes | TiB | % of used | ours? |
|---|---|---|---|---|
| hunyuan (81 sub-users) | 354265456145525 | 322.203 | 48.33% | shared, 0.0002 TiB ours |
| cyanbi | 61410007257528 | 55.852 | 8.38% | no |
| leoxjhuang | 57630538600569 | 52.415 | 7.86% | no |
| yiyuanzhou | 47820394241264 | 43.492 | 6.52% | no |
| xiaokunyuan | 33993823043740 | 30.917 | 4.64% | no |
| ddingtang | 30484511822948 | 27.726 | 4.16% | no |
| **pighzliu_code** | **25263945749288** | **22.977** | **3.45%** | **YES** |
| hankyzhao | 23959005235205 | 21.791 | 3.27% | no |
| moryhuang_wza | 19662951504694 | 17.883 | 2.68% | no |
| mrliliu | 15567303496072 | 14.158 | 2.12% | no |
| eganhong | 9082779374231 | 8.261 | 1.24% | no |
| shebin | 7222279593021 | 6.569 | 0.99% | no |
| chenluclli | 6333933605590 | 5.761 | 0.86% | no |
| common | 5732087660844 | 5.213 | 0.78% | no |
| rukizheng | 5491516865772 | 4.995 | 0.75% | no |
| ptm_resources | 4339736663330 | 3.947 | 0.59% | no |
| ferman | 3302662893256 | 3.004 | 0.45% | no |
| kwinsheng | 2744794816414 | 2.496 | 0.37% | no |
| leyiye | 2596403762206 | 2.361 | 0.35% | no |
| weilluo | 2388153407170 | 2.172 | 0.33% | no |
| leopoldzeng | 2246999675385 | 2.044 | 0.31% | no |
| macroliu | 1896929414589 | 1.725 | 0.26% | no |
| mingjihan | 1243993409166 | 1.131 | 0.17% | no |
| jensengeng | 1121970358398 | 1.020 | 0.15% | no |
| caseykwang | 1076854040664 | 0.979 | 0.15% | no |
| models | 756209332222 | 0.688 | 0.10% | no (shared model zoo) |
| yifei | 656055894093 | 0.597 | 0.09% | no |
| ptm_v2_ci | 587056977434 | 0.534 | 0.08% | no |
| ddylanwang | 506432326652 | 0.461 | 0.07% | no |
| a3bv3_stage2_5pp | 427616559992 | 0.389 | 0.06% | no |
| guanbinxu | 366125636076 | 0.333 | 0.05% | no |
| junkangchen | 291896024799 | 0.265 | 0.04% | no |
| boyyang | 279953671801 | 0.255 | 0.04% | no |
| tmp_copy | 266441018395 | 0.242 | 0.04% | no |
| mangopan | 197255543468 | 0.179 | 0.03% | no |
| lsy | 194705992273 | 0.177 | 0.03% | no |
| rickrmlu | 187312416065 | 0.170 | 0.03% | no |
| rayying | 184037446836 | 0.167 | 0.03% | no |
| taiji_offline_inference_model_copy | 116748991225 | 0.106 | 0.02% | no |
| **out_llama** (share root) | **101871860736** | **0.093** | **0.01%** | **probably YES** — §4 |
| qwen3_5_27b | 55575985615 | 0.051 | 0.01% | no |
| jeffreyjgao | 16683820293 | 0.015 | 0.00% | no |
| envs | 9690173642 | 0.009 | — | no |
| qinweiyang | 9168600012 | 0.008 | — | no |
| baatarbu / jenshu / rriesu / clavischen | 5319742856 / 2886643772 / 2839667924 / 2486961296 | ≤0.005 | — | no |
| iverli / yuntaonie / profiler_analyse | 1058613396 / 905028072 / 357948281 | ~0 | — | no |
| public_resources | 142394985 | 0.000 | — | no |
| charliecli / suzhongling / .Trash-0 / moningchen | 41291498 / 28504143 / 11890786 / 1118952 | ~0 | — | no |
| **claude_memcalc** | **535472** | 0.000 | — | **YES (ours, trivial)** |
| **pzongghzliu_code** | **146486** | 0.000 | — | **YES (ours, trivial: .openclaw)** |
| gpu_load.log / squad / nq (files) | 100 / 16 / 13 | 0 | — | ? |
| wangyj, tmp, test, rainyxsong, hunyuan_infer, gangyiwang, apdcephfs_zwfy6 | 0 | 0 | — | no |
| **SUM of top-level rbytes** | **732958101794104** | **666.622** | | |
| (df Used) | 732744900411392 | 666.428 | 100% | |

Note the sum here *exceeds* df Used by 213 GB (0.03%) — expected, since rstat on a 667 TiB tree
with 5 nodes writing is not instantaneous in either direction. Immaterial at this precision.

Distinct identities: 69 top-level + 81 inside `hunyuan/` (48 unique to it) = **117 distinct names.**

**`hunyuan/` on zwfy6 is the single dominant fact of this disk: 322.2 TiB = 48.3% of everything.**
Its top consumers, none of them us:
```
rukizheng     79.580 TiB      caseykwang    26.079 TiB      kwinsheng     16.481 TiB
macroliu      56.302 TiB      chenluclli    24.189 TiB      qinweiyang    11.736 TiB
ddylanwang    48.389 TiB      shuyongtan    19.087 TiB      junkangchen    9.053 TiB
```
Any one of those nine is bigger than our entire zwfy6 footprint.

## 3. Our own footprint, broken down (exact)

**wzc1 — `pighzliu_code` 17.048 TiB (parent) / 17.394 TiB (children-sum, 120 entries):**
```
Mixture-of-Memory                    6467825870745   5.882 TiB   <- the git repo + outputs
out_llama                            5155972021221   4.689 TiB   <- pruning/mask experiment outputs
data                                 4031770641067   3.667 TiB   <- tokenized corpora
outputs                              1454101033546   1.322 TiB
models                               1174911445854   1.069 TiB
out_llama_tokenmatched_slorb          361489378999   0.329 TiB
out_llama_tokenmatched_noslorb        214611372776   0.195 TiB
dllm_draft                            113093677397   0.103 TiB
MemoryLLM-source                       39143846446   0.036 TiB
out_llama_alps_slorb_gate0{,b}      2x 28126391970   0.052 TiB   <- byte-identical pair, suspicious
(+ 110 smaller entries)
```
Inside `Mixture-of-Memory` (5.882 TiB): `outputs` 4.865 TiB, `data` 0.895 TiB, `external`
0.065 TiB, `MemLong` 0.028 TiB, rest <0.02 TiB each.

**zwfy6 — `pighzliu_code` 22.977 TiB (parent == children-sum to 130 KB, so fully propagated):**
```
Mixture-of-Memory                   11129132636923  10.122 TiB
out_llama                            4306509853839   3.917 TiB
dllm_draft                           2813089400444   2.558 TiB
data                                 2330438508802   2.120 TiB
outputs                              1696529081378   1.543 TiB
models                               1192782207705   1.085 TiB
MemoryLLM                             921418692225   0.838 TiB
MemLong-Reproduce                     460924287133   0.419 TiB
dllm_draft_104                        196575243923   0.179 TiB
baselines                              94480199452   0.086 TiB
```
Inside `Mixture-of-Memory` (10.122 TiB): `outputs` 7.742 TiB, `distill_cache` 0.861 TiB,
`data` 0.516 TiB, `MemLong` 0.273 TiB, `models` 0.252 TiB, `watchdog_ckpts` 0.226 TiB.

**The five LIVE runs (must not be touched) total 2.206 TiB — 5.5% of our 40.0 TiB combined:**
```
wzc1  outputs/olmo2_probe2_7B_keep10fresh2         585144334902   0.532 TiB
zwfy6 outputs/olmo2_probe2_7B_keep12fresh2         526404587471   0.479 TiB
zwfy6 outputs/olmo2_probe2_7B_keep8fresh2          455362613877   0.414 TiB
zwfy6 outputs/paperC_qwen3base_heal_k8f2           418986636001   0.381 TiB
      (.212 keep14fresh2_distill lives on wzc1 tree, not separately isolated here)
```
So the live runs are NOT what is filling either disk. Even a maximally aggressive cleanup that
spares them loses almost nothing.

## 4. Attribution caveat: the stray `out_llama` at each share root

`/apdcephfs_wzc1/share_304376610/out_llama` (0.064 TiB) and
`/apdcephfs_zwfy6/share_304376610/out_llama` (0.093 TiB) sit at the *share root*, not under
`pighzliu_code`. Evidence they are ours:
- wzc1 copy holds exactly one dir,
  `models_facebook--opt-2.7b_mask-unstructured_s0.5_m-hessian_ratio_20260123_205128`, containing
  `legacy_ckpt.pt` (35230344651 B) + `model.pt` (35230342537 B) + `args.json`/`config.json`/`eval.json`.
- Our own `pighzliu_code/out_llama/` uses the *same* naming convention
  (`models_facebook--opt-2.7b_mask-unstructured_s0.5_m-hessian_obd_20260125_160930`, etc.), and
  our scripts write `out_llama*` output dirs (`scripts/_run_sparseforge_tokenmatched.sh:64`,
  `scripts/_run_alps_slorb_gate0.sh:72`).
- The exact dirname at the share root does **not** exist under `pighzliu_code/out_llama/`
  (verified: `ls` -> No such file or directory), so it is not a duplicate — it looks like a run
  launched with a relative `out_dir` from the wrong CWD.
- `m-hessian_ratio_` vs our usual `m-hessian_obd_` differs, so I cannot prove it is ours from
  naming alone. Marked **probably ours, unconfirmed**. It is 0.157 TiB across both disks —
  0.4% of our footprint, so the ambiguity does not move any conclusion.

Also ours but negligible: `hunyuan/pighzliu` (0.0006 TiB wzc1 / 0.0002 TiB zwfy6, just
`basic_train_pighzliu_2026071*` job dirs), `claude_memcalc` (535 KB), `pzongghzliu_code` (146 KB).

## 5. THE ANSWER: our share, and what deleting everything would buy

### wzc1 — WE ARE A MAJOR OWNER. Cleanup here is worth doing.

Our total = `pighzliu_code` + `hunyuan/pighzliu` + stray `out_llama`:
| basis | our bytes | TiB | % of df Used |
|---|---|---|---|
| conservative (parent rbytes) | 18815579281275 | **17.113** | **15.60%** |
| best (children-sum, catches un-propagated writes) | 19195913384111 | **17.459** | **15.92%** |

**If we deleted 100% of what we own on wzc1:**
- avail 9.916 TiB -> **27.03-27.38 TiB** (a **2.73-2.76x increase in free space**)
- Use% **91.7% -> 77.1-77.4%**

That is materially decisive. Deleting even a *quarter* of our wzc1 footprint (4.3 TiB) would
take avail from 9.92 to 14.2 TiB (+43%) and Use% from 91.7% to 88.4%.

### zwfy6 — WE ARE A BYSTANDER. Cleanup here barely registers.

Our total = `pighzliu_code` + `hunyuan/pighzliu` + `pzongghzliu_code` + stray `out_llama`
= 25366076314688 B = **23.070 TiB = 3.462% of df Used**.

**If we deleted 100% of what we own on zwfy6:**
- avail 21.854 TiB -> **44.92 TiB** (2.06x — sounds good, but only because avail is already a
  thin 3.2% sliver of a 689 T disk)
- Use% **96.8% -> 93.47%** — a **3.3 percentage point** move. Still red.

To be blunt: on zwfy6, `hunyuan/rukizheng` alone (79.58 TiB) is **3.4x our entire footprint**,
and `hunyuan/` as a whole (322.2 TiB) is **14x** it. **We cannot fix zwfy6 by deleting our own
data.** Nuking everything we own — including all five live training runs and every paper's
evidence — would still leave that disk at 93.5% full. zwfy6 pressure is somebody else's problem
and needs a conversation with the share owners, not a cleanup on our side.

### Bottom line for the decision

- **wzc1: cleanup IS worth doing.** We are the #2 consumer at 15.6%, only 0.23 TiB behind #1
  (`eachwang`). Our footprint is the single biggest lever we personally control on this disk, and
  we are *not* an innocent bystander here. The old "our ckpts are ~1%" note is wrong by ~16x
  for this disk.
- **zwfy6: cleanup is NOT worth doing for space reasons.** 3.46% share, 3.3pp of Use% at the
  absolute maximum, at the cost of destroying three live runs. The old note's spirit is right for
  *this* disk (though the specific "96% / dozen colleagues" framing is still wrong: it's 48.3%
  in one shared `hunyuan/` tree spread over 81 sub-users, and 117 distinct identities total).
- **The two disks need opposite decisions.** Any plan that treats "the disk is full" as one
  problem will get zwfy6 wrong.

## 6. What I did NOT do / limits

- No `rm`, `mv`, `truncate`, or any write to any measured path. The only file I created is this
  one. No GPU touched, no `nvidia-smi`, nothing launched or killed.
- I did not descend into other users' trees beyond one level (`hunyuan/` sub-users) — not needed
  for an ownership question, and it is not our data to inventory.
- Ownership is inferred from **directory naming**, not from uid: everything on both shares is
  `root:root`, so `find -user` is useless here. A directory named `eachwang` is attributed to
  eachwang. This is the same convention the share itself uses and the only signal available.
- `df -i` on wzc1 reports `IUsed/IFree` as `-` (dop-fuse does not expose inode accounting), so I
  cannot say whether either disk is under inode pressure as opposed to byte pressure.
- The 0.32% wzc1 rstat lag means our wzc1 number could be understated by up to ~0.35 TiB; I
  report both bounds above and the conclusion is identical either way.
- Two byte-identical dirs `out_llama_alps_slorb_gate0` and `out_llama_alps_slorb_gate0b`
  (28126391970 B each) look like a possible duplicate pair worth a look — but that is the
  what-can-we-delete lane's call, not mine, and it is only 0.026 TiB.
