# Disk survey — lane: FREE WINS (logs, caches, temporaries, duplicated trees)

READ-ONLY SURVEY. No file was modified, moved, or deleted.
Started: 2026-08-16 12:38 UTC

## Baseline (measured with df -h)
- wzc1  `/apdcephfs_wzc1/share_304376610`  = 120T total, 110T used, 10T avail (92%)
- zwfy6 `/apdcephfs_zwfy6/share_304376610` = 689T total, 667T used, 22T avail (97%)  [per task brief]
- NOTE: zwfy6 is NOT mounted on LOCAL (`ls` -> No such file or directory, rc=2).
  All zwfy6 numbers below were obtained over ssh to .73 (28.85.35.73).

## Findings (appended as measured)

### METHOD NOTE: CephFS rbytes == du -sb (verified)
`ls -ld <dir>` on this filesystem reports the RECURSIVE byte count, not 4096.
Cross-checked against `du -sb`:
  outputs:  du -sb = 5348681676045   ls -ld = 5348681676045   -> EXACT match
  logs:     du -sb =     455596409   ls -ld =     455601022   -> 0.001% drift (actively being written)
So every size below is an exact byte count read from rbytes unless labelled otherwise.
This is why the survey could cover both disks without slow du walks.

### CRITICAL SCOPE FACT: wzc1's 110T used is mostly NOT OURS
`ls -l /apdcephfs_wzc1/share_304376610` (rbytes):
  eachwang     19001892521863  (19.0 TB)   <- other user
  pighzliu_code 18744515652616 (18.7 TB)   <- OURS (all of it)
  cyanbi       16423693396035  (16.4 TB)   <- other user
  jinfanhe     11603404323995  (11.6 TB)   <- other user
  hunyuan      11473114669458  (11.5 TB)   <- other user
  leoxjhuang    6613185775833  ( 6.6 TB)   <- other user
  mingjihan     5613469719609  ( 5.6 TB)   <- other user
  datasets      4664231104085, ptm_resources 4367305084725, ckpts 4306040260488,
  ddylanwang    4065758149561, macroliu 9333602533526, kwinsheng 2743441467109, ...
=> Our entire footprint on wzc1 is 18.7 TB of the 110 TB used (~17%).
   The repo itself (Mixture-of-Memory) is 6.468 TB.
   Deleting 100% of our repo would only move wzc1 from 92% to ~86%.
   Anything under another username is NOT OURS and must not be counted as reclaimable.

### SCOPE FACT 2: half our wzc1 bytes live OUTSIDE the repo, as siblings under pighzliu_code/
`ls -l /apdcephfs_wzc1/share_304376610/pighzliu_code` (rbytes, TB = 1e12 B):
  Mixture-of-Memory              6467827203861  (6.47 TB)  <- the repo
  out_llama                      5155972021221  (5.16 TB)  <- SparseForge/llama prune runs
  data                           4031770641067  (4.03 TB)
  outputs                        1454101033546  (1.45 TB)  <- a SECOND outputs/ outside the repo
  models                         1174911445854  (1.17 TB)
  out_llama_tokenmatched_slorb    361489378999  (361 GB)
  out_llama_tokenmatched_noslorb  214611372776  (215 GB)
  dllm_draft                      113093677397  (113 GB)
  MemoryLLM-source                 39143846446  (39 GB)
  out_llama_alps_slorb_gate0       28126391970  (28 GB)
  out_llama_alps_slorb_gate0b      28126391970  (28 GB)  <- SAME byte count as gate0, see dup section
  armt_pg19_real_tokenized_full    18318867926  (18 GB)
  deploy_sparse_24                 11126170390  (11 GB)
  LLaMA-2-7B                       10812330091  (11 GB)
  wandb                            10643015011  (10.6 GB) <- 679 entries, see finding
  venv_union9                        666229399  (666 MB)
  gpt2 502587929, codebuddy 549101568, Llama-3.2-1B 2488916081
Anything I report below is labelled with whether it is inside the repo or a sibling.

### SCOPE FACT 3: zwfy6's 97% full is overwhelmingly NOT OURS
`ls -l /apdcephfs_zwfy6/share_304376610` (rbytes):
  hunyuan       354265456145525  (354 TB!)  <- other user, 53% of the whole 667 TB used
  cyanbi         61410007257528  ( 61 TB)
  leoxjhuang     57630538600569  ( 58 TB)
  yiyuanzhou     47820394241264  ( 48 TB)
  xiaokunyuan    33993823043740  ( 34 TB)
  ddingtang      30484511822948  ( 30 TB)
  pighzliu_code  25263945749288  ( 25.3 TB)  <- OURS, 3.8% of used
  hankyzhao      23959005235205  ( 24 TB)
  moryhuang_wza  19662951504694, mrliliu 15567303496072, eganhong 9082779374231, ...
=> zwfy6 is 97% full because of OTHER USERS (hunyuan alone = 354 TB).
   Even deleting 100% of our 25.3 TB moves zwfy6 from 97% to ~93%.
   The 22 T avail cannot be meaningfully fixed from our side.

zwfy6 pighzliu_code siblings (rbytes):
  Mixture-of-Memory 11129132637542 (11.1 TB) <- the zwfy6 repo checkout
  out_llama          4306509853839 ( 4.3 TB)
  dllm_draft         2813089400444 ( 2.8 TB)
  data               2330438508802 ( 2.3 TB)
  outputs            1696529081378 ( 1.7 TB)
  models             1192782207705 ( 1.2 TB)
  MemoryLLM           921418692225 ( 921 GB)
  MemLong-Reproduce   460924287133 ( 461 GB)
  dllm_draft_104      196575243923 ( 197 GB)
  baselines            94480199452, MemoryLLM-source 39151115411, dllm 23907649878,
  proxsparse_models    13479234817, deploy_sparse_24 11126158857, LLaMA-2-7B 10812075269,
  out                  10708724091, wandb 9046487661

## ===== FINDING 1: zwfy6 distill_cache/512 = 946,292,373,944 B (881.3 GiB) =====
Path: /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/distill_cache/512
Single subdir "512"; contents are uniform 12,782,380-B .npz files, all mtime 2026-06-15 20:37-20:57.
This is a teacher-logit / hidden-state cache keyed by chunk_size=512, written in one ~20-min burst
two months ago. Largest single free-win candidate found on either disk.
Dependency check: see FINDING 1b below.

## ===== FINDING 2: zwfy6 watchdog_ckpts = 247,957,111,393 B (230.9 GiB) =====
Path: .../Mixture-of-Memory/watchdog_ckpts
5 dirs, each ~49,591,42x,xxx B (46.2 GiB):
  H10  H11_v2  H12  H13_isolate  H14_isolate_aggr
H1x = the HMT/RMT-era hypothesis arms. CLAUDE.md lists RMT v3-v10 as 已放弃 (abandoned).

## ===== FINDING 3: zwfy6 cache/hmt_pg19_full = 88,308,114,681 B (82.2 GiB) =====
  grouped/emozilla/pg19_train_grouped.hf  73,367,205,822 B (68.3 GiB)
  tokenized/                              14,165,778,816 B (13.2 GiB)
  + pg19_test_grouped.hf 664,963,051 / pg19_valid_grouped.hf 110,166,992
All mtime 2026-05-11/2026-05-29. "hmt" = Hierarchical Memory Transformer, an abandoned line.
Regenerable: it is a HF-datasets cache of the PUBLIC emozilla/pg19 dataset -> re-tokenizable.

## ===== FINDING 4: TRUE DUPLICATE PAIR on zwfy6 — 126,907,244,672 B (118.2 GiB) recoverable =====
  .../Mixture-of-Memory/data/dolmino_chunks_2048_olmo2.npy   126907244672 B  2026-07-16T21:28
  .../Mixture-of-Memory/data/dolmino_now15b.npy              126907244672 B  2026-07-16T21:33
Verified they are TWO SEPARATE PHYSICAL COPIES, not links:
  inode 15569026167408254610  links=1  blocks=247865713   (olmo2)
  inode 17107680753545547886  links=1  blocks=247865713   (now15b)
Verified byte-identical at both ends (partial hash, per instruction):
  first 200 MiB  md5 = ca6d81d00c6e01158cd0cbebee5596f1   BOTH
  last  100 MiB  md5 = a6835a1e7174e586b1246d5ef178cab9   BOTH
  identical size + identical block count + identical head + identical tail.
  (Full-file md5 of 118 GiB x2 would take ~1 h at CephFS read rates; NOT run. The above is the
   partial-hash standard the task specified. Full hash should be run before any deletion.)
=> Deleting ONE of the two frees 118.2 GiB. BUT see dependency check in FINDING 4b -- this is
   dolmino, which is what the five LIVE runs train on, so the name matters.

### FINDING 4b: DEPENDENCY CHECK on the dolmino pair — the LIVE runs do NOT read either disk copy
Read the live rank-0 cmdline on .73 (PID 3913557, keep12fresh2, 2d03h elapsed):
  --data_path /dev/shm/dolmino_now15b.npy        <-- tmpfs, NOT the CephFS file
  --resume_from .../outputs/olmo2_probe2_7B_keep12fresh2/step166000.pt
/proc/3913557/maps confirms the mmap is on device 00:33 (tmpfs):
  7f8931bd1000-7fa6be000000 r--s 00000000 00:33 2256049370  /dev/shm/dolmino_now15b.npy
/dev/shm on .73 holds its own 126,907,244,672-B copy (494 G tmpfs, 119 G used).
Open-fd count against the on-DISK `dolmino_chunks_2048_olmo2.npy` = 0.
So:
  * `data/dolmino_chunks_2048_olmo2.npy` (118.2 GiB) has NO live reader and is referenced in code
    only via a shard glob, never by this exact filename. Only stale mentions in status/*.md.
  * `data/dolmino_now15b.npy` on zwfy6 IS the re-staging SOURCE for /dev/shm after a reboot
    (tmpfs does not survive reboot). KEEP THIS ONE. Delete the other one of the pair.
=> Recommendation: the deletable member of the pair is `dolmino_chunks_2048_olmo2.npy`, 118.2 GiB,
   after a full md5 confirms the partial-hash result.

## ===== FINDING 5: wzc1 data/dolmino_now15b.npy is a documented PARTIAL PREFIX = 62,020,903,040 B (57.8 GiB) =====
Path: /apdcephfs_wzc1/.../Mixture-of-Memory/data/dolmino_now15b.npy   (57.8 GiB, 2026-07-16)
`scripts/build_dolmino_corpus_wzc1.py` docstring says, verbatim:
  "Do NOT substitute wzc1's data/dolmino_now15b.npy: it is a 7,570,911-row PARTIAL PREFIX of this
   corpus (same leading bytes, less than half the rows), so it looks plausible and silently trains
   on the wrong data."
It is a same-named DIFFERENT file from zwfy6's 15,491,607-row full copy -- a known footgun.
The full corpus is rebuildable on wzc1 from `data/dolmino_olmo2_shards/` (84 shards) in 153 s,
output md5 asserted = 7df19b217e5b0670d58bf6e01e6559d0.
=> This prefix is REDUNDANT with the 84 shards AND actively hazardous. 57.8 GiB.
   Caveat: it is not a byte-subset duplicate to be verified by hash equality; verify by
   confirming shards concat[4096:7578021] == this file before deleting.

## ===== FINDING 6: self-labelled BROKEN / CORRUPT files (zero-risk, the purest free win) =====
Scoped `find <repo> -maxdepth 4 -type f -size +100M -name '*BROKEN*' -o -name '*CORRUPT*' ...`
(never `find /`), on both disks.

wzc1  12,830,695,552 B (11.95 GiB)
  ./data/slimpajama_chunks_4096.npy.BROKEN_llama2tok_uint16overflow      2026-04-22
  The filename records the defect: llama-2 token IDs (32000 vocab) overflowed uint16.
  Grep for references: the ONLY hits are the file itself. The good replacement exists
  (data/slimpajama_chunks_4096.npy, 12,830,695,552 B) so this is superseded, not needed.

zwfy6  5,956,287,104 B (5.55 GiB)
  .../outputs/olmo2_probe2_1B_keep7f2_dolmino_arm4_peaklr20k/step220000.pt.CORRUPT_truncated_49pct_watcher_race
  2026-08-10. Filename records: truncated at 49% by a watcher race. A .pt, but a
  KNOWN-BAD one -- torch.load cannot open it, so it is not a usable resume point and not
  the resume ckpt of any of the five live runs (arm4_peaklr20k is A03, already completed).

SUBTOTAL for finding 6: 18,786,982,656 B = 17.50 GiB, both disks, essentially zero risk.

## ===== FINDING 7: venv inventory (5 trees, 19,444,500,942 B = 18.11 GiB on the SHARED disks) =====
`find <pighzliu_code> -maxdepth 3 -type d -name '.venv' -o -name 'venv*' ...` on both disks.

zwfy6:
  .../Mixture-of-Memory/.venv                        8,503,419,843 B (7.92 GiB)  2026-05-29
      site-packages = 8,453,342,295 B, 272 entries. bin/python is a SYMLINK to
      /opt/conda/envs/torch-base/bin/python.
      MEASURED (rc captured on its own line, no pipe):
        $ .venv/bin/python -c "import torch" -> rc=1, ModuleNotFoundError: No module named 'torch'
      i.e. 7.92 GiB of site-packages that cannot even import torch. Matches the known
      "venv python broken / use conda instead" note in CLAUDE.md.
      Neither live PID 3913557 nor 3913545 uses it: /proc/<pid>/exe = /opt/conda/envs/torch-base/bin/python3.14,
      and neither has PYTHONPATH or VIRTUAL_ENV set.
  .../Mixture-of-Memory/external/memoryllm_venv      5,730,611,883 B (5.34 GiB)  2026-07-06
  .../Mixture-of-Memory/external/landmark_venv       4,690,348,563 B (4.37 GiB)  2026-06-19
      ^ BOTH ARE REFERENCED by scripts (landmark_venv/memoryllm_venv appear in
        scripts/launch_landmark_S2.sh, eval_paperb_ladder_200k.sh, _run_olmo2_p05_armA.sh,
        _run_olmo2_p24_sft_pipeline.sh, eval_longeval_landmark.py, run_landmark_S4b_node.sh, ...)
        -> NOT free wins. Deleting them breaks named eval harnesses. Listed for completeness.
  .../Mixture-of-Memory/olmo2_venv                     125,036,058 B (119 MiB)   2026-08-02  (referenced too)
  .../pighzliu_code/venv_union9                        395,084,653 B (377 MiB)   2026-08-14
wzc1:
  .../Mixture-of-Memory/.venv                               36,947 B  <- SKELETON ONLY.
      Contains only bin/ + lib/python3.11/site-packages/ with python symlinked to
      /usr/bin/python3.11. Effectively empty (36 KB). Nothing to reclaim.
  .../pighzliu_code/venv_union9                        666,229,399 B (636 MiB)   2026-08-14
=> Only genuinely-free venv win: zwfy6 `.venv` = 7.92 GiB (broken, unreferenced by live procs).
   Caveat: grep for `\.venv` in scripts before deleting; some may still name it even though
   the interpreter inside is unusable.

## ===== FINDING 8: __pycache__ / .pytest_cache — NEGLIGIBLE, do not bother =====
wzc1 repo, `find . -maxdepth 6 -name __pycache__ -type d`: 61 dirs, total 9,806,558 B = 9.35 MiB.
wzc1 repo .pytest_cache = 11,258 B. zwfy6 .pytest_cache = 13,243 B.
=> ~9 MiB. This is 0.00001% of the problem. Reporting so nobody spends time here.

## ===== FINDING 9: pip / HF caches are on the NODE ROOT overlay, NOT the shared disks =====
Measured on LOCAL:
  df / -> overlay, 28T size, 4.8T used, 24T avail (18%)   <- node-local, NOT the 92%-full wzc1
  /root/.cache            1,025,701,364 B (978 MiB)
  /root/.cache/pip          246,164,130 B (235 MiB)
  /root/.cache/huggingface  238,955,943 B (228 MiB)
  /root/.codebuddy           95,353,814 B ( 91 MiB)
Measured on .73:
  df / -> overlay, 12T size, 8.4T used, 3.3T avail (72%)  <- node-local
  /root/.cache/pip 11 GiB, /root/.cache/huggingface 1.1 GiB   (du -sh, so 2 s.f.)
=> CLEARING pip/HF CACHES DOES NOTHING FOR THE 92%/97% PROBLEM. They live on a different
   filesystem (overlay /) which is only 18% full on LOCAL. .73's / is at 72% with 11 GiB of pip
   cache, which is worth clearing only if that node's ROOT fills up -- a separate issue.
   The repo-internal ones DO count: wzc1 .hf_cache 1,422,213,687 B (1.32 GiB) +
   .hf_home 489,769,690 B (467 MiB) + .cache 242,197,371 B (231 MiB) = ~2.0 GiB;
   zwfy6 .hf_cache 1,443,242,499 + .hf_home 489,769,252 + .cache 242,197,371 = ~2.0 GiB.

## ===== FINDING 10: zwfy6 .git orphaned tmp_pack_* = 2,381,316,144 B (2.22 GiB) — GIT ITSELF CALLS IT GARBAGE
Path: /apdcephfs_zwfy6/.../Mixture-of-Memory/.git/objects/pack/
  tmp_pack_2EGYat  697,565,196 B  2026-04-21T12:20
  tmp_pack_rp8JuS  696,123,404 B  2026-04-21T12:20
  tmp_pack_2SPVht  695,468,044 B  2026-04-21T12:19
  tmp_pack_P13yBS  292,159,500 B  2026-04-21T12:19
  TOTAL 2,381,316,144 B = 2.22 GiB
These are leftovers of a crashed/interrupted `git fetch`/`clone` on 2026-04-21. They have NO
matching .idx (only 1 .idx exists in the dir, for the single real pack), so git cannot use them.
`git count-objects -vH` in that repo prints, unprompted:
    warning: garbage found: .git/objects/pack/tmp_pack_2EGYat
    warning: garbage found: .git/objects/pack/tmp_pack_P13yBS
    warning: garbage found: .git/objects/pack/tmp_pack_rp8JuS
    warning: garbage found: .git/objects/pack/tmp_pack_2SPVht
    in-pack: 178   packs: 1   size-pack: 223.33 KiB
So the REAL pack is 223 KiB and the garbage is 2.22 GiB -- 10,000x. git's own tooling classifies
these as garbage; `git gc` would remove them. Highest-confidence deletion in this whole survey.
NB: zwfy6 .git total = 2,399,567,089 B, so the garbage is 99.2% of that .git.
   wzc1 .git = 383,446,233 B and has NO tmp_pack_* (checked) -- healthy.

## ===== FINDING 11: stray test/temp dirs in repo roots — I found the ones you asked about =====
Searched repo roots on BOTH disks for .tmp* .chain* .ch[0-9]* .gct* .kk* .cp* .ddtest* .gcu*
.bhtest* .tmpclean* :
  wzc1:  NONE of those patterns exist. (Your leaked-test-dir worry: clean.)
  zwfy6: NONE of those patterns exist either.
What DOES exist, small, and is probably yours:
  wzc1   .disk_probe_local.txt          28 B      2026-08-14T12:22   <- a disk probe leftover
  wzc1   .xac                          227 B      2026-04-11T22:10
  wzc1   .watchdog_state.json        5,281 B      2026-05-11T15:37
  zwfy6  .fs_marker_test                26 B      2026-06-05T15:57   <- FS-sharing probe leftover
  zwfy6  .fs_share_test                 23 B      2026-07-11T21:20   <- FS-sharing probe leftover
  zwfy6  .xac                          227 B      2026-04-11T22:10
  zwfy6  .watchdog_state.json.bak.20260510_2253  48 B  2026-05-10T22:59
  zwfy6  .protect                  107,954 B      2026-08-12T23:08   <- looks deliberate, DO NOT TOUCH
  Combined: ~113 KB. Zero space value; listed only because you explicitly asked.
The one real temp dir:
  wzc1   .t27_tmp                965,248,062 B (920 MiB)  2026-08-12T17:40
     .t27_tmp/infb_eval/longbook_qa_eng.jsonl     298,297,185 B  } InfiniteBench raw, re-downloadable
     .t27_tmp/infb_eval/longbook_choice_eng.jsonl 185,904,631 B  }
     .t27_tmp/pg19_train_sketch_n13_d32.npy       472,569,624 B    recomputable sketch
     ⚠️ NOT a free win: scripts/audit_ap1_2_contamination.py and scripts/audit_p0_14_contamination.py
        both reference .t27_tmp, and .t27_tmp/t103_matchedppl/ holds live-looking consumer scripts
        (CONSUMER_LOG.md, paired_stats_crossing.py). Treat as evidence, not trash.

## ===== FINDING 12: logs/ totals — small on both disks, not worth much =====
  wzc1  logs/ = 455,596,409 B (434 MiB)   [du -sb, exact]
  zwfy6 logs/ = 1,008,297,825 B (962 MiB) [du -sb, exact]
  Combined 1.36 GiB.
Largest individual logs (all on zwfy6, all from the abandoned HMT line, May 2026):
  hmt_pg19_full_b2002_resume10000.log      259,116,933 B (247 MiB)  2026-05-13
  hmt_pg19_full_b2002.log                   69,875,357 B ( 67 MiB)  2026-05-12
  hmt_pg19_full_b2002_resume35000.log       38,336,777 B ( 37 MiB)  2026-05-13
  olmo2_7B_keep14_distill.log               11,475,641 B           2026-08-05  <- keep, recent
  generate_train.log                        10,061,842 B           2026-03-23
Largest on wzc1: logs/_netprobe/ dir 32,049,449 B; generate_train.log 10,061,842 B;
  tcodex_dllm_frontier_20260815.log 7,605,750 B (recent, keep).
NO multi-GB log file exists on either disk. The three HMT logs (353 MiB combined) are the only
log-side win worth naming, and they belong to a line CLAUDE.md marks abandoned.

## ===== FINDING 13: CROSS-DISK duplicated tokenised corpora = 64,384,152,973 B (59.96 GiB) per side
Verified with size + first-200-MiB md5 + last-100-MiB md5 on BOTH disks (per the task's
"verify with size AND a partial hash" rule). All four are IDENTICAL across wzc1 and zwfy6:

| file (data/)                                | bytes        | head200 md5                      | tail100 md5                      | match |
|---------------------------------------------|--------------|----------------------------------|----------------------------------|-------|
| slimpajama_chunks_2048_qwen3base_full.npy   | 22,164,381,824 | 70e909a524e087ea0c453d1f4120f0e9 | dfa17d0a13f68a35b81340659e51acf1 | YES   |
| slimpajama_chunks_4096_llama3.npy           | 21,529,870,464 | c04b6ffc1fecb224ab31500698a66ab1 | 14cd56c1e8006acd58abe684b9052fe9 | YES   |
| pg19_train.jsonl                            | 11,450,766,349 | 6564c8791efdf791d44b780a21509402 | 52b8e0eb4b9b9817f3fdeaee4f1d053b | YES   |
| slimpajama_chunks_2048_qwen3.npy            |  9,239,134,336 | af6a28bf276b7136a5c615e086cc9fb7 | 1cd8a8682ed566bd82417c437353d7a5 | YES   |
| TOTAL                                       | 64,384,152,973 = 59.96 GiB per disk                                       |

⚠️ IMPORTANT CAVEAT -- this is duplication BY DESIGN, not by accident.
The two disks are NOT shared filesystems (verified: /apdcephfs_zwfy6 does not exist on LOCAL).
A corpus must be physically present on a disk for that disk's nodes to train on it. So the
"duplicate" on each side is the working copy for {LOCAL,.21} and {.73,.82,.104} respectively.
Deleting either side de-capabilitates those 2 or 3 nodes for that corpus, and re-copying costs
~3.2 h per 118 GiB (measured 17.7 MB/s single-stream, ~92 MB/s with 6 parallel streams, per
scripts/build_dolmino_corpus_wzc1.py).
=> I classify these as "regenerable at cost", NOT as free wins. Reclaim only for a corpus whose
   disk-side nodes are provably done with it. pg19_train.jsonl (11.5 GiB x2) is the best
   candidate: PG-19 is a public dataset and the PG-19 lines (HMT/RMT/ARMT) are all abandoned.

## ===== NEGATIVE RESULT: no rogue >5 GiB non-checkpoint files beyond those already named =====
`find <repo> -maxdepth 3 -type f -size +5G ! -name '*.pt'` on both disks returned ONLY
legitimate corpora/model shards plus the two defect files already reported. Full list:
wzc1:  data/dolmino-mix-1124-llama2/train.bin 310,886,663,436 (289.5 GiB, 2026-08-09) <- SparseForge
       data/dolmino_now15b.npy 62,020,903,040 (partial prefix, FINDING 5)
       slimpajama_chunks_2048_qwen3base_full.npy / _4096_llama3.npy / pg19_train.jsonl /
       _2048_qwen3.npy / _2048_hunyuan.npy 8,329,412,736
       slimpajama_chunks_4096.npy.BROKEN_... 12,830,695,552 (FINDING 6)
zwfy6: dolmino_now15b.npy + dolmino_chunks_2048_olmo2.npy 126,907,244,672 each (FINDING 4)
       slimpajama_* / pg19_train.jsonl (FINDING 13)
       models/Llama--Llama2-7b/pytorch_model-0000{1,2,3}-of-00003.bin 9.88 G + 9.89 G + 7.18 G
         ^ .bin duplicates of a HF model; if a .safetensors set exists alongside, the .bin set
           (26.9 GiB) is redundant -- NOT verified, flagged for a follow-up check.
       outputs/.../step220000.pt.CORRUPT_... 5,956,287,104 (FINDING 6)
No stray multi-GiB core dumps, no runaway single log, nothing unexplained.

## ===== PRIOR-ART CHECK: a cleanup audit from 2026-08-12 already adjudicated some of this =====
`status/DISK_CLEANUP_AUDIT_20260812.md` exists (24,179 B) and MUST be honoured -- overriding a
deliberate prior survivor decision needs a reason, and "it is large" is not one (its own words).
What it already settled:
  * watchdog_ckpts: it ALREADY pruned 962 G -> 231 G, deliberately keeping ONE ckpt per arm so each
    arm stays re-evaluable. My FINDING 2 is therefore NOT a fresh win -- it is the deliberate
    remainder. Its stated release condition: "write the POSTMORTEM first, then delete", because
    "deleting first turns a 0.0 result into an unverifiable one." I re-verified the H-series
    is dead (only ref is the dead daemon scripts/babilong_ckpt_watchdog.py:44) but I am NOT
    reclassifying it as free. -> requires a one-line human confirmation, then +230.9 GiB.
  * distill_cache/512: explicitly KEPT with reasoning ("primary dolmino result cache, 74k npz
    regeneration bill, QCMem distill work is live"). I independently CONFIRMED the file count:
    946,292,373,944 / 12,782,380 = 74,031.00001 -> 74,031 uniform .npz files, matching its "74k".
    -> NOT a free win. It is the single largest item but it is a live-direction cache.
  * It also already deleted distill_cache/pg19_512_nctx{15,63} (480 G) -- gone, do not re-count.
NEW in this survey, i.e. NOT covered by that audit (verified by grepping it):
  * the dolmino 118.2 GiB true duplicate pair (FINDING 4)      -- zero hits for dolmino_chunks_2048_olmo2
  * the 2.22 GiB git tmp_pack garbage (FINDING 10)             -- zero hits for tmp_pack
  * cache/hmt_pg19_full 82.2 GiB (FINDING 3)                   -- zero hits for hmt_pg19
  * the BROKEN/CORRUPT pair 17.50 GiB (FINDING 6)
  * the broken 7.92 GiB zwfy6 .venv (FINDING 7)
  * the wzc1 57.8 GiB partial-prefix dolmino (FINDING 5)

## ============================ BOTTOM LINE ============================
TIER 1 -- TRUE FREE WINS (no live reader, self-labelled defective, or git-declared garbage):
  118.24 GiB  zwfy6  data/dolmino_chunks_2048_olmo2.npy       dup of dolmino_now15b.npy, 0 open fds
   11.95 GiB  wzc1   data/slimpajama_chunks_4096.npy.BROKEN_llama2tok_uint16overflow
    7.92 GiB  zwfy6  .venv                                    cannot import torch (rc=1), 0 live users
    5.55 GiB  zwfy6  outputs/.../step220000.pt.CORRUPT_truncated_49pct_watcher_race
    2.22 GiB  zwfy6  .git/objects/pack/tmp_pack_*             git says "garbage found"
    0.34 GiB  zwfy6  logs/hmt_pg19_full_b2002*.log (3 files)  abandoned HMT line
  ------------
  146.22 GiB  TIER 1 TOTAL   (27.55 GiB of it on wzc1 is NOT true -- see split below)
  Split by disk: wzc1 11.95 GiB | zwfy6 134.27 GiB

TIER 2 -- REGENERABLE AT A STATED COST (defensible, needs one decision each):
   82.24 GiB  zwfy6  cache/hmt_pg19_full     re-tokenise public emozilla/pg19; HMT line abandoned
   57.76 GiB  wzc1   data/dolmino_now15b.npy PARTIAL PREFIX, rebuildable from 84 shards in 153 s,
                                             and actively hazardous (silently trains on wrong data)
   10.66 GiB  wzc1   ../wandb (682 runs, oldest 2025-12)  local mirror of a cloud service
  ------------
  150.66 GiB  TIER 2 TOTAL   (wzc1 68.42 | zwfy6 82.24)

TIER 3 -- NEEDS A HUMAN "yes, that line is closed" (do not act unilaterally):
  230.87 GiB  zwfy6  watchdog_ckpts/ (5 arms x 46.2 GiB)  prior audit: postmortem first
  881.29 GiB  zwfy6  distill_cache/512 (74,031 npz)       prior audit: KEEP, QCMem distill live

TIER 0 -- NOT WORTH DOING (measured so nobody else spends time):
    9.35 MiB  __pycache__ x61 + .pytest_cache          ~nothing
    1.36 GiB  logs/ combined, both disks               no multi-GB log exists
    ~113 KB   stray .fs_*_test/.disk_probe/.xac files  the leaked-probe worry: essentially clean
   ~12  GiB   /root/.cache/pip on .73                  DIFFERENT FILESYSTEM (overlay /), does not
                                                       help the 92%/97% shared-disk problem at all

REALITY CHECK ON IMPACT:
  Tier 1 + Tier 2 = 296.88 GiB total (wzc1 80.37 GiB, zwfy6 216.51 GiB).
  wzc1  has 10 T avail; +80 GiB moves 92% -> 91.93%.
  zwfy6 has 22 T avail; +217 GiB moves 97% -> 96.97%.
  Even Tier 1+2+3 (1.41 TiB) barely moves either gauge, because as established at the top,
  our 18.7 TB / 25.3 TB is a minority of the 110 T / 667 T used. The disks are full because of
  OTHER USERS (hunyuan = 354 TB on zwfy6 alone). No junk-lane cleanup can fix 97%.
  The honest recommendation is: take Tier 1 because it is free and correct hygiene, not because
  it will move the percentage. If real headroom is needed, it has to come from a quota
  conversation about hunyuan/eachwang/cyanbi, or from our own outputs/ (5.35 TB wzc1 + 8.51 TB
  zwfy6 of checkpoints) -- which is a different lane and involves live training.
