# Mixture-of-Memory Legacy Manifest

Audited at: `2026-08-20T23:36:13+08:00`

Audited source commit: `79e9e48b3ade3d282042907e1f9558364330200e`

This manifest separates four different claims that were previously easy to
conflate:

1. an object exists in Git;
2. an object is materialized in the sparse working tree;
3. an external dataset/checkpoint was historically recorded on a cluster disk;
4. that external asset is reachable and hash-verified from the current host.

Only the first two are established for all repository assets below. External
assets are never marked available merely because an old document names them.

## Repository integrity

- Branch at audit: `research/continue-20260819`.
- Remote `main` at audit: `6c09dff51d3aca94701a93968c71a3fd5861079b`.
- The audited source commit is one commit ahead of remote `main`.
- `git fsck --full --no-reflogs`: pass.
- `git rev-list --objects --missing=print HEAD`: zero missing objects.
- Git contains 36,113 tracked paths. Large result trees remain recoverable from
  Git even when they are not materialized by sparse checkout.
- Core inheritance directories materialized for the active working tree:
  `paperA/`, `paperB/`, `paperC/`, `paperC_research/`, `paperD_research/`,
  `paper/`, `proposal/`, `legacy/`, `method_sources/`, `paper_results/`,
  `report/`, `collab_handoff/`, `src/`, `scripts/`, `status/`, and supporting
  docs/config/tests.

## Paper assets

| Identity | Canonical tree | State at audit | Canonical PDF |
|---|---|---|---|
| Paper A — *CoMem: Reusing Transformer Depth across Queries with Persistent Intermediate Residuals* | `paperA/`, tree `f710a0f22cf8938bf75dacc24d9ffeeb6eaeab1f`, 1,454 files | Full source, PDFs, venue versions, figures, artifacts, review history and follow-up TODOs are in Git. | `paperA/main.pdf`, SHA-256 `9451a471b0e007616452f99545ba861712d81c01a176dd5761c0445e01ef9c97` |
| Paper B — *Auditing Recovery after Depth Pruning: Perplexity, Evaluation Interfaces, and Construction Matter* | `paperB/`, tree `58748c208ba2c6ab19e5a4730e27fd94485c715e`, 1,093 files | Full source, current/final PDFs, submission packages, anonymous artifact and review snapshots are in Git. The old reproducibility manifest is not trustworthy; see Known inconsistencies. | `paperB/main.pdf`, SHA-256 `2d57964ef31fe3cbd62c22f930df5950ded0fa12b2110d57de8e023231712741` |
| Paper C — *Null Calibration for Multiple-Choice Evaluation: When “Above Chance” Fails the Input-Blind Floor* | `paperC/`, tree `b1690a12034db2b8c2a015b08c43747cbd8e15dc`, 524 files | Full ICLR source, evidence, gates and frozen review rounds `00`–`06` are in Git. Root PDF is byte-identical to round 06 `submission_complete`. | `paperC/main.pdf`, SHA-256 `9dbbaf8ca10395b26d679bf954429e1a051ef3b3f8924e6454b784aa0c3ace75` |
| Unlettered CoMem-extension draft — *Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory* | `paper/`, tree `c9e4b7ab585738fc4b4ccb1ed6299fda002bc74a`, 25 files | LaTeX draft exists; no committed PDF. Treat as a draft asset, not as current Paper D. | none |
| Historical Paper-D research | `paperD_research/`, tree `a551a8f0b66c041b8d0b4ead35c8d412ab4a5bfa`, 126 files | Research reports and 117 JSON/CSV artifacts exist, but there is no LaTeX manuscript or PDF. The cross-family stitching direction was killed and released the Paper-D name. | none |

Paper D must not be inferred from a historical directory name. The naming
authority is `proposal/PAPER_LETTER_ASSIGNMENT.md`: the old stitching direction
is dead and a new Paper D had not been promoted at the audited commit.

## Proposal and research assets

| Tree | Git tree | Files | Logical bytes | Working-tree policy |
|---|---|---:|---:|---|
| `proposal/` | `4541a7124885cb51bb8b4837b2119bab01f83d80` | 787 | 22,588,457 | materialized |
| `legacy/` | `ba39233d5cc51cad6b01b2c7e9fac4cf0b240f6d` | 522 | 5,020,100 | materialized |
| `method_sources/` | `c02606feba4f78ae8e76dff302d4d249c8351334` | 476 | 37,396,102 | materialized |
| `experiments_archive/` | `401c88989a2f79954b55d9202520ace2c4887567` | 1,056 | 2,156,217,084 | Git object only by default |
| `babilong_results/` | `d428ce739d496e119e66ec39436483d8c651365e` | 21,755 | 60,339,660 | Git object only by default |
| `ruler_results/` | `326d5d180627102554fbc4b23ee761b5df61e3a5` | 4,495 | 29,697,916 | Git object only by default |
| `longbench_results/` | `a471e9dcc3de9179c19ec5d087a658db5901f248` | 610 | 18,128,335 | Git object only by default |
| `longeval_results/` | `1942e4813e2a7b0c79e1f560e99b9d954c41a6c4` | 280 | 1,387,121 | Git object only by default |
| `bench_results/` | `fc39e04a76f16aa295cda6c9eca712f9bec1d96e` | 819 | 28,663,918 | Git object only by default |

“Git object only” means the tree is fully present in `.git` and can be restored
with sparse-checkout. It does not mean the asset is missing.

## Pinned external dependencies

The parent repository now records and successfully initializes these exact
gitlinks:

| Path | Upstream | Commit |
|---|---|---|
| `locomo/` | `https://github.com/snap-research/locomo.git` | `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` |
| `third_party/HMT-pytorch/` | `https://github.com/OswaldHe/HMT-pytorch.git` | `177da4b7787557dd8475f9be62f09b5d19516cb7` |
| `third_party/recurrent-memory-transformer/` | `https://github.com/booydar/recurrent-memory-transformer.git` | `9d0ebe1778687995697fe68e886bc1dcf0e45e1c` |

`third_party/babilong-pkg/` is different: it is ignored and is not a gitlink.
The official upstream is `https://github.com/booydar/babilong`; the historically
recorded local checkout was `f09a184b43316a751d5059e13de7c557b6daca86`.
`scripts/setup_third_party.sh` now restores that exact detached revision. The
scorer remains an external prerequisite, but its recovery is pinned and
mechanical rather than an undocumented manual clone.

## External data and checkpoints

The following two datasets are pinned by the historical Paper-B manifest but
are not present in this clone:

| Path | Recorded SHA-256 | Current-host state |
|---|---|---|
| `data/dolmino_now15b.npy` | `4c1a2c899568714e859ba8429cc2ec393ff9c91599429f6e14a77aecd689a41b` | unavailable |
| `data/dolmino_now_val.npy` | `ee36248dc8e4a79c1ffda16272f9e44a0675eb0dea5c9920f2f0735e570f22e8` | unavailable |

Model weights, `.pt` checkpoints, Hugging Face caches, local virtual
environments, and raw cluster outputs are intentionally ignored by Git. The
historical disk inventory is `status/DISK_DECISION_20260816.md`; the checkpoint
retention/deletion ledgers are `status/CKPT_CLEANUP_VETOED_370.txt` and
`status/CKPT_CLEANUP_EXECUTED_579.txt`.

At audit time, the old `/apdcephfs_wzc1/...` and `/apdcephfs_zwfy6/...` roots
named by those records were not mounted on the current host. Therefore this
manifest does not claim that the external assets were deleted; it claims only
that they could not be reached or re-hashed here.

## Known inconsistencies and inheritance rules

1. `paperB/REPRO_SHA256.txt` is historical, not a current integrity baseline.
   `paperB/REPRO_SHA256_AUDIT_20260816.md` measured 3/9 matches and 6/9
   mismatches on the old disk. On this clone, the two referenced files that are
   present (`main.pdf` and `anonymous_artifact/SHA256SUMS.txt`) both mismatch
   the historical manifest. Do not silently replace the old record.
2. `paperC/state/paper_state.json` still says round 3/review in flight, while
   `paperC/review_rounds/round_06/submission_complete/` and the byte-identical
   root PDF exist. New work must treat round 06 plus the root tree as the
   physical latest asset and repair the stale state explicitly.
3. `proposal/check_stale_absence_claims.py` reports one stale assertion:
   B09 says `RELATED_WORK.md` is absent although the file exists. Preserve the
   historical record and add a dated superseding field rather than rewriting
   append-only history.
4. A directory named `paperD_research/` is not evidence of a live Paper D.
   Read `proposal/PAPER_LETTER_ASSIGNMENT.md` before assigning a Paper-D worker.
5. A new worker may inherit a claim only after resolving its canonical paper
   tree, tree SHA, evidence paths, external dependencies and open integrity
   issues from this manifest and `LEGACY_MANIFEST.json`.
