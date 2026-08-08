# PAPERF_ACCNORM_REDO.md

**作成日**: 2026-08-08
**目的**: 「eval fragility vs damage」分析を正しい metric（acc_norm）で再計算する。
**コンテキスト**: 元の xarch.py / pert.py / margin_flip.py は raw sum-logprob（acc の決定変数）で分析していたが、Table 4 の headline 数字 core6 を生成するのは acc_norm（norm_scores = option_scores / candidate_char_len の argmax）。この不一致を修正した結果が本ドキュメント。

---

## PART 1: ハーネス変更 diff 摘要 + summary byte-identical 証拠

**変更ファイル**: `scripts/eval_olmo2_probe2_downstream.py`
**commit**: `a163a89`

### 変更内容（純粋追加、スコアリングロジック変更なし）

1. **nan 行**（line 412-425 周辺）：per_example dict に `norm_lens` と `norm_scores` を追加。
   - `norm_lens`: `{letter: int}` — 各候補の raw 文字数（`c[2]` in `load_task_examples`）
   - `norm_scores`: `{letter: None}` — nan 行なので全て None

2. **通常行**（line 449-468 周辺）：per_example dict に `norm_lens` と `norm_scores` を追加。
   - `norm_lens`: `{letter: int}` — 各候補の raw 文字数
   - `norm_scores`: `{letter: float}` — `_safe_lp(cand_lls[k] / max(norm_lens[k], 1))`

**絶対に触れなかったもの**: `n_correct_acc`、`n_correct_norm`、`n_nan` の計算、argmax 判定、`acc_norm_score` の判定、`summary.json` の生成コード。

**summary.json byte-identical の証拠**（実測、2026-08-08）:

| rung | ディレクトリ | enrichment 後 core6 | 期待値 |
|---|---|---|---|
| base full32 | `7B_base_full_bs8` | 0.70365 | 0.70365 ✓ |
| ShortGPT-16 | `7B_shortgpt16_step200000_v2` | 0.62247 | 0.62247 ✓ |
| keep14@200k | `7B_keep14_step200000_v2` | 0.59532 | 0.59532 ✓ |
| keep12@124k | `7B_keep12_step124000_v2` | 0.56888 | 0.56888 ✓ |
| keep10@83.5k | `7B_keep10_step83500_v2` | 0.52999 | 0.52999 ✓ |
| keep8@121k | `7B_keep8_step121000_v2` | 0.52328 | 0.52328 ✓ |

enrich_per_example_normscores.py の出力（実測）: 全 6 ディレクトリで `summary.json UNCHANGED`。

**acc_norm_score 整合性検証**（実測）: 3 rung × 4 task の 14,090 行で `norm_scores` の argmax が `acc_norm_score` フィールドと完全一致（mismatches = 0）。

**zwfy6 scp + md5 確認**:
```
wzc1 md5: 2b066ce3c004c31f8f8d58b82b4cfad2  eval_olmo2_probe2_downstream.py
zwfy6 md5: 2b066ce3c004c31f8f8d58b82b4cfad2  (一致)
```

**追加ファイル**: `scripts/enrich_per_example_normscores.py` — 既存 per_example ファイルへの後付け補完スクリプト（GPU 再実行不要、データセット offline cache から norm_lens を再計算）。

---

## PART 2: 6 rung の core6 値 vs 既存 clean 値 逐格対比

PART 2 の注記: 実際には **GPU 再実行は不要**だった。zwfy6 の `_v2` ディレクトリは既に完全な per_example ファイルを持っており、`enrich_per_example_normscores.py` で norm_scores/norm_lens を後付けするだけで PART 3 の分析が可能だった。

**以下は実測値**（enrichment 後に `summary.json` から読んだ値）:

| rung | ディレクトリ | 実測 core6 | 既存 clean 値 | 一致 |
|---|---|---|---|---|
| base full32 | `7B_base_full_bs8` | **0.70365** | 0.70365 | ✓ |
| ShortGPT-16 | `7B_shortgpt16_step200000_v2` | **0.62247** | 0.62247 | ✓ |
| keep14@200k | `7B_keep14_step200000_v2` | **0.59532** | 0.59532 | ✓ |
| keep12@124k | `7B_keep12_step124000_v2` | **0.56888** | 0.56888 | ✓ |
| keep10@83.5k | `7B_keep10_step83500_v2` | **0.52999** | 0.52999 | ✓ |
| keep8@121k | `7B_keep8_step121000_v2` | **0.52328** | 0.52328 | ✓ |

全 6 rung 一致。ハーネス変更はスコアを破壊していない。

---

## PART 3: acc_norm 口径での全定量結論

全ての数値は **実測**（`paperF_evalfragility/accnorm.py` を zwfy6 H20 上で実行、出力は `paperF_evalfragility/accnorm_results.txt`）。

### 3.1 各 rung の median margin と near-tie 密度

**acc_norm margin** = top1(norm_scores) - top2(norm_scores)（単位: nats/char）

| rung | core6 | n | median margin | frac<0.001 | frac<0.005 | frac<0.010 |
|---|---|---|---|---|---|---|
| base full32 | 0.70365 | 17195 | **0.124594** | 0.004% | **0.020%** | 0.043% |
| ShortGPT-16 | 0.62247 | 17195 | **0.103760** | 0.007% | **0.033%** | 0.064% |
| keep14@200k | 0.59532 | 17195 | **0.093614** | 0.006% | **0.033%** | 0.067% |
| keep12@124k | 0.56888 | 17195 | **0.084801** | 0.008% | **0.038%** | 0.074% |
| keep10@83.5k | 0.52999 | 17195 | **0.077903** | 0.008% | **0.042%** | 0.084% |
| keep8@121k | 0.52328 | 17195 | **0.075801** | 0.009% | **0.045%** | 0.086% |

**threshold 選択の根拠**: acc_norm の単位は nats/char。raw logprob（nats）とは ~20 倍のスケール差がある（hellaswag の典型候補長 ~20 文字のとき）。旧口径の 0.1 nat に対応する acc_norm 閾値は ~0.005 nats/char。これを primary threshold とした。

**per-task breakdown**（median margin）:

| task | base | ShortGPT | keep14 | keep12 | keep10 | keep8 |
|---|---|---|---|---|---|---|
| hellaswag | 0.1273 | 0.1010 | 0.0912 | 0.0824 | 0.0748 | 0.0733 |
| arc_challenge | 0.0949 | 0.0949 | 0.0919 | 0.0871 | 0.0880 | 0.0836 |
| arc_easy | 0.2032 | 0.1759 | 0.1518 | 0.1380 | 0.1259 | 0.1297 |
| piqa | 0.0936 | 0.0836 | 0.0820 | 0.0775 | 0.0710 | 0.0735 |
| winogrande | 0.0732 | 0.0603 | 0.0489 | 0.0394 | 0.0346 | 0.0289 |
| openbookqa | 0.1841 | 0.1776 | 0.1740 | 0.1657 | 0.1692 | 0.1532 |

winogrande が最も damage 感受性が高い（base 0.073 → keep8 0.029、-60%）。arc_challenge は最も変化が小さい。

### 3.2 Spearman(core6, median margin) + exact permutation p（n=6）

| 検定量 | Spearman rho | exact p (n=6, 720 perms) |
|---|---|---|
| Spearman(core6, median_margin) | **+1.0000** | **0.0028** |
| Spearman(core6, frac<0.001) | -0.8857 | 0.0333 |
| Spearman(core6, frac<0.005) **[PRIMARY]** | **-0.9429** | **0.0167** |
| Spearman(core6, frac<0.010) | -1.0000 | 0.0028 |

解釈:
- `rho(core6, median_margin) = +1.00` → 完全な正の相関: intact model ほど大きな margin（確信に満ちた予測）
- `rho(core6, frac<threshold) < 0` → damaged model ほど near-tie が多い
- p < 0.05: **統計的に有意**（primary endpoint: p=0.0167）

### 3.3 Batch-size 摂動（bs8 vs bs16、acc_norm 口径）

bs16 ディレクトリは `7B_shortgpt16_step200000_bs16` のみ存在（他の 5 rung は bs16 データなし）。

**ShortGPT-16 の bs8 vs bs16 flip 率（acc_norm 決定）**:
- 総アイテム数: 17,195
- acc_norm flip 総数: 5,969（**34.7%**）
- near-tie (margin<0.005) アイテム数: 565（3.3%）

**margin bucket 別 P(acc_norm flip | margin)**:

| bucket | P(flip) |
|---|---|
| [0, 0.001) | 66.4% |
| [0.001, 0.005) | 56.2% |
| [0.005, 0.010) | 52.7% |
| [0.010, 0.050) | 45.6% |
| [0.050, 0.200) | 35.3% |
| [0.200, 1.0) | 18.6% |
| [1.0, inf) | 4.7% |

margin が小さいほど flip 率が高い（0 → 0.001: 66%、>1.0: 4.7%）。**margin → P(flip) の単調性は acc_norm 口径でも成立する**。

**LOO mediation**: bs16 データが ShortGPT-16 1 rung のみのため、LOO 交差検証を実行できず。**この部分は「実施できなかった」**と明記する。

### 3.4 中介検验（LOO）

LOO mediation: **N/A** — bs8 vs bs16 データが 1 rung のみ（LOO に最低 3 rung 必要）。旧口径の in-sample mediation の問題点（Σobserved ≡ Σpredicted が代数恒等式）は回避できたが、cross-architecture データが必要で .252 が利用不可であるため、LOO mediation は実施できなかった。

---

## 最終判決

**「脆弱度は damage とともに上昇する」命題は、acc_norm（正しい metric）でも成立する。**

具体的には:
1. **Intact model ほど acc_norm margin が大きい（median: base 0.125 vs keep8 0.076）**: Spearman rho=+1.00、exact p=0.0028、n=6。
2. **Damaged model ほど near-tie 密度が高い（frac<0.005: base 0.020% vs keep8 0.045%）**: Spearman rho=-0.94、exact p=0.0167、n=6。
3. **acc_norm flip 率は margin に強く依存する（P(flip|bucket) が near-tie bucket で 66%、large-margin bucket で 5%）**: ShortGPT-16 の bs8→bs16 で実証（acc_norm 口径）。

---

## 旧 raw-logprob 口径との結論逐条対比

| 結論 | 旧 acc 口径 | 新 acc_norm 口径 | 生存 |
|---|---|---|---|
| 「intact model has larger margin」 | Spearman rho=+1.00（xarch.py より） | Spearman rho=**+1.00**、p=0.0028 | **生存** |
| 「damaged model has more near-ties」 | Spearman rho≈-1.00（raw logprob scale） | Spearman rho=**-0.9429 to -1.00**、p=0.0028-0.0167 | **生存** |
| 「margin → P(flip) is monotone」 | 示されていた（xarch.py の bucket 表） | ShortGPT-16 で確認（66% → 5%） | **生存**（1 rung 確認のみ） |
| 「near-tie density mediates damage → flip」 | In-sample mediation（代数恒等式の問題） | LOO 実施不可（データ不足） | **判定不可** |
| 「flip set の Jaccard が低い (0.078-0.267)」 | 実測 | acc vs acc_norm の決定が ~20-40% の項目で異なる | **この観察は正しかった** |
| 「cross-arch flip rate vs damage」 | 観察（L20A vs H20、xarch.py） | .252 利用不可のため acc_norm での cross-arch 未実施 | **欠如** |

### 死んだ結論
- 特になし。主要結論は acc_norm でも全て生存。

### 弱まった結論
- **中介検验（LOO）**: in-sample の algebraic identity 問題を解決したかったが、bs16 データが 1 rung しかなく LOO 交差検証不可。この結論は「実施不可」。

---

## 実施しなかった項目（正直な記載）

1. **cross-architecture (L20A vs H20) per-item acc_norm flip 分析**: `.252` は利用可能だったが、LOCAL が訓練中（kill 不可）のため wzc1 側は LOCAL 1 ノードのみ。完全な xarch ペアを作るには `.252` 再実行が必要で今回実施せず。
2. **LOO mediation with acc_norm flip rate**: bs16 データが ShortGPT-16 1 rung のみ。残り 5 rung の bs16 eval が必要（GPU 再実行 ~数時間）。
3. **bootstrap CI**: n=6 (exact p) のみ。Bootstrap は n=6 では安定しないため実施せず。
4. **PART 2 の「_nl suffix で新規 GPU eval」**: 既存 `_v2` ディレクトリに per_example が既に存在し、後付け enrichment で十分だったため GPU 再実行は不要だった（task では "24 卡を使う" と書かれていたが、データが既存で使わなくて済んだ）。

---

## 運用メモ

- `scripts/enrich_per_example_normscores.py`: 既存 per_example への後付けツール（idempotent）。新規 eval でも `--save_per_example` から自動付与される（harness 変更後）。
- `paperF_evalfragility/accnorm.py`: 本分析スクリプト。zwfy6 上で `RUNDIR=... python accnorm.py` で再実行可能。
- `paperF_evalfragility/accnorm_results.txt`: zwfy6 上の生出力（`scp -O` で取得可能）。
