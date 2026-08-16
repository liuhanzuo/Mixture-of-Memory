#!/usr/bin/env python3
"""Independent re-derivation of the designated-damaged denominators (paperC, 2026-08-16).

Recomputes, from the raw evidence JSONs and WITHOUT reading any rollup block:
  (A) off-MMLU OLMo-2 counts: above-floor and above-CHANCE, for every arm x benchmark;
  (B) MMLU-Pro OLMo-2 counts: above-floor and above-chance, for every arm;
  (C) the same for the three non-OLMo families, to check whether their damaged_rungs
      lists match what the manuscript designates.

Purpose: the rollup blocks hardcode damaged_rungs, which silently defines the
denominators quoted in the prose (14/15, 10/15, 0/15, 0/60). This script derives
the counts from first principles for any arm set, so the effect of the hardcoding
is measurable rather than assumed.
"""
import json, os, sys, argparse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(HERE, "evidence")

OFFMMLU = os.path.join(EV, "second_mc_benchmark", "gate2_letter_content_nulls.json")
OFFMMLU_XF = os.path.join(EV, "second_mc_benchmark_crossfamily", "gate2_crossfamily_nulls.json")
MMLUPRO = os.path.join(EV, "mmlu_scale_power", "mmlu_pro_power_nulls_v2.json")

SMALL5 = ["arc_challenge", "arc_easy", "openbookqa", "commonsense_qa", "piqa"]
NEG_CONTROL = ["winogrande"]


def load(p):
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- off-MMLU OLMo-2
def offmmlu_olmo2_cells():
    """Yield one dict per (arm, benchmark) cell with floor- and chance-side numbers,
    derived from tasks.<bench>.arms.<arm>.interfaces.letter, not from any rollup."""
    d = load(OFFMMLU)
    out = []
    for bench in SMALL5 + NEG_CONTROL:
        t = d["tasks"][bench]
        ln = t["letter_null"]
        floor = ln["best_constant_acc"]
        floor_name = "best_constant_" + ln["best_constant_letter"]
        chance = ln["chance_1_over_nopt"]
        for arm, av in t["arms"].items():
            L = av["interfaces"]["letter"]
            acc = L["acc"]
            vn = L["vs_null"]
            assert floor_name in vn, (bench, arm, list(vn))
            v = vn[floor_name]
            # independent recomputation of the floor delta from acc and floor
            my_delta = (acc - floor) * 100.0
            assert abs(my_delta - v["delta_pp"]) < 1e-6, (bench, arm, my_delta, v["delta_pp"])
            lo, hi = v["ci95_pp"]
            out.append(dict(
                family="olmo2_7b", bench=bench, arm=arm, acc=acc,
                is_neg_control=bool(t["is_negative_control"]),
                n=av["shard_integrity"]["n_scored"],
                floor=floor, floor_name=floor_name,
                floor_delta_pp=my_delta,
                floor_hw_pp=(hi - lo) / 2.0,
                floor_p=v["boot_p"],
                floor_ci_excl_0=(lo > 0 or hi < 0),
                floor_above_point=(my_delta > 0),
                floor_above_sig=(my_delta > 0 and lo > 0),
                verdict_in_file=v["verdict"],
                chance=chance,
                chance_delta_pp=(acc - chance) * 100.0,
                chance_above_point=(acc > chance),
            ))
    return out


# ------------------------------------------------------------- off-MMLU non-OLMo
def offmmlu_nonolmo_cells():
    d = load(OFFMMLU_XF)
    out = []
    tln = d["task_letter_nulls"]
    for fam, fv in d["content_nulls_by_family"].items():
        pass
    # arms live under d['families'][fam]['tasks'?] -- discover shape
    return d


# ---------------------------------------------------------------- MMLU-Pro cells
def mmlupro_cells():
    d = load(MMLUPRO)
    ln = d["letter_null"]
    floor = ln["best_constant_acc"]
    chance_mean = ln["chance_mean_1_over_nopt"]
    chance_naive = ln["chance_naive_1_over_max_nopt"]
    h2h = d["head_to_head_248"]
    out = []
    fams = [("olmo2_7b", d["olmo2"])] + [(k, v) for k, v in d["crossfamily"].items()]
    for fam, fv in fams:
        for arm, av in fv["rungs"].items():
            L = av["interfaces"]["letter"]
            acc = L["acc"]
            rec = L["vs_null"]["best_constant_A"]
            my_delta = (acc - floor) * 100.0
            assert abs(my_delta - rec["delta_pp"]) < 1e-6, (fam, arm, my_delta, rec["delta_pp"])
            row = dict(family=fam, arm=arm, acc=acc, n_layers=fv["n_layers"],
                       regime=fv["regime"], floor=floor,
                       floor_delta_pp=my_delta,
                       chance_mean=chance_mean, chance_naive=chance_naive,
                       chance_mean_delta_pp=(acc - chance_mean) * 100.0,
                       chance_naive_delta_pp=(acc - chance_naive) * 100.0,
                       chance_mean_above_point=(acc > chance_mean),
                       chance_naive_above_point=(acc > chance_naive))
            lo, hi = rec["ci95_pp"]
            row["file_half_width_pp"] = rec.get("ci95_half_width_pp", (hi-lo)/2.0)
            row["file_boot_p"] = rec["boot_p"]
            row["file_verdict"] = rec["verdict"]
            row["floor_ci95_pp"] = rec["ci95_pp"]
            row["floor_above_sig"] = (my_delta > 0 and lo > 0)
            row["floor_above_point"] = (my_delta > 0)
            row["n"] = av["shard_integrity"]["n_scored"]
            out.append(row)
    return out, h2h, d


def fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_out", default=None)
    a = ap.parse_args()

    report = {"what": "independent re-derivation of designated-damaged denominators",
              "date": "2026-08-16"}

    # ============ A. off-MMLU OLMo-2 ============
    cells = offmmlu_olmo2_cells()
    arms = sorted({c["arm"] for c in cells})
    print("=" * 100)
    print("A. OFF-MMLU OLMo-2 (5 small benchmarks; winogrande = negative control, shown separately)")
    print("=" * 100)
    print(f"{'arm':<28}{'bench':<17}{'acc':>9}{'floor':>9}{'dFloor':>10}{'hw':>8}{'p':>9}"
          f"{'aboveF':>8}{'chance':>8}{'dChance':>10}{'aboveC':>8}")
    for arm in arms:
        for c in [x for x in cells if x["arm"] == arm and not x["is_neg_control"]]:
            print(f"{c['arm']:<28}{c['bench']:<17}{c['acc']:>9.4f}{c['floor']:>9.4f}"
                  f"{c['floor_delta_pp']:>+10.3f}{c['floor_hw_pp']:>8.3f}{c['floor_p']:>9.4f}"
                  f"{str(c['floor_above_sig']):>8}{c['chance']:>8.4f}"
                  f"{c['chance_delta_pp']:>+10.3f}{str(c['chance_above_point']):>8}")

    def count(armset, key):
        sel = [c for c in cells if c["arm"] in armset and not c["is_neg_control"]]
        return sum(1 for c in sel if c[key]), len(sel)

    AS_REPORTED = ["7B_keep12_step124000", "7B_keep10_step83500", "7B_keep8_step121000"]
    EXCLUDED = ["7B_shortgpt16_step200000", "7B_keep14_step200000"]
    RESTORED = AS_REPORTED + EXCLUDED

    print()
    print("--- off-MMLU counts (5 benchmarks, winogrande excluded as negative control) ---")
    for label, armset in [("as reported (keep12,keep10,keep8)", AS_REPORTED),
                          ("+ shortgpt16 + keep14", RESTORED),
                          ("+ shortgpt16 only", AS_REPORTED + ["7B_shortgpt16_step200000"]),
                          ("+ keep14 only", AS_REPORTED + ["7B_keep14_step200000"])]:
        af, nf = count(armset, "floor_above_sig")
        afp, _ = count(armset, "floor_above_point")
        ac, nc = count(armset, "chance_above_point")
        print(f"  {label:<38} above-floor(sig) {af}/{nf}   above-floor(point) {afp}/{nf}"
              f"   ABOVE-CHANCE {ac}/{nc}")
    report["offmmlu_olmo2"] = dict(
        cells=cells,
        as_reported_arms=AS_REPORTED, excluded_arms=EXCLUDED,
        as_reported=dict(above_floor_sig=count(AS_REPORTED, "floor_above_sig"),
                         above_floor_point=count(AS_REPORTED, "floor_above_point"),
                         above_chance_point=count(AS_REPORTED, "chance_above_point")),
        restored=dict(above_floor_sig=count(RESTORED, "floor_above_sig"),
                      above_floor_point=count(RESTORED, "floor_above_point"),
                      above_chance_point=count(RESTORED, "chance_above_point")),
    )

    # negative control, for completeness
    print("\n--- winogrande (negative control, NOT in any denominator) ---")
    for c in [x for x in cells if x["is_neg_control"]]:
        print(f"  {c['arm']:<28}{c['acc']:>9.4f} dFloor {c['floor_delta_pp']:>+8.3f} "
              f"p={c['floor_p']:.4f} aboveF={c['floor_above_sig']} "
              f"dChance {c['chance_delta_pp']:>+8.3f} aboveC={c['chance_above_point']}")

    # ============ B. MMLU-Pro ============
    mp, h2h, raw = mmlupro_cells()
    print()
    print("=" * 100)
    print("B. MMLU-Pro, all families, all arms (floor = best-constant A = %.6f)" % mp[0]["floor"])
    print("=" * 100)
    print(f"{'family':<16}{'arm':<12}{'nL':>4}{'acc':>10}{'dFloor':>10}{'hw':>8}{'p':>9}"
          f"{'verdict':<22}{'dChanceMean':>12}{'aboveCmean':>11}{'aboveCnaive':>12}")
    for r in mp:
        print(f"{r['family']:<16}{r['arm']:<12}{r['n_layers']:>4}{r['acc']:>10.6f}"
              f"{r['floor_delta_pp']:>+10.4f}{r.get('file_half_width_pp', float('nan')):>8.3f}"
              f"{r.get('file_boot_p', float('nan')):>9.4f}{str(r.get('file_verdict','?')):<22}"
              f"{r['chance_mean_delta_pp']:>+12.4f}{str(r['chance_mean_above_point']):>11}"
              f"{str(r['chance_naive_above_point']):>12}")
    report["mmlupro_cells"] = mp

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(report, f, indent=1)
        print("\nwrote", a.json_out)


if __name__ == "__main__":
    main()
