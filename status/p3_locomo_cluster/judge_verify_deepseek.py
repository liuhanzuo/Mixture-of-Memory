#!/usr/bin/env python
"""P3.3(b) — LoCoMo judge verification with a SECOND, independent LLM judge.

Reviewer critique: the +4.81 CoMem>KV-Direct LoCoMo advantage rests entirely on
a single GPT-4o judge. This re-judges a stratified sample of the SAME questions
(for BOTH the CoMem and KV-Direct runs) with an independent model family
(deepseek-v3 on the same maas endpoint), then reports:
  (1) judge agreement % and Cohen's kappa between gpt-4o (cached) and deepseek-v3;
  (2) whether the CoMem>KVD gap REPLICATES under the second judge.

Uses the verbatim _JUDGE_TEMPLATE + verdict parsing from
scripts/eval_qcmem_locomo.py so the second judge sees exactly the gpt-4o prompt.
maas is reachable only via the hy-proxy in .env (proxies loaded below).
"""
import glob
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
COMEM = os.path.join(ROOT, "locomo_results", "qcmem_8b_iter_chatFALSE")
KVD = os.path.join(ROOT, "locomo_results", "kvdirect_8b_chatFALSE")
SECOND_JUDGE = "deepseek-v3"
PER_CAT = 50          # stratified target per cat1-4  -> ~200 ids, ~400 judge calls
SEED = 777
WORKERS = 16

_JUDGE_TEMPLATE = (
    "You are grading a model's answer against the gold answer for a question "
    "about a long, multi-session dialogue (the LoCoMo benchmark).\n\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Model answer: {pred}\n\n"
    "Grade whether the model answer is CORRECT. It is CORRECT if it conveys the "
    "same key information as the gold answer (a semantic match), even if phrased "
    "differently, more verbosely, or with extra correct context. It is WRONG if "
    "it contradicts the gold answer, omits the key information, or is empty / "
    "refuses when an answer exists. For date/time answers, accept any unambiguous "
    "equivalent phrasing.\n\n"
    "Respond with ONLY one word: CORRECT or WRONG."
)


def load_env(p=os.path.join(os.path.dirname(ROOT), ".env")):
    # .env lives at project root; ROOT here is .../status/p3_locomo_cluster
    proot = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/.env"
    path = proot if os.path.exists(proot) else p
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v.strip()
    for up, lo in [("HTTP_PROXY", "http_proxy"), ("HTTPS_PROXY", "https_proxy")]:
        if os.environ.get(up):
            os.environ[lo] = os.environ[up]


def load_run(d):
    rec = {}
    for f in sorted(glob.glob(os.path.join(d, "preds_shard*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rec[o["id"]] = {"cat": int(o.get("category", -1)),
                            "q": o.get("question", ""),
                            "answers": o.get("answers", []),
                            "pred": o.get("pred", ""),
                            "gpt4o": None}
    for line in open(os.path.join(d, "judge_cache.jsonl")):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o["id"] in rec:
            rec[o["id"]]["gpt4o"] = float(o["judge"])
    return rec


def judge_one(question, golds, pred, model, base, key, retries=4):
    gold = " OR ".join(str(g) for g in golds if str(g).strip()) or "(none)"
    prompt = _JUDGE_TEMPLATE.format(question=question, gold=gold, pred=pred or "")
    body = {"model": model, "stream": False, "seed": 1,
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = base.rstrip("/") + "/chat/completions"
    backoff = 2.0
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=60)
            if r.status_code == 200:
                txt = r.json()["choices"][0]["message"]["content"].strip()
                up = txt.upper()
                if up.startswith("CORRECT"):
                    return 1.0
                if up.startswith("WRONG"):
                    return 0.0
                if "CORRECT" in up and "WRONG" not in up:
                    return 1.0
                if "WRONG" in up and "CORRECT" not in up:
                    return 0.0
                return 0.0
            if r.status_code not in (429, 500, 502, 503, 504):
                return None
        except Exception:
            if attempt == retries - 1:
                return None
        time.sleep(backoff)
        backoff *= 2
    return None


def cohen_kappa(pairs):
    """pairs: list of (a,b) in {0,1}. Return (po, pe, kappa)."""
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    a1 = sum(a for a, _ in pairs) / n
    b1 = sum(b for _, b in pairs) / n
    pe = a1 * b1 + (1 - a1) * (1 - b1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 1.0
    return po, pe, kappa


def main():
    load_env()
    base = os.environ["OPENAI_BASE_URL"]
    key = os.environ["OPENAI_API_KEY"]

    comem = load_run(COMEM)
    kvd = load_run(KVD)
    common = [i for i in comem if i in kvd
              and comem[i]["gpt4o"] is not None and kvd[i]["gpt4o"] is not None
              and comem[i]["cat"] in (1, 2, 3, 4)]

    by_cat = defaultdict(list)
    for i in common:
        by_cat[comem[i]["cat"]].append(i)
    rng = random.Random(SEED)
    sample = []
    for c in sorted(by_cat):
        ids = sorted(by_cat[c])
        rng.shuffle(ids)
        sample.extend(ids[:PER_CAT])
    print(f"common cat1-4 judged ids={len(common)}  sampled={len(sample)} "
          f"(per-cat target {PER_CAT}); second judge = {SECOND_JUDGE}")

    # build judge tasks: (id, run) for both runs
    tasks = []
    for i in sample:
        tasks.append((i, "comem", comem[i]))
        tasks.append((i, "kvd", kvd[i]))

    ds = {}  # (id,run) -> deepseek verdict

    def work(t):
        i, run, rec = t
        v = judge_one(rec["q"], rec["answers"], rec["pred"], SECOND_JUDGE, base, key)
        return (i, run), v

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        for f in as_completed(futs):
            k, v = f.result()
            ds[k] = v
            done += 1
            if done % 50 == 0:
                print(f"  judged {done}/{len(tasks)}")

    # agreement / kappa over all (id,run) with a valid deepseek verdict
    pairs, n_fail = [], 0
    for i in sample:
        for run, rec in (("comem", comem[i]), ("kvd", kvd[i])):
            dv = ds.get((i, run))
            if dv is None:
                n_fail += 1
                continue
            pairs.append((int(rec["gpt4o"]), int(dv)))
    po, pe, kappa = cohen_kappa(pairs)

    # headline replication: CoMem vs KVD judge on the sample under each judge
    def rate(run, judge):
        vals = []
        for i in sample:
            rec = comem[i] if run == "comem" else kvd[i]
            v = rec["gpt4o"] if judge == "gpt4o" else ds.get((i, run))
            if v is not None:
                vals.append(v)
        return 100.0 * sum(vals) / len(vals), len(vals)

    g_c, ng_c = rate("comem", "gpt4o"); g_k, ng_k = rate("kvd", "gpt4o")
    d_c, nd_c = rate("comem", "deepseek"); d_k, nd_k = rate("kvd", "deepseek")

    print("\n===== JUDGE AGREEMENT (gpt-4o cached vs deepseek-v3) =====")
    print(f" n valid pairs = {len(pairs)}  (failed calls = {n_fail})")
    print(f" agreement (po) = {100*po:.1f}%   expected-by-chance (pe) = {100*pe:.1f}%")
    print(f" Cohen's kappa  = {kappa:.3f}")
    print("\n===== HEADLINE REPLICATION on the sample =====")
    print(f"            gpt-4o(cached)     deepseek-v3")
    print(f" CoMem      {g_c:6.2f} (n={ng_c})   {d_c:6.2f} (n={nd_c})")
    print(f" KV-Direct  {g_k:6.2f} (n={ng_k})   {d_k:6.2f} (n={nd_k})")
    print(f" diff       {g_c-g_k:+6.2f}          {d_c-d_k:+6.2f}")

    out = {
        "second_judge": SECOND_JUDGE, "per_cat": PER_CAT, "seed": SEED,
        "n_common_cat14": len(common), "n_sampled_ids": len(sample),
        "n_judge_calls": len(tasks), "n_failed_calls": n_fail,
        "agreement_po": round(po, 4), "chance_pe": round(pe, 4),
        "cohen_kappa": round(kappa, 4),
        "sample_gpt4o": {"comem": round(g_c, 2), "kvd": round(g_k, 2),
                         "diff": round(g_c - g_k, 2)},
        "sample_deepseek": {"comem": round(d_c, 2), "kvd": round(d_k, 2),
                            "diff": round(d_c - d_k, 2)},
    }
    with open(os.path.join(ROOT, "judge_verify_deepseek_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n[done] wrote judge_verify_deepseek_result.json")


if __name__ == "__main__":
    main()
