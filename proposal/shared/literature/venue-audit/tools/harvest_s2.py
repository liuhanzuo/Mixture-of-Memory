#!/usr/bin/env python
"""Harvest S2 venue metadata with aggressive backoff. Must reach HTTP 200 or mark UNRESOLVED."""
import json, os, time, sys
import urllib.request, urllib.error

PROXY = "http://hy-proxy.woa.com:3128"
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY

IDS = """2411.15558 2606.07978 2606.16897 2607.25663 2510.18871 2605.11416 2605.02105 2602.11137
2606.09932 2602.14486 2410.06981 2503.04429 2312.02730 2109.08406 2502.05795 2310.04680 2506.00288
2407.17467 2403.17887 2403.03853 2402.02834 2304.01373 2312.12141 2601.13580 2506.11389 2509.06518
2502.13794 2508.08011 2402.05913 2511.03270 2509.01213 2505.20155 2307.01163 2006.05987 2004.14975
2410.06225 2410.11654 2312.15166 2401.02415 2410.02330 2403.19135 2407.16286 2210.10041 2403.17919
2406.11753 2505.23811 2510.10071 2601.20009 2404.07066""".split()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_results.json")
FIELDS = "title,venue,publicationVenue,externalIds,year,publicationTypes,journal"

results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))


def fetch(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "venue-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, str(e)


def save():
    json.dump(results, open(OUT, "w"), indent=1, ensure_ascii=False)


# Multiple passes; each pass retries only the not-yet-200 ids, with growing sleep.
for rnd in range(1, 9):
    todo = [i for i in IDS if results.get(i, {}).get("_http") != 200]
    if not todo:
        break
    print(f"=== round {rnd}: {len(todo)} todo", flush=True)
    base_sleep = min(4 + rnd * 3, 25)
    for aid in todo:
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}?fields={FIELDS}"
        code, body = fetch(url)
        rec = results.get(aid, {})
        rec["_http"] = code
        rec["_attempts"] = rec.get("_attempts", 0) + 1
        if code == 200:
            try:
                rec.update(json.loads(body))
            except Exception as e:
                rec["_parse_err"] = str(e)
        elif code == 404:
            rec["_note"] = "S2 404 = not indexed"
        else:
            rec["_body"] = body[:200]
        results[aid] = rec
        save()
        print(f"  {aid} -> {code}", flush=True)
        time.sleep(base_sleep)
    time.sleep(20)

save()
got = sum(1 for i in IDS if results.get(i, {}).get("_http") == 200)
nf = sum(1 for i in IDS if results.get(i, {}).get("_http") == 404)
print(f"DONE s2: 200={got} 404={nf} total={len(IDS)}", flush=True)
