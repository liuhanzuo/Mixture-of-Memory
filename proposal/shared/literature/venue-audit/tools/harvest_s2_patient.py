#!/usr/bin/env python
"""Patient S2 retry for still-429 ids: long sleeps, many rounds. Also Crossref as 4th channel."""
import json, os, time, urllib.parse
import urllib.request, urllib.error

PROXY = "http://hy-proxy.woa.com:3128"
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
HERE = os.path.dirname(os.path.abspath(__file__))
S2F = os.path.join(HERE, "s2_results.json")
FIELDS = "title,venue,publicationVenue,externalIds,year,publicationTypes,journal"


def fetch(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": "venue-audit/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:200]
    except Exception as e:
        return -1, str(e)


for rnd in range(1, 26):
    res = json.load(open(S2F))
    todo = [k for k, v in res.items() if v.get("_http") != 200]
    if not todo:
        print("ALL 200", flush=True)
        break
    print(f"== patient round {rnd}: {len(todo)} todo", flush=True)
    for aid in todo:
        code, body = fetch(f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}?fields={FIELDS}")
        res = json.load(open(S2F))
        rec = res.get(aid, {})
        rec["_http"] = code
        rec["_attempts"] = rec.get("_attempts", 0) + 1
        if code == 200:
            try:
                rec.update(json.loads(body))
            except Exception as e:
                rec["_parse_err"] = str(e)
        elif code == 404:
            rec["_note"] = "S2 404 not indexed"
        res[aid] = rec
        json.dump(res, open(S2F, "w"), indent=1, ensure_ascii=False)
        if code == 200:
            pv = rec.get("publicationVenue") or {}
            print(f"  OK {aid} venue={rec.get('venue')} type={pv.get('type')}", flush=True)
        time.sleep(35)
    time.sleep(60)
print("DONE patient-s2", flush=True)
