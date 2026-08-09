#!/usr/bin/env python
"""Fill remaining S2 gaps via search/match endpoint (separate quota from paper/arXiv:)."""
import json, os, time, urllib.parse
import urllib.request, urllib.error

PROXY = "http://hy-proxy.woa.com:3128"
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
HERE = os.path.dirname(os.path.abspath(__file__))
ABS = json.load(open(os.path.join(HERE, "arxiv_abs.json")))
OUT = os.path.join(HERE, "s2_match.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
S2 = json.load(open(os.path.join(HERE, "s2_results.json")))

todo = [k for k, v in S2.items() if v.get("_http") != 200 and k not in res]
print("todo", len(todo), flush=True)
for aid in todo:
    t = ABS.get(aid, {}).get("citation_title")
    if not t:
        continue
    url = ("https://api.semanticscholar.org/graph/v1/paper/search/match?query="
           + urllib.parse.quote(t) + "&fields=title,venue,publicationVenue,year,externalIds,publicationTypes")
    req = urllib.request.Request(url, headers={"User-Agent": "venue-audit/3"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            code, body = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        code, body = e.code, e.read().decode("utf-8", "replace")[:200]
    except Exception as e:
        code, body = -1, str(e)
    rec = {"_http": code}
    if code == 200:
        try:
            data = json.loads(body).get("data", [])
            if data:
                rec.update(data[0])
        except Exception as e:
            rec["_parse_err"] = str(e)
    else:
        rec["_body"] = body[:150]
    res[aid] = rec
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    pv = rec.get("publicationVenue")
    print(aid, code, "| venue=", repr(rec.get("venue")), "| pv=", (pv or {}).get("name") if pv else None,
          "| type=", (pv or {}).get("type") if pv else None,
          "| xid_ok=", (rec.get("externalIds") or {}).get("ArXiv") == aid, flush=True)
    time.sleep(11)
print("DONE s2-match", flush=True)
