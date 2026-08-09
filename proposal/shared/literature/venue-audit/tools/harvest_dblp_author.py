#!/usr/bin/env python
"""Retitle-aware author-based DBLP sweep for papers with no peer-reviewed hit yet.
Lesson from 2310.04680: title-search misses papers RETITLED for the camera-ready.
So for each unresolved id we search DBLP by AUTHOR SET and look for non-CoRR entries
in the plausible year window.
"""
import json, os, re, html, time, urllib.parse
import urllib.request, urllib.error

PROXY = "http://hy-proxy.woa.com:3128"
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY
HERE = os.path.dirname(os.path.abspath(__file__))
ABS = json.load(open(os.path.join(HERE, "arxiv_abs.json")))
HTMLDIR = os.path.join(HERE, "abs_html")

# ids with NO peer-reviewed evidence from title-DBLP or accepted-OpenReview
UNRES = """2411.15558 2606.07978 2606.16897 2607.25663 2510.18871 2605.11416 2606.09932
2410.06981 2312.02730 2310.04680 2506.00288 2403.03853 2402.02834 2601.13580 2506.11389
2509.06518 2502.13794 2508.08011 2511.03270 2509.01213 2505.20155 2006.05987 2004.14975
2410.06225 2410.11654 2312.15166 2401.02415 2410.02330 2407.16286 2210.10041 2312.12141
2406.11753 2510.10071 2601.20009 2404.07066 2605.02105 2602.11137 2602.14486""".split()


def get(url, tries=4):
    for t in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 venue-audit"})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(6 * (t + 1)); continue
            return e.code, ""
        except Exception:
            time.sleep(4 * (t + 1))
    return -1, ""


def authors(aid):
    p = os.path.join(HTMLDIR, aid + ".html")
    if not os.path.exists(p):
        return []
    h = open(p, encoding="utf-8", errors="replace").read()
    return [html.unescape(m) for m in re.findall(r'name="citation_author" content="([^"]*)"', h)]


OUT = os.path.join(HERE, "dblp_author.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for aid in UNRES:
    if aid in res and res[aid].get("_http") == 200:
        continue
    au = authors(aid)
    if not au:
        res[aid] = {"_http": -1, "_note": "no authors parsed"}
        continue
    # DBLP wants "First Last"; citation_author is "Last, First"
    def norm(a):
        if "," in a:
            l, f = a.split(",", 1)
            return f.strip() + " " + l.strip()
        return a.strip()
    names = [norm(a) for a in au]
    # query = first author + last author surname (distinctive, keeps query short)
    q = " ".join(names[:1] + ([names[-1].split()[-1]] if len(names) > 1 else []))
    code, body = get("https://dblp.org/search/publ/api?q=" + urllib.parse.quote(q) + "&format=json&h=60")
    rec = {"_http": code, "query": q, "authors": names, "nonCorr": []}
    if code == 200:
        try:
            hits = json.loads(body)["result"]["hits"].get("hit", [])
            for hh in hits:
                i = hh.get("info", {})
                k = i.get("key", "")
                if "journals/corr" in k:
                    continue
                if k.startswith(("conf/", "journals/")):
                    rec["nonCorr"].append({"key": k, "venue": i.get("venue"),
                                           "year": i.get("year"), "title": i.get("title")})
        except Exception as e:
            rec["_parse_err"] = str(e)
    res[aid] = rec
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"### {aid} q='{q}' http={code} nonCorr={len(rec['nonCorr'])}", flush=True)
    for x in rec["nonCorr"][:12]:
        print("    ", x["key"], "|", x["venue"], x["year"], "|", (x["title"] or "")[:78], flush=True)
    time.sleep(1.8)

print("DONE dblp-author", flush=True)
