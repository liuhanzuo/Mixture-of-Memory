#!/usr/bin/env python
"""Channel 2+3: DBLP search (venue-authoritative) + OpenReview API2 (venueid).
Also re-parses arXiv abs pages with FIXED comments regex (class="tablecell comments mathjax").
"""
import json, os, re, html, time, urllib.parse
import urllib.request, urllib.error

PROXY = "http://hy-proxy.woa.com:3128"
os.environ["http_proxy"] = PROXY
os.environ["https_proxy"] = PROXY

HERE = os.path.dirname(os.path.abspath(__file__))
ABS = json.load(open(os.path.join(HERE, "arxiv_abs.json")))
HTMLDIR = os.path.join(HERE, "abs_html")
os.makedirs(HTMLDIR, exist_ok=True)

IDS = """2411.15558 2606.07978 2606.16897 2607.25663 2510.18871 2605.11416 2605.02105 2602.11137
2606.09932 2602.14486 2410.06981 2503.04429 2312.02730 2109.08406 2502.05795 2310.04680 2506.00288
2407.17467 2403.17887 2403.03853 2402.02834 2304.01373 2312.12141 2601.13580 2506.11389 2509.06518
2502.13794 2508.08011 2402.05913 2511.03270 2509.01213 2505.20155 2307.01163 2006.05987 2004.14975
2410.06225 2410.11654 2312.15166 2401.02415 2410.02330 2403.19135 2407.16286 2210.10041 2403.17919
2406.11753 2505.23811 2510.10071 2601.20009 2404.07066""".split()


def get(url, tries=3, timeout=45):
    for t in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 venue-audit"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503):
                time.sleep(6 * (t + 1))
                continue
            return e.code, ""
        except Exception:
            time.sleep(4 * (t + 1))
    return -1, ""


# ---------- pass A: re-fetch abs html (cache) + fixed comments/jref parse ----------
def cell(h, cls):
    m = re.search(r'<td class="tablecell %s[^"]*">(.*?)</td>' % cls, h, re.S)
    if not m:
        return None
    v = re.sub(r"<[^>]+>", " ", m.group(1))
    v = re.sub(r"\s+", " ", html.unescape(v)).strip()
    return v or None


def meta(h, name):
    m = re.search(r'name="%s"\s+content="([^"]*)"' % name, h)
    return html.unescape(m.group(1)) if m else None


for aid in IDS:
    p = os.path.join(HTMLDIR, aid + ".html")
    if not os.path.exists(p) or os.path.getsize(p) < 2000:
        code, h = get(f"https://arxiv.org/abs/{aid}")
        if code != 200:
            print("ABSFAIL", aid, code, flush=True)
            continue
        open(p, "w").write(h)
        time.sleep(2.0)
    h = open(p, encoding="utf-8", errors="replace").read()
    rec = ABS.get(aid, {})
    rec["_http"] = 200
    rec["citation_title"] = meta(h, "citation_title")
    rec["citation_journal_title"] = meta(h, "citation_journal_title")
    rec["comments"] = cell(h, "comments")
    rec["jref"] = cell(h, "jref")
    rec["doi"] = cell(h, "doi")
    ABS[aid] = rec
json.dump(ABS, open(os.path.join(HERE, "arxiv_abs.json"), "w"), indent=1, ensure_ascii=False)
print("=== abs reparsed ===", flush=True)
for aid in IDS:
    r = ABS.get(aid, {})
    print(aid, "| CMT:", (r.get("comments") or "-")[:95], "| JREF:", r.get("jref") or "-", flush=True)

# ---------- pass B: DBLP by title ----------
DB = os.path.join(HERE, "dblp.json")
dblp = json.load(open(DB)) if os.path.exists(DB) else {}
print("\n=== DBLP ===", flush=True)
for aid in IDS:
    if aid in dblp and dblp[aid].get("_http") == 200:
        continue
    title = (ABS.get(aid, {}) or {}).get("citation_title")
    if not title:
        dblp[aid] = {"_http": -1, "_note": "no title"}
        continue
    q = urllib.parse.quote(re.sub(r"[^\w\s:-]", " ", title)[:120])
    code, body = get(f"https://dblp.org/search/publ/api?q={q}&format=json&h=8")
    rec = {"_http": code, "hits": []}
    if code == 200:
        try:
            hits = json.loads(body)["result"]["hits"].get("hit", [])
            for hh in hits:
                i = hh.get("info", {})
                rec["hits"].append({
                    "title": i.get("title"), "venue": i.get("venue"),
                    "year": i.get("year"), "type": i.get("type"),
                    "key": i.get("key"), "doi": i.get("doi"),
                })
        except Exception as e:
            rec["_parse_err"] = str(e)
    dblp[aid] = rec
    json.dump(dblp, open(DB, "w"), indent=1, ensure_ascii=False)
    nonc = [x for x in rec["hits"] if x.get("key", "").startswith(("conf/", "journals/")) and "journals/corr" not in x.get("key", "")]
    print(aid, code, "|", (title or "")[:50], "| PEER:", [(x["key"], x["venue"], x["year"]) for x in nonc][:3], flush=True)
    time.sleep(1.8)

# ---------- pass C: OpenReview API2 by title ----------
OR = os.path.join(HERE, "openreview.json")
orv = json.load(open(OR)) if os.path.exists(OR) else {}
print("\n=== OPENREVIEW ===", flush=True)
for aid in IDS:
    if aid in orv and orv[aid].get("_http") == 200:
        continue
    title = (ABS.get(aid, {}) or {}).get("citation_title")
    if not title:
        orv[aid] = {"_http": -1}
        continue
    q = urllib.parse.quote(title[:120])
    code, body = get(f"https://api2.openreview.net/notes/search?query={q}&limit=15")
    rec = {"_http": code, "hits": []}
    if code == 200:
        try:
            for n in json.loads(body).get("notes", []):
                c = n.get("content", {})
                def g(k):
                    v = c.get(k)
                    return v.get("value") if isinstance(v, dict) else v
                t = (g("title") or "")
                if t and t.lower().strip()[:55] == title.lower().strip()[:55]:
                    rec["hits"].append({"title": t, "venue": g("venue"),
                                        "venueid": g("venueid"), "domain": n.get("domain")})
        except Exception as e:
            rec["_parse_err"] = str(e)
    orv[aid] = rec
    json.dump(orv, open(OR, "w"), indent=1, ensure_ascii=False)
    accepted = [x for x in rec["hits"] if x.get("venueid") and "Withdrawn" not in str(x.get("venueid"))
                and "Rejected" not in str(x.get("venueid")) and "dblp.org" not in str(x.get("venueid"))
                and "Desk" not in str(x.get("venueid"))]
    print(aid, code, "|", (title or "")[:45], "| ACC:", [(x["venue"], x["venueid"]) for x in accepted][:3], flush=True)
    time.sleep(1.5)

print("\nDONE dblp+openreview", flush=True)
