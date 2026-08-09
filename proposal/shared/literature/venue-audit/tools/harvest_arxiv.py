#!/usr/bin/env python
"""Fetch arXiv abs pages: citation_title, citation_journal_title, COMMENT, journal_ref, DOI."""
import json, os, re, html, time
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "arxiv_abs.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 venue-audit"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception as e:
        return -1, str(e)


def meta(h, name):
    m = re.search(r'name="%s"\s+content="([^"]*)"' % name, h)
    return html.unescape(m.group(1)) if m else None


def cell(h, cls):
    m = re.search(r'<td class="tablecell %s"[^>]*>(.*?)</td>' % cls, h, re.S)
    if not m:
        return None
    v = re.sub(r"<[^>]+>", " ", m.group(1))
    return re.sub(r"\s+", " ", html.unescape(v)).strip()


for aid in IDS:
    if res.get(aid, {}).get("_http") == 200:
        continue
    code, h = get(f"https://arxiv.org/abs/{aid}")
    rec = {"_http": code}
    if code == 200:
        rec["citation_title"] = meta(h, "citation_title")
        rec["citation_journal_title"] = meta(h, "citation_journal_title")
        rec["citation_date"] = meta(h, "citation_date")
        rec["citation_doi"] = meta(h, "citation_doi")
        rec["comments"] = cell(h, "comments")
        rec["jref"] = cell(h, "jref")
        rec["doi"] = cell(h, "doi")
    res[aid] = rec
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(aid, code, "|", (rec.get("citation_title") or "")[:60], "| JREF:", rec.get("jref"),
          "| COMMENT:", (rec.get("comments") or "")[:110], flush=True)
    time.sleep(2.5)

print("DONE arxiv:", sum(1 for i in IDS if res.get(i, {}).get("_http") == 200), "/", len(IDS), flush=True)
