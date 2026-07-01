#!/usr/bin/env python
"""Readout-wall LOWER BOUND: feed the base model ONLY the supporting-fact
sentence(s) + question (no 16k haystack, no memory chain) and score.

This isolates the base Llama-3-8B's intrinsic ability to answer qa5 when the
evidence is trivially present and short. If it still fails a lot here, the
"readout wall" is a base-model reasoning/instruction-following limit, NOT a
long-context / memory-architecture limit -> direction c must train the backbone,
not the selector or the reforward window.

Reuses _locate_qa5_supporting_fact to reconstruct the gold fact sentence, then
builds a tiny context = just that sentence (plus a few bAbI filler sentences
around it for realism, controlled by --filler).

Official scoring (compare_answers). qa5 only.
"""
import sys, os, re, csv, argparse
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "third_party/babilong-pkg")
sys.path.insert(0, "scripts")
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from babilong.metrics import TASK_LABELS, compare_answers

# reuse the SF locator + question parser from the probe
import importlib.util
_spec = importlib.util.spec_from_file_location("pf", "scripts/probe_fullchain_oracle_qa5.py")
pf = importlib.util.module_from_spec(_spec); sys.modules["pf"] = pf
try:
    _spec.loader.exec_module(pf)
except SystemExit:
    pass


def extract_sf_sentence(input_text, question, answer):
    """Return the gold supporting-fact sentence string, or None."""
    q_elem = pf._parse_qa5_question(question)
    if not q_elem:
        return None
    pats = []
    V = pf._TRANSFER_VERB_PAT
    a = re.escape(answer.strip())
    t = q_elem.get('type')
    if t == 'who_receiver':
        pats.append(rf"{re.escape(q_elem['giver'])} {V} (?:the |a )?{re.escape(q_elem['obj'])} to {a}")
    elif t == 'what_obj':
        pats.append(rf"{re.escape(q_elem['giver'])} {V} (?:the |a )?{a} to {re.escape(q_elem['receiver'])}")
    elif t == 'who_giver':
        pats.append(rf"{a} {V} (?:the |a )?{re.escape(q_elem['obj'])} to \w+")
    elif t == 'who_giver_to':
        pats.append(rf"{a} {V} (?:the |a )?{re.escape(q_elem['obj'])} to {re.escape(q_elem['receiver'])}")
    elif t == 'who_receiver2':
        pats.append(rf"\w+ {V} (?:the |a )?{re.escape(q_elem['obj'])} to {a}")
    for pat in pats:
        m = re.search(pat, input_text, re.IGNORECASE)
        if m:
            # extend to sentence end
            end = input_text.find('.', m.end())
            return input_text[m.start():(end + 1 if end > 0 else m.end())].strip()
    return None


# bAbI-style sentence patterns (give-events + movements + pickups)
_BABI_PAT = re.compile(
    r'[A-Z][a-z]+ (?:went (?:back )?to|journeyed to|travelled to|moved to|'
    r'got|grabbed|took|picked up|dropped|left|put down|discarded|'
    r'gave|handed|passed)(?: back)? (?:the |a |to )?[a-z]+(?: to [A-Z][a-z]+)?\.')


def extract_babi_sentences(input_text):
    """Return the list of clean bAbI sentences embedded in the babilong sample,
    in original order (give-events, movements, pickups). No PG19 noise."""
    return [m.group().strip() for m in _BABI_PAT.finditer(input_text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", default="16k")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--filler", type=int, default=0, help="bAbI filler sentences to add around the SF (0=SF only)")
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--results", default="babilong_results/readout_floor_qa5")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(args.device).eval()
    arrow = f"/root/.cache/huggingface/datasets/RMT-team___babilong/{args.length}/0.0.0/ee0d588794c7ac098062ee0d247c733d62e94fe2/babilong-qa5.arrow"
    ds = Dataset.from_file(arrow)
    L = TASK_LABELS['qa5']
    prompt_cfg = {
        "instruction": DEFAULT_PROMPTS['qa5']["instruction"],
        "examples":    DEFAULT_PROMPTS['qa5']["examples"],
        "post_prompt": DEFAULT_PROMPTS['qa5']["post_prompt"],
    }
    os.makedirs(args.results, exist_ok=True)
    rows = []
    n_sf = 0
    for i in range(min(args.limit, len(ds))):
        ex = ds[i]
        sf = extract_sf_sentence(ex['input'], ex['question'], ex['target'])
        if not sf:
            continue
        n_sf += 1
        if args.filler <= 0:
            context = sf  # SF sentence only
        else:
            # bAbI-only distraction: SF + up to N real bAbI sentences from this
            # sample, in original order, NO PG19 noise. Isolates base-model
            # robustness to competing bAbI facts without long-context/mem-chain.
            babi = extract_babi_sentences(ex['input'])
            sf_norm = re.sub(r'\s+', ' ', sf).strip()
            babi = [re.sub(r'\s+', ' ', b).strip() for b in babi]
            # ensure SF is present
            if sf_norm not in babi:
                babi = babi + [sf_norm]
            # keep SF + first N others (preserve order, SF stays where it is)
            others = [b for b in babi if b != sf_norm]
            keep = set(others[:args.filler]) | {sf_norm}
            context = " ".join(b for b in babi if b in keep)
        full = get_formatted_input(context, ex['question'], prompt_cfg["examples"],
                                   prompt_cfg["instruction"], prompt_cfg["post_prompt"],
                                   template=DEFAULT_TEMPLATE)
        ids = tok(full, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                  min_new_tokens=1, begin_suppress_tokens=[tok.eos_token_id],
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        rows.append({"target": ex['target'], "output": gen, "question": ex['question']})

    acc = 100 * sum(compare_answers(r['target'], r['output'], r['question'], L) for r in rows) / max(1, len(rows))
    # stop-fix
    def sfix(o):
        o = re.split(r'(?i)\bquestion\b|\n', o)[0]
        p = re.split(r'(?i)answer\s*:', o)
        return (p[0] + 'answer:' + p[1]) if len(p) >= 3 else o
    acc_fix = 100 * sum(compare_answers(r['target'], sfix(r['output']), r['question'], L) for r in rows) / max(1, len(rows))
    with open(f"{args.results}/floor_{args.length}_f{args.filler}_n{len(rows)}.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["target", "output", "question"]); w.writeheader(); w.writerows(rows)
    print(f"[floor] qa5 {args.length} filler={args.filler} (SF + N bAbI distractors, no PG19/mem-chain), n={len(rows)} (sf_located={n_sf})")
    print(f"  raw-official = {acc:.0f}   stop-fix = {acc_fix:.0f}")
    print(f"  -> 这是基座读出能力的下界(证据trivially在场, 无haystack/无mem-chain)")


if __name__ == "__main__":
    main()
