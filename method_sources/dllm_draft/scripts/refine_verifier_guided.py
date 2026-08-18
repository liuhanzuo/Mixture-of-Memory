#!/usr/bin/env python
"""
Verifier-guided iterative refinement harness for Dream-Coder-Instruct-7B on
HumanEval+. Given a run's per-task solutions, execute each against the
EvalPlus base_input tests, then for tasks that FAIL, do a second diffusion
pass under one of two refinement policies:

  restart : reseed the entire canvas from the prompt at temperature T=0.6
            (Reflexion-adjacent restart baseline).
  remask  : take the failed draft, re-mask the last `remask_frac` fraction of
            its tokens, and re-diffuse for --refine-steps.

Then re-grade the resulting solutions vs the original run.

This is the ARM-A vs ARM-B version of the SotA-scan direction (2)
"verifier-guided subtree collapse" idea. Arm C (typed subtree collapse) needs
the scaffold checkpoint, which is only on the remote node -- so we build A/B
first and add C when the ckpt lands.

The verifier used here is EvalPlus's own grader (`untrusted_check`) run on the
VISIBLE (base) tests only, with ground-truth output comparison. This is a WEAK
verifier relative to final scoring, which additionally uses the hidden `plus`
tests -- intentionally so: it is what a real Reflexion-style loop would have
access to, and it lets us test whether visible-test feedback lifts hidden-test
pass@1.

NOTE (2026-08-07): before this revision the verifier was a hand-rolled sandbox
that called `entry_point(*args)` and counted "did not raise" as a pass, never
comparing return values to expected output. Empty/stub solutions therefore
scored full marks and were routed to `keep`. All refinement runs produced
before this fix are invalid; see DLLM_RESULTS_20260807.md for the retraction.
"""
import argparse, ast, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

from evalplus.eval import PASS, untrusted_check

def read_jsonl(p):
    return [json.loads(l) for l in open(p)]

def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")

def sandbox_exec(solution: str, prompt: str, task: dict, dataset: str, gt: dict):
    """
    Return dict {ok: bool, error: str|None, n_tests: int, n_pass: int}.

    Uses EvalPlus's OWN grader (`untrusted_check`) on the VISIBLE (base) tests,
    comparing against ground-truth expected outputs.

    HISTORY -- this replaces a hand-rolled sandbox that only checked whether
    `entry_point(*args)` raised. That discarded the return value and never
    compared against expected output, so any non-crashing function scored a
    full pass. An EMPTY solution scored 7/7 on HumanEval/0 (docstring-only
    stub returns None without raising). Every run produced before this fix has
    an inflated `prior_ok`/`kept` population and must be re-run, not patched.
    """
    tid = task["task_id"]
    code = prompt + "\n" + solution
    ref = gt.get(tid)
    if ref is None:
        return {"ok": False, "error": "NO_GROUNDTRUTH", "n_tests": 0, "n_pass": 0}
    try:
        status, details = untrusted_check(
            dataset, code, task["base_input"], task["entry_point"],
            expected=ref["base"], atol=task["atol"],
            ref_time=ref["base_time"], fast_check=False,
            min_time_limit=1.0, gt_time_limit_factor=4.0,
        )
    except Exception as e:
        return {"ok": False, "error": f"GRADER_ERR: {type(e).__name__}: {e}", "n_tests": 0, "n_pass": 0}
    details = list(details) if details is not None else []
    n_tests = len(details)
    n_pass = int(sum(bool(d) for d in details))
    first_fail = None
    for i, d in enumerate(details):
        if not d:
            first_fail = f"test[{i}]: output mismatch or error"
            break
    return {"ok": status == PASS and n_tests > 0 and n_pass == n_tests,
            "error": first_fail if status != PASS else None,
            "n_tests": n_tests, "n_pass": n_pass}


def parseable(text: str) -> bool:
    try:
        ast.parse(text); return True
    except SyntaxError: return False


def extract_python(text: str) -> str:
    fences = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fences: text = max(fences, key=len)
    text = text.strip()
    starts = [m.start() for m in re.finditer(r"(?m)^(?:async\s+def|def|from|import|@)\s*", text)]
    if starts: text = text[min(starts):]
    return text.rstrip() + ("\n" if text else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-run", required=True, help="Path to a run dir containing solutions.jsonl (the failing-task pool comes from here)")
    ap.add_argument("--dataset", choices=("humaneval","mbpp"), default="humaneval")
    ap.add_argument("--policy", choices=("restart","remask"), required=True)
    ap.add_argument("--refine-steps", type=int, default=256)
    ap.add_argument("--refine-temp", type=float, default=0.6)
    ap.add_argument("--remask-frac", type=float, default=0.5,
                    help="Fraction of tokens to remask (remask policy only)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--checkpoint", default="models/Dream-Coder-v0-Instruct-7B")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK","0"))
    local_rank = int(os.environ.get("LOCAL_RANK","0"))
    world_size = int(os.environ.get("WORLD_SIZE","1"))

    # 1. Read input run's solutions
    input_run = Path(args.input_run)
    sols = {r["task_id"]: r["solution"] for r in read_jsonl(input_run/"solutions.jsonl")}

    # 2. Load EvalPlus HE+ or MBPP+ with base_input, plus ground-truth expected outputs
    from evalplus.evaluate import get_groundtruth
    from evalplus.eval import MBPP_OUTPUT_NOT_NONE_TASKS
    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
        he = get_human_eval_plus()
        gt = get_groundtruth(he, get_human_eval_plus_hash(), [])
    else:
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
        he = get_mbpp_plus()
        gt = get_groundtruth(he, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)
    tasks = list(he.items())  # already ordered by id
    if args.limit: tasks = tasks[:args.limit]

    # 3. Score visible tests to find FAILING tasks (this is the verifier)
    assigned = [(tid, task) for idx, (tid, task) in enumerate(tasks) if idx % world_size == rank]

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / f"metrics.rank{rank:02d}.jsonl"
    solutions_path = outdir / f"solutions.rank{rank:02d}.jsonl"
    metrics_f = open(metrics_path, "w")
    solutions_f = open(solutions_path, "w")

    # Lazy import model
    import torch
    from transformers import AutoTokenizer, AutoModel
    from scaffold_coder.tokenizer_utils import extend_tokenizer, edit_source_token_ids

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    print(f"[rank {rank}] loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True, local_files_only=True,
                                      low_cpu_mem_usage=True).to(device).eval()
    suppressed_ids = tuple(sorted(set(
        e.token_id for e in extend_tokenizer(tokenizer)
    ) | set(edit_source_token_ids(tokenizer))))
    def hook(step, x, logits):
        if suppressed_ids:
            logits[..., list(suppressed_ids)] = torch.finfo(logits.dtype).min
        return logits

    def diffuse(prompt_text: str, canvas_init_ids: list = None, steps: int = None, temp: float = None) -> str:
        inputs = tokenizer.apply_chat_template(
            [{"role":"user","content":prompt_text}],
            return_tensors="pt", return_dict=True, add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device)
        attn = inputs.attention_mask.to(device)
        with torch.inference_mode():
            output = model.diffusion_generate(
                input_ids, attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                output_history=False, return_dict_in_generate=True,
                steps=steps or args.refine_steps,
                temperature=temp if temp is not None else args.refine_temp,
                top_p=0.95, alg="entropy", alg_temp=0.0,
                generation_logits_hook_func=hook,
            )
        gen_ids = output.sequences[0, input_ids.shape[1]:].tolist()
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    def remask_and_diffuse(prompt_text: str, prev_solution: str) -> str:
        """Encode prev, remask last remask_frac of its tokens, run diffusion.
        Note: current Dream sampler doesn't support arbitrary init canvas from outside,
        so as a proxy we truncate prev_solution to (1-frac) then let diffusion continue
        from there — this is functionally equivalent to remasking the tail."""
        prev = extract_python(prev_solution)
        toks = tokenizer(prev, add_special_tokens=False).input_ids
        keep = int(len(toks) * (1.0 - args.remask_frac))
        prefix = tokenizer.decode(toks[:keep], skip_special_tokens=True)
        aug_prompt = prompt_text + "\n\n[Previous attempt (partial):]\n```python\n" + prefix + "\n```\n[Continue and correct if wrong.]"
        return diffuse(aug_prompt, steps=args.refine_steps, temp=args.refine_temp)

    for tid, task in assigned:
        prior_sol = sols.get(tid, "")
        entry_pt = task["entry_point"]
        prior_verdict = sandbox_exec(prior_sol, task["prompt"], task, args.dataset, gt)

        if prior_verdict["ok"]:
            # already passes visible tests -> keep as-is
            new_sol = prior_sol
            action = "keep"
            new_verdict = prior_verdict
            elapsed = 0.0
        else:
            # apply refinement policy
            t0 = time.perf_counter()
            try:
                base_prompt = ("Write a complete Python solution for the following function. "
                               "Return only Python code and preserve the required function name.\n\n"
                               + task["prompt"])
                if args.policy == "restart":
                    raw = diffuse(base_prompt, steps=args.refine_steps, temp=args.refine_temp)
                else:  # remask
                    raw = remask_and_diffuse(base_prompt, prior_sol)
                new_sol = extract_python(raw)
                action = args.policy
                new_verdict = sandbox_exec(new_sol, task["prompt"], task, args.dataset, gt)
            except Exception as e:
                new_sol = prior_sol
                action = "error"
                new_verdict = prior_verdict
                new_verdict["error"] = f"REFINE_ERR: {e}"
            elapsed = time.perf_counter() - t0

        # if refinement worse, fall back
        if action != "keep" and prior_verdict["n_pass"] > new_verdict["n_pass"]:
            new_sol = prior_sol; action += "_reverted"; new_verdict = prior_verdict

        solutions_f.write(json.dumps({"task_id": tid, "solution": new_sol})+"\n"); solutions_f.flush()
        metrics_f.write(json.dumps({
            "task_id": tid,
            "action": action,
            "prior_ok": prior_verdict["ok"],
            "prior_npass": prior_verdict["n_pass"],
            "prior_ntests": prior_verdict["n_tests"],
            "prior_err": prior_verdict["error"],
            "new_ok": new_verdict["ok"],
            "new_npass": new_verdict["n_pass"],
            "new_err": new_verdict.get("error"),
            "refine_seconds": round(elapsed,2),
            "policy": args.policy,
        })+"\n"); metrics_f.flush()

    solutions_f.close(); metrics_f.close()
    print(f"[rank {rank}] done", flush=True)


if __name__ == "__main__":
    main()
