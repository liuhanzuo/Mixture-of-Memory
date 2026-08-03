#!/usr/bin/env python
"""Paper A — P1.9 dense-retriever + native-prompting standard RAG reference.

NEW FILE (2026-08-03). A zero-TRAINING reference harness that answers "how does a
deployment-realistic standard RAG (frozen public DENSE retriever + full-depth
recompute reader) compare to CoMem's mid-depth resume, on the SAME examples?".

It does NOT replace, and is NOT conflated with, the matched-BM25 ``j=0`` reference
(config #2, the RAG-recompute upper bound whose selector is the flagship lexical
iter_bm25) nor MemoryLLM. P1.9 swaps ONLY the SELECTOR — lexical BM25 -> a frozen
public dense retriever (BGE-large-en-v1.5) — while keeping EVERYTHING else on the
config #2 j0-RAG reader byte-for-byte:

  * SAME reader:   models/Qwen3-8b-local, NO LoRA, resume_j=0 (full 36-layer
                   recompute over the selected pack), sink=bos, chunk_size=512,
                   bf16 + sdpa + seed 42.
  * SAME examples: each (task,length) sample is built by the UNMODIFIED QCMem
                   driver primitives (eval_qcmem_babilong / eval_qcmem_longeval /
                   eval_qcmem_locomo / eval_ruler_qcmem + eval_ruler_mem_space),
                   with the identical seed / shard convention, so example i here
                   == example i in the BM25 j=0 and CoMem runs (1:1 pairing).
  * SAME reader interface: the dense-selected top-k DOCUMENT-ABSOLUTE chunk
                   indices are fed as ``needle_chunk_set`` into the unmodified
                   ``eval_qcmem_babilong.qcmem_generate`` with ``selector="oracle"``.
                   The oracle branch packs EXACTLY the supplied indices (see
                   ``_select_context_chunk_indices``), so the read pack differs
                   from config #2 j0-RAG ONLY in which chunks the selector chose.

This module IMPORTS the shared eval modules; it MODIFIES NONE of them.

--------------------------------------------------------------------------------
Report decomposition (per (family,task,length), --mode aggregate):
  * recall@k          — fraction of samples whose GOLD SUPPORT chunk (located by
                        the family's own oracle-needle locator, INDEPENDENTLY of
                        the answer) lands in the dense top-k pack.
  * reader accuracy CONDITIONAL-ON-HIT  (reader quality given retrieval succeeded)
  * reader accuracy CONDITIONAL-ON-MISS (does full-depth recompute salvage a miss)
  * end-to-end quality (BABILong/LongEval/RULER: accuracy; LoCoMo: F1 + substr-acc,
                        judge fields emitted for an offline GPT-4o pass)
  * retrieval latency (ms/query: dense encode of all context chunks + query + rank)
  * index size        (bytes = n_ctx_chunks x embed_dim x dtype_bytes)
  + Wilson 95% CI for each proportion.

Full retriever provenance (model / corpus-index / distance metric / pooling /
normalization / revision / weight sha256 / hardware) is written to every cell's
config JSON and to the aggregate, per the acceptance bar.

--------------------------------------------------------------------------------
Usage (one shard = one (family,task,length,shard) cell):
    python scripts/eval_p1_9_dense_rag.py --mode run \
        --family babilong --task qa1 --length 8k \
        --model_path models/Qwen3-8b-local \
        --retriever_path models/bge-large-en-v1.5 \
        --topk 12 --chunk_size 512 --resume_j 0 --limit 100 \
        --num_shards 4 --shard_index 0 \
        --output_dir bench_results/p1_9_dense_rag \
        --index_dir retrieval_results/p1_9_dense

Aggregate (CPU, after all shards of the requested cells finish):
    python scripts/eval_p1_9_dense_rag.py --mode aggregate \
        --output_dir bench_results/p1_9_dense_rag \
        --require_family babilong:qa1,qa2 longeval: locomo: ruler:niah_multikey_1

The 8-GPU task-pool launcher is scripts/_run_p1_9_dense_rag_8gpu.sh (DRY by
default; RUN=1 to execute on a free diskB H20 node).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import socket
import sys
import time
import zlib
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- unmodified shared eval modules (imported, never edited) ----------------
import scripts.eval_qcmem_babilong as qcb            # noqa: E402
import scripts.eval_qcmem_longeval as qle            # noqa: E402
import scripts.eval_qcmem_locomo as qlo              # noqa: E402
import scripts.eval_ruler_qcmem as qru               # noqa: E402
import scripts.eval_ruler_mem_space as ruler         # noqa: E402

qcmem_generate = qcb.qcmem_generate
QCMemModel = qcb.QCMemModel
harness = qcb.harness  # babilong needle locator + csv writer

# ---- frozen dense-retriever provenance (fail-closed gate) -------------------
# BGE-large-en-v1.5, downloaded 2026-08-03 via hy-proxy. CLS pooling, L2-norm,
# cosine (== dot after norm). Query instruction prepended, passages raw.
EXPECTED_BGE_SHA256 = \
    "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
EXPECTED_BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# dense retriever (frozen BGE, pure transformers — no sentence-transformers dep)
# --------------------------------------------------------------------------- #
class DenseRetriever:
    """Frozen BGE-large-en-v1.5 encoder. CLS pooling, L2-normalized, cosine.

    Passages are encoded WITHOUT instruction; the query is encoded WITH the
    official BGE query instruction. Both truncated to the model's 512-token
    position budget (standard dense-RAG behaviour; documented in the provenance).
    """

    def __init__(self, retriever_path: str, device, dtype, allow_sha_mismatch=False):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.path = retriever_path
        self.device = device
        self.dtype = dtype

        weight = os.path.join(retriever_path, "model.safetensors")
        self.weight_sha256 = _sha256_file(weight) if os.path.exists(weight) else None
        self.sha_ok = (self.weight_sha256 == EXPECTED_BGE_SHA256)
        if not self.sha_ok and not allow_sha_mismatch:
            raise RuntimeError(
                f"[p1.9][ABORT] retriever weight sha256 {self.weight_sha256} != "
                f"expected {EXPECTED_BGE_SHA256} (BGE-large-en-v1.5 "
                f"rev {EXPECTED_BGE_REVISION}). Pass --allow_retriever_sha_mismatch "
                f"only if you deliberately swapped the retriever.")

        # pooling contract read straight off the checkpoint (fail-closed on CLS).
        pool_cfg = os.path.join(retriever_path, "1_Pooling", "config.json")
        self.pooling = "cls"
        if os.path.exists(pool_cfg):
            with open(pool_cfg) as f:
                pc = json.load(f)
            if not pc.get("pooling_mode_cls_token", False):
                raise RuntimeError(
                    f"[p1.9][ABORT] retriever pooling config {pc} is not CLS; this "
                    f"harness hard-codes the BGE CLS+L2+cosine contract.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            retriever_path, local_files_only=True)
        self.max_len = int(getattr(self.tokenizer, "model_max_length", 512) or 512)
        if self.max_len > 512 or self.max_len <= 0:
            self.max_len = 512
        self.model = AutoModel.from_pretrained(
            retriever_path, torch_dtype=dtype, local_files_only=True
        ).to(device).eval()
        self.hidden = int(self.model.config.hidden_size)
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.batch_size = 64

    @property
    def dtype_bytes(self) -> int:
        return 2 if self.dtype in (self.torch.float16, self.torch.bfloat16) else 4

    def _encode_texts(self, texts, is_query: bool):
        """Return an L2-normalized [N, hidden] float tensor (CLS pooling)."""
        torch = self.torch
        if is_query:
            texts = [f"{BGE_QUERY_INSTRUCTION} {t}" for t in texts]
        embs = []
        with torch.no_grad():
            for s in range(0, len(texts), self.batch_size):
                batch = texts[s:s + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_len, return_tensors="pt").to(self.device)
                out = self.model(**enc)
                cls = out.last_hidden_state[:, 0]                # CLS token
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                embs.append(cls.float().cpu())
        if not embs:
            return torch.zeros((0, self.hidden), dtype=torch.float32)
        return torch.cat(embs, dim=0)

    def select_topk(self, context_texts, query_text, topk):
        """Return (sorted top-k DOCUMENT-ABSOLUTE chunk indices,
                   ranked_scores dict, retrieval_latency_ms, index_bytes).

        ``context_texts[i]`` is chunk i (doc order); scores are cosine == dot of
        L2-normalized CLS embeddings; ties broken by ascending index (stable).
        """
        torch = self.torch
        n = len(context_texts)
        if n == 0:
            return [], {}, 0.0, 0
        t0 = time.perf_counter()
        ctx_emb = self._encode_texts(context_texts, is_query=False)   # [n, d]
        q_emb = self._encode_texts([query_text], is_query=True)       # [1, d]
        sims = (ctx_emb @ q_emb[0]).tolist()                          # cosine
        k = max(0, min(int(topk), n))
        # stable: sort by (-score, index) so ties keep document order.
        order = sorted(range(n), key=lambda i: (-sims[i], i))
        sel = sorted(order[:k])
        lat_ms = (time.perf_counter() - t0) * 1000.0
        index_bytes = n * self.hidden * self.dtype_bytes
        scores = {i: round(sims[i], 6) for i in range(n)}
        return sel, scores, round(lat_ms, 3), index_bytes


# --------------------------------------------------------------------------- #
# per-family sample iterators — build EXACTLY the config#2 j0-RAG / CoMem sample
# for shard s, yielding a uniform record the runner consumes. Nothing about the
# example construction is re-invented: each branch mirrors the unmodified driver.
# --------------------------------------------------------------------------- #
class Sample:
    __slots__ = ("sid", "input_ids", "query_text", "gold_probes",
                 "score_ctx", "raw")

    def __init__(self, sid, input_ids, query_text, gold_probes, score_ctx, raw):
        self.sid = sid                 # stable pairing id (matches driver)
        self.input_ids = input_ids     # [1, L] LongTensor on device
        self.query_text = query_text   # dense-retriever query string
        self.gold_probes = gold_probes  # list[str] for recall locator
        self.score_ctx = score_ctx     # dict of family-specific scoring info
        self.raw = raw                 # extra passthrough (locomo answers etc.)


def _encode_prompt(prompt, tokenizer, device, use_chat, enable_thinking,
                   gen_boundary_ids):
    """Reader input ids — byte-identical to the QCMem drivers' encode
    (add_special_tokens=True; chat-template only the input, gen prefix appended
    at the query boundary via gen_boundary_ids in the native variant)."""
    import torch
    if use_chat:
        messages = [{"role": "user", "content": prompt}]
        add_gen = gen_boundary_ids is None
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_gen,
                enable_thinking=enable_thinking)
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_gen)
    ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    return ids.to(device)


def iter_babilong(args, tokenizer, device, use_chat, enable_thinking,
                  gen_boundary_ids):
    from babilong.prompts import (DEFAULT_PROMPTS, DEFAULT_TEMPLATE,
                                   get_formatted_input)
    task = args.task
    task_data = qcb._load_babilong_task(args.dataset_name, args.length, task)
    prompt_cfg = {
        "instruction": DEFAULT_PROMPTS[task]["instruction"],
        "examples": DEFAULT_PROMPTS[task]["examples"],
        "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"],
        "template": DEFAULT_TEMPLATE,
    }
    n = len(task_data)
    if args.limit > 0:
        n = min(n, args.limit)
    idxs = list(range(n))[args.shard_index::args.num_shards]
    for idx in idxs:
        sample = task_data[idx]
        target = sample["target"]
        question = sample["question"]
        input_text = get_formatted_input(
            sample["input"], question, prompt_cfg["examples"],
            prompt_cfg["instruction"], prompt_cfg["post_prompt"],
            template=prompt_cfg["template"])
        input_ids = _encode_prompt(input_text, tokenizer, device, use_chat,
                                   enable_thinking, gen_boundary_ids)
        yield Sample(
            sid=f"{task}_{args.length}_i{idx}",
            input_ids=input_ids,
            query_text=(question or "").strip(),
            gold_probes=[target],           # qa1 answer-string == support span
            score_ctx={"target": target, "question": question, "task": task},
            raw=None)


def iter_longeval(args, tokenizer, device, use_chat, enable_thinking,
                  gen_boundary_ids):
    length = args.length
    if length not in qle._LENGTH_TOKENS:
        raise ValueError(f"unknown longeval length {length}")
    target_tokens = qle._LENGTH_TOKENS[length]
    length_seed = args.seed + (zlib.crc32(length.encode()) % 100000)
    n = args.limit
    idxs = list(range(n))[args.shard_index::args.num_shards]
    for i in idxs:
        rng = random.Random(length_seed * 1000 + i)
        prompt, expected, target_label, n_lines = qle.build_lines_prompt(
            target_tokens, tokenizer, rng)
        input_ids = _encode_prompt(prompt, tokenizer, device, use_chat,
                                   enable_thinking, gen_boundary_ids)
        yield Sample(
            sid=f"{length}_i{i}",
            input_ids=input_ids,
            query_text=qle._bm25_query(target_label),   # paired with BM25 j=0
            gold_probes=[expected, f"<{expected}>", f"line {target_label}"],
            score_ctx={"expected": expected, "label": target_label},
            raw=None)


def iter_locomo(args, tokenizer, device, use_chat, enable_thinking,
                gen_boundary_ids):
    data_path = args.locomo_data
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)
    samples = qlo.build_locomo_samples(data_path)
    if args.locomo_categories:
        cats = {int(c) for c in args.locomo_categories.split(",") if c.strip()}
        samples = [s for s in samples if s["category"] in cats]
    if args.limit > 0:
        samples = samples[:args.limit]
    shard = samples[args.shard_index::args.num_shards]
    for sample in shard:
        prompt = sample["prompt"]
        input_ids = _encode_prompt(prompt, tokenizer, device, use_chat,
                                   enable_thinking, gen_boundary_ids)
        # recall gold probes: evidence turn texts + gold answer (== oracle locator)
        probes = list(sample.get("evidence_texts") or [])
        if sample.get("answers"):
            probes.append(sample["answers"][0])
        yield Sample(
            sid=sample["id"],
            input_ids=input_ids,
            query_text=sample["question"],
            gold_probes=probes,
            score_ctx={"answers": sample["answers"],
                       "category": sample["category"],
                       "is_abstention": sample["is_abstention"],
                       "question": sample["question"]},
            raw=sample)


def iter_ruler(args, tokenizer, device, use_chat, enable_thinking,
               gen_boundary_ids):
    import random as _random
    task = qru._resolve_task(args.task)
    length = args.length
    if length not in ruler._LENGTH_TOKENS:
        raise ValueError(f"unknown ruler length {length}")
    target_tokens = ruler._LENGTH_TOKENS[length]
    base_seed = args.seed + (zlib.crc32(f"{task}\x00{length}".encode()) % 100000)
    vt_icl = None
    if task == "variable_tracking":
        vt_icl = ruler._make_vt_icl(_random.Random(base_seed + 777), 4)
    idxs = list(range(args.limit))[args.shard_index::args.num_shards]
    idxs_set = set(idxs)
    for i in range(args.limit):
        rng = _random.Random(base_seed * 1000 + i)
        prompt, answers, gold_needle = ruler._build_sample(
            task, target_tokens, tokenizer, rng, vt_icl)
        if i not in idxs_set:
            continue
        bare_q = qru._bare_question(prompt)
        input_ids = _encode_prompt(prompt, tokenizer, device, use_chat,
                                   enable_thinking, gen_boundary_ids)
        probes = ([gold_needle] if gold_needle else []) + list(answers or [])
        yield Sample(
            sid=f"{task}_{length}_i{i}",
            input_ids=input_ids,
            query_text=bare_q,
            gold_probes=probes,
            score_ctx={"answers": answers, "task": task,
                       "gold_needle": gold_needle},
            raw=None)


_FAMILY_ITER = {
    "babilong": iter_babilong,
    "longeval": iter_longeval,
    "locomo": iter_locomo,
    "ruler": iter_ruler,
}


# --------------------------------------------------------------------------- #
# per-family scoring (mirrors each unmodified driver's judgement)
# --------------------------------------------------------------------------- #
def _score_babilong(output, ctx):
    from babilong.metrics import TASK_LABELS, compare_answers
    correct = bool(compare_answers(ctx["target"], output, ctx["question"],
                                   TASK_LABELS[ctx["task"]]))
    return {"correct": int(correct), "metric": "compare_answers"}


def _score_longeval(output, ctx):
    pred = qle.extract_prediction(output)
    return {"correct": int(pred == ctx["expected"]), "pred": pred,
            "metric": "exact_register"}


def _score_locomo(output, ctx):
    answers = ctx["answers"]
    is_abst = ctx["is_abstention"]
    if is_abst:
        refused = bool(qlo._REFUSAL_RE.search(output)) or output.strip() == ""
        return {"f1": None, "acc": float(refused), "is_abstention": True,
                "pred": output, "metric": "abstention"}
    f1 = qlo.compute_f1_multi(output, answers)
    em = qlo.compute_em_multi(output, answers)
    acc = qlo.substring_acc(output, answers)
    return {"f1": f1, "em": em, "acc": acc, "is_abstention": False,
            "pred": output, "metric": "f1+substr"}


def _score_ruler(output, ctx):
    rec = ruler._string_match_all_one(output, ctx["answers"])
    return {"recall": rec, "correct": int(rec >= 1.0), "metric": "string_match_all"}


_FAMILY_SCORE = {
    "babilong": _score_babilong,
    "longeval": _score_longeval,
    "locomo": _score_locomo,
    "ruler": _score_ruler,
}


# --------------------------------------------------------------------------- #
# recall locator — gold SUPPORT chunk set, decided INDEPENDENTLY of the answer.
# --------------------------------------------------------------------------- #
def _gold_chunk_set(input_ids, gold_probes, tokenizer, chunk_size):
    """Union of document-absolute chunk indices that contain any gold probe
    (support-span locator). Returns (set_or_None). None == unlocatable -> the
    sample is EXCLUDED from the recall denominator (never inferred from answer)."""
    chunks = set()
    for probe in gold_probes:
        probe = (probe or "").strip()
        if not probe:
            continue
        got = harness._locate_needle_chunks(input_ids, probe, tokenizer, chunk_size)
        if got:
            chunks |= got
    return chunks or None


# --------------------------------------------------------------------------- #
# run mode — one (family,task,length,shard) cell
# --------------------------------------------------------------------------- #
def run_cell(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.resume_j != 0:
        raise SystemExit(f"[p1.9][ABORT] P1.9 reader is the full-depth j=0 "
                         f"RAG-recompute reader; --resume_j must be 0 "
                         f"(got {args.resume_j}).")
    if args.lora_adapter:
        raise SystemExit(f"[p1.9][ABORT] P1.9 reader is training-free (NO LoRA); "
                         f"--lora_adapter must be empty (got {args.lora_adapter!r}).")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    retriever_path = args.retriever_path
    if not os.path.isabs(retriever_path):
        retriever_path = os.path.join(PROJECT_ROOT, retriever_path)

    use_chat = (args.reader_prompt == "native")

    print(f"[p1.9] family={args.family} task={args.task} length={args.length} "
          f"shard={args.shard_index}/{args.num_shards} reader_prompt={args.reader_prompt}")
    print(f"[p1.9] reader={model_path} (NO LoRA, resume_j=0, sink={args.sink_tokens}, "
          f"chunk={args.chunk_size}, topk={args.topk}, {args.dtype}/{args.attn_impl})")

    # ---- reader (Qwen3-8B, full-depth, no adapter) --------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True).to(device).eval()
    qc = QCMemModel(model, resume_j=0)

    gen_boundary_ids = None
    enable_thinking = args.enable_thinking
    if use_chat:
        gen_boundary_ids = qcb._chat_generation_boundary_ids(
            tokenizer, enable_thinking)

    # ---- frozen dense retriever ---------------------------------------------
    retriever = DenseRetriever(
        retriever_path, device, dtype,
        allow_sha_mismatch=args.allow_retriever_sha_mismatch)
    print(f"[p1.9] retriever=BGE-large-en-v1.5 sha_ok={retriever.sha_ok} "
          f"sha256={retriever.weight_sha256} pooling={retriever.pooling} "
          f"hidden={retriever.hidden} max_len={retriever.max_len}")

    provenance = {
        "retriever_model": "BAAI/bge-large-en-v1.5",
        "retriever_path": retriever_path,
        "retriever_revision": EXPECTED_BGE_REVISION,
        "retriever_weight_sha256": retriever.weight_sha256,
        "retriever_weight_sha256_expected": EXPECTED_BGE_SHA256,
        "retriever_sha_ok": retriever.sha_ok,
        "pooling": "cls",
        "normalization": "l2",
        "distance_metric": "cosine (dot of L2-normalized CLS)",
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "passage_instruction": "",
        "retriever_max_tokens": retriever.max_len,
        "retriever_dtype": args.dtype,
        "hidden_dim": retriever.hidden,
        "index_type": "flat brute-force cosine (exact, per-query rebuild)",
        "hardware": {
            "node": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else None),
        },
    }
    reader_cfg = {
        "reader_model_path": model_path,
        "reader_lora_adapter": None,
        "resume_j": 0,
        "sink_tokens": args.sink_tokens,
        "chunk_size": args.chunk_size,
        "topk": args.topk,
        "selector": "dense_bge (via oracle needle_chunk_set injection)",
        "reader_prompt": args.reader_prompt,
        "use_chat_template": use_chat,
        "enable_thinking": enable_thinking,
        "add_special_tokens": True,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "seed": args.seed,
        "num_layers": int(model.config.num_hidden_layers),
    }

    outdir = Path(args.output_dir) / f"{args.family}"
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
    cell = f"{args.task or args.family}_{args.length}"
    outfile = outdir / f"{cell}_{args.reader_prompt}{shard_tag}.jsonl"
    cfgfile = outdir / f"{cell}_{args.reader_prompt}{shard_tag}.config.json"

    index_dir = Path(args.index_dir) / f"{args.family}"
    index_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    it = _FAMILY_ITER[args.family](
        args, tokenizer, device, use_chat, enable_thinking, gen_boundary_ids)
    scorer = _FAMILY_SCORE[args.family]

    records = []
    index_sizes = []
    t_start = time.time()
    for pos, s in enumerate(it):
        # dense chunking IDENTICAL to qcmem_generate: tokens.split(chunk_size),
        # context = chunks[:-1], query chunk = chunks[-1].
        tokens = s.input_ids[0]
        chunks = list(tokens.split(args.chunk_size))
        context_chunks = chunks[:-1]
        n_ctx = len(context_chunks)
        # decode each context chunk to text for the (text-space) dense retriever.
        ctx_texts = [tokenizer.decode(c.tolist(), skip_special_tokens=True)
                     for c in context_chunks]
        sel_idx, scores, lat_ms, index_bytes = retriever.select_topk(
            ctx_texts, s.query_text, args.topk)
        index_sizes.append(index_bytes)

        # recall: gold support chunk in the dense top-k pack? (answer-independent)
        gold_set = _gold_chunk_set(s.input_ids, s.gold_probes, tokenizer,
                                   args.chunk_size)
        if gold_set is None:
            recall_hit = None                    # unlocatable -> excluded
        else:
            gold_in_ctx = {c for c in gold_set if 0 <= c < n_ctx}
            recall_hit = int(bool(gold_in_ctx & set(sel_idx))) \
                if gold_in_ctx else None

        # reader: feed dense-selected indices as oracle needle_chunk_set.
        bare_q_ids = tokenizer.encode(s.query_text or "", add_special_tokens=False)
        gen_stats = {}
        try:
            output = qcmem_generate(
                qc=qc, tokenizer=tokenizer, input_ids=s.input_ids,
                chunk_size=args.chunk_size, max_new_tokens=args.max_new_tokens,
                selector="oracle", topk=args.topk,
                sink_tokens=args.sink_tokens,
                needle_chunk_set=set(sel_idx), bare_question_ids=bare_q_ids,
                no_retrieval=False, stats=gen_stats,
                gen_boundary_ids=gen_boundary_ids)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            output = "[OOM]"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sc = scorer(output, s.score_ctx)
        rec = {
            "id": s.sid,
            "family": args.family, "task": args.task, "length": args.length,
            "query_text": s.query_text,
            "n_context_chunks": n_ctx,
            "dense_sel_idx": sel_idx,
            "gold_chunk_set": (sorted(gold_set) if gold_set else None),
            "recall_hit": recall_hit,
            "read_len": gen_stats.get("read_len"),
            "n_selected_chunks": gen_stats.get("n_selected_chunks"),
            "retrieval_latency_ms": lat_ms,
            "index_bytes": index_bytes,
            "output": output,
            "input_ids_sha256": _sha256_str(",".join(map(str, tokens.tolist()))),
            "pack_sel_sha256": _sha256_str(",".join(map(str, sel_idx))),
        }
        rec.update(sc)
        records.append(rec)

        if (pos + 1) % 10 == 0 or True:
            with open(outfile, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(outfile, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # per-cell index manifest (sizes; embeddings themselves not persisted by
    # default — the flat index is rebuilt per query and its size reported).
    mean_bytes = (sum(index_sizes) / len(index_sizes)) if index_sizes else 0
    with open(index_dir / f"{cell}_{args.reader_prompt}{shard_tag}.index.json",
              "w") as f:
        json.dump({
            "cell": cell, "shard": shard_tag or "single",
            "n_samples": len(records),
            "embed_dim": retriever.hidden,
            "dtype": args.dtype, "dtype_bytes": retriever.dtype_bytes,
            "metric": "cosine", "index_type": provenance["index_type"],
            "index_bytes_mean": round(mean_bytes, 1),
            "index_bytes_min": min(index_sizes) if index_sizes else 0,
            "index_bytes_max": max(index_sizes) if index_sizes else 0,
        }, f, indent=2)

    cfg = {
        "status": "completed", "family": args.family, "task": args.task,
        "length": args.length, "n": len(records),
        "n_requested": args.limit,
        "sharding": {"num_shards": args.num_shards,
                     "shard_index": args.shard_index},
        "reader": reader_cfg,
        "retriever_provenance": provenance,
        "recall_definition": ("gold support-span chunk (family oracle locator) in "
                              "dense top-k pack; decided INDEPENDENTLY of answer; "
                              "unlocatable gold -> excluded from recall denom"),
        "pairing": ("same seed/shard/chunk_size as config#2 j0-RAG + CoMem; "
                    "input_ids_sha256 + pack_sel_sha256 recorded per example"),
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    with open(cfgfile, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[p1.9] wrote {len(records)} records -> {outfile}")
    print(f"[p1.9] cell config -> {cfgfile}")


# --------------------------------------------------------------------------- #
# aggregate mode — merge shards, decompose recall/readout, Wilson CIs.
# --------------------------------------------------------------------------- #
def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(p, 4), round(max(0.0, centre - half), 4),
            round(min(1.0, centre + half), 4))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(args):
    root = Path(args.output_dir)
    # required cells guard: "family:task1,task2 ..." (empty task -> family-level)
    required = {}
    for spec in (args.require_family or []):
        fam, _, tasks = spec.partition(":")
        required[fam] = [t for t in tasks.split(",") if t] or [None]

    summary = {}
    seen_cells = set()
    for fam_dir in sorted(root.glob("*")):
        if not fam_dir.is_dir():
            continue
        fam = fam_dir.name
        # group jsonl shards by cell key (task_length_prompt).
        cells = {}
        for jf in fam_dir.glob("*.jsonl"):
            name = jf.name
            # strip _shard{a}of{b}.jsonl / .jsonl
            base = name[:-len(".jsonl")]
            if "_shard" in base:
                base = base[:base.rfind("_shard")]
            cells.setdefault(base, []).append(jf)
        for cellkey, files in sorted(cells.items()):
            recs = []
            seen_ids = set()
            for jf in sorted(files):
                with open(jf) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        if r["id"] in seen_ids:
                            continue
                        seen_ids.add(r["id"])
                        recs.append(r)
            if not recs:
                continue
            seen_cells.add((fam, cellkey))
            n = len(recs)
            # recall (locatable subset only)
            loc = [r for r in recs if r.get("recall_hit") is not None]
            hits = sum(r["recall_hit"] for r in loc)
            recall_p, recall_lo, recall_hi = _wilson(hits, len(loc))

            def _acc(r):
                if fam == "locomo":
                    return r.get("acc")
                if "correct" in r:
                    return r["correct"]
                if "recall" in r:
                    return r["recall"]
                return None

            e2e = _mean([_acc(r) for r in recs])
            hit_subset = [r for r in loc if r["recall_hit"] == 1]
            miss_subset = [r for r in loc if r["recall_hit"] == 0]
            acc_hit = _mean([_acc(r) for r in hit_subset])
            acc_miss = _mean([_acc(r) for r in miss_subset])
            entry = {
                "n": n,
                "n_locatable": len(loc),
                "recall_at_k": recall_p,
                "recall_ci95": [recall_lo, recall_hi],
                "end_to_end": e2e,
                "acc_cond_on_hit": acc_hit,
                "n_hit": len(hit_subset),
                "acc_cond_on_miss": acc_miss,
                "n_miss": len(miss_subset),
                "retrieval_latency_ms_mean":
                    _mean([r.get("retrieval_latency_ms") for r in recs]),
                "read_len_mean": _mean([r.get("read_len") for r in recs]),
                "index_bytes_mean": _mean([r.get("index_bytes") for r in recs]),
            }
            if fam == "locomo":
                entry["f1_mean"] = _mean([r.get("f1") for r in recs
                                          if not r.get("is_abstention")])
            summary.setdefault(fam, {})[cellkey] = entry

    # ---- all-tasks-reported fail-closed guard -------------------------------
    missing = []
    for fam, tasks in required.items():
        present_cells = summary.get(fam, {})
        for t in tasks:
            hit = any((t is None) or ck.startswith(f"{t}_") or (f"_{t}_" in ck)
                      or ck.startswith(t)
                      for ck in present_cells) if present_cells else False
            if not hit:
                missing.append(f"{fam}:{t or '<any>'}")
    guard_ok = not missing

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(root),
        "all_tasks_reported": guard_ok,
        "missing_required_cells": missing,
        "summary": summary,
    }
    aggfile = root / "aggregate.json"
    with open(aggfile, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[p1.9] aggregate -> {aggfile}")
    if not guard_ok:
        print(f"[p1.9][FAIL-CLOSED] required cells missing: {missing}")
        sys.exit(5)


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Paper A P1.9 dense-retriever + native-prompting RAG reference")
    p.add_argument("--mode", choices=["run", "aggregate", "provenance"],
                   default="run")
    p.add_argument("--family", choices=list(_FAMILY_ITER),
                   help="benchmark family (run mode)")
    p.add_argument("--task", type=str, default=None,
                   help="babilong qa1/qa2 ; ruler niah_multikey_1 ; "
                        "unused for longeval/locomo")
    p.add_argument("--length", type=str, default="8k")
    p.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    p.add_argument("--retriever_path", type=str,
                   default="models/bge-large-en-v1.5")
    p.add_argument("--lora_adapter", type=str, default="",
                   help="MUST be empty — P1.9 reader is training-free (j=0).")
    p.add_argument("--resume_j", type=int, default=0,
                   help="MUST be 0 — P1.9 reader is the full-depth RAG-recompute.")
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--sink_tokens", type=str, default="bos",
                   choices=["bos", "none"])
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--reader_prompt", choices=["plain", "native"], default="plain",
                   help="plain = unified no-chat main protocol (chat_template off, "
                        "the config#2 j0-RAG口径). native = reader "
                        "native-prompt/template-sensitivity variant (chat template "
                        "on, no-think generation boundary).")
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    p.add_argument("--locomo_data", type=str, default="locomo/data/locomo10.json")
    p.add_argument("--locomo_categories", type=str, default=None)
    p.add_argument("--output_dir", type=str,
                   default="bench_results/p1_9_dense_rag")
    p.add_argument("--index_dir", type=str, default="retrieval_results/p1_9_dense")
    p.add_argument("--allow_retriever_sha_mismatch", action="store_true",
                   default=False,
                   help="bypass the BGE weight sha fail-closed gate (audit only).")
    p.add_argument("--require_family", nargs="+", default=None,
                   help="aggregate all-tasks guard: 'family:task1,task2' specs "
                        "(empty task list == family-level presence).")
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == "run":
        if not args.family:
            build_parser().error("--family is required in run mode")
        if args.num_shards < 1:
            build_parser().error("--num_shards must be >= 1")
        if not (0 <= args.shard_index < args.num_shards):
            build_parser().error("--shard_index out of range")
        run_cell(args)
    elif args.mode == "aggregate":
        aggregate(args)
    elif args.mode == "provenance":
        weight = os.path.join(
            args.retriever_path if os.path.isabs(args.retriever_path)
            else os.path.join(PROJECT_ROOT, args.retriever_path),
            "model.safetensors")
        sha = _sha256_file(weight) if os.path.exists(weight) else None
        ok = (sha == EXPECTED_BGE_SHA256)
        print(json.dumps({
            "retriever_path": args.retriever_path,
            "weight_sha256": sha,
            "expected_sha256": EXPECTED_BGE_SHA256,
            "revision": EXPECTED_BGE_REVISION,
            "sha_ok": ok,
        }, indent=2))
        sys.exit(0 if ok else 6)


if __name__ == "__main__":
    main()
