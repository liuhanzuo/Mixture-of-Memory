"""BABILong task-specific IterableDataset for Memory-Space SFT.

Yields BABILong supervised samples (prompt + answer) where labels are -100
on every token EXCEPT the answer-text portion, so HF's CausalLM loss only
back-propagates on the answer tokens (answer-only loss).

Critical design constraint
--------------------------
The prompt MUST be byte-for-byte identical to what
``scripts/run_babilong_mem_space.py`` constructs at evaluation time, except
that we additionally append the gold answer text after it.  Eval-time prompt
construction (run_babilong_mem_space.py:514-527):

    input_text = get_formatted_input(
        context, question, examples, instruction, post_prompt,
        template=DEFAULT_TEMPLATE,
    )
    if args.use_chat_template:
        messages = [{"role": "user", "content": input_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, ...)

We replicate that exactly, then concatenate the answer text afterwards as a
separate ``encode(answer, add_special_tokens=False)`` call so prompt_len /
answer_len are well-defined and we can mask labels precisely.

Reference: status/PENDING_TASKS.md "Fix 2 — BABILong task-specific SFT".
"""
from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.utils.data


# --------------------------------------------------------------------------- #
# BABILong package import
# --------------------------------------------------------------------------- #

# Mirror the path setup used by scripts/run_babilong_mem_space.py:50-57 so
# this module works whether or not third_party/babilong-pkg is already on
# sys.path.
_BABILONG_ROOTS = [
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "third_party", "babilong-pkg",
    ),
    "/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg",
    "/apdcephfs_zwfy6/share_303098609/pighzliu_code/babilong",
    "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong",
]
for _root in _BABILONG_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

try:
    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402
except ImportError as _e:
    DEFAULT_PROMPTS = None
    DEFAULT_TEMPLATE = None
    get_formatted_input = None
    _BABILONG_IMPORT_ERROR = _e
else:
    _BABILONG_IMPORT_ERROR = None


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class BABILongTrainDataset(torch.utils.data.IterableDataset):
    """Infinite stream of BABILong supervised samples for mem_space SFT.

    Each yielded sample is a dict with:
        ``input_ids``: 1-D long tensor [<= max_seq_len]   (prompt + answer)
        ``labels``:    1-D long tensor [<= max_seq_len]   (-100 on prompt,
                       gold tokens on answer span)
        ``is_niah``:   False (signals "BABILong" path to the train loop;
                       loss treatment is identical in spirit — answer-only).
        ``is_babilong``: True (explicit flag)
        ``task``:        e.g. "qa1"
        ``length``:      e.g. "1k"

    Args:
        dataset_name:    HuggingFace dataset id (default ``RMT-team/babilong``).
        tasks:           Which qa-tasks to sample from (uniform mixture).
        lengths:         Which length splits to sample from (uniform mixture).
                         External code (curriculum scheduler) can swap this list
                         in-place between training stages.
        tokenizer:       HF tokenizer.  pad_token must exist (we set it to eos
                         if missing, mirroring train_mem_space_pg19.py:502).
        max_seq_len:     Hard cap on the returned sequence length (left-truncate
                         the context portion if needed; never truncate the
                         answer).
        seed:            Base RNG seed; per-DDP-worker seeded from seed+worker_id
                         (same convention as ``NIAHIterableDataset``).
        use_chat_template: If True, wrap the formatted prompt with the
                           tokenizer's chat template (mirrors
                           run_babilong_mem_space.py --use_chat_template flag).
                           Required when the backbone is an Instruct model.
        use_instruction:   Include DEFAULT_PROMPTS[task]['instruction'].
        use_examples:      Include DEFAULT_PROMPTS[task]['examples'].
        use_post_prompt:   Include DEFAULT_PROMPTS[task]['post_prompt'].
        limit_per_cell:    If >0, only sample from the first N rows of each
                           (task, length) split (useful for fast smoke tests).
                           0 = use all rows.
    """

    def __init__(
        self,
        tokenizer: Any,
        dataset_name: str = "RMT-team/babilong",
        tasks: Optional[List[str]] = None,
        lengths: Optional[List[str]] = None,
        max_seq_len: int = 4096,
        seed: int = 42,
        use_chat_template: bool = False,
        use_instruction: bool = True,
        use_examples: bool = True,
        use_post_prompt: bool = True,
        limit_per_cell: int = 0,
    ) -> None:
        super().__init__()
        if _BABILONG_IMPORT_ERROR is not None:
            raise ImportError(
                "babilong package not importable; tried "
                f"{_BABILONG_ROOTS}.  Original error: {_BABILONG_IMPORT_ERROR}"
            )

        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.tasks = list(tasks) if tasks else ["qa1"]
        self.lengths = list(lengths) if lengths else ["1k", "2k"]
        self.max_seq_len = int(max_seq_len)
        self.seed = int(seed)
        self.use_chat_template = bool(use_chat_template)
        self.use_instruction = bool(use_instruction)
        self.use_examples = bool(use_examples)
        self.use_post_prompt = bool(use_post_prompt)
        self.limit_per_cell = int(limit_per_cell)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._pad_id: int = tokenizer.pad_token_id

        for t in self.tasks:
            if t not in DEFAULT_PROMPTS:
                raise ValueError(
                    f"Unknown BABILong task {t!r}. "
                    f"Known tasks: {sorted(DEFAULT_PROMPTS.keys())}"
                )

        # Cached HF dataset splits: {(task, length) -> list-like of rows}.
        # Built eagerly here at __init__ rather than lazily in _load_split.
        # ----------------------------------------------------------------------
        # Why eager: in distributed training, lazy loading caused the whole
        # group to deadlock — when each rank lazily called load_dataset on the
        # FIRST iter() of a (task, length) it had not seen, the 8 ranks would
        # race the HF cache flock and stall (observed 2026-05-15: rank 1 + 7
        # active, ranks 0,2-6 idle for 5+ min after the rank-0 prefetch
        # barrier). Eagerly building the cache at __init__ time is sequential
        # within each rank but predictable: all ranks run the same load
        # sequence after the same outer barrier, so the work is bounded and
        # cache contention is one-shot.
        self._cache: Dict[tuple, Any] = {}
        self._eager_load_all_splits()

    # --------------------------------------------------------------------- #
    # Eager dataset loading at init
    # --------------------------------------------------------------------- #

    def _eager_load_all_splits(self) -> None:
        """Load every (task, length) cell into ``self._cache`` once at init.

        Avoids per-iter lazy loads that deadlock under distributed training.
        """
        import datasets  # noqa: WPS433
        for length in self.lengths:
            try:
                data = datasets.load_dataset(self.dataset_name, length)
            except Exception:
                # Defer the error to first iter; some test envs may not need
                # every (task, length) combination.
                continue
            for task in self.tasks:
                if task in data:
                    self._cache[(task, length)] = data[task]

    # --------------------------------------------------------------------- #
    # Lazy data loading
    # --------------------------------------------------------------------- #

    def _load_split(self, task: str, length: str) -> Any:
        """Load one (task, length) cell from the BABILong HF dataset.

        Cached on (task, length) so repeated draws reuse the same in-memory
        Arrow shard.
        """
        key = (task, length)
        if key in self._cache:
            return self._cache[key]
        # Local import: avoid bringing `datasets` into module-level imports so
        # syntax-only smoke tests can succeed without the dependency installed.
        import datasets  # noqa: WPS433
        data = datasets.load_dataset(self.dataset_name, length)
        task_data = data[task]
        self._cache[key] = task_data
        return task_data

    # --------------------------------------------------------------------- #
    # Prompt construction (eval-parity)
    # --------------------------------------------------------------------- #

    def _build_prompt_text(self, task: str, sample: Dict[str, Any]) -> str:
        """Construct the EVAL-parity prompt text for one BABILong row.

        Must match scripts/run_babilong_mem_space.py:451-527 byte-for-byte.
        """
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if self.use_instruction else "",
            "examples":    DEFAULT_PROMPTS[task]["examples"]    if self.use_examples    else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if self.use_post_prompt else "",
        }
        input_text = get_formatted_input(
            sample["input"],          # context
            sample["question"],       # question
            prompt_cfg["examples"],
            prompt_cfg["instruction"],
            prompt_cfg["post_prompt"],
            template=DEFAULT_TEMPLATE,
        )
        if self.use_chat_template:
            messages = [{"role": "user", "content": input_text}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return input_text

    @staticmethod
    def _build_answer_text(task: str, target: str) -> str:
        """Render the gold answer in the format BABILong's post_prompt asks for.

        We re-use the wording from DEFAULT_PROMPTS[task]['examples'] so the
        answer text matches the style the prompt asks the model to produce.

        Falls back to plain ``target`` for unknown tasks.
        """
        target = str(target).strip()
        if task == "qa1":
            # post_prompt expects: "The most recent location of 'person' is 'location'."
            # ``sample['question']`` ends with a phrase like "Where is Charlie?";
            # the example answers start with "The most recent location of <person> is".
            # We don't have ``person`` directly here, so we let the loss simply
            # supervise the location word — but to maximise eval/train
            # consistency we wrap it in the canonical phrasing whenever the
            # caller can pass the person.  At dataset level we don't know it,
            # so output the bare target string and rely on the prompt's
            # post_prompt text to push the model toward the canonical wrapper.
            return target
        if task == "qa2":
            return target
        if task == "qa3":
            return target
        if task == "qa4":
            return target
        if task == "qa5":
            return target
        return target

    # --------------------------------------------------------------------- #
    # IterableDataset protocol
    # --------------------------------------------------------------------- #

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id: int = worker_info.id
            num_workers: int = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        rng = random.Random(self.seed + worker_id)

        # Optional: also de-correlate across DDP ranks if RANK env is set.
        try:
            ddp_rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            ddp_rank = 0
        rng = random.Random(self.seed + worker_id + 10_007 * ddp_rank)

        while True:
            task = rng.choice(self.tasks)
            length = rng.choice(self.lengths)

            try:
                split = self._load_split(task, length)
            except Exception as exc:  # pragma: no cover (network / HF errors)
                # If we can't load a (task, length) cell, drop it from the
                # rotation and keep going.  Silent skip keeps the trainer
                # alive when one length isn't downloaded yet.
                if (task, length) not in self._cache:
                    self._cache[(task, length)] = None
                continue
            if split is None or len(split) == 0:
                continue

            n_rows = len(split)
            if self.limit_per_cell > 0:
                n_rows = min(n_rows, self.limit_per_cell)
            row_idx = rng.randrange(n_rows)
            sample = split[row_idx]

            try:
                prompt_text = self._build_prompt_text(task, sample)
            except Exception:
                continue
            answer_text = self._build_answer_text(task, sample["target"])

            built = self._tokenize_with_answer_mask(prompt_text, answer_text)
            if built is None:
                continue
            input_ids, labels, prompt_len, ans_len = built

            yield {
                "input_ids": input_ids,
                "labels":    labels,
                "is_niah":   False,
                "is_babilong": True,
                "task":      task,
                "length":    length,
                "prompt_len": prompt_len,
                "answer_len": ans_len,
            }

    # --------------------------------------------------------------------- #
    # Tokenisation + answer-only label masking
    # --------------------------------------------------------------------- #

    def _tokenize_with_answer_mask(
        self,
        prompt_text: str,
        answer_text: str,
    ) -> Optional[tuple]:
        """Tokenise prompt + " " + answer, build labels masking out prompt tokens.

        Strategy
        --------
        1.  Tokenise prompt with ``add_special_tokens=True`` (matches eval).
        2.  Tokenise " " + answer with ``add_special_tokens=False`` so we get
            the answer-only token IDs (and the leading space lets BPE/SP avoid
            merging the answer with the prompt's last token).
        3.  Append eos_token_id to the answer so we supervise an end-of-answer
            token (helps generation stop cleanly at eval).
        4.  Concatenate; build labels = -100 for the prompt portion and =
            input_ids for the answer portion.
        5.  If concatenated length > max_seq_len, left-truncate the prompt
            (we keep the question/post_prompt at the end of the prompt and
            preserve the entire answer span).

        Returns:
            (input_ids, labels, prompt_len, answer_len)
            or ``None`` if the answer alone exceeds max_seq_len (we drop the
            sample rather than pollute training with a label-less example).
        """
        prompt_ids: List[int] = self.tokenizer.encode(
            prompt_text, add_special_tokens=True,
        )
        answer_ids: List[int] = self.tokenizer.encode(
            " " + answer_text, add_special_tokens=False,
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            answer_ids = list(answer_ids) + [int(eos_id)]

        ans_len = len(answer_ids)
        if ans_len <= 0 or ans_len >= self.max_seq_len:
            # Answer alone is too long (or empty) — drop sample.
            return None

        # Left-truncate the prompt if needed so prompt+answer fit max_seq_len.
        max_prompt_len = self.max_seq_len - ans_len
        if len(prompt_ids) > max_prompt_len:
            # Keep the *tail* of the prompt (question + post_prompt + chat
            # template suffix) — the head (instruction + examples + start of
            # context) is the safest to drop because the post_prompt already
            # restates the task.  Without keeping the bos/special, this is OK
            # because we still set add_special_tokens=True above; left-trunc
            # may drop the BOS, which is acceptable for SFT (matches what eval
            # would see for a >max_seq_len sample anyway).
            prompt_ids = prompt_ids[-max_prompt_len:]

        prompt_len = len(prompt_ids)
        full_ids = prompt_ids + answer_ids
        total_len = len(full_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.full((total_len,), -100, dtype=torch.long)
        # Unmask the answer span ONLY.
        labels[prompt_len: prompt_len + ans_len] = input_ids[prompt_len: prompt_len + ans_len]

        return input_ids, labels, prompt_len, ans_len


# --------------------------------------------------------------------------- #
# Collate fn (variable-length, pad-right)
# --------------------------------------------------------------------------- #


def babilong_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Right-pad input_ids with 0 and labels with -100.

    Mirrors ``niah_collate_fn`` in ``niah_dataset.py`` but for BABILong; we
    keep the ``is_niah`` / ``is_babilong`` flags around so the training loop
    can dispatch on the sample type.
    """
    input_ids_list = [s["input_ids"] for s in batch]
    labels_list    = [s["labels"]    for s in batch]
    max_len = max(t.shape[0] for t in input_ids_list)

    padded_ids: List[torch.Tensor] = []
    padded_labels: List[torch.Tensor] = []
    for ids, lbl in zip(input_ids_list, labels_list):
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        padded_ids.append(ids)
        padded_labels.append(lbl)

    out: Dict[str, Any] = {
        "input_ids":   torch.stack(padded_ids, dim=0),
        "labels":      torch.stack(padded_labels, dim=0),
        "is_niah":     False,
        "is_babilong": True,
        "task":        [s.get("task", "") for s in batch],
        "length":      [s.get("length", "") for s in batch],
        "prompt_len":  [s.get("prompt_len", 0) for s in batch],
        "answer_len":  [s.get("answer_len", 0) for s in batch],
    }
    return out
