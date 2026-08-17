#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from timeit import default_timer
from typing import Optional

import requests
import tqdm
from pydantic import BaseModel



APP_ID = os.environ.get('APP_ID')  
APP_KEY = os.environ.get('APP_KEY')  
MODEL_NAME = os.environ.get('MODEL_NAME', 'api_azure_openai_gpt-5.4-2026-03-05')  


# ── 参数封装 ───────────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    app_id: str = APP_ID 
    app_key: str = APP_KEY
    base_url: str = "http://llm-api.model-eval.woa.com"
    model: str = MODEL_NAME
    timeout: int = 1200
    request_timeout: int = 600
    max_tokens: int = 128000
    reasoning_effort: str = "high"
    max_retries: int = 5
    retry_base_sleep: float = 2.0

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    @property
    def headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.app_id}:{self.app_key}",
        }


# ── 线程安全工具 ───────────────────────────────────────────────────────

_print_lock = threading.Lock()
_write_lock = threading.Lock()


def safe_print(message: str, *args, **kwargs):
    with _print_lock:
        print(message, *args, **kwargs)


def safe_write(fh, text: str):
    with _write_lock:
        fh.write(text)
        fh.flush()


# ── 核心调用（带重试 + 指数退避 + 异常捕获）──────────────────────────

def call_host_model(prompt: str, config: LLMConfig) -> tuple:
    """
    返回 (status_code, response_json | None)。
    内部最多 config.max_retries 次重试，每次失败后指数退避。
    """
    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "timeout": config.timeout,
        "max_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort,
    }

    last_exc = None
    for attempt in range(1, config.max_retries + 1):
        try:
            r = requests.post(
                config.url,
                headers=config.headers,
                json=body,
                timeout=config.request_timeout,
            )
            status = r.status_code
            data = r.json() if r.text else None

            if status == 200 and data and data.get("choices"):
                return status, data

            safe_print(
                f"[attempt {attempt}/{config.max_retries}] "
                f"status={status}, body={r.text[:200]}",
                file=sys.stderr,
            )
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            safe_print(
                f"[attempt {attempt}/{config.max_retries}] timeout: {exc}",
                file=sys.stderr,
            )
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            safe_print(
                f"[attempt {attempt}/{config.max_retries}] connection error: {exc}",
                file=sys.stderr,
            )
        except Exception as exc:
            last_exc = exc
            safe_print(
                f"[attempt {attempt}/{config.max_retries}] unexpected error: {exc}",
                file=sys.stderr,
            )

        if attempt < config.max_retries:
            sleep_sec = config.retry_base_sleep * (2 ** (attempt - 1))
            time.sleep(sleep_sec)

    safe_print(
        f"[FAILED] all {config.max_retries} attempts exhausted. last_exc={last_exc}",
        file=sys.stderr,
    )
    return -1, None


# ── MD5 去重工具 ──────────────────────────────────────────────────────

def calc_md5(text: str) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ── Worker（线程池任务单元）─────────────────────────────────────────

def worker(sample: dict, prompt_key: str, config: LLMConfig, output_fh) -> bool:
    """处理单条样本，成功返回 True。"""
    prompt = sample.get(prompt_key, "")
    if not prompt:
        safe_print("[WARN] empty prompt, skip", file=sys.stderr)
        return False

    start = default_timer()
    status, data = call_host_model(prompt, config)

    if status == 200 and data and data.get("choices"):
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            safe_print(f"[ERROR] parse response failed: {exc}", file=sys.stderr)
            return False

        sample["server_response"] = answer
        sample["cost"] = round(default_timer() - start, 3)
        out_line = json.dumps(sample, ensure_ascii=False) + "\n"
        safe_write(output_fh, out_line)
        return True

    sample["server_response"] = None
    return False


# ── CLI 参数定义 ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPT-5 distill caller with concurrency")
    p.add_argument("--input_file", required=True, help="输入 jsonl 文件路径")
    p.add_argument("--output_file", required=True, help="输出 jsonl 文件路径")
    p.add_argument("--prompt_key", default="prompt", help="jsonl 中 prompt 字段名")
    p.add_argument("--model", default="api_azure_openai_gpt-5.4-2026-03-05",
                    help="模型名称")
    p.add_argument("--num_jobs", type=int, default=5, help="并发线程数")
    p.add_argument("--max_retries", type=int, default=5, help="单请求最大重试次数")
    p.add_argument("--retry_base_sleep", type=float, default=2.0,
                    help="重试基础睡眠秒数（指数退避）")
    p.add_argument("--max_tokens", type=int, default=128000)
    p.add_argument("--reasoning_effort", default="high",
                    choices=["low", "medium", "high"])
    p.add_argument("--request_timeout", type=int, default=600,
                    help="单次 HTTP 超时(秒)")
    p.add_argument("--is_continue", action="store_true", default=False,
                    help="断点续跑：跳过 output 中已完成的样本")
    p.add_argument("--tqdm", action="store_true", default=False, help="显示进度条")
    p.add_argument("--task_retry_limit", type=int, default=3,
                    help="全局失败样本重试轮次上限")
    return p


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    config = LLMConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        max_retries=args.max_retries,
        retry_base_sleep=args.retry_base_sleep,
        request_timeout=args.request_timeout,
    )

    # ── 断点续跑：收集已完成样本的 prompt md5 ──
    processed_ids: set = set()
    if args.is_continue and os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    processed_ids.add(calc_md5(d.get(args.prompt_key, "")))
                except json.JSONDecodeError:
                    pass
        safe_print(
            f"[INFO] continue mode: {len(processed_ids)} samples already done",
            file=sys.stderr,
        )

    # ── 读取输入，过滤已处理 ──
    all_samples = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            pid = calc_md5(sample.get(args.prompt_key, ""))
            if pid not in processed_ids:
                all_samples.append(sample)

    safe_print(
        f"[INFO] {len(all_samples)} samples to process, "
        f"{len(processed_ids)} skipped (already done)",
        file=sys.stderr,
    )

    if not all_samples:
        safe_print("[INFO] nothing to do, exit.", file=sys.stderr)
        return

    output_fh = open(
        args.output_file,
        "a" if args.is_continue else "w",
        encoding="utf-8",
    )

    # ── 线程池并发处理，支持多轮重试失败样本 ──
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.num_jobs)
    remain_samples = all_samples
    success_total = 0

    for round_idx in range(1, args.task_retry_limit + 1):
        if not remain_samples:
            break

        safe_print(
            f"[INFO] round {round_idx}: {len(remain_samples)} samples",
            file=sys.stderr,
        )

        futures = {
            executor.submit(worker, sample, args.prompt_key, config, output_fh): sample
            for sample in remain_samples
        }

        success_count = 0
        fail_count = 0
        failed_samples = []

        pbar = tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"round-{round_idx}",
            disable=not args.tqdm,
        )
        for future in pbar:
            sample = futures[future]
            try:
                ok = future.result()
            except Exception as exc:
                safe_print(
                    f"[ERROR] worker exception: {exc}\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                ok = False

            if ok:
                success_count += 1
            else:
                failed_samples.append(sample)
                fail_count += 1

            if args.tqdm:
                rate = success_count / max(success_count + fail_count, 1)
                pbar.set_postfix(success=success_count, rate=f"{rate:.1%}")

        success_total += success_count
        safe_print(
            f"[INFO] round {round_idx} done: "
            f"success={success_count}, failed={fail_count}",
            file=sys.stderr,
        )

        if not failed_samples:
            break
        remain_samples = failed_samples

    output_fh.close()
    executor.shutdown(wait=True)

    total = len(all_samples)
    safe_print(
        f"[SUMMARY] {success_total}/{total} succeeded "
        f"({success_total / max(total, 1):.1%}), "
        f"{total - success_total} failed",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
