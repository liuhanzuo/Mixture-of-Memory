#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 task_id 批量下载 Langfuse trace 数据到本地 JSON 文件.

用途：
  伴生评估 / 评测任务开启了 Langfuse 追踪（extra_info.enable_traj=true）后，
  需要把该任务下所有 trace（可能成百上千条）落盘为 JSON 文件，供后续离线分析。

实现方式（重要）：
  直接用 requests 调 Langfuse Public API（Basic Auth: base64(public_key:secret_key)），
  不走 `npx langfuse-cli` 子进程。原因：`npx`/`langfuse-cli` 每次调用都要起一个 Node 子
  进程（即使全局安装了 `langfuse` 命令，仍有进程启动开销），批量下载上千条时这部分固定
  开销会被放大成数十分钟的纯浪费；而 requests.Session 用同一个 HTTP keep-alive 连接，
  纯 Python 函数调用零进程开销，在大批量场景下明显更快。
  接口：
    GET {host}/api/public/traces?tags=<tag>&limit=<limit>&page=<page>  —— 列表
    GET {host}/api/public/traces/{trace_id}                           —— 单条详情

核心能力：
  1. 并发拉取（两个独立并发池，上限固定，不做自适应爬升）：
     - 列表分页：固定 --list-concurrency（默认 6）并发拉页
     - trace 详情：初始/上限并发固定为 --initial-concurrency（默认 16），
       遇到限流/高失败率仍会自动降并发止损，但绝不会爬升超过该上限
  2. 断点续传 —— 每个 trace 落盘为独立文件 <output_dir>/<trace_id>.json，并维护
     <output_dir>/manifest.json 记录每条的下载状态（ok/fail/pending）。脚本可反复
     以相同参数重跑：已成功的（本地文件存在且是合法 JSON）自动跳过，只重试
     pending/fail 的，不会重复消耗时间/触发不必要的请求。
  3. 单条重试 —— 每条下载内部自带最多 3 次退避重试，命中限流信号（HTTP 429）
     会额外等待再重试。
  4. 超大数据量封顶 —— 列表最多拉 --max-pages 页（默认 300 页 × 每页固定 50 条
     = 15000 条上限，每页条数为内部固定值，不对外暴露为参数）。达到上限只打印
     警告，**不阻断、不报错**，按已拉到的部分继续下载；如需完整数据，可用
     --from-timestamp 缩小时间范围分批多次运行，或加大 --max-pages。
  5. 时间范围过滤（--from-timestamp / --to-timestamp，统一用 UTC）—— 传给 Langfuse
     `GET /api/public/traces` 的 `fromTimestamp`/`toTimestamp` 查询参数。Langfuse 按
     task_id 过滤 trace 目前是全表扫描 tags 字段，数据量大时很慢；加上时间范围可以让
     服务端提前收窄扫描范围，明显提速，官方也建议两个参数成对使用（而非只传下界、
     任由上界隐式取"此刻"）。
     - **两个参数最终都会被脚本内部规范化为 UTC（带 Z 后缀）再发给 Langfuse**，
       调用方不需要自己做时区换算：
         · 传入的字符串若已带时区标识（如 "...Z" 或 "...+08:00"）→ 直接换算成 UTC；
         · 传入的字符串若不带时区标识（裸字符串，如太极接口 create_time 字段的格式
           "2026-07-14 22:57:10"）→ **默认假定为北京时间（GMT+8）**（可用
           --assume-tz-offset-hours 覆盖），换算成 UTC 后再用。
       这避免了"裸字符串直接拼 Z 当成 UTC"的常见错误（会把 GMT+8 时间误当 UTC，
       导致 fromTimestamp 偏早 8 小时，扫描范围多出 8 小时冗余）。
     - fromTimestamp 语义是"下界"，只保证不早于该时间的 trace 都在结果里，传的
       时间只要不晚于 trace 真实产生时间即可，不需要精确对齐——直接把任务的
       create_time 原样传入即可，脚本会自动按 GMT+8 换算成 UTC。
     - toTimestamp 若不传，脚本会在启动时自动取当前 UTC 时刻**一次性冻结**作为
       上界，同一次运行内所有分页请求共享这个固定值（而不是每次请求各自取
       "此刻"）。这一点对仍在运行、trace 还在持续产生的任务很重要：如果不冻结，
       每页各自的隐式上界会随请求时刻漂移，可能导致分页结果不一致，也会破坏
       断点续传的确定性（两次运行拼出来的不是同一个时间窗口快照）。

  6. 打包交付（由必填参数 --fs-access 驱动 --archive-format 的默认值）—— 全部
     下载完成后，视调用方环境决定是否把散落的 <trace_id>.json 合并成单个文件，
     避免在网页端 Agent（如 knot）里出现"下载 300 条数据要点 300 次下载按钮"的
     问题；同时避免在能直接访问文件系统的场景里做多余打包（磁盘翻倍 + 多一步解压）。
     - --fs-access（必填，无默认值，取值 direct|download-only）：调用方（Agent）
       必须显式声明自己此刻能否直接访问用户的文件系统查看/使用散文件：
         · direct        —— 能（如本地桌面环境，output_dir 本身用户就能直接打开查看），
                            --archive-format 默认变为 none（不打包，省磁盘、省一步解压）
         · download-only —— 不能（如网页端 Agent，只能靠"生成一个文件供用户点击
                            下载"这种交付方式），--archive-format 默认变为 zip；
                            此模式下脚本会拒绝 --archive-format=none（硬校验，
                            防止误配置导致用户要点 N 次下载按钮）
     - --archive-format（可选，zip|jsonl|none）：显式传入时覆盖上面的默认值，
       但 download-only + none 的非法组合仍会被拒绝并报错退出。
     - zip：把 output_dir 下所有 <trace_id>.json（+ manifest.json）打包成一个 .zip，
       解压后仍是逐条独立 JSON，兼容现有的按 trace_id 查找习惯。
     - jsonl：把所有 trace 详情合并成一个 .jsonl（每行一个 trace 的完整 JSON），
       免解压，适合直接喂给下游离线分析脚本。
     - none：不打包，保留原来的逐条散文件（direct 场景的默认行为）。
     打包产物默认命名 `<output_dir>/../<task_id>_traces.<ext>`（与 output_dir 同级，
     避免归档文件被误认为是待处理的散文件之一）。

用法：
  python3 download_langfuse_traces.py \\
    --task-id 92667 \\
    --output-dir ./langfuse_traces/92667 \\
    --public-key pk-lf-... \\
    --secret-key sk-lf-... \\
    --host http://langfuse-taiji-api.woa.com \\
    --fs-access download-only \\
    --from-timestamp "2026-07-14 22:57:10" \\
    [--to-timestamp "2026-07-15 08:00:00"] \\
    [--assume-tz-offset-hours 8] \\
    [--archive-format zip] \\
    [--initial-concurrency 16] [--min-concurrency 1] [--max-concurrency 16] \\
    [--list-concurrency 6] [--max-pages 300] [--skip-list]

  # --fs-access 必填，Agent 必须先判断自己此刻是否能直接访问用户文件系统：
  #   本地桌面/命令行环境（用户能直接打开 output_dir 看到散文件）→ direct
  #   网页端 Agent/只能靠"生成文件给用户点击下载"的环境           → download-only
  # --from-timestamp 直接传太极接口返回的 create_time 原始字符串即可（GMT+8 裸字符串），
  # 脚本会自动换算成 UTC，不需要手动拼 "T"/"Z" 或做 -8 小时的时区运算。
  # download-only 默认自动打包为 <task_id>_traces.zip，只需交付这一个文件给用户；
  # direct 默认不打包，直接把 output_dir 路径告知用户即可。

  # 中断后 / 想重试失败的，原样重跑同一条命令即可（自动跳过已完成的，--fs-access 仍须传）：
  python3 download_langfuse_traces.py --task-id 92667 --output-dir ./langfuse_traces/92667 \\
    --public-key pk-lf-... --secret-key sk-lf-... --host http://langfuse-taiji-api.woa.com \\
    --fs-access download-only

输出：
  <output_dir>/trace_list_page{N}.json  —— 各页原始列表响应
  <output_dir>/<trace_id>.json—— 每条 trace 的完整详情（散文件，中间产物）
  <output_dir>/manifest.json            —— 下载状态清单（ok/fail/pending + 失败原因）
  <output_dir 同级目录>/<task_id>_traces.zip 或 .jsonl —— --fs-access=download-only（或
    显式指定 --archive-format=zip/jsonl）时的最终交付物（见"打包交付"）；
    --fs-access=direct 且未显式覆盖 --archive-format 时不生成，直接以
    <output_dir> 下的散文件为最终产物

退出码：
  0 —— 全部成功（若指定了打包，同时表示打包成功）
  1 —— 仍有失败条目（manifest.json 里可查看具体是哪些 trace_id 及失败原因）
  （达到 --max-pages 截断上限不算失败，只是 stderr 打印警告，退出码仍按下载结果判定）
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests

DEFAULT_TIMEOUT = 60
LIST_PAGE_LIMIT = 50  # 列表接口每页条数，固定值，不对外暴露为CLI 参数。
                       # 原因：断点续传靠"某页条数 < LIST_PAGE_LIMIT"判断是否为末页
                       # 来做完整性校验；如果这个值可配置且中途被改动，复用磁盘上
                       # 用旧值拉取的缓存页时判断会错乱（旧页条数是按旧 limit 拉的，
                       # 用新 limit 去比较会误判末页/非末页）。固定为常量彻底消除
                       # 这个隐患，也和Langfuse 服务端实际表现一致（不需要调大/调小）。

_TZ_SUFFIX_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def normalize_to_utc_iso(ts_str, assume_offset_hours=8):
    """把用户/接口传入的时间字符串统一规范化为 UTC ISO8601（带毫秒 + Z 后缀）。

    规则：
      - 已带时区标识（结尾是 "Z" 或 "+HH:MM"/"-HH:MM"/"+HHMM"）→ 按其自带时区解析，
        换算成 UTC。
      - 不带时区标识的裸字符串（如太极接口 create_time 字段格式
        "2026-07-14 22:57:10" 或 "2026-07-14T22:57:10"）→ 默认视为
        assume_offset_hours（默认 8，即 GMT+8/北京时间），换算成 UTC 再返回。

    这避免了"裸字符串直接拼 Z 当成 UTC"的常见错误——create_time 后端序列化用的是
    GMT+8，若不做换算直接标 Z，会把时间误判早了 assume_offset_hours 小时。
    """
    if ts_str is None:
        return None
    s = ts_str.strip()
    has_tz = bool(_TZ_SUFFIX_RE.search(s))
    # 统一成 "YYYY-MM-DDTHH:MM:SS[.ffffff][tz]" 供 fromisoformat 解析
    s_norm = s.replace(" ", "T", 1) if "T" not in s else s
    if has_tz:
        s_iso = s_norm[:-1] + "+00:00" if s_norm.endswith("Z") else s_norm
        # 补齐 "+0800" 这种无冒号偏移为 "+08:00"
        m = re.search(r"([+-])(\d{2})(\d{2})$", s_iso)
        if m:
            s_iso = s_iso[: m.start()] + f"{m.group(1)}{m.group(2)}:{m.group(3)}"
        dt = datetime.fromisoformat(s_iso)
    else:
        dt = datetime.fromisoformat(s_norm)
        dt = dt.replace(tzinfo=timezone(timedelta(hours=assume_offset_hours)))
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_session(public_key, secret_key):
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    session = requests.Session()
    session.headers.update({"Authorization": f"Basic {token}"})
    return session


def is_rate_limited(exc, resp):
    if resp is not None and resp.status_code == 429:
        return True
    if isinstance(exc, requests.exceptions.RequestException) and resp is not None:
        return resp.status_code in (429, 503)
    return False


def fetch_one_page(session, host, task_id, page, from_timestamp=None, to_timestamp=None):
    """拉取单页列表，返回 (body_or_None, err_str)。"""
    params = {"tags": f"task_id: {task_id}", "limit": LIST_PAGE_LIMIT, "page": page}
    if from_timestamp:
        params["fromTimestamp"] = from_timestamp
    if to_timestamp:
        params["toTimestamp"] = to_timestamp
    return get_json(session, f"{host.rstrip('/')}/api/public/traces", params=params)


def fetch_trace_list(session, host, task_id, output_dir, max_pages, list_concurrency,
                      from_timestamp=None, to_timestamp=None):
    """分页拉取 trace 列表，每页落盘。

    先顺序读取磁盘上已缓存且完整（该页条数 < LIST_PAGE_LIMIT，说明是最后一页）的页，
    实现断点续传；剩余未缓存的页用 --list-concurrency 并发拉取，每批按页码顺序
    检查，一旦发现某页条数 < LIST_PAGE_LIMIT（到达末页）立即停止，不再继续拉后面的页。
    达到 max_pages 上限时只打印警告，不报错，按已拉到的部分继续往下走。

    to_timestamp 应在调用方（main）里一次性冻结（默认取脚本启动时刻），保证同一次
    运行内所有分页请求共享同一个稳定的时间窗口快照，不随每次请求各自漂移。

    返回按页码排序的原始 JSON 列表（未去重的各页 body）。
    """
    pages = {}

    # ---- 第一步：顺序消费磁盘缓存，找到断点续传的起点 ----
    page = 1
    reached_end = False
    while page <= max_pages:
        out_path = os.path.join(output_dir, f"trace_list_page{page}.json")
        if not os.path.exists(out_path):
            break
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                body = json.load(f)
            except json.JSONDecodeError:
                break  # 文件损坏，从这页重新拉取
        pages[page] = body
        n = count_items_in_page(body)
        print(f"[list] page {page} 已存在，跳过重新拉取（{n} 条）")
        if n < LIST_PAGE_LIMIT:
            reached_end = True
            break
        page += 1
    start_page = page

    # ---- 第二步：并发拉取剩余页（固定 list_concurrency 并发，不做自适应） ----
    while not reached_end and start_page <= max_pages:
        batch = list(range(start_page, min(start_page + list_concurrency, max_pages + 1)))
        results = {}
        with ThreadPoolExecutor(max_workers=list_concurrency) as pool:
            futures = {
                pool.submit(fetch_one_page, session, host, task_id, p, from_timestamp, to_timestamp): p
                for p in batch
            }
            for fut in as_completed(futures):
                p = futures[fut]
                body, err = fut.result()
                if body is None:
                    print(f"[list] page {p} 拉取失败: {err}", file=sys.stderr)
                    continue
                results[p] = body

        # 按页码顺序落盘 + 检查是否到达末页，一旦遇到短页立即停止（丢弃该批次里更靠后的页）
        for p in sorted(results.keys()):
            body = results[p]
            out_path = os.path.join(output_dir, f"trace_list_page{p}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False)
            pages[p] = body
            n = count_items_in_page(body)
            print(f"[list] page {p} 拉取完成，{n} 条")
            if n < LIST_PAGE_LIMIT:
                reached_end = True
                break
        start_page += list_concurrency

    if not reached_end and start_page > max_pages:
        print(f"[list] ⚠️ 已达到 --max-pages={max_pages} 上限（每页 {LIST_PAGE_LIMIT} 条，"
              f"最多 {max_pages * LIST_PAGE_LIMIT} 条），可能仍有更多数据未拉取。"
              f"如需完整数据，可用 --from-timestamp 缩小时间范围分批多次运行，或加大 --max-pages。",
              file=sys.stderr)

    return [pages[p] for p in sorted(pages.keys())]


def get_json(session, url, params=None, retries=5, timeout=DEFAULT_TIMEOUT):
    """单次 GET 请求，内部自带退避重试。返回 (json_or_None, error_str)。"""
    last_err = ""
    for attempt in range(1, retries + 1):
        resp = None
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json(), ""
        except requests.exceptions.RequestException as e:
            last_err = str(e)[:300]
            rate_limited = is_rate_limited(e, resp)
            if attempt < retries:
                time.sleep(2 * attempt + (3 if rate_limited else 0))
            else:
                return None, last_err
        except json.JSONDecodeError as e:
            last_err = f"invalid json response: {e}"
            if attempt < retries:
                time.sleep(2 * attempt)
            else:
                return None, last_err
    return None, last_err


def count_items_in_page(body):
    return len(extract_ids_from_page(body))


def extract_ids_from_page(body):
    """从一页列表响应里提取 trace id 列表，兼容几种常见的包装结构。"""
    candidates = []
    if isinstance(body, dict):
        for path in (
            ("data",),
            ("body", "data"),
            ("data", "data"),
            ("traces",),
            ("items",),
        ):
            node = body
            ok = True
            for key in path:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    ok = False
                    break
            if ok and isinstance(node, list):
                candidates = node
                break
    ids = []
    for item in candidates:
        if isinstance(item, dict) and "id" in item:
            ids.append(item["id"])
    return ids


def load_manifest(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_manifest(path, manifest):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def download_one_trace(session, host, trace_id, output_dir):
    out_path = os.path.join(output_dir, f"{trace_id}.json")
    url = f"{host.rstrip('/')}/api/public/traces/{trace_id}"
    body, err = get_json(session, url)
    if body is None:
        rate_limited = "429" in err or "503" in err
        return False, rate_limited, err
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    return True, False, ""


def archive_as_zip(output_dir, trace_ids, task_id, dest_path, include_manifest=True):
    """把output_dir 下所有已成功下载的 <trace_id>.json 打包成一个 zip。

    zip 内部结构：
      traces/<trace_id>.json   —— 逐条 trace 详情（保留原始文件名，解压后可按id 查找）
      manifest.json            —— 下载状态清单（可选）

    只打包实际存在且能被解析为合法 JSON 的文件，跳过缺失/损坏的（不阻断打包流程）。
    返回 (zip 内文件数, 跳过数)。
    """
    packed = 0
    skipped = 0
    with zipfile.ZipFile(dest_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for tid in trace_ids:
            src = os.path.join(output_dir, f"{tid}.json")
            if not os.path.exists(src):
                skipped += 1
                continue
            zf.write(src, arcname=f"traces/{tid}.json")
            packed += 1
        if include_manifest:
            manifest_path = os.path.join(output_dir, "manifest.json")
            if os.path.exists(manifest_path):
                zf.write(manifest_path, arcname="manifest.json")
    return packed, skipped


def archive_as_jsonl(output_dir, trace_ids, task_id, dest_path):
    """把 output_dir 下所有已成功下载的 <trace_id>.json 合并成一个 .jsonl（每行一条完整 trace）。

    免解压、可直接被下游脚本按行流式处理，适合离线分析场景。
    只写入实际存在且能被解析为合法 JSON 的文件，跳过缺失/损坏的（不阻断打包流程）。
    返回 (写入行数, 跳过数)。
    """
    packed = 0
    skipped = 0
    with open(dest_path, "w", encoding="utf-8") as out_f:
        for tid in trace_ids:
            src = os.path.join(output_dir, f"{tid}.json")
            if not os.path.exists(src):
                skipped += 1
                continue
            try:
                with open(src, "r", encoding="utf-8") as f:
                    body = json.load(f)
            except (json.JSONDecodeError, OSError):
                skipped += 1
                continue
            out_f.write(json.dumps(body, ensure_ascii=False))
            out_f.write("\n")
            packed += 1
    return packed, skipped


def finalize_and_archive(args, manifest, trace_ids, total):
    """下载流程结束后统一收尾：打印失败退出码判定 + 按 --archive-format 打包成单文件交付。

    archive_format 的默认值已在 main() 里根据 --fs-access 推导好，这里只按最终
    取值执行，不再关心 fs_access 本身。

    返回最终退出码（0=全部成功且打包成功；1=仍有下载失败或打包失败）。
    """
    final_fail = [tid for tid, m in manifest.items() if m.get("status") == "fail"]
    exit_code = 1 if final_fail else 0

    if args.archive_format == "none":
        abs_output_dir = os.path.abspath(args.output_dir)
        if not final_fail:
            print(f"[archive] --fs-access={args.fs_access}，不打包，最终产物是以下目录下的 "
                  f"{total} 个散文件（<trace_id>.json）：")
        else:
            print(f"[archive] --fs-access={args.fs_access}，不打包；{len(final_fail)} 条下载失败，"
                  f"其余成功的散文件（<trace_id>.json）在以下目录：")
        print(f"[archive] 下载路径：{abs_output_dir}")
        return exit_code

    ext = "zip" if args.archive_format == "zip" else "jsonl"
    dest_path = args.archive_path or os.path.join(
        os.path.dirname(os.path.abspath(args.output_dir.rstrip("/"))) or ".",
        f"{args.task_id}_traces.{ext}",
    )

    if args.archive_format == "zip":
        packed, skipped = archive_as_zip(args.output_dir, trace_ids, args.task_id, dest_path)
    else:
        packed, skipped = archive_as_jsonl(args.output_dir, trace_ids, args.task_id, dest_path)

    if packed == 0:
        print(f"[archive] ⚠️ 打包失败：没有任何可打包的 trace 文件（{skipped} 条缺失/损坏），"
              f"未生成 {os.path.abspath(dest_path)}", file=sys.stderr)
        return 1

    abs_dest_path = os.path.abspath(dest_path)
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"\n[archive] ✅ 已打包 {packed} 条 trace 到单个文件：{abs_dest_path}（{size_mb:.1f} MB）"
          + (f"，{skipped} 条因缺失/损坏被跳过" if skipped else ""))
    print(f"[archive] 下载路径：{abs_dest_path}")
    print(f"[archive] 只需交付这一个文件（{ext.upper()}），不需要逐条下载 {os.path.abspath(args.output_dir)} 下的散文件")
    return exit_code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-id", required=True, help="评估任务 task_id")
    ap.add_argument("--output-dir", required=True, help="下载输出目录")
    ap.add_argument("--public-key", required=True, help="Langfuse public key")
    ap.add_argument("--secret-key", required=True, help="Langfuse secret key")
    ap.add_argument("--host", required=True, help="Langfuse host，如 http://langfuse-taiji-api.woa.com")
    ap.add_argument("--from-timestamp", default=None,
                     help="可选，时间下界，用于收窄服务端扫描范围提速。可直接传太极接口 create_time"
                          "的原始字符串（如 '2026-07-14 22:57:10'，裸字符串默认按 GMT+8 解释）或"
                          "带时区的 ISO 8601（如 '2026-07-14T22:57:10Z'）——脚本会统一换算成 UTC"
                          "再传给 Langfuse，不需要手动做时区转换。建议直传任务的 create_time。")
    ap.add_argument("--to-timestamp", default=None,
                     help="可选，时间上界，格式规则与 --from-timestamp 相同（裸字符串默认按 GMT+8"
                          "解释，脚本统一换算成 UTC）。不传则在脚本启动时自动取当前 UTC 时刻并一次性"
                          "冻结（不是每次请求各自取'此刻'），保证同一次运行内所有分页请求共享同一个"
                          "稳定的时间窗口快照，避免仍在运行中的任务因持续产生新 trace 导致分页结果漂移。")
    ap.add_argument("--assume-tz-offset-hours", type=float, default=8,
                     help="--from-timestamp/--to-timestamp 传入不带时区标识的裸字符串时，假定其"
                          "所属时区相对 UTC 的偏移小时数，默认 8（对应太极接口 create_time 用的"
                          "GMT+8/北京时间）。带时区标识的字符串忽略此参数，按自带时区解析。")
    ap.add_argument("--initial-concurrency", type=int, default=16, help="trace 详情下载的初始并发数，默认 16")
    ap.add_argument("--min-concurrency", type=int, default=1, help="trace 详情下载的最低并发数，默认 1")
    ap.add_argument("--max-concurrency", type=int, default=16,
                     help="trace 详情下载并发数的硬上限（自适应降并发后可爬升，但不超过该值），默认 16")
    ap.add_argument("--list-concurrency", type=int, default=6, help="列表分页拉取的固定并发数，默认 6")
    ap.add_argument("--max-pages", type=int, default=300,
                     help=f"列表最多拉取的页数上限，默认 300（配合固定每页 {LIST_PAGE_LIMIT} 条，"
                          f"即最多 {300 * LIST_PAGE_LIMIT} 条）。达到上限只打印警告不报错，按已拉到的部分继续下载。")
    ap.add_argument("--fail-rate-threshold", type=float, default=0.15,
                     help="单批次失败率超过该阈值即降并发，默认 0.15（15%%）")
    ap.add_argument("--skip-list", action="store_true", help="跳过拉列表阶段，直接用 output_dir 里已有的 trace_list_page*.json")
    ap.add_argument("--fs-access", required=True, choices=["direct", "download-only"],
                     help="必填。调用方（Agent）此刻能否直接访问用户的文件系统："
                          "direct=能（如本地桌面环境，output_dir 本身用户就能直接打开查看），"
                          "--archive-format 默认变为 none；"
                          "download-only=不能（如网页端 Agent，只能靠生成文件给用户点击下载），"
                          "--archive-format 默认变为 zip，且禁止显式指定 none。"
                          "这个判断由调用方自身对所处环境的认知决定，脚本不做任何环境探测。")
    ap.add_argument("--archive-format", choices=["zip", "jsonl", "none"], default=None,
                     help="下载完成后的打包格式。不传则按 --fs-access 推导默认值"
                          "（direct→none，download-only→zip）。显式传入时覆盖推导值，"
                          "但 download-only 下不允许显式传none（会报错退出）。"
                          "zip=打包成单个 .zip（解压后仍是逐条 JSON）；"
                          "jsonl=合并成单个 .jsonl（每行一条，免解压）；"
                          "none=不打包，保留 output_dir 下的散文件。")
    ap.add_argument("--archive-path", default=None,
                     help="打包产物的输出路径，默认 <output_dir 的父目录>/<task_id>_traces.<ext>"
                          "（与 output_dir 同级，避免和散文件混在一起）。--archive-format=none 时忽略。")
    args = ap.parse_args()

    # --archive-format 未显式传入时，按 --fs-access 推导默认值：
    #   direct（能直接访问文件系统）→ none（不打包，省磁盘、省一步解压）
    #   download-only（只能靠生成文件给用户下载）→ zip（必须打包成单文件交付）
    if args.archive_format is None:
        args.archive_format = "none" if args.fs_access == "direct" else "zip"

    # 硬校验：download-only 场景下不允许打包成 none，否则会退化成"让用户点 N 次
    # 下载按钮"的问题，这是本参数存在的核心目的，不能被误配置绕过。
    if args.fs_access == "download-only" and args.archive_format == "none":
        print("[error] --fs-access=download-only 时不允许 --archive-format=none："
              "该组合意味着 Agent 只能生成文件给用户点击下载，却选择不打包成单文件，"
              "会导致用户需要逐条点击下载 output_dir 下的散文件。请改用 zip 或 jsonl，"
              "或者如果调用方确实能直接访问用户文件系统，应改传 --fs-access=direct。",
              file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.output_dir, exist_ok=True)
    session = build_session(args.public_key, args.secret_key)

    # 统一把 from/to timestamp 规范化为 UTC ISO8601：
    # - 传入值已带时区标识 → 按其自带时区换算成 UTC
    # - 传入值是裸字符串（无时区标识，如太极 create_time 原始格式）→ 按
    #   --assume-tz-offset-hours（默认 8，GMT+8）解释后换算成 UTC
    # - --to-timestamp 未传 → 在此一次性冻结当前 UTC 时刻，保证同一次运行内所有
    #   分页请求共享同一个稳定的时间窗口快照，不随每次请求各自漂移。
    from_timestamp = normalize_to_utc_iso(args.from_timestamp, args.assume_tz_offset_hours)
    to_timestamp = normalize_to_utc_iso(args.to_timestamp, args.assume_tz_offset_hours) \
        if args.to_timestamp else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    print(f"[init] 直连 Langfuse API（requests + Basic Auth），host={args.host}"
          + (f"，from_timestamp={from_timestamp}(UTC，原始输入={args.from_timestamp})" if from_timestamp else "")
          + f"，to_timestamp={to_timestamp}(UTC)"
          + (" (脚本启动时刻自动冻结)" if not args.to_timestamp else f"，原始输入={args.to_timestamp}"))

    # ---- Step 1: 拉列表 ----
    if args.skip_list:
        pages = []
        page = 1
        while True:
            fpath = os.path.join(args.output_dir, f"trace_list_page{page}.json")
            if not os.path.exists(fpath):
                break
            with open(fpath, "r", encoding="utf-8") as f:
                pages.append(json.load(f))
            page += 1
        print(f"[list] --skip-list 模式，读取到已有 {page - 1} 页")
    else:
        pages = fetch_trace_list(
            session, args.host, args.task_id, args.output_dir,
            args.max_pages, args.list_concurrency,
            from_timestamp=from_timestamp, to_timestamp=to_timestamp,
        )

    trace_ids = []
    for p in pages:
        trace_ids.extend(extract_ids_from_page(p))
    trace_ids = list(dict.fromkeys(trace_ids))  # 去重且保序
    total = len(trace_ids)
    print(f"[list] 共 {total} 条 trace 待下载")
    if total == 0:
        print("[list] 未提取到任何 trace_id，请检查列表响应结构或 --tags 过滤条件是否正确", file=sys.stderr)
        sys.exit(1)

    # ---- Step 2: 断点续传初始化 manifest ----
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = load_manifest(manifest_path)
    for tid in trace_ids:
        out_path = os.path.join(args.output_dir, f"{tid}.json")
        if tid in manifest and manifest[tid].get("status") == "ok" and os.path.exists(out_path):
            continue  # 已确认成功，跳过
        if os.path.exists(out_path):
            # 文件存在但 manifest 未记录 ok，校验一下是否是合法 JSON（兼容之前非本脚本下载的产物）
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    json.load(f)
                manifest[tid] = {"status": "ok", "attempts": manifest.get(tid, {}).get("attempts", 0)}
                continue
            except (json.JSONDecodeError, OSError):
                pass
        manifest.setdefault(tid, {"status": "pending", "attempts": 0})
    save_manifest(manifest_path, manifest)

    pending = [tid for tid in trace_ids if manifest[tid]["status"] != "ok"]
    skipped = total - len(pending)
    if skipped:
        print(f"[resume] 断点续传：{skipped} 条已存在且校验通过，跳过；剩余 {len(pending)} 条需下载")
    if not pending:
        print(f"[done] 全部 {total} 条均已下载完成，无需处理")
        sys.exit(finalize_and_archive(args, manifest, trace_ids, total))

    # ---- Step 3: 自适应并发下载 ----
    concurrency = max(args.min_concurrency, min(args.initial_concurrency, args.max_concurrency))
    idx = 0
    done_count = skipped
    while idx < len(pending):
        batch = pending[idx: idx + concurrency]
        batch_fail = 0
        batch_rate_limited = False
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(download_one_trace, session, args.host, tid, args.output_dir): tid for tid in batch}
            for fut in as_completed(futures):
                tid = futures[fut]
                ok, rate_limited, err = fut.result()
                manifest[tid]["attempts"] = manifest[tid].get("attempts", 0) + 1
                if ok:
                    manifest[tid]["status"] = "ok"
                    manifest[tid].pop("error", None)
                else:
                    manifest[tid]["status"] = "fail"
                    manifest[tid]["error"] = err
                    batch_fail += 1
                    if rate_limited:
                        batch_rate_limited = True
        done_count += len(batch)
        save_manifest(manifest_path, manifest)

        fail_rate = batch_fail / len(batch) if batch else 0
        print(f"[download] 批次完成：{len(batch)} 条（并发={concurrency}），"
              f"失败 {batch_fail} 条（{fail_rate:.0%}），累计进度 {done_count}/{total}")

        if batch_rate_limited or fail_rate > args.fail_rate_threshold:
            new_concurrency = max(args.min_concurrency, concurrency // 2)
            if new_concurrency != concurrency:
                print(f"[adaptive] 检测到{'限流' if batch_rate_limited else '失败率过高'}，"
                      f"并发数 {concurrency} → {new_concurrency}")
                concurrency = new_concurrency
            print("[adaptive] 退避等待 5s 再继续")
            time.sleep(5)
        elif fail_rate == 0 and concurrency < min(args.initial_concurrency, args.max_concurrency):
            new_concurrency = min(concurrency + 2, args.initial_concurrency, args.max_concurrency)
            if new_concurrency != concurrency:
                print(f"[adaptive] 批次全部成功，并发数 {concurrency} → {new_concurrency}（谨慎爬升）")
                concurrency = new_concurrency

        idx += len(batch)

    # ---- Step 4: 汇总 + 打包交付 ----
    final_fail = [tid for tid, m in manifest.items() if m["status"] == "fail"]
    final_ok = total - len(final_fail)
    print(f"\n[summary] 完成：{final_ok}/{total} 成功，{len(final_fail)} 失败")
    if final_fail:
        print(f"[summary] 失败的 trace_id（详情见 {manifest_path}）：{final_fail[:20]}"
              f"{' ...' if len(final_fail) > 20 else ''}")
        print("[summary] 重跑同一条命令即可只重试失败的（已成功的会被跳过）")
    sys.exit(finalize_and_archive(args, manifest, trace_ids, total))


if __name__ == "__main__":
    main()
