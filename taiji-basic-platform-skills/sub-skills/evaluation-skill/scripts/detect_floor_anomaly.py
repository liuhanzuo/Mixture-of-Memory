#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一下限异常检测脚本 v1

对评测任务的所有 bench 文件执行三大类下限检测：
  1. TOOL_CALL_ANOMALY  — tool call 格式异常（7 种子类型）【仅 Agent 类 Bench】
  2. RED_LINE_PROB      — 红线问题（REPEAT / UNREADABLE_CHAR / MARKDOWN_ERR）【全部 Bench】
  3. THINK_TAG_ILLEGAL  — think 标签异常（6 种子类型）【全部 Bench】
     对齐 analyze_think_or_answer_pattern_stage1 的检测逻辑：
       A. EMPTY_THINK        — think模式下 reasoning_content 为空
       B. MISSING_</think>   — think模式下缺少闭合标签(response为空)
       C. REDUNDANT_</think> — response中残留</think>(存在多个)
       D. REDUNDANT_<think>  — reasoning/response中残留<think>(存在多个)
       E. EMPTY_RESP         — 模型RESPONSE为空但finish_reason=stop
       + NOTHINK_HAS_<think> — nothink模式下出现think标签
     前置过滤: finish_reason=length 时跳过检测(截断场景)
     标签变体兼容: <think:6124c78e> / </think:6124c78e>

输入：评测数据目录（含 *.jsonl.gz 文件 + trajectory/ 子目录）
输出：
  1. floor_anomaly_metrics.json — 指标汇总
  2. {output_dir}/{sub_type}.jsonl — 每种子类型的 badcase 集合

用法：
  python3 detect_floor_anomaly.py \\
    --data_dir /path/to/1403 \\
    --output_dir /data/home/bettyleihe/1403/anomaly_cases \\
    --task_id 27491 \\
    [--debug]                    # debug 模式每个 bench 只跑前 100 条
    [--detect_types TYPE1,TYPE2] # 指定检测类型，默认 ALL
                                 # 可选: TOOL_CALL_ANOMALY,RED_LINE_PROB,THINK_TAG_ILLEGAL,ALL
    [--bench_name NAME1,NAME2]   # 指定 bench 名称过滤，默认 ALL
                                 # 示例: CLBench,prbench_finance
"""

import argparse
import gzip
import json
import os
import re
import io
import sys
from collections import defaultdict, Counter
from pathlib import Path

def _ensure_utf8_io():
    if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '') != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer') and getattr(sys.stderr, 'encoding', '') != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import zstandard
    HAS_ZST = True
except ImportError:
    HAS_ZST = False
    print("[WARN] zstandard not installed, agent trajectory parsing will be skipped", file=sys.stderr)


# ============================================================
# Agent Bench 定义
# ============================================================

AGENT_BENCH_PREFIXES = [
    "swe_bench_verified",
    "swe_bench_multilingual",
    "swe_bench_pro",
    "terminal_bench_2_0",
    "hyeval_browsecomp-zh",
    "hyeval_widesearch",
    "hyeval_finsearchcomp-t2_and_t3",
    "hyeval_seal-0",
    "webarena",
    "hyeval_browsecomp-subset-150",
    "tau2",
]


def is_agent_bench(filename):
    """判断文件是否为 Agent 类 Bench"""
    basename = os.path.basename(filename)
    return any(basename.startswith(p) for p in AGENT_BENCH_PREFIXES)


# ============================================================
# 通用工具函数（对齐 detect_toolcall_anomaly.py / detect_floor.py）
# ============================================================

def first_text(field):
    """从 [[str, ...], ...] 嵌套结构中取第一个非空字符串。"""
    if not field or not isinstance(field, list):
        return ""
    for group in field:
        if isinstance(group, list):
            for x in group:
                if x and str(x).strip():
                    return str(x)
        elif group and str(group).strip():
            return str(group)
    return ""


def extract_finish_reason(payload):
    """从 usage 字段提取 finish_reason（取第一个 pass）。"""
    usages = payload.get("usage", [])
    if not usages:
        return ""
    u = usages[0]
    if isinstance(u, list):
        u = u[0] if u else {}
    if not isinstance(u, dict):
        return ""
    return (u.get("finish_reason") or "").lower()


_SKIP_FINISH_REASONS = frozenset({"length", "content_filter", "safety"})


# ============================================================
# TOOL_CALL_ANOMALY 检测
# ============================================================

_RE_TOOL_CALLS_CLOSE_AT_END = re.compile(r'</tool_calls(?::[0-9a-fA-F]+)?>\s*$', re.IGNORECASE)
_RE_TOOL_SEP = re.compile(r'<tool_sep(?::[0-9a-fA-F]+)?>', re.IGNORECASE)

_TOOLCALL_PAIR_PATTERNS = [
    re.compile(r'<tool_call(?::[0-9a-fA-F]+)?>.*?</tool_call(?::[0-9a-fA-F]+)?>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<arg_key(?::[0-9a-fA-F]+)?>.*?</arg_key(?::[0-9a-fA-F]+)?>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<arg_value(?::[0-9a-fA-F]+)?>.*?</arg_value(?::[0-9a-fA-F]+)?>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<function=[^>]+>.*?</function>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<parameter=[^>]+>.*?</parameter>', re.IGNORECASE | re.DOTALL),
]


def _contains_toolcall_markup(text):
    """检测文本中是否包含 tool call 原始标记。"""
    if not text:
        return False
    if _RE_TOOL_CALLS_CLOSE_AT_END.search(text):
        return True
    if _RE_TOOL_SEP.search(text):
        return True
    return any(p.search(text) for p in _TOOLCALL_PAIR_PATTERNS)


def extract_tool_names_from_tools(tools):
    """从 tools 定义列表中提取可用工具名称集合。"""
    if not isinstance(tools, list) or not tools:
        return None
    names = set()
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function") or {}
            name = fn.get("name")
            if name:
                names.add(name)
    return names or None


def extract_tool_calls(payload):
    """从 payload 中提取 tool_calls 列表。兼容多种格式。"""
    tc = payload.get("tool_calls")
    if isinstance(tc, list) and tc:
        return tc

    responses = payload.get("responses", [])
    if isinstance(responses, list):
        for group in responses:
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, dict) and "tool_calls" in item:
                        return item["tool_calls"]
            elif isinstance(group, dict) and "tool_calls" in group:
                return group["tool_calls"]
    return []


def extract_tool_choice(payload):
    """提取 tool_choice 设置。"""
    tc = payload.get("tool_choice")
    if tc is not None:
        if isinstance(tc, str):
            return tc.lower()
        if isinstance(tc, dict):
            return "specific"
    return ""


def detect_toolcall_anomalies_single(content, reasoning, tool_calls, finish_reason, tools, tool_choice):
    """对单次 LLM 调用执行全部 7 类 tool call 异常检测。
    返回 list of (subtype, desc, detail)
    完整对齐 detect_toolcall_anomaly.py 的 detect_toolcall_anomalies()
    """
    anomalies = []
    available_tools = extract_tool_names_from_tools(tools)
    has_tools_defined = available_tools is not None

    if finish_reason in _SKIP_FINISH_REASONS:
        return anomalies

    # 1. CONTENT_LEAKAGE -> TOOL_CALL_LEAKAGE
    leaked_in = []
    if _contains_toolcall_markup(content):
        leaked_in.append("content")
    if _contains_toolcall_markup(reasoning):
        leaked_in.append("reasoning")
    if leaked_in:
        snippet = ""
        text_to_check = content if "content" in leaked_in else reasoning
        for pat in [_RE_TOOL_CALLS_CLOSE_AT_END, _RE_TOOL_SEP] + _TOOLCALL_PAIR_PATTERNS:
            m = pat.search(text_to_check)
            if m:
                start = max(0, m.start() - 30)
                end = min(len(text_to_check), m.end() + 30)
                snippet = text_to_check[start:end]
                break
        anomalies.append(("TOOL_CALL_LEAKAGE",
                          f"content/reasoning 中泄漏 tool call 标记 (in: {leaked_in})",
                          {"leaked_in": leaked_in, "snippet": snippet[:200]}))

    # 2. REASONING_ONLY
    if reasoning and not content.strip() and not tool_calls:
        anomalies.append(("REASONING_ONLY",
                          f"只有推理内容(len={len(reasoning)}), 无 content 也无 tool_calls(可能缺think结束符)",
                          {"reasoning_len": len(reasoning)}))

    # 3. TOOL_CALLS_FIELD_EMPTY
    if finish_reason == "tool_calls" and not tool_calls:
        anomalies.append(("TOOL_CALLS_FIELD_EMPTY",
                          "finish_reason=tool_calls 但 tool_calls 为空",
                          {}))

    # 4. STOP_WITHOUT_TOOL_CALL — 仅 tool_choice=required 时触发
    if finish_reason == "stop" and not tool_calls and has_tools_defined:
        if tool_choice == "required":
            anomalies.append(("STOP_WITHOUT_TOOL_CALL",
                              f"有 tools 且 tool_choice=required 但 stop 时未调用",
                              {"tool_choice": "required", "confirmed": True}))

    # 5. TOOL_CALL_DESPITE_NONE
    if tool_calls and tool_choice == "none":
        anomalies.append(("TOOL_CALL_DESPITE_NONE",
                          f"tool_choice=none 但模型仍调用了 {len(tool_calls)} 个工具",
                          {"n_tool_calls": len(tool_calls)}))

    # 6. JSON_PARSE_ERROR
    json_errors = []
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") or tc
        arguments = fn.get("arguments")
        if arguments is not None and isinstance(arguments, str):
            try:
                json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                json_errors.append({
                    "index": i,
                    "function_name": fn.get("name", "?"),
                    "arguments_snippet": arguments[:200],
                })
    if json_errors:
        anomalies.append(("JSON_PARSE_ERROR",
                          f"tool_calls arguments JSON 解析失败: {len(json_errors)}处",
                          {"count": len(json_errors), "details": json_errors[:5]}))

    # 7. HALLUCINATED_TOOL
    if available_tools:
        hallucinated = []
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function") or tc
            name = fn.get("name")
            if name and name not in available_tools:
                hallucinated.append({
                    "index": i,
                    "called": name,
                    "available": sorted(available_tools),
                })
        if hallucinated:
            anomalies.append(("HALLUCINATED_TOOL",
                              f"调用了不在 tools 列表中的函数: {[h['called'] for h in hallucinated[:3]]}",
                              {"count": len(hallucinated), "details": hallucinated[:5]}))

    return anomalies


# ============================================================
# RED_LINE_PROB — REPEAT 检测（完整对齐 detect_floor.py）
# ============================================================

REPEAT_UNIT_MIN = 2
REPEAT_UNIT_MAX = 20
REPEAT_SCAN_MAX_CHARS = 120000

# finish_reason=length 时使用低阈值（典型的复读导致截断）
REPEAT_MIN_RUN_LENGTH = 8
# finish_reason!=length 时使用高阈值（避免正常文本短重复被误判）
REPEAT_MIN_RUN_NORMAL = 50

_REPEAT_PATTERNS_CACHE = {}


def _get_repeat_patterns(min_run):
    if min_run in _REPEAT_PATTERNS_CACHE:
        return _REPEAT_PATTERNS_CACHE[min_run]
    pats = []
    for n in range(REPEAT_UNIT_MIN, REPEAT_UNIT_MAX + 1):
        p = re.compile(r"(.{%d})\1{%d,}" % (n, min_run - 1), re.DOTALL)
        pats.append((n, p))
    _REPEAT_PATTERNS_CACHE[min_run] = pats
    return pats


def _is_meaningful_unit(unit):
    """判断重复单元是否属于"语义片段"。仅由数字/符号/空白组成的不算。"""
    if not unit.strip():
        return False
    for ch in unit:
        if ch.isalpha():
            return True
    return False


def detect_mechanical_repeat(text, finish_reason=""):
    """检测机械复读。返回 (matched, sample_dict_or_None)。

    策略：
    - finish_reason=length: 使用低阈值(8次)，因为复读导致截断是典型下限问题
    - 其他情况: 使用高阈值(50次)，只有结尾明显的重复 pattern（复读机现象）才算
    """
    if not text:
        return False, None
    n_text = len(text)
    if n_text > REPEAT_SCAN_MAX_CHARS:
        half = REPEAT_SCAN_MAX_CHARS // 2
        scan_text = text[:half] + text[-half:]
    else:
        scan_text = text

    min_run = REPEAT_MIN_RUN_LENGTH if finish_reason == "length" else REPEAT_MIN_RUN_NORMAL
    patterns = _get_repeat_patterns(min_run)

    # 优先扫描文本尾部（复读机通常在结尾持续重复）
    best_match = None
    for n, pat in patterns:
        for m in pat.finditer(scan_text):
            unit = m.group(1)
            if not _is_meaningful_unit(unit):
                continue
            run = (m.end() - m.start()) // n
            # 对于 finish_reason!=length，要求重复出现在文本尾部附近（最后20%区域）
            if finish_reason != "length":
                tail_start = int(len(scan_text) * 0.8)
                if m.end() < tail_start:
                    continue
            unit_preview = unit if len(unit) <= 40 else unit[:37] + "..."
            return True, {"unit": unit_preview, "unit_len": len(unit), "run": run, "pos": m.start()}
    return False, None


# ============================================================
# RED_LINE_PROB — UNREADABLE_CHAR 检测（完整对齐 text_quality_check.py + 新增 U+FFFD 检测）
# ============================================================

_LATIN_MOJIBAKE_PAIR = re.compile(r"[\u00C0-\u00FF][\u0080-\u00BF]")
_GBK_FAMOUS = re.compile(r"锟斤拷|锘垮|锘锛")
_BIDI_CONTROLS = re.compile(r"[\u202A-\u202E\u2066-\u2069]")
_ZERO_WIDTH = re.compile(r"[\u200B\u200C\u200E\u200F\u2060]")
_CTRL_NORMAL = {0x09, 0x0A, 0x0B, 0x0C, 0x0D}
_PRIVATE_OR_SPECIAL = re.compile(r"[\uE000-\uF8FF\uFFF0-\uFFFF\uFE30-\uFE4F]")
_CJK = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")

_FENCED_BLOCK_RE = re.compile(r"(?m)^(`{3,}|~{3,})[^\n]*\n.*?\n\1[ \t]*$", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`+[^`\n]+`+")


def _strip_code_content(text):
    """将代码块和行内代码内容替换为空格，保留行号。"""
    text = _FENCED_BLOCK_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    text = _INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), text)
    return text


def detect_unreadable_char(text):
    """检测不可读字符/乱码，返回 (has_issue, reasons_list)。
    完整对齐 text_quality_check.py 的 is_garbled (GARBLED 维度)。
    新增：包含 U+FFFD（�）即判为下限问题。
    """
    if not text:
        return False, []
    reasons = []
    text_no_code = _strip_code_content(text)

    # U+FFFD 替换字符（新增检测点 + 原始脚本已有）
    n_repl = text_no_code.count("\uFFFD")
    if n_repl > 0:
        reasons.append(f"包含 {n_repl} 个 U+FFFD 替换字符 (原始字节已丢失, 不可恢复)")

    # GBK 典型乱码
    famous_hits = _GBK_FAMOUS.findall(text_no_code)
    if famous_hits:
        reasons.append(f"包含典型 GBK 乱码字符 (锟斤拷/锘 等) {len(famous_hits)} 处")

    # Latin-1 mojibake
    pairs = _LATIN_MOJIBAKE_PAIR.findall(text_no_code)
    if pairs:
        try:
            recovered = text_no_code.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            recovered = ""
        cjk = _CJK.findall(recovered)
        if len(cjk) >= 2:
            reasons.append(
                f"疑似 UTF-8 被当作 Latin-1 读 (mojibake): "
                f"特征字符对 {len(pairs)} 个, 局部还原后包含 {len(cjk)} 个汉字"
            )
        elif len(pairs) >= 3:
            reasons.append(
                f"疑似 mojibake: 出现 {len(pairs)} 处罕见的高位字符连续模式"
            )

    # Unicode 双向控制字符 (Trojan Source)
    bidi = _BIDI_CONTROLS.findall(text)
    if bidi:
        reasons.append(f"包含 {len(bidi)} 个 Unicode 双向控制字符 (Trojan Source 风险)")

    # 零宽/方向标记
    body_for_zw = text[1:] if text.startswith("\uFEFF") else text
    mid_bom = body_for_zw.count("\uFEFF")
    zw_others = _ZERO_WIDTH.findall(body_for_zw)
    if zw_others:
        reasons.append(f"包含 {len(zw_others)} 个零宽/方向标记字符 (U+200B-200F / U+2060)")
    if mid_bom > 0:
        reasons.append(f"文件中部出现 {mid_bom} 个 BOM (U+FEFF 不在文件起始位置)")

    # NUL
    if "\x00" in text:
        reasons.append(f"包含 {text.count(chr(0))} 个 NUL 字符")

    # 非常规控制字符
    weird_ctrl = [c for c in text if ord(c) < 0x20 and ord(c) not in _CTRL_NORMAL]
    if weird_ctrl:
        reasons.append(f"包含 {len(weird_ctrl)} 个非常规控制字符 (ord < 32)")

    # 私有区字符
    sus = _PRIVATE_OR_SPECIAL.findall(text)
    if len(sus) >= 3 and len(text) > 0 and len(sus) / len(text) > 0.02:
        reasons.append(
            f"包含 {len(sus)} 个私有区/特殊区字符 (占比 {len(sus)/len(text):.1%})"
        )

    return bool(reasons), reasons


# ============================================================
# RED_LINE_PROB — MARKDOWN_ERR 检测
# 综合 check_markdown.py + markdown_issue.py 的最佳实践
# ============================================================


def _md_strip_code_blocks(text):
    """将 fenced code block (``` / ~~~) 内容替换为空行，保持行号不变。
    返回 (stripped_text, is_fence_unclosed)。
    对齐 markdown_issue.py 的 strip_code_blocks，但同时检测未闭合。
    """
    out = []
    in_code = False
    fence_char = None
    fence_len = 0
    for line in text.split("\n"):
        stripped = line.lstrip()
        m = re.match(r"^(`{3,}|~{3,})", stripped)
        if m:
            ch = m.group(1)[0]
            mlen = len(m.group(1))
            if not in_code:
                in_code = True
                fence_char = ch
                fence_len = mlen
                out.append("")
                continue
            if ch == fence_char and mlen >= fence_len:
                in_code = False
                fence_char = None
                out.append("")
                continue
        out.append("" if in_code else line)
    return "\n".join(out), in_code


def _md_strip_math_blocks(text):
    """将 \\[...\\] 多行显示数学块和行内 \\(...\\) 替换为空，保持行号不变。
    对齐 markdown_issue.py 的 strip_math_blocks。
    """
    out = []
    in_math = False
    for line in text.split("\n"):
        stripped = line.strip()
        if not in_math:
            if stripped == r"\[":
                in_math = True
                out.append("")
                continue
            cleaned = re.sub(r"\\\[[\s\S]*?\\\]", "", line)
            out.append(cleaned)
        else:
            if stripped == r"\]":
                in_math = False
            out.append("")
    result = "\n".join(out)
    result = re.sub(r"\\\(.*?\\\)", "", result)
    return result


def _md_strip_inline_code(line):
    """把行内 `...` 替换为空字符串。"""
    return re.sub(r"`[^`\n]*`", "", line)


# 编译一次复用
_HR_VALID_RE = re.compile(
    r'^([-]{3,}|[*]{3,}|[_]{3,}|'
    r'[-]\s*[-]\s*[-][-\s]*|[*]\s*[*]\s*[*][*\s]*|[_]\s*[_]\s*[_][_\s]*)$'
)
_HEADING_NO_SPACE_RE = re.compile(r"^(#{1,6})([^ #\s\t])")
_HEADING_TOO_DEEP_RE = re.compile(r"^(#{7,})\s")
_EMPHASIS_FLANKING_OPEN_RE = re.compile(r"\*\* +([^*\n]+?)\*\*")
_EMPHASIS_FLANKING_CLOSE_RE = re.compile(r"\*\*([^*\n]*?) +\*\*")


def detect_markdown_err(text):
    """检测 markdown 渲染问题，返回 (has_issue, reasons_list)。
    综合 check_markdown.py + markdown_issue.py 最佳实践。
    """
    if not text or len(text.strip()) < 10:
        return False, []
    reasons = []
    lines = text.split('\n')
    total_lines = len(lines)

    # ── 0. 基础设施：剥除代码块和数学块 ──
    no_code, fence_unclosed = _md_strip_code_blocks(text)
    no_code_no_math = _md_strip_math_blocks(no_code)

    # ── 1. 代码块检测 ──
    if fence_unclosed:
        reasons.append("代码块未闭合（``` 不配对）")

    # 行内代码检测 —— 暂时关闭（误召回率高）
    # in_blockquote_fence = False
    # for line in no_code.split('\n'):
    #     stripped_bq = re.sub(r"^(>\s*)+", "", line)
    #     if re.match(r"^`{3,}", stripped_bq):
    #         in_blockquote_fence = not in_blockquote_fence
    #         continue
    #     if in_blockquote_fence:
    #         continue
    #     cleaned = line.replace(r"\`", "")
    #     n = cleaned.count("`")
    #     if n and n % 2 != 0:
    #         reasons.append("行内代码未闭合（` 不配对）")
    #         break

    # ── 2. 标题格式（在代码块外检测） ──
    no_code_lines = no_code.split('\n')
    for i, line in enumerate(no_code_lines):
        if not line.strip():
            continue
        # 超深标题：仅 #{7,} 后跟空格才算（纯 ### 装饰线不报）
        if _HEADING_TOO_DEEP_RE.match(line):
            if "标题级别超过6级" not in reasons:
                reasons.append("标题级别超过6级")

        # # 后缺空格
        hm = _HEADING_NO_SPACE_RE.match(line)
        if hm:
            stripped_line = line.strip()
            hashes = hm.group(1)
            # 排除社交媒体 hashtag（行内多个 #word）
            if len(hashes) == 1 and re.search(r'#\S+\s+#\S+', stripped_line):
                continue
            # 排除 ###TAG### 类标记
            if re.match(r'^#{1,6}[A-Za-z_]+#{1,6}$', stripped_line):
                continue
            # 排除 CSS 选择器（#id { 或 #id,）
            if re.match(r'^#\w+\s*[{,;:]', stripped_line):
                continue
            # 排除步骤号（#1 #2）
            if re.match(r'^#\d+\b', stripped_line) and len(hashes) == 1:
                continue
            if "标题 # 后缺少空格" not in reasons:
                reasons.append("标题 # 后缺少空格")

    # ── 3. 链接和图片（在代码块外检测） ──
    text_for_link = no_code
    idx = 0
    while idx < len(text_for_link):
        if text_for_link[idx] == '[':
            is_image = idx > 0 and text_for_link[idx-1] == '!'
            j = idx + 1
            bracket_depth = 1
            while j < len(text_for_link) and bracket_depth > 0:
                if text_for_link[j] == '[':
                    bracket_depth += 1
                elif text_for_link[j] == ']':
                    bracket_depth -= 1
                j += 1
            if bracket_depth == 0:
                if j < len(text_for_link) and text_for_link[j] == '(':
                    k = j + 1
                    paren_depth = 1
                    while k < len(text_for_link) and paren_depth > 0:
                        if text_for_link[k] == '(':
                            paren_depth += 1
                        elif text_for_link[k] == ')':
                            paren_depth -= 1
                        k += 1
                    if paren_depth != 0:
                        issue_type = "图片格式不完整（缺少闭合括号）" if is_image else "链接格式不完整（缺少闭合括号）"
                        if issue_type not in reasons:
                            reasons.append(issue_type)
            else:
                rest = text_for_link[idx:idx+100]
                if re.search(r'\]\s*\(', rest):
                    issue_type = "图片格式不完整" if is_image else "链接格式不完整"
                    if issue_type not in reasons:
                        reasons.append(issue_type)
            idx = j if bracket_depth == 0 else idx + 1
        else:
            idx += 1

    # ── 4. 文本样式（粗体、删除线）—— 暂时关闭（误召回率高） ──
    # text_style = re.sub(r'`[^`]+`', lambda m: ' ' * len(m.group()), no_code)
    #
    # # 粗体 ** 检测：排除 glob 路径
    # text_for_bold = text_style
    # text_for_bold = re.sub(r'\*\*/[^\s]*', '    ', text_for_bold)
    # text_for_bold = re.sub(r'[^\s]*\*\*/', '    ', text_for_bold)
    # text_check_bold = re.sub(r'\*\*[^\s*][^*]*[^\s*]\*\*', '    ', text_for_bold)
    # text_check_bold = re.sub(r'\*\*[^\s*]+\*\*', '    ', text_check_bold)
    # double_asterisks = len(re.findall(r'(?<!\*)\*\*(?!\*)', text_check_bold))
    # if double_asterisks % 2 != 0:
    #     reasons.append("粗体标记未闭合（** 不配对）")
    #
    # # 粗体 __ 检测：排除 Python 魔术方法
    # text_for_underscore = text_style
    # text_for_underscore = re.sub(r'__\w+__', '    ', text_for_underscore)
    # text_check_underscore = re.sub(r'__[^\s_][^_]*[^\s_]__', '    ', text_for_underscore)
    # text_check_underscore = re.sub(r'__[^\s_]+__', '    ', text_check_underscore)
    # double_underscores = len(re.findall(r'(?<!_)__(?!_)', text_check_underscore))
    # if double_underscores % 2 != 0:
    #     reasons.append("粗体标记未闭合（__ 不配对）")
    #
    # # 删除线 ~~ 检测
    # text_check_strike = re.sub(r'~~[^\s~][^~]*[^\s~]~~', '    ', text_style)
    # text_check_strike = re.sub(r'~~[^\s~]+~~', '    ', text_check_strike)
    # strikethrough = len(re.findall(r'(?<!~)~~(?!~)', text_check_strike))
    # if strikethrough % 2 != 0:
    #     reasons.append("删除线标记未闭合（~~ 不配对）")

    # ── 5. 分割线 / Setext 检测 ──
    yaml_front_matter_end = -1
    if total_lines > 0 and lines[0].strip() == '---':
        for fi in range(1, total_lines):
            if lines[fi].strip() == '---':
                yaml_front_matter_end = fi
                break

    for i, line in enumerate(lines):
        # 跳过代码块内
        if not no_code_lines[i].strip() and lines[i].strip():
            continue
        stripped = line.strip()
        if not stripped:
            continue
        valid_hr = _HR_VALID_RE.match(stripped)
        if valid_hr:
            if yaml_front_matter_end >= 0 and (i == 0 or i == yaml_front_matter_end):
                continue

            # Setext h2 检测：--- / === 前一行是非空纯文本 → 渲染为 <h2>/<h1> 导致异常加粗
            if i > 0 and (stripped.startswith('-') or stripped.startswith('=')):
                prev_line = lines[i-1].strip()
                if not prev_line:
                    continue
                # 基础排除（与 check_markdown.py 对齐）
                if prev_line.startswith('#') or prev_line.startswith('>') \
                   or prev_line.startswith('|') or re.match(r'^[-*_\s]{3,}$', prev_line):
                    continue

                # ── 白名单排除（降低误召回） ──

                # W1: 上一行是列表项（有序 / 无序）→ 不触发 Setext
                if re.match(r'^\s*(\d+\.|[-*+])\s', prev_line):
                    continue

                # W2: 上一行是整行粗体 **...** 或 __...__（作为装饰标题）
                if (re.match(r'^\*\*[^*]+\*\*\s*$', prev_line) or
                        re.match(r'^__[^_]+__\s*$', prev_line)):
                    continue

                # W3: 下一行是 ATX 标题（--- 是章节分隔）
                if i + 1 < total_lines:
                    next_line = lines[i+1].strip()
                    if re.match(r'^#{1,6}\s', next_line):
                        continue
                    # W4: 下一行是有序列表（--- 引出列表的分隔线）
                    if re.match(r'^\d+\.\s', next_line):
                        continue

                if "---前缺少空行导致Setext标题异常加粗" not in reasons:
                    reasons.append("---前缺少空行导致Setext标题异常加粗")
                continue

    return bool(reasons), reasons


# ============================================================
# THINK_TAG_ILLEGAL 检测（对齐 analyze_think_or_answer_pattern_stage1）
# ============================================================

_THINK_BEGIN_TOKENS = ["<think>", "<think:6124c78e>"]
_THINK_END_TOKENS = ["</think>", "</think:6124c78e>"]


def _contains_any_think_begin(text):
    """检测文本中是否包含任何 think 开始标签变体。"""
    return any(tok in text for tok in _THINK_BEGIN_TOKENS)


def _contains_any_think_end(text):
    """检测文本中是否包含任何 think 结束标签变体。"""
    return any(tok in text for tok in _THINK_END_TOKENS)


def detect_think_tag_issues(response_text, thinking_text, think_mode, finish_reason=""):
    """
    检测 think 标签异常。
    think_mode: 'think' or 'nothink'
    finish_reason: 模型的 finish_reason，用于截断前置过滤
    返回 list of (subtype, desc)

    对齐 analyze_think_or_answer_pattern_stage1 的检测逻辑：
      A. EMPTY_THINK        — think模式下，reasoning_content为空
      B. MISSING_</think>   — think模式下，排除超长截断后，answer为空（推断缺少结束符）
      C. REDUNDANT_</think> — response中出现</think>（平台按第一个</think>切分，后续为多余）
      D. REDUNDANT_<think>  — reasoning/response中出现<think>（平台已消费首个，后续为多余）
      E. EMPTY_RESP         — 模型RESPONSE为空但finish_reason=stop
      + NOTHINK_HAS_<think> — nothink 模式下出现 think 标签
    """
    issues = []
    if not response_text and not thinking_text:
        return issues

    # 截断前置过滤：finish_reason=length 说明是截断导致的，跳过 think 标签检测
    if finish_reason == "length":
        return issues

    combined = (thinking_text or "") + (response_text or "")

    # --- nothink 模式 ---
    if think_mode == "nothink":
        if _contains_any_think_begin(combined) or _contains_any_think_end(combined):
            issues.append(("NOTHINK_HAS_<think>",
                           "nothink模式下response中出现<think>或</think>标签"))
        # E: nothink 模式下的空答案
        if finish_reason == "stop" and (not response_text or not response_text.strip()):
            issues.append(("EMPTY_RESP", "模型RESPONSE为空但finish_reason=stop"))
        return issues

    # --- think 模式 ---

    # A: EMPTY_THINK — think模式下 reasoning_content 为空
    if not thinking_text or not thinking_text.strip():
        issues.append(("EMPTY_THINK", "think模式下reasoning_content为空"))

    # B: MISSING_</think> — think模式下 response 为空（推断缺少结束符导致全部内容被归入 reasoning）
    if not response_text or not response_text.strip():
        if finish_reason == "stop":
            issues.append(("MISSING_</think>",
                           "think模式下response为空(推断缺少</think>闭合，全部内容归入reasoning)"))
            issues.append(("EMPTY_RESP", "模型RESPONSE为空但finish_reason=stop"))
        return issues

    # C: REDUNDANT_</think> — response 中包含 </think>（平台按首个</think>切分，残留在 response 中说明有多余）
    if _contains_any_think_end(response_text):
        issues.append(("REDUNDANT_</think>",
                       "response中残留</think>标签(平台按首个切分,说明存在多个)"))

    # D: REDUNDANT_<think> — reasoning 或 response 中出现 <think>（平台已消费首个，后续为多余）
    if _contains_any_think_begin(combined):
        issues.append(("REDUNDANT_<think>",
                       "reasoning/response中残留<think>标签(说明存在多个)"))

    # E: EMPTY_RESP — response 非空已在上面 B 的分支中处理过，此处为兜底
    # （正常走到这里 response_text 已非空，不会触发）

    return issues


# ============================================================
# Trajectory 解析（agent 类任务）
# ============================================================

def resolve_trajectory_path(payload):
    """从 payload 中解析实际可读的轨迹文件路径"""
    td = payload.get("trial_details")
    if not isinstance(td, dict):
        return None
    ti = td.get("trajectory_info")
    if not isinstance(ti, dict):
        return None
    tp = ti.get("trajectory_path", "")
    if not tp:
        return None
    if os.path.isfile(tp):
        return tp
    if os.path.isfile(tp + ".zst"):
        return tp + ".zst"
    return None


def parse_trajectory_spans(traj_path):
    """
    解析轨迹文件，返回每轮 LLM 调用的输出信息列表。
    每个元素: {turn, content, reasoning_content, tool_calls, finish_reason, tools, tool_choice}
    """
    if not HAS_ZST:
        return []

    try:
        if traj_path.endswith(".zst"):
            dctx = zstandard.ZstdDecompressor()
            with open(traj_path, "rb") as fh:
                raw = dctx.decompress(fh.read())
            lines = raw.decode("utf-8").strip().split("\n")
        else:
            with open(traj_path, "r", encoding="utf-8") as fh:
                lines = fh.read().strip().split("\n")
    except Exception:
        return []

    start_spans = []
    update_spans = []

    for line in lines:
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        span_type = s.get("type")
        name = s.get("name", "")

        if span_type == "START" and name == "openai_completion":
            start_spans.append(s)
        elif span_type == "UPDATE":
            attrs = s.get("attributes")
            if not isinstance(attrs, dict):
                continue
            outputs = attrs.get("outputs")
            if not isinstance(outputs, dict):
                continue
            choices = outputs.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                continue
            msg = first_choice.get("message") or first_choice.get("delta")
            if msg is None:
                continue
            update_spans.append((s, msg, first_choice.get("finish_reason", "")))

    results = []

    def _make_msg_fingerprint(msgs):
        """轻量级 fingerprint: (消息数, 最后一条消息的 sha1)"""
        if not msgs:
            return (0, "")
        import hashlib
        last_json = json.dumps(msgs[-1], ensure_ascii=False, sort_keys=True)
        sha1 = hashlib.sha1(last_json.encode("utf-8")).hexdigest()
        return (len(msgs), sha1)

    last_fingerprint = None
    current_step_id = 0

    for turn_idx, (span, msg, fr) in enumerate(update_spans):
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        tool_calls = msg.get("tool_calls", []) or []
        span_id = span.get("span_id", "") or span.get("context", {}).get("span_id", "")

        tools = []
        tool_choice = ""
        messages = []
        if turn_idx < len(start_spans):
            start_attrs = start_spans[turn_idx].get("attributes", {})
            if isinstance(start_attrs, dict):
                inp = start_attrs.get("inputs")
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except Exception:
                        inp = None
                if isinstance(inp, dict):
                    kw = inp.get("kwargs") or {}
                    tools = inp.get("tools") or kw.get("tools") or []
                    tc_raw = inp.get("tool_choice") or kw.get("tool_choice")
                    if isinstance(tc_raw, str):
                        tool_choice = tc_raw.lower()
                    messages = inp.get("messages", []) or []

        fp = _make_msg_fingerprint(messages)
        is_retry = (last_fingerprint is not None and fp == last_fingerprint)
        last_fingerprint = fp

        if not is_retry:
            current_step_id += 1

        results.append({
            "turn": turn_idx,
            "span_id": span_id,
            "step_id": current_step_id,
            "is_retry": is_retry,
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
            "finish_reason": fr or "",
            "tools": tools,
            "tool_choice": tool_choice,
        })

    return results


# ============================================================
# Badcase 构建（保留完整原始数据 + anomaly_results）
# ============================================================

_SUBTYPE_DESC = {
    "TOOL_CALL_LEAKAGE": "content/reasoning 中泄漏 tool call 标记",
    "HALLUCINATED_TOOL": "模型调用了不存在的工具",
    "JSON_PARSE_ERROR": "tool_calls 参数 JSON 解析失败",
    "REASONING_ONLY": "模型只输出推理内容,无 content 也无 tool_calls(可能缺think结束符)",
    "STOP_WITHOUT_TOOL_CALL": "finish_reason=stop 但未发起 tool call",
    "TOOL_CALLS_FIELD_EMPTY": "finish_reason=tool_calls 但 tool_calls 为空",
    "TOOL_CALL_DESPITE_NONE": "tool_choice=none 但模型仍发起 tool call",
    "REPEAT": "模型输出中存在机械复读",
    "UNREADABLE_CHAR": "模型输出含不可读字符/乱码",
    "MARKDOWN_ERR": "Markdown 渲染问题(代码块/公式/标签不闭合/---前缺换行等)",
    "EMPTY_THINK": "think模式下reasoning_content为空",
    "MISSING_</think>": "think模式下缺少</think>闭合(response为空,全部内容归入reasoning)",
    "REDUNDANT_</think>": "response中残留</think>标签(存在多个</think>)",
    "REDUNDANT_<think>": "reasoning/response中残留<think>标签(存在多个<think>)",
    "EMPTY_RESP": "模型RESPONSE为空但finish_reason=stop",
    "NOTHINK_HAS_<think>": "nothink模式下response出现<think>或</think>标签",
}


def _get_text_snippet(text, reason, max_len=200):
    """根据 reason 内容尝试提取相关的文本片段，确保 snippet 覆盖问题位置"""
    if not text:
        return ""
    if "复读" in reason or "REPEAT" in reason:
        return text[-max_len:] if len(text) > max_len else text
    if "---" in reason or "Setext" in reason:
        lines = text.split("\n")
        for i, l in enumerate(lines):
            stripped = l.strip()
            if not re.match(r"^[-]{3,}$", stripped):
                continue
            if i == 0:
                continue
            prev = lines[i-1].strip()
            if not prev:
                continue
            if prev.startswith('#') or prev.startswith('>') \
               or prev.startswith('|') or re.match(r'^[-*_\s]{3,}$', prev):
                continue
            if re.match(r'^\s*(\d+\.|[-*+])\s', prev):
                continue
            if re.match(r'^\*\*[^*]+\*\*\s*$', prev) or re.match(r'^__[^_]+__\s*$', prev):
                continue
            if i + 1 < len(lines):
                nxt = lines[i+1].strip()
                if re.match(r'^#{1,6}\s', nxt) or re.match(r'^\d+\.\s', nxt):
                    continue
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            return "\n".join(lines[start:end])[:max_len]
    if "标题" in reason or "# 后" in reason:
        lines = text.split("\n")
        for l in lines:
            if re.match(r"^#{1,}[^ #\t]", l) or re.match(r"^#{7,}\s", l):
                return l[:max_len]

    if "公式" in reason or "$$" in reason:
        # 定位第一个 $$ 或孤立 $ 的位置
        m = re.search(r"\$\$|\$", text)
        if m:
            half = max_len // 2
            start = max(0, m.start() - half)
            end = min(len(text), m.end() + half)
            return text[start:end]

    if "flanking" in reason:
        # 定位 ** 前后有空格的位置
        m = re.search(r"\*\*\s+\S|\S\s+\*\*", text)
        if m:
            half = max_len // 2
            start = max(0, m.start() - half)
            end = min(len(text), m.end() + half)
            return text[start:end]

    # UNREADABLE_CHAR: 根据具体异常类型定位问题字符并提取上下文
    snippet = _extract_unreadable_snippet(text, reason, max_len)
    if snippet is not None:
        return snippet

    return text[:max_len]


# 各类不可读字符的搜索 pattern（与 detect_unreadable_char 对齐）
_UNREADABLE_SNIPPET_PATTERNS = [
    ("U+FFFD", re.compile(r"\uFFFD")),
    ("GBK", re.compile(r"锟斤拷|锟斤|锘")),
    ("mojibake", _LATIN_MOJIBAKE_PAIR),
    ("双向控制", _BIDI_CONTROLS),
    ("零宽", _ZERO_WIDTH),
    ("BOM", re.compile(r"\uFEFF")),
    ("NUL", re.compile(r"\x00")),
    ("控制字符", re.compile(r"[\x01-\x08\x0e-\x1f]")),
]


def _extract_unreadable_snippet(text, reason, max_len=200):
    """针对 UNREADABLE_CHAR 类 reason，定位问题字符并返回包含其上下文的 snippet。
    如果 reason 与已知 pattern 匹配，则搜索第一个命中位置并提取前后文；否则返回 None。
    """
    for keyword, pattern in _UNREADABLE_SNIPPET_PATTERNS:
        if keyword in reason:
            m = pattern.search(text)
            if m:
                half = max_len // 2
                start = max(0, m.start() - half)
                end = min(len(text), m.end() + half)
                return text[start:end]
    return None


def build_non_agent_badcase(record, payload, anomaly_type, anomaly_subtype, anomaly_details):
    """构建非 agent case 的 badcase，保留完整原始 record + anomaly_results"""
    bc = dict(record)
    bc["anomaly_results"] = {
        "anomaly_type": anomaly_type,
        "anomaly_subtype": anomaly_subtype,
        "desc": _SUBTYPE_DESC.get(anomaly_subtype, anomaly_subtype),
        "is_agent_task": False,
        "anomaly_details": anomaly_details,
    }
    return bc


def build_agent_badcase(record, payload, anomaly_type, anomaly_subtype,
                        anomaly_counts, error_requests, error_steps,
                        anomaly_details_steps, retry_requests, retry_steps,
                        total_requests, total_steps):
    """构建 agent case 的 badcase，保留完整原始 record + anomaly_results"""
    bc = dict(record)
    bc["anomaly_results"] = {
        "anomaly_type": anomaly_type,
        "anomaly_subtype": anomaly_subtype,
        "desc": _SUBTYPE_DESC.get(anomaly_subtype, anomaly_subtype),
        "is_agent_task": True,
        "anomaly_counts": anomaly_counts,
        "error_requests": error_requests,
        "error_steps": error_steps,
        "anomaly_details": anomaly_details_steps,
        "retry_requests": retry_requests,
        "retry_steps": retry_steps,
        "total_requests": total_requests,
        "total_steps": total_steps,
    }
    return bc


# ============================================================
# 非 Agent Case 检测（RED_LINE_PROB + THINK_TAG_ILLEGAL）
# ============================================================

def detect_non_agent_case(record, payload, think_mode, task_id, bench_file, bench_name, detect_types=None):
    """对非 agent case 执行 RED_LINE_PROB + THINK_TAG_ILLEGAL 检测。
    每种 anomaly_subtype 生成一条 badcase。
    """
    badcases = []

    if payload.get("__infer_status__") == "infer_failed":
        return badcases

    response_text = first_text(payload.get("responses"))
    thinking_text = first_text(payload.get("thinking_responses"))
    finish_reason = extract_finish_reason(payload)

    def _make(atype, asub, details):
        bc = build_non_agent_badcase(record, payload, atype, asub, details)
        badcases.append(bc)

    # --- RED_LINE_PROB ---
    if detect_types is None or "RED_LINE_PROB" in detect_types:
        # REPEAT (response)
        hit_resp, detail_resp = detect_mechanical_repeat(response_text, finish_reason)
        if hit_resp:
            details = [{
                "reason": f"回答机械复读: 重复单元'{detail_resp['unit']}'连续重复{detail_resp['run']}次",
                "snippet": (response_text or "")[-200:]
            }]
            # REPEAT (thinking) — 同一 subtype 合入同一 badcase
            hit_cot, detail_cot = detect_mechanical_repeat(thinking_text, finish_reason)
            if hit_cot:
                details.append({
                    "reason": f"思维链机械复读: 重复单元'{detail_cot['unit']}'连续重复{detail_cot['run']}次",
                    "snippet": (thinking_text or "")[-200:]
                })
            _make("RED_LINE_PROB", "REPEAT", details)
        else:
            hit_cot, detail_cot = detect_mechanical_repeat(thinking_text, finish_reason)
            if hit_cot:
                _make("RED_LINE_PROB", "REPEAT", [{
                    "reason": f"思维链机械复读: 重复单元'{detail_cot['unit']}'连续重复{detail_cot['run']}次",
                    "snippet": (thinking_text or "")[-200:]
                }])

        # UNREADABLE_CHAR
        has_unread, unread_reasons = detect_unreadable_char(response_text)
        if has_unread:
            details = []
            for r in unread_reasons:
                details.append({
                    "reason": r,
                    "snippet": _get_text_snippet(response_text, r)
                })
            _make("RED_LINE_PROB", "UNREADABLE_CHAR", details)

        # MARKDOWN_ERR
        has_md_err, md_reasons = detect_markdown_err(response_text)
        if has_md_err:
            details = []
            for r in md_reasons:
                details.append({
                    "reason": r,
                    "snippet": _get_text_snippet(response_text, r)
                })
            _make("RED_LINE_PROB", "MARKDOWN_ERR", details)

    # --- THINK_TAG_ILLEGAL ---
    if detect_types is None or "THINK_TAG_ILLEGAL" in detect_types:
        think_issues = detect_think_tag_issues(response_text, thinking_text, think_mode, finish_reason)
        by_sub = defaultdict(list)
        for sub, desc in think_issues:
            by_sub[sub].append({
                "reason": desc,
                "snippet": _get_text_snippet(response_text, desc)
            })
        for sub, details in by_sub.items():
            _make("THINK_TAG_ILLEGAL", sub, details)

    return badcases


# ============================================================
# Agent Case 检测（TOOL_CALL_ANOMALY + RED_LINE_PROB + THINK_TAG_ILLEGAL）
# ============================================================

_SKIP_FINISH_REASONS_AGENT = frozenset({"length", "content_filter"})

def detect_agent_case(record, payload, think_mode, task_id, bench_file, bench_name, detect_types=None):
    """对 agent case 执行基于轨迹的逐轮检测。
    参照 toolcall_anomaly.py 的统计方式，按 step 粒度记录 anomaly_details。
    每种 anomaly_subtype 生成一条 badcase。
    """
    badcases = []

    if payload.get("__infer_status__") == "infer_failed":
        return badcases

    traj_path = resolve_trajectory_path(payload)
    if not traj_path:
        return badcases

    turns = parse_trajectory_spans(traj_path)
    if not turns:
        return badcases

    total_requests = len(turns)
    total_steps = max((t["step_id"] for t in turns), default=0)

    # ---------- 逐轮检测，收集每轮的所有异常 ----------
    per_turn_anomalies = []
    last_content = ""
    last_reasoning = ""
    last_finish = ""

    for turn_info in turns:
        content = turn_info["content"]
        reasoning = turn_info["reasoning_content"]
        tool_calls = turn_info["tool_calls"]
        fr = turn_info["finish_reason"]
        tools = turn_info["tools"]
        tool_choice = turn_info["tool_choice"]
        last_content = content or last_content
        last_reasoning = reasoning or last_reasoning
        last_finish = fr or last_finish

        turn_anomalies = []

        if fr in _SKIP_FINISH_REASONS_AGENT:
            per_turn_anomalies.append(turn_anomalies)
            continue

        # TOOL_CALL_ANOMALY
        if detect_types is None or "TOOL_CALL_ANOMALY" in detect_types:
            tc_results = detect_toolcall_anomalies_single(
                content, reasoning, tool_calls, fr, tools, tool_choice)
            for sub, desc, detail in tc_results:
                turn_anomalies.append(("TOOL_CALL_ANOMALY", sub))

        # RED_LINE_PROB — REPEAT
        if detect_types is None or "RED_LINE_PROB" in detect_types:
            hit_resp, _ = detect_mechanical_repeat(content, fr)
            if hit_resp:
                turn_anomalies.append(("RED_LINE_PROB", "REPEAT"))
            hit_cot, _ = detect_mechanical_repeat(reasoning, fr)
            if hit_cot:
                turn_anomalies.append(("RED_LINE_PROB", "REPEAT"))

        # THINK_TAG_ILLEGAL — 中间轮次仅检测标签泄漏类(C/D/NOTHINK)
        if detect_types is None or "THINK_TAG_ILLEGAL" in detect_types:
            combined_turn = (reasoning or "") + (content or "")
            if think_mode == "nothink":
                if _contains_any_think_begin(combined_turn) or _contains_any_think_end(combined_turn):
                    turn_anomalies.append(("THINK_TAG_ILLEGAL", "NOTHINK_HAS_<think>"))
            else:
                if _contains_any_think_end(content or ""):
                    turn_anomalies.append(("THINK_TAG_ILLEGAL", "REDUNDANT_</think>"))
                if _contains_any_think_begin(combined_turn):
                    turn_anomalies.append(("THINK_TAG_ILLEGAL", "REDUNDANT_<think>"))

        per_turn_anomalies.append(turn_anomalies)

    # 最后一轮: UNREADABLE_CHAR / MARKDOWN_ERR 只检测最后一轮
    # 最后一轮: THINK_TAG_ILLEGAL 完整检测所有子类型
    last_turn_extra = []
    if detect_types is None or "RED_LINE_PROB" in detect_types:
        has_unread, _ = detect_unreadable_char(last_content)
        if has_unread:
            last_turn_extra.append(("RED_LINE_PROB", "UNREADABLE_CHAR"))
        has_md, _ = detect_markdown_err(last_content)
        if has_md:
            last_turn_extra.append(("RED_LINE_PROB", "MARKDOWN_ERR"))

    if detect_types is None or "THINK_TAG_ILLEGAL" in detect_types:
        think_issues = detect_think_tag_issues(last_content, last_reasoning, think_mode, last_finish)
        for sub, desc in think_issues:
            last_turn_extra.append(("THINK_TAG_ILLEGAL", sub))

    if per_turn_anomalies and last_turn_extra:
        per_turn_anomalies[-1].extend(last_turn_extra)
    elif last_turn_extra and not per_turn_anomalies:
        per_turn_anomalies.append(last_turn_extra)

    # ---------- 收集所有出现过的 subtype ----------
    all_subtypes = set()
    for ta_list in per_turn_anomalies:
        for atype, asub in ta_list:
            all_subtypes.add((atype, asub))

    if not all_subtypes:
        return badcases

    # ---------- 为每种 subtype 构建 agent badcase ----------
    for anomaly_type, anomaly_subtype in sorted(all_subtypes):
        anomaly_counts = 0
        error_request_set = set()
        error_step_set = set()
        retry_request_set = set()
        retry_step_set = set()
        step_details = []

        # caused_retry: 如果下一轮是 retry 且本轮有该 subtype 异常，则 caused_retry=True
        for idx, turn_info in enumerate(turns):
            turn_anomaly_list = per_turn_anomalies[idx] if idx < len(per_turn_anomalies) else []
            subtypes_this_turn = [asub for (at, asub) in turn_anomaly_list if at == anomaly_type]
            has_this_subtype = anomaly_subtype in subtypes_this_turn

            is_retry = turn_info["is_retry"]
            next_is_retry = (idx + 1 < len(turns) and turns[idx + 1]["is_retry"])
            caused_retry = has_this_subtype and next_is_retry

            if has_this_subtype:
                anomaly_counts += 1
                error_request_set.add(idx)
                error_step_set.add(turn_info["step_id"])

            if is_retry:
                retry_request_set.add(idx)
                retry_step_set.add(turn_info["step_id"])

            if has_this_subtype:
                step_details.append({
                    "caused_retry": caused_retry,
                    "is_retry": is_retry,
                    "span_id": turn_info["span_id"],
                    "step_id": turn_info["step_id"],
                })

        bc = build_agent_badcase(
            record, payload, anomaly_type, anomaly_subtype,
            anomaly_counts=anomaly_counts,
            error_requests=len(error_request_set),
            error_steps=len(error_step_set),
            anomaly_details_steps=step_details,
            retry_requests=len(retry_request_set),
            retry_steps=len(retry_step_set),
            total_requests=total_requests,
            total_steps=total_steps,
        )
        badcases.append(bc)

    return badcases


# ============================================================
# 主流程
# ============================================================

def determine_think_mode(data_dir):
    """从非 agent 文件的第一条数据确定 think 模式"""
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".jsonl.gz"):
            continue
        if is_agent_bench(f):
            continue
        filepath = os.path.join(data_dir, f)
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as fh:
                line = fh.readline()
                if line:
                    record = json.loads(line)
                    payload = record.get("payload", {})
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    mi = payload.get("model_input", [[]])
                    if mi and isinstance(mi, list) and mi[0]:
                        m0 = mi[0][0] if isinstance(mi[0], list) and mi[0] else mi[0]
                        if isinstance(m0, dict):
                            kte = m0.get("chat_template_kwargs", {})
                            re_val = kte.get("reasoning_effort", "")
                            if re_val and str(re_val).lower() in ("no_think"):
                                return "nothink"
                            else:
                                return "think"
        except Exception:
            continue
    return "think"


def parse_bench_name(filename):
    """从文件名中提取 bench 名称（格式: BenchName__evId__task_taskId.jsonl.gz）"""
    parts = filename.split("__")
    if len(parts) >= 2:
        return parts[0]
    return filename.replace(".jsonl.gz", "")


def process_file(filepath, task_id, think_mode, debug_limit=None, detect_types=None,
                  on_case=None, on_bench_done=None):
    """处理单个 .jsonl.gz 文件。

    Args:
        debug_limit: 如果设置，每个文件只处理前 N 条。
        detect_types: 要检测的下限问题类型集合。
        on_case: 回调 on_case(bc)，每检测到一条 badcase 立即调用（流式写入）。
        on_bench_done: 回调 on_bench_done(badcases_list)，当前 bench 检测完后
                       批量调用（按 bench 写入）。

    三种写入模式:
        - on_case=None, on_bench_done=None → 不写 badcase，仅返回统计
        - on_case=fn → 每条 badcase 立即回调
        - on_bench_done=fn → 每个 bench 检测完后批量回调

    返回 (total_cases, subtype_counts)
    """
    filename = os.path.basename(filepath)
    bench_name = parse_bench_name(filename)
    is_agent = is_agent_bench(filepath)
    total = 0
    subtype_counts = Counter()
    bench_badcases = [] if on_bench_done else None

    try:
        with gzip.open(filepath, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                payload = record.get("payload", {})
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                total += 1

                if is_agent:
                    cases = detect_agent_case(record, payload, think_mode, task_id, filename, bench_name, detect_types)
                else:
                    cases = detect_non_agent_case(record, payload, think_mode, task_id, filename, bench_name, detect_types)

                for bc in cases:
                    ar = bc.get("anomaly_results", {})
                    subtype_counts[ar.get("anomaly_subtype", "UNKNOWN")] += 1
                    if on_case:
                        on_case(bc)
                    if bench_badcases is not None:
                        bench_badcases.append(bc)

                if debug_limit and total >= debug_limit:
                    break
    except Exception as e:
        print(f"  [ERROR] Failed to process {filename}: {e}", file=sys.stderr)

    if on_bench_done and bench_badcases:
        on_bench_done(bench_badcases)

    return total, subtype_counts


def _sanitize(obj):
    """递归清理字符串中的 surrogate 字符"""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="统一下限异常检测 v2")
    parser.add_argument("--data_dir", required=True, help="评测数据目录（含 *.jsonl.gz）")
    parser.add_argument("--output_dir", required=True, help="badcase 输出目录")
    parser.add_argument("--task_id", type=int, required=True, help="评测任务 ID")
    parser.add_argument("--debug", action="store_true", help="Debug模式：每个bench只跑前100条")
    parser.add_argument("--detect_types", type=str, default="ALL",
                        help="指定检测的下限问题类型，逗号分隔。"
                             "可选: TOOL_CALL_ANOMALY,RED_LINE_PROB,THINK_TAG_ILLEGAL,ALL。"
                             "默认 ALL（全部检测）。")
    parser.add_argument("--bench_name", type=str, default="ALL",
                        help="指定要检测的 bench 名称（文件名前缀），逗号分隔。"
                             "ALL 表示检测所有 bench。"
                             "示例: --bench_name CLBench,prbench_finance")
    parser.add_argument("--write_mode", type=str, default="per_bench",
                        choices=["immediate", "per_bench", "none"],
                        help="badcase 写入模式: "
                             "immediate=每条立即写入, "
                             "per_bench=每个bench检测完后批量写入(默认), "
                             "none=不写badcase文件")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    task_id = args.task_id
    debug_limit = 100 if args.debug else None

    ALL_TYPES = {"TOOL_CALL_ANOMALY", "RED_LINE_PROB", "THINK_TAG_ILLEGAL"}
    if args.detect_types.upper() == "ALL":
        detect_types = ALL_TYPES
    else:
        detect_types = set(t.strip() for t in args.detect_types.split(","))
        invalid = detect_types - ALL_TYPES
        if invalid:
            parser.error(f"未知的检测类型: {invalid}。可选: TOOL_CALL_ANOMALY, RED_LINE_PROB, THINK_TAG_ILLEGAL, ALL")

    if args.bench_name.upper() == "ALL":
        bench_filter = None
    else:
        bench_filter = [b.strip() for b in args.bench_name.split(",") if b.strip()]

    os.makedirs(output_dir, exist_ok=True)

    think_mode = determine_think_mode(data_dir)
    print(f"Think mode: {think_mode}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Task ID: {task_id}")
    print(f"Detect types: {', '.join(sorted(detect_types))}")
    bench_filter_display = ', '.join(bench_filter) if bench_filter else 'ALL'
    print(f"Bench filter: {bench_filter_display}".encode("utf-8", errors="replace").decode("utf-8"))
    if debug_limit:
        print(f"Debug mode: ON (max {debug_limit} cases per bench)")
    print()

    all_gz_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".jsonl.gz")
    ])

    if bench_filter:
        gz_files = []
        for fp in all_gz_files:
            fname = os.path.basename(fp)
            bench = parse_bench_name(fname)
            if any(b in bench for b in bench_filter):
                gz_files.append(fp)
        print(f"Found {len(all_gz_files)} total bench files, {len(gz_files)} matched filter")
    else:
        gz_files = all_gz_files
        print(f"Found {len(gz_files)} bench files")
    agent_count = sum(1 for f in gz_files if is_agent_bench(f))
    std_count = len(gz_files) - agent_count
    print(f"  Agent benches: {agent_count}")
    print(f"  Standard benches: {std_count}")
    print()

    write_mode = args.write_mode
    print(f"Write mode: {write_mode}")
    print()

    global_total = 0
    global_subtypes = Counter()
    badcase_counts_by_subtype = Counter()

    def _safe_subtype(subtype):
        return subtype.replace("/", "_").replace("<", "_").replace(">", "_")

    # ---------- 写入基础设施 ----------
    _file_handles = {}

    def _get_fh(subtype):
        safe_sub = _safe_subtype(subtype)
        if safe_sub not in _file_handles:
            out_path = os.path.join(output_dir, f"{safe_sub}.jsonl")
            _file_handles[safe_sub] = open(out_path, "w", encoding="utf-8")
        return _file_handles[safe_sub]

    def _write_single(bc):
        """immediate 模式：每条 badcase 立即写入"""
        ar = bc.get("anomaly_results", {})
        subtype = ar.get("anomaly_subtype", "UNKNOWN")
        fh = _get_fh(subtype)
        fh.write(json.dumps(_sanitize(bc), ensure_ascii=False) + "\n")
        fh.flush()
        badcase_counts_by_subtype[subtype] += 1

    def _write_batch(badcases):
        """per_bench 模式：一个 bench 检测完后批量写入"""
        for bc in badcases:
            ar = bc.get("anomaly_results", {})
            subtype = ar.get("anomaly_subtype", "UNKNOWN")
            fh = _get_fh(subtype)
            fh.write(json.dumps(_sanitize(bc), ensure_ascii=False) + "\n")
            badcase_counts_by_subtype[subtype] += 1
        # 批量写完后统一 flush
        for fh in _file_handles.values():
            fh.flush()

    # 根据 write_mode 选择回调
    on_case = _write_single if write_mode == "immediate" else None
    on_bench_done = _write_batch if write_mode == "per_bench" else None

    try:
        for filepath in gz_files:
            filename = os.path.basename(filepath)
            is_agent = is_agent_bench(filepath)
            tag = "[AGENT]" if is_agent else "[STANDARD]"
            safe_name = filename.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")

            if not is_agent and detect_types == {"TOOL_CALL_ANOMALY"}:
                print(f"  {tag} {safe_name} ... SKIP (非Agent，仅检测TOOL_CALL_ANOMALY)")
                continue
            if is_agent and "TOOL_CALL_ANOMALY" not in detect_types \
               and "RED_LINE_PROB" not in detect_types \
               and "THINK_TAG_ILLEGAL" not in detect_types:
                print(f"  {tag} {safe_name} ... SKIP (无适用检测类型)")
                continue

            print(f"  {tag} {safe_name} ...", end="", flush=True)

            total, subtype_counts = process_file(
                filepath, task_id, think_mode, debug_limit, detect_types,
                on_case=on_case, on_bench_done=on_bench_done
            )
            global_total += total
            global_subtypes += subtype_counts

            anomaly_count = sum(subtype_counts.values())
            suffix = f" (限{debug_limit})" if debug_limit and total >= debug_limit else ""
            print(f" {total} cases{suffix}, {anomaly_count} anomalies")
            if subtype_counts:
                for sub, cnt in sorted(subtype_counts.items()):
                    print(f"    {sub}: {cnt}")
    finally:
        for fh in _file_handles.values():
            fh.close()

    print(f"\n{'='*60}")
    if write_mode == "none":
        print("Badcase writing disabled (--write_mode none)")
    else:
        print(f"Badcase files written to {output_dir}/")
        for subtype, cnt in sorted(badcase_counts_by_subtype.items()):
            print(f"  {_safe_subtype(subtype)}.jsonl: {cnt} cases")

    # 构建指标 JSON
    TYPE_SUBTYPE_MAP = {
        "TOOL_CALL_ANOMALY": [
            "TOOL_CALL_LEAKAGE", "HALLUCINATED_TOOL", "JSON_PARSE_ERROR",
            "REASONING_ONLY", "STOP_WITHOUT_TOOL_CALL", "TOOL_CALLS_FIELD_EMPTY",
            "TOOL_CALL_DESPITE_NONE"
        ],
        "RED_LINE_PROB": [
            "REPEAT", "UNREADABLE_CHAR", "MARKDOWN_ERR"
        ],
        "THINK_TAG_ILLEGAL": [
            "EMPTY_THINK", "MISSING_</think>", "REDUNDANT_</think>",
            "REDUNDANT_<think>", "EMPTY_RESP", "NOTHINK_HAS_<think>"
        ],
    }

    desc_map = {
        "TOOL_CALL_LEAKAGE": "content/reasoning 中泄漏 tool call 标记",
        "HALLUCINATED_TOOL": "调用了不在 tools 列表中的函数",
        "JSON_PARSE_ERROR": "tool_calls arguments JSON 解析失败",
        "REASONING_ONLY": "只有推理内容, 无 content 也无 tool_calls(可能缺think结束符)",
        "STOP_WITHOUT_TOOL_CALL": "有 tools 但 stop 时未调用",
        "TOOL_CALLS_FIELD_EMPTY": "finish_reason=tool_calls 但 tool_calls 为空",
        "TOOL_CALL_DESPITE_NONE": "tool_choice=none 但仍调用了工具",
        "REPEAT": "机械复读(response或thinking)",
        "UNREADABLE_CHAR": "模型输出含不可读/乱码字符(含U+FFFD)",
        "MARKDOWN_ERR": "Markdown 渲染问题(代码块/公式/标签不闭合/---后缺换行等)",
        "EMPTY_THINK": "think模式下reasoning_content为空",
        "MISSING_</think>": "think模式下缺少</think>闭合(response为空,全部内容归入reasoning)",
        "REDUNDANT_</think>": "response中残留</think>标签(存在多个</think>)",
        "REDUNDANT_<think>": "reasoning/response中残留<think>标签(存在多个<think>)",
        "EMPTY_RESP": "模型RESPONSE为空但finish_reason=stop",
        "NOTHINK_HAS_<think>": "nothink模式下出现<think>或</think>",
    }

    metrics = []
    for atype, subtypes in TYPE_SUBTYPE_MAP.items():
        measures = []
        for sub in subtypes:
            cnt = global_subtypes.get(sub, 0)
            if think_mode == "think" and sub == "NOTHINK_HAS_<think>":
                measures.append({
                    "name": sub, "count": 0, "ratio": 0.0,
                    "desc": "不适用(当前为think模式)"
                })
            elif think_mode == "nothink" and sub in ("EMPTY_THINK", "MISSING_</think>", "REDUNDANT_</think>", "REDUNDANT_<think>"):
                measures.append({
                    "name": sub, "count": 0, "ratio": 0.0,
                    "desc": "不适用(当前为nothink模式)"
                })
            else:
                measures.append({
                    "name": sub,
                    "count": cnt,
                    "ratio": round(cnt / global_total, 6) if global_total > 0 else 0.0,
                    "desc": desc_map.get(sub, sub),
                })
        metrics.append({"type": atype, "measures": measures})

    metrics_path = os.path.join(output_dir, "floor_anomaly_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(metrics), f, ensure_ascii=False, indent=2)
    print(f"\nMetrics written to {metrics_path}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases: {global_total}")
    print(f"Think mode: {think_mode}")
    if debug_limit:
        print(f"Debug mode: ON (max {debug_limit} cases per bench)")
    total_anomalies = sum(global_subtypes.values())
    if global_total > 0:
        print(f"Total anomalies: {total_anomalies} ({total_anomalies/global_total*100:.2f}%)")
    else:
        print(f"Total anomalies: 0")
    print()
    for atype, subtypes in TYPE_SUBTYPE_MAP.items():
        type_total = sum(global_subtypes.get(s, 0) for s in subtypes)
        scope = "仅Agent Bench" if atype == "TOOL_CALL_ANOMALY" else "全部Bench"
        print(f"  {atype} ({scope}): {type_total}")
        for sub in subtypes:
            cnt = global_subtypes.get(sub, 0)
            if cnt > 0:
                print(f"    {sub}: {cnt} ({cnt/global_total*100:.3f}%)")


if __name__ == "__main__":
    _ensure_utf8_io()
    main()
