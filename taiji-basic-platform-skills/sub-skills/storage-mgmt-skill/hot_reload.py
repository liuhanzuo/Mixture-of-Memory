#!/usr/bin/env python3
"""hot_reload.py — independently published Taiji skill hot reloader.

机制（4 步）：
  1. TTL 检查：距上次检查不到 TTL_HOURS → 直接退出（零网络）
  2. GET /api/hot-reload/skills/{skill_key} 获取该独立 skill 的 current version
  3. 比对已装版本：一样 → 刷新 TTL 后退出
  4. 不一样 → 探测 skill 目录可写性：
        不可写 → 提示 + 刷新 TTL 后退出（不下载）
        可写   → 下载整个 ZIP → 解压覆盖 → 记录版本号

设计要点：
  - 仅用标准库（urllib/zipfile/json），Agent 环境零依赖
  - 网络/服务异常一律静默跳过，用旧版继续（不阻塞用户任务）
  - --force 跳过 TTL 立即检查
  - 解压做路径遍历防护
  - 保留本地凭证文件（PRESERVE_FILES），不被 ZIP 覆盖
  - 状态文件（TTL/版本号）落在用户可写目录，不随 skill 目录权限失效；
    只读安装下「正确地放弃」而非反复重下（见 _state_dir / _skill_dir_writable）

配置（按优先级）：
  - 环境变量 TAIJI_SKILLS_MANAGER_URL
  - 同目录 skills_manager.json 的 "base_url" 字段
  - 默认 DEFAULT_BASE_URL
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

# ---- 配置 -------------------------------------------------------------------

BASIC_SKILL_KEY = "taiji-basic-platform-skills"
# This file is copied into every independently installable skill.
# The basic copy keeps PACKAGE for backward-compatible imports/tests.
SKILL_KEY = "storage-mgmt-skill"
PACKAGE = SKILL_KEY
DEFAULT_BASE_URL = "http://taiji-skills-manager.woa.com"
TTL_HOURS = 5 / 60  # 5 分钟
HTTP_TIMEOUT = 10  # 秒

SKILL_DIR = Path(__file__).resolve().parent

# .skill_commit 仍留在 skill 目录：它由 ZIP 携带（pack.py 未排除），且被
# connect_mcp.py 的 _local_skill_commit() 按 skill 根目录读取。只读安装下写不
# 进去只会丢一个诊断 header，不会像状态文件那样引发重复下载。
COMMIT_FILE = SKILL_DIR / ".skill_commit"
CONFIG_FILE = SKILL_DIR / "skills_manager.json"
CREDENTIALS_FILE = Path.home() / ".config" / "taiji" / "credentials.json"

# 状态文件（TTL 时间戳、已装版本号）落在用户可写目录，而非 skill 目录内。
#
# 为什么不能放 skill 目录：skill 常被装在共享/只读路径下（如他人的
# /apdcephfs_*/share_*/<user>/.claude/...，目录属 root）。状态文件跟着 skill
# 目录就继承了它的权限，写不进去 → _local_version() 恒为空 → 永远判定「需要
# 更新」→ 每次调用都重新下载全量 ZIP，形成不会自愈的死循环。
#
# 环境变量 TAIJI_SKILL_MANAGER_HOME，默认 ~/.taiji-skill-manager。
DEFAULT_MANAGER_HOME = "~/.taiji-skill-manager"


def _find_basic_root() -> Path | None:
    """Find a valid enclosing basic package, not merely a nearby reloader.

    A sub-skill nested in the basic ZIP must update the basic ZIP as a whole.
    The same sub-skill installed by itself has no such parent and updates only
    its own directory.
    """
    try:
        for candidate in (SKILL_DIR, *SKILL_DIR.parents):
            if (
                candidate.name == BASIC_SKILL_KEY
                and (candidate / "SKILL.md").is_file()
                and (candidate / "hot_reload.py").is_file()
                and (candidate / "sub-skills").is_dir()
            ):
                return candidate
    except Exception:
        pass
    return None


def _delegated_to_basic() -> bool:
    basic_root = _find_basic_root()
    if basic_root is None:
        return False
    try:
        return SKILL_DIR.resolve() != basic_root.resolve()
    except Exception:
        return SKILL_DIR != basic_root


def _state_dir() -> Path:
    """本 skill 安装对应的状态目录（按 skill 路径 hash 隔离）。

    hash 的作用：同一台机器上可能存在多份 skill 安装（自己一份、共享一份），
    版本各不相同，必须各记各账；也避免多个用户共读同一份 skill 时互相覆盖
    对方的 TTL 时间戳。
    """
    home = os.environ.get("TAIJI_SKILL_MANAGER_HOME") or DEFAULT_MANAGER_HOME
    base = Path(os.path.expandvars(os.path.expanduser(str(home))))
    key = hashlib.sha1(str(SKILL_DIR).encode("utf-8")).hexdigest()[:16]
    return base / "hot_reload" / key


def _cache_file() -> Path:
    return _state_dir() / ".update_cache"


def _version_file() -> Path:
    return _state_dir() / ".skill_version"


def _legacy_cache_file() -> Path:
    """旧位置的 TTL 文件：只回退读、不再写。"""
    return SKILL_DIR / ".update_cache"


def _legacy_version_file() -> Path:
    """旧位置的版本号文件：只回退读、不再写。

    回退的意义：老安装的版本号还在 skill 目录里，若直接忽略会在升级那一刻
    误判「本地无版本」而白下载一次全量 ZIP。
    """
    return SKILL_DIR / ".skill_version"


# 本地凭证/运行时文件：解压时保留，不被 ZIP 内容覆盖。
# .skill_version / .update_cache 已迁出 skill 目录，这里仍保留是为了兼容
# 迁移期的老安装（旧位置文件不该被 ZIP 覆盖或全量镜像删掉）。
PRESERVE_FILES = {
    "env_config.json",
    ".skill_version",
    ".skill_commit",
    ".update_cache",
    ".skill_pins.json",
}


def _log(msg: str) -> None:
    print(f"[taiji-skills] {msg}", file=sys.stderr)


def _base_url() -> str:
    env = os.environ.get("TAIJI_SKILLS_MANAGER_URL")
    if env:
        return env.rstrip("/")
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if cfg.get("base_url"):
                return str(cfg["base_url"]).rstrip("/")
        except Exception:
            pass
    return DEFAULT_BASE_URL.rstrip("/")


# ---- TTL --------------------------------------------------------------------

def _read_ts(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _ttl_valid() -> bool:
    # skill 目录优先（原位置）；读不到再看 manager home（只读安装的回退）。
    last = _read_ts(_legacy_cache_file())
    if last is None:
        last = _read_ts(_cache_file())
    if last is None:
        return False
    return (time.time() - last) < TTL_HOURS * 3600


def _touch_ttl() -> None:
    # 优先写 skill 目录（原位置），写不进去再落到 manager home。
    # 只读安装下 skill 目录无写权限，manager home 保证可写，TTL 才真正生效。
    for path in (_legacy_cache_file(), _cache_file()):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(time.time()), encoding="utf-8")
            return
        except Exception:
            continue


def _write_commit(commit: str) -> None:
    """把服务端返回的 git_commit 落盘（供 connect_mcp.py 作为 header 传出）。

    best-effort：写失败一律吞掉，绝不影响热更新主链路。
    """
    if not commit:
        return
    try:
        COMMIT_FILE.write_text(commit, encoding="utf-8")
    except Exception:
        pass


def _local_version() -> str:
    """已装版本号：skill 目录优先，回退读 manager home（只读安装的回退）。"""
    for path in (_legacy_version_file(), _version_file()):
        try:
            if path.is_file():
                version = path.read_text(encoding="utf-8").strip()
                if version:
                    return version
        except Exception:
            continue
    return ""


def _write_local_version(version: str) -> None:
    """记录已装版本号。优先写 skill 目录，写不进去落到 manager home。
    best-effort：全部失败静默跳过，不影响主链路。"""
    for path in (_legacy_version_file(), _version_file()):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(version, encoding="utf-8")
            return
        except Exception:
            continue


# ---- 可写性探测 -------------------------------------------------------------

def _skill_dir_writable() -> bool:
    """探测 SKILL_DIR 能否写入——决定是否值得下载 ZIP。

    用「真的建一个临时文件再删掉」而不是 os.access(W_OK)：后者在 NFS/Ceph
    等网络文件系统上会说谎（返回可写，实际 EACCES）。
    """
    try:
        probe = tempfile.NamedTemporaryFile(dir=str(SKILL_DIR), prefix=".taiji_probe_")
    except Exception:
        return False
    try:
        probe.close()
    except Exception:
        pass
    return True


# ---- 网络 -------------------------------------------------------------------

def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": f"taiji-hot-reload/{SKILL_KEY}"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_token() -> str:
    """Best-effort 读取 PAT；任何本地异常都降级为空 Token。"""
    try:
        token = os.environ.get("TAIJI_PAT_TOKEN", "").strip()
        if token:
            return token
    except Exception:
        pass

    try:
        if CREDENTIALS_FILE.is_file():
            credentials = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
            token = str(credentials.get("pat_token") or "").strip()
            if token:
                return token
    except Exception:
        pass
    return ""


def _download_bytes(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT * 3) as resp:
        return resp.read()


def _http_get_bytes(url: str) -> bytes:
    """下载 ZIP；有 PAT 时携带，Token/Header 异常时降级匿名重试。"""
    headers = {"User-Agent": f"taiji-hot-reload/{SKILL_KEY}"}
    token = _api_token()
    if not token:
        return _download_bytes(url, headers)

    try:
        return _download_bytes(url, {**headers, "taiji_api_token": token})
    except urllib.error.HTTPError as exc:
        # Token 被网关/服务端拒绝时保持旧匿名下载语义；其他 HTTP 错误
        # 通常与 Token 无关，交给主流程静默跳过，避免无意义的二次请求。
        if exc.code not in {400, 401, 403, 431}:
            raise
    except (ValueError, UnicodeError):
        # urllib 会在 Header 含非法字符时本地抛错，匿名重试不影响主链路。
        pass

    return _download_bytes(url, headers)


# ---- 解压覆盖 ---------------------------------------------------------------

def _safe_extract_and_overwrite(zip_bytes: bytes) -> None:
    """解压 ZIP（内部路径为 '<package>/...'）到临时目录，做路径遍历检查后
    以「全量镜像」方式同步到 SKILL_DIR：

      - ZIP 里的文件 → 覆盖写入本地
      - 本地存在、但 ZIP 里没有的文件 → 删除（保证本地 == 上游快照，
        下线/改名的 sub-skill 不会残留旧文件）
      - PRESERVE_FILES 中的文件（按文件名匹配）始终保留，不被覆盖也不被删除

    路径遍历防护：拒绝绝对路径 / 含 .. 组件 / resolve 越界。
    """
    with tempfile.TemporaryDirectory(prefix="taiji_skill_") as tmp:
        tmp_path = Path(tmp).resolve()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                norm = name.replace("\\", "/")
                if norm.startswith("/") or ".." in Path(norm).parts:
                    raise ValueError(f"ZIP 含非法路径: {name}")
                dest = (tmp_path / name).resolve()
                if not str(dest).startswith(str(tmp_path)):
                    raise ValueError(f"ZIP 含非法路径: {name}")
            zf.extractall(tmp_path)

        # ZIP 顶层是 the independently published skill key.
        extracted_root = tmp_path / SKILL_KEY
        if not extracted_root.is_dir():
            # 兜底：若没有 package 前缀，直接用 tmp_path
            extracted_root = tmp_path

        # 备份要保留的本地文件
        preserved: dict[str, bytes] = {}
        for fname in PRESERVE_FILES:
            fp = SKILL_DIR / fname
            if fp.exists() and fp.is_file():
                try:
                    preserved[fname] = fp.read_bytes()
                except Exception:
                    pass

        # 1) 收集新 ZIP 的相对路径清单（作为"期望的最终状态"）
        expected_rel: set[str] = set()
        for src in extracted_root.rglob("*"):
            if src.is_file():
                expected_rel.add(src.relative_to(extracted_root).as_posix())

        # 2) 覆盖复制：ZIP 里的文件写入本地（PRESERVE_FILES 跳过）。
        #    先写 <name>.tmp 再原子 rename，避免 Agent 并发读到半写的文件。
        #    rename 失败（如跨文件系统）降级为直接覆盖，保住 best-effort 语义。
        #    崩溃残留的 .tmp 会被下面 step 3 的全量镜像删除自动清掉。
        for src in extracted_root.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(extracted_root)
            if rel.name in PRESERVE_FILES:
                continue  # 保留本地版本，跳过 ZIP 里的
            target = SKILL_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_target = target.with_name(target.name + ".tmp")
            shutil.copy2(src, tmp_target)
            try:
                tmp_target.replace(target)  # os.rename：同一 fs 上原子
            except OSError:
                shutil.copy2(src, target)   # 降级：直接覆盖
                try:
                    tmp_target.unlink()
                except Exception:
                    pass

        # 3) 全量镜像：删除本地存在、但新 ZIP 里没有的文件（PRESERVE_FILES 除外）
        for local in SKILL_DIR.rglob("*"):
            if not local.is_file():
                continue
            rel = local.relative_to(SKILL_DIR)
            if rel.name in PRESERVE_FILES:
                continue  # 受保护文件永不删
            if rel.as_posix() not in expected_rel:
                try:
                    local.unlink()
                except Exception:
                    pass

        # 4) 清理因删除而变空的目录（自底向上）
        for d in sorted(
            (p for p in SKILL_DIR.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        ):
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                except Exception:
                    pass
            except Exception:
                pass

        # 5) 恢复保留文件（防止被误删/误覆盖）
        for fname, data in preserved.items():
            try:
                (SKILL_DIR / fname).write_bytes(data)
            except Exception:
                pass


# ---- 主流程 -----------------------------------------------------------------

def run(force: bool = False) -> int:
    if os.environ.get("TAIJI_NO_HOT_RELOAD"):
        return 0

    # A reloader packaged inside basic is only a compatibility entry point.
    # Delegate to the basic root so its state, version probe and ZIP overwrite
    # remain isolated from standalone installations of this same sub-skill.
    if _delegated_to_basic():
        basic_root = _find_basic_root()
        if basic_root is not None:
            try:
                args = [sys.executable, str(basic_root / "hot_reload.py")]
                if force:
                    args.append("--force")
                return subprocess.run(args, check=False).returncode
            except Exception:
                return 0

    if not force and _ttl_valid():
        return 0  # TTL 内，秒过

    base = _base_url()
    try:
        info = _http_get_json(f"{base}/api/hot-reload/skills/{urllib.parse.quote(SKILL_KEY, safe='')}")
    except Exception:
        # 服务不可用 → 静默跳过，用旧版
        return 0

    remote_version = str(info.get("version", ""))
    remote_ts = str(info.get("version_ts", ""))
    if not remote_version:
        return 0

    # 落盘 git_commit（bad case 版本归属用）。写在 hash 比对之前，
    # 「已是最新」与「下载更新」两条路径都能覆盖；.skill_commit 在
    # PRESERVE_FILES 中，后续解压会备份并恢复，不会丢。
    _write_commit(str(info.get("git_commit", "")))

    if remote_version == _local_version():
        _touch_ttl()
        return 0  # 已是最新

    # 目录不可写 → 更新注定失败，直接放弃：不下载、不解压。
    #
    # 这是本函数最重要的一道闸。历史故障：skill 装在只读共享目录时，下载
    # 成功但解压必然 PermissionError，版本号也写不下去，于是每次调用都重新
    # 下载同一个 ZIP —— 曾累计约 9GB 无效流量、上万次请求。
    # 放在版本检查之后，是为了让提示能带上「本地 X → 最新 Y」这个当时最缺
    # 的诊断信息；同时 ZIP 一个字节都不会下。
    if not _skill_dir_writable():
        _log(
            f"skill 目录不可写，跳过更新（本地 {_local_version() or '未知'} → "
            f"最新 {remote_ts or remote_version}）：{SKILL_DIR}"
        )
        _touch_ttl()  # 记账，避免每次调用都重新探测+查版本
        return 0

    # 需要更新：下载整个 ZIP
    zip_url = info.get("zip_url") or f"/api/hot-reload/skills/{urllib.parse.quote(SKILL_KEY, safe='')}/download"
    if zip_url.startswith("/"):
        zip_url = base + zip_url
    try:
        zip_bytes = _http_get_bytes(zip_url)
    except Exception:
        return 0  # 下载失败，用旧版

    try:
        _safe_extract_and_overwrite(zip_bytes)
    except Exception as e:
        _log(f"更新解压失败，保持旧版：{e}")
        return 0

    _write_local_version(remote_version)
    _touch_ttl()
    _log(f"已更新到 {remote_ts or remote_version}")
    return 0


def main() -> int:
    force = "--force" in sys.argv[1:]
    return run(force=force)


if __name__ == "__main__":
    raise SystemExit(main())
