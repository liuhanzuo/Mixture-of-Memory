"""
时间工具模块。

提供:
- ISO 8601 时间戳
- 耗时计算
- 上下文管理器 Timer
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

logger = logging.getLogger(__name__)


def now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。

    Returns:
        如 "2025-01-15T08:30:00Z".
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_local_iso() -> str:
    """返回当前本地时间的 ISO 8601 字符串。

    Returns:
        如 "2025-01-15T16:30:00".
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def elapsed_ms(start: float) -> float:
    """计算从 start 到现在的耗时 (毫秒)。

    Args:
        start: time.monotonic() 返回的起始时间.

    Returns:
        耗时毫秒数.
    """
    return (time.monotonic() - start) * 1000.0


class Timer:
    """简易计时器, 支持上下文管理器和手动 start/stop。

    Usage::

        # 上下文管理器模式
        with Timer("L1 update") as t:
            do_l1_update()
        print(f"Elapsed: {t.elapsed_ms:.1f}ms")

        # 手动模式
        t = Timer("My task")
        t.start()
        do_something()
        t.stop()
        print(t.elapsed_ms)
    """

    def __init__(self, name: str = "Timer", log: bool = True):
        """
        Args:
            name: 计时器名称 (用于日志).
            log: 是否在结束时自动打印日志.
        """
        self.name = name
        self.log = log
        self._start: float = 0.0
        self._end: float = 0.0
        self._running: bool = False

    def start(self) -> "Timer":
        """开始计时。"""
        self._start = time.monotonic()
        self._running = True
        return self

    def stop(self) -> float:
        """停止计时并返回耗时毫秒数。"""
        self._end = time.monotonic()
        self._running = False
        ms = self.elapsed_ms
        if self.log:
            logger.debug(f"[Timer] {self.name}: {ms:.2f}ms")
        return ms

    @property
    def elapsed_ms(self) -> float:
        """当前耗时 (毫秒)。如果仍在运行, 返回截至当前的耗时。"""
        end = self._end if not self._running else time.monotonic()
        return (end - self._start) * 1000.0

    @property
    def elapsed_s(self) -> float:
        """当前耗时 (秒)。"""
        return self.elapsed_ms / 1000.0

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self._running else f"{self.elapsed_ms:.2f}ms"
        return f"Timer({self.name!r}, {status})"


@contextmanager
def log_elapsed(name: str, level: int = logging.DEBUG) -> Generator[None, None, None]:
    """上下文管理器: 自动记录代码块的执行耗时。

    Usage::

        with log_elapsed("data loading"):
            load_data()

    Args:
        name: 操作名称.
        level: 日志级别.
    """
    t0 = time.monotonic()
    yield
    ms = (time.monotonic() - t0) * 1000.0
    logger.log(level, f"[Elapsed] {name}: {ms:.2f}ms")
