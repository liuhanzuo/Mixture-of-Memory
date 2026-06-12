"""
HeartbeatMonitor: 训练过程心跳监控工具。

功能:
    - 实时记录训练状态到 heartbeat.json
    - 检测超时未更新的任务
    - 支持 OpenClaw heartbeat 读取

Usage:
    # 在训练脚本中集成
    from heartbeat_monitor import HeartbeatMonitor
    monitor = HeartbeatMonitor("outputs/my_run/heartbeat.json")
    monitor.update("running", progress=0.5, metrics={"loss": 2.3})

    # 检查任务状态
    python -c "from heartbeat_monitor import HeartbeatMonitor; \
               h=HeartbeatMonitor(); print('Stale:', h.check_stale_tasks())"
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class HeartbeatMonitor:
    """轻量级训练心跳监控器。

    将训练状态定期写入 JSON 文件, 供外部监控工具 (如 OpenClaw heartbeat) 读取。

    Attributes:
        heartbeat_file: 心跳文件路径.
        start_time: 监控开始时间.
    """

    def __init__(self, heartbeat_file: str = "heartbeat.json"):
        self.heartbeat_file = heartbeat_file
        self.start_time = time.time()
        # 确保目录存在
        Path(heartbeat_file).parent.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        status: str,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """更新心跳文件。

        Args:
            status: 任务状态 ("running", "completed", "error", "idle").
            progress: 进度 [0.0, 1.0].
            metrics: 训练指标 (loss, lr, epoch, batch 等).
            extra: 其他自定义信息.
        """
        heartbeat_data = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "uptime_seconds": int(time.time() - self.start_time),
            "progress": progress,
            "metrics": metrics or {},
            "extra": extra or {},
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 原子写入: 先写临时文件, 再 rename (避免读到半写状态)
        tmp_file = self.heartbeat_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(heartbeat_data, f, indent=2, ensure_ascii=False)
        
        # rename 是原子的 (POSIX)
        try:
            os.replace(tmp_file, self.heartbeat_file)
        except OSError:
            # fallback: 直接写 (Windows 或跨文件系统)
            with open(self.heartbeat_file, "w", encoding="utf-8") as f:
                json.dump(heartbeat_data, f, indent=2, ensure_ascii=False)

    def read(self) -> dict[str, Any] | None:
        """读取当前心跳状态。

        Returns:
            心跳数据字典, 或 None (文件不存在).
        """
        if not os.path.exists(self.heartbeat_file):
            return None
        try:
            with open(self.heartbeat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def check_stale_tasks(self, timeout_minutes: int = 30) -> bool:
        """检查是否有超时未更新的任务。

        Args:
            timeout_minutes: 超时阈值 (分钟).

        Returns:
            True 表示任务可能已停止/挂起, False 表示正常.
        """
        data = self.read()
        if data is None:
            return True  # 心跳文件不存在，认为有异常

        if data.get("status") in ("completed", "error"):
            return False  # 已完成或报错不算超时

        try:
            last_update = datetime.fromisoformat(data["timestamp"])
            time_diff = (datetime.now() - last_update).total_seconds() / 60
            return time_diff > timeout_minutes
        except (ValueError, KeyError):
            return True

    def get_progress_summary(self) -> str:
        """获取人类可读的进度摘要。"""
        data = self.read()
        if data is None:
            return "无心跳记录"

        status = data.get("status", "unknown")
        progress = data.get("progress")
        metrics = data.get("metrics", {})
        last_update = data.get("last_update", "unknown")
        uptime = data.get("uptime_seconds", 0)

        parts = [f"状态: {status}"]
        if progress is not None:
            parts.append(f"进度: {progress:.1%}")
        if metrics:
            loss = metrics.get("loss")
            if loss is not None:
                parts.append(f"loss: {loss:.4f}")
            lr = metrics.get("learning_rate")
            if lr is not None:
                parts.append(f"lr: {lr:.2e}")
            epoch = metrics.get("epoch")
            batch = metrics.get("batch")
            if epoch is not None or batch is not None:
                parts.append(f"epoch/batch: {epoch}/{batch}")
        parts.append(f"运行: {uptime // 60}分钟")
        parts.append(f"最后更新: {last_update}")

        return " | ".join(parts)

    def mark_completed(self, extra: dict[str, Any] | None = None) -> None:
        """标记任务完成。"""
        self.update("completed", progress=1.0, extra=extra)

    def mark_error(self, error_msg: str, extra: dict[str, Any] | None = None) -> None:
        """标记任务出错。"""
        error_extra = {"error": error_msg}
        if extra:
            error_extra.update(extra)
        self.update("error", extra=error_extra)
