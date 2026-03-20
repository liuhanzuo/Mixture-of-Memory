#!/usr/bin/env python3
"""
run_chat.py — MoM Agent 对话运行脚本。

支持两种模式:
1. 交互式: 从终端输入对话
2. 脚本式: 从 JSON 文件加载对话

Usage::

    # 交互式 (默认)
    python scripts/run_chat.py --config-name swa_mom

    # 从文件加载
    python scripts/run_chat.py --config-name swa_mom mode=file input_file=data/raw/demo.json

    # 快速 demo (内置示例对话)
    python scripts/run_chat.py --config-name swa_mom mode=demo
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from omegaconf import DictConfig, OmegaConf

from src.agents.memory_agent import MemoryAgent, AgentConfig
from src.agents.session_runner import SessionRunner
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed

logger = logging.getLogger(__name__)


# ---- 内置 demo 对话 ----
DEMO_CONVERSATIONS = [
    {
        "session_id": "demo_session_001",
        "messages": [
            "你好，我叫小明，最近在研究大语言模型的稀疏化训练。",
            "我习惯用Python和PyTorch，偏好结构化的技术回答。",
            "帮我解释一下SWA（Sliding Window Attention）的原理。",
            "我的项目叫MoM-Agent，目标是为SWA模型提供层次化记忆补偿。",
            "对了，我现在住在深圳。",
            "Transformer里的RoPE位置编码是怎么工作的？",
            "回到MoM的话题，L1用的是衰减关联矩阵，你还记得吗？",
            "我之前说住在深圳，其实搬到北京了，更新一下。",
            "总结一下你对我的了解。",
        ],
    },
]


def build_agent_from_config(cfg: DictConfig) -> MemoryAgent:
    """从 OmegaConf 配置构建 MemoryAgent。"""
    # 提取记忆层开关
    mem_cfg = cfg.get("experiment", {}).get("memory", {})
    enable_l1 = mem_cfg.get("l1", {}).get("enabled", True)
    enable_l2 = mem_cfg.get("l2", {}).get("enabled", True)
    enable_l3 = mem_cfg.get("l3", {}).get("enabled", True)

    agent_config = AgentConfig(
        enable_l1=enable_l1,
        enable_l2=enable_l2,
        enable_l3=enable_l3,
    )

    agent = MemoryAgent(config=agent_config)
    return agent


def run_demo(cfg: DictConfig) -> None:
    """运行内置 demo 对话。"""
    agent = build_agent_from_config(cfg)
    runner = SessionRunner(agent)

    output_dir = cfg.get("experiment", {}).get("run", {}).get("output_dir", "outputs/runs/demo")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for conv in DEMO_CONVERSATIONS:
        logger.info(f"运行 demo 对话: {conv['session_id']}")
        trace = runner.run_conversation(
            messages=conv["messages"],
            session_id=conv["session_id"],
        )

        # 保存 trace
        trace_path = output_path / f"{trace.session_id}.json"
        trace.save(trace_path)

        # 打印摘要
        print(f"\n{'='*60}")
        print(f"Session: {trace.session_id}")
        print(f"Turns: {len(trace.turns)}")
        print(f"Total time: {trace.total_time_ms:.1f}ms")
        print(f"{'='*60}")
        for i, (user_msg, reply) in enumerate(zip(trace.user_messages, trace.agent_replies)):
            print(f"\n👤 User [{i+1}]: {user_msg}")
            print(f"🤖 Agent: {reply}")

    # 导出画像
    profile_path = output_path / "profile.md"
    agent.export_profile(str(profile_path), format="markdown")
    print(f"\n📋 Profile exported to {profile_path}")

    # 打印统计
    stats = agent.get_stats()
    print(f"\n📊 Stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")


def run_interactive(cfg: DictConfig) -> None:
    """运行交互式对话。"""
    agent = build_agent_from_config(cfg)
    runner = SessionRunner(agent)

    output_dir = cfg.get("experiment", {}).get("run", {}).get("output_dir", "outputs/runs/interactive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trace = runner.run_interactive(
        greeting="欢迎使用 MoM Agent! 我是一个带有层次化记忆的智能助手。",
    )

    # 保存 trace
    trace_path = output_path / f"{trace.session_id}.json"
    trace.save(trace_path)


def run_from_file(cfg: DictConfig, input_file: str) -> None:
    """从 JSON 文件加载对话并运行。"""
    agent = build_agent_from_config(cfg)
    runner = SessionRunner(agent)

    output_dir = cfg.get("experiment", {}).get("run", {}).get("output_dir", "outputs/runs/file")

    traces = runner.run_from_file(
        input_path=input_file,
        output_dir=output_dir,
    )

    print(f"\n✅ 完成 {len(traces)} 个会话的运行。")
    for trace in traces:
        print(f"   Session {trace.session_id}: {len(trace.turns)} turns, {trace.total_time_ms:.1f}ms")


def main() -> None:
    """主入口。

    简化版配置加载 (不依赖 Hydra decorator, 方便脚本运行)。
    """
    import argparse

    parser = argparse.ArgumentParser(description="MoM Agent Chat Runner")
    parser.add_argument(
        "--config-name", type=str, default="swa_mom",
        help="实验配置名 (对应 configs/exp/ 下的文件)",
    )
    parser.add_argument(
        "--mode", type=str, default="demo",
        choices=["demo", "interactive", "file"],
        help="运行模式: demo / interactive / file",
    )
    parser.add_argument(
        "--input-file", type=str, default=None,
        help="脚本模式的输入 JSON 文件路径",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="日志级别",
    )
    args = parser.parse_args()

    # 设置日志
    setup_logging(level=args.log_level)

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    config_path = _PROJECT_ROOT / "configs" / "exp" / f"{args.config_name}.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(str(config_path))
        logger.info(f"加载配置: {config_path}")
    else:
        logger.warning(f"配置文件 {config_path} 不存在, 使用默认配置。")
        cfg = OmegaConf.create({
            "experiment": {
                "name": args.config_name,
                "memory": {"l1": {"enabled": True}, "l2": {"enabled": True}, "l3": {"enabled": True}},
                "run": {"seed": args.seed, "output_dir": f"outputs/runs/{args.config_name}"},
            }
        })

    # 运行
    if args.mode == "demo":
        run_demo(cfg)
    elif args.mode == "interactive":
        run_interactive(cfg)
    elif args.mode == "file":
        if args.input_file is None:
            print("❌ 文件模式需要指定 --input-file 参数。")
            sys.exit(1)
        run_from_file(cfg, args.input_file)


if __name__ == "__main__":
    main()
