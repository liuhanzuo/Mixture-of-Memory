#!/usr/bin/env python3
"""
run_eval.py — MoM Agent 评测脚本。

运行合成基准任务并输出评测报告:
- synthetic_update: 记忆更新评测
- profile_bench: 用户画像评测
- longhorizon_chat: 长程对话评测

Usage::

    python scripts/run_eval.py --config-name swa_mom
    python scripts/run_eval.py --config-name swa_only
    python scripts/run_eval.py --config-name swa_mom --tasks synthetic_update profile_bench
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from omegaconf import DictConfig, OmegaConf

from src.agents.memory_agent import MemoryAgent, AgentConfig
from src.agents.session_runner import SessionRunner
from src.backbone import build_backbone_from_config
from src.tasks.synthetic_update_task import SyntheticUpdateTask
from src.tasks.profile_task import ProfileTask
from src.tasks.longhorizon_chat_task import LongHorizonChatTask
from src.eval.cost_eval import CostEvaluator
from src.utils.logging import setup_logging
from src.utils.seeds import set_seed

logger = logging.getLogger(__name__)


def build_agent_from_config(cfg: DictConfig) -> MemoryAgent:
    """从配置构建 MemoryAgent (含 backbone 加载)。

    流程:
    1. 从实验配置解析 backbone 配置并加载模型 + tokenizer
    2. 从实验配置提取记忆层开关
    3. 构建 MemoryAgent 并注入 backbone
    """
    # ---- Step 1: 加载 backbone ---- #
    config_dir = _PROJECT_ROOT / "configs"
    try:
        backbone = build_backbone_from_config(cfg, config_dir=config_dir)
        tokenizer = backbone.get_tokenizer()
        logger.info(
            f"Backbone 加载完成: type={type(backbone).__name__}, "
            f"debug={backbone.is_debug()}, tokenizer={'✓' if tokenizer else '✗'}"
        )
    except Exception as e:
        logger.error(f"Backbone 加载失败: {e}", exc_info=True)
        logger.warning("将使用无 backbone 的规则模式运行。")
        backbone = None
        tokenizer = None

    # ---- Step 2: 提取记忆层开关 ---- #
    mem_cfg = cfg.get("experiment", {}).get("memory", {})
    enable_l1 = mem_cfg.get("l1", {}).get("enabled", True)
    enable_l2 = mem_cfg.get("l2", {}).get("enabled", True)
    enable_l3 = mem_cfg.get("l3", {}).get("enabled", True)

    agent_config = AgentConfig(
        enable_l1=enable_l1,
        enable_l2=enable_l2,
        enable_l3=enable_l3,
    )

    # ---- Step 3: 构建 agent ---- #
    return MemoryAgent(
        config=agent_config,
        backbone=backbone,
        tokenizer=tokenizer,
    )


# ------------------------------------------------------------------ #
#  评测任务执行器
# ------------------------------------------------------------------ #

def eval_synthetic_update(
    agent: MemoryAgent,
    seed: int = 42,
    num_samples: int = 50,
) -> dict[str, Any]:
    """运行 synthetic_update 评测。"""
    logger.info("=" * 50)
    logger.info("开始 Synthetic Update 评测")
    logger.info("=" * 50)

    task = SyntheticUpdateTask(num_samples=num_samples, seed=seed)
    samples = task.generate_samples()

    predictions: list[str] = []
    traces: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        # 将 sample 的对话喂给 agent
        runner = SessionRunner(agent)
        messages = [content for role, content in sample.conversation if role == "user"]
        messages.append(sample.query)

        trace = runner.run_conversation(
            messages=messages,
            session_id=f"update_{sample.sample_id}",
        )
        traces.append(trace.to_dict())

        # 最后一轮的回复作为预测
        if trace.agent_replies:
            predictions.append(trace.agent_replies[-1])
        else:
            predictions.append("")

        agent.reset(keep_l3=False)

        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(samples)}")

    # 使用 task 自带的评估
    report = task.evaluate_batch(samples, predictions)

    # 开销评估
    cost_eval = CostEvaluator()
    cost_report = cost_eval.evaluate_batch(traces)
    report["cost"] = cost_report["metrics"]

    logger.info(f"Synthetic Update 评测完成: overall_accuracy={report['overall_accuracy']:.2%}")
    return report


def eval_profile(
    agent: MemoryAgent,
    seed: int = 42,
    num_samples: int = 30,
) -> dict[str, Any]:
    """运行 profile 评测。"""
    logger.info("=" * 50)
    logger.info("开始 Profile 评测")
    logger.info("=" * 50)

    task = ProfileTask(num_samples=num_samples, seed=seed)
    samples = task.generate_samples()

    predictions: list[str] = []
    traces: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        runner = SessionRunner(agent)
        messages = [content for role, content in sample.conversation if role == "user"]
        messages.append(sample.query)

        trace = runner.run_conversation(
            messages=messages,
            session_id=f"profile_{sample.sample_id}",
        )
        traces.append(trace.to_dict())

        if trace.agent_replies:
            predictions.append(trace.agent_replies[-1])
        else:
            predictions.append("")

        agent.reset(keep_l3=False)

        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(samples)}")

    report = task.evaluate_batch(samples, predictions)

    cost_eval = CostEvaluator()
    cost_report = cost_eval.evaluate_batch(traces)
    report["cost"] = cost_report["metrics"]

    logger.info(f"Profile 评测完成: avg_precision={report['avg_precision']:.2%}")
    return report


def eval_longhorizon_chat(
    agent: MemoryAgent,
    seed: int = 42,
    num_samples: int = 30,
) -> dict[str, Any]:
    """运行 longhorizon_chat 评测。"""
    logger.info("=" * 50)
    logger.info("开始 Long-Horizon Chat 评测")
    logger.info("=" * 50)

    task = LongHorizonChatTask(num_samples=num_samples, seed=seed)
    samples = task.generate_samples()

    predictions: list[str] = []
    traces: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        runner = SessionRunner(agent)
        messages = [content for role, content in sample.conversation if role == "user"]
        messages.append(sample.query)

        trace = runner.run_conversation(
            messages=messages,
            session_id=f"longhorizon_{sample.sample_id}",
        )
        traces.append(trace.to_dict())

        if trace.agent_replies:
            predictions.append(trace.agent_replies[-1])
        else:
            predictions.append("")

        agent.reset(keep_l3=False)

        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(samples)}")

    report = task.evaluate_batch(samples, predictions)

    cost_eval = CostEvaluator()
    cost_report = cost_eval.evaluate_batch(traces)
    report["cost"] = cost_report["metrics"]

    logger.info(f"Long-Horizon Chat 评测完成: overall_accuracy={report['overall_accuracy']:.2%}")
    return report


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

TASK_REGISTRY = {
    "synthetic_update": eval_synthetic_update,
    "profile_bench": eval_profile,
    "longhorizon_chat": eval_longhorizon_chat,
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MoM Agent Evaluation Runner")
    parser.add_argument(
        "--config-name", type=str, default="swa_mom",
        help="实验配置名",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="指定要运行的评测任务 (默认运行配置中指定的全部任务)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=20, help="每个任务的样本数")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    set_seed(args.seed)

    # 加载配置
    config_path = _PROJECT_ROOT / "configs" / "exp" / f"{args.config_name}.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(str(config_path))
    else:
        logger.warning(f"配置文件 {config_path} 不存在, 使用默认配置。")
        cfg = OmegaConf.create({
            "experiment": {
                "name": args.config_name,
                "memory": {"l1": {"enabled": True}, "l2": {"enabled": True}, "l3": {"enabled": True}},
                "eval_tasks": ["synthetic_update", "profile_bench", "longhorizon_chat"],
                "run": {"seed": args.seed, "output_dir": f"outputs/runs/{args.config_name}"},
            }
        })

    # 确定要运行的任务
    if args.tasks:
        task_names = args.tasks
    else:
        task_names = OmegaConf.to_container(
            cfg.get("experiment", {}).get("eval_tasks", []),
            resolve=True,
        )
        if not task_names:
            task_names = list(TASK_REGISTRY.keys())

    # 构建 agent
    agent = build_agent_from_config(cfg)

    # 运行评测
    exp_name = cfg.get("experiment", {}).get("name", args.config_name)
    output_dir = Path(
        cfg.get("experiment", {}).get("run", {}).get("output_dir", f"outputs/runs/{exp_name}")
    )
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_reports: dict[str, Any] = {}
    t_start = time.monotonic()

    for task_name in task_names:
        if task_name not in TASK_REGISTRY:
            logger.warning(f"未知任务: {task_name}, 跳过。")
            continue

        eval_fn = TASK_REGISTRY[task_name]
        report = eval_fn(agent, seed=args.seed, num_samples=args.num_samples)
        all_reports[task_name] = report

    total_time = time.monotonic() - t_start

    # 保存汇总报告
    summary = {
        "experiment": exp_name,
        "config": args.config_name,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "total_time_s": total_time,
        "tasks": {},
    }

    for task_name, report in all_reports.items():
        # 提取核心指标 (排除 details 等大对象)
        task_summary: dict[str, Any] = {}
        for k, v in report.items():
            if k not in ("details", "results", "snapshots"):
                task_summary[k] = v
        summary["tasks"][task_name] = task_summary

    report_path = metrics_dir / f"{exp_name}_eval.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"📊 评测报告: {exp_name}")
    print(f"{'='*60}")
    print(f"配置: {args.config_name}")
    print(f"随机种子: {args.seed}")
    print(f"总耗时: {total_time:.1f}s")
    print()

    for task_name, report in all_reports.items():
        print(f"  📋 {task_name}:")
        if "overall_accuracy" in report:
            print(f"     Overall Accuracy: {report['overall_accuracy']:.2%}")
        if "avg_precision" in report:
            print(f"     Avg Precision: {report['avg_precision']:.2%}")
        if "type_accuracy" in report:
            for t, acc in report["type_accuracy"].items():
                print(f"     {t}: {acc:.2%}")
        if "type_avg_precision" in report:
            for t, acc in report["type_avg_precision"].items():
                print(f"     {t}: {acc:.2%}")
        print()

    print(f"📁 报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
