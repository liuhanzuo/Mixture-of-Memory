#!/usr/bin/env python3
"""
run_ablation.py — MoM Agent 消融实验脚本。

自动遍历所有实验配置 (swa_only, swa_l1, swa_l1_l2, swa_mom, fullattn_baseline)，
对每个配置运行全部评测任务，最后生成横向对比表。

Usage::

    python scripts/run_ablation.py
    python scripts/run_ablation.py --configs swa_only swa_mom
    python scripts/run_ablation.py --num-samples 10 --seed 42
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

# ---- 默认配置列表 (按消融顺序) ----
DEFAULT_CONFIGS = [
    "swa_only",
    "swa_l1",
    "swa_l1_l2",
    "swa_mom",
    "fullattn_baseline",
]


def build_agent(cfg: DictConfig) -> MemoryAgent:
    """从配置构建 agent (含 backbone 加载)。"""
    # 加载 backbone
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

    mem_cfg = cfg.get("experiment", {}).get("memory", {})
    return MemoryAgent(
        config=AgentConfig(
            enable_l1=mem_cfg.get("l1", {}).get("enabled", False),
            enable_l2=mem_cfg.get("l2", {}).get("enabled", False),
            enable_l3=mem_cfg.get("l3", {}).get("enabled", False),
        ),
        backbone=backbone,
        tokenizer=tokenizer,
    )


def run_single_config(
    config_name: str,
    seed: int = 42,
    num_samples: int = 20,
) -> dict[str, Any]:
    """运行单个配置的全部评测任务，返回汇总指标。"""
    config_path = _PROJECT_ROOT / "configs" / "exp" / f"{config_name}.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(str(config_path))
    else:
        logger.warning(f"配置 {config_path} 不存在, 使用默认 (全部启用)。")
        cfg = OmegaConf.create({
            "experiment": {
                "name": config_name,
                "memory": {"l1": {"enabled": True}, "l2": {"enabled": True}, "l3": {"enabled": True}},
            }
        })

    agent = build_agent(cfg)
    results: dict[str, Any] = {"config": config_name}

    # L3 不应跨独立 sample 累积：每个 sample 是独立测试用例
    # L3 的跨 session 积累应在同一 sample 的多轮对话内生效

    # ---- Synthetic Update ----
    logger.info(f"[{config_name}] Running synthetic_update...")
    task_update = SyntheticUpdateTask(num_samples=num_samples, seed=seed)
    samples = task_update.generate_samples()
    predictions = _run_task_samples(agent, samples, keep_l3=False)
    report = task_update.evaluate_batch(samples, predictions)
    results["update_accuracy"] = report["overall_accuracy"]
    results["update_type_accuracy"] = report.get("type_accuracy", {})

    # ---- Profile ----
    logger.info(f"[{config_name}] Running profile_bench...")
    task_profile = ProfileTask(num_samples=num_samples, seed=seed)
    p_samples = task_profile.generate_samples()
    p_predictions = _run_task_samples(agent, p_samples, sample_type="profile", keep_l3=False)
    p_report = task_profile.evaluate_batch(p_samples, p_predictions)
    results["profile_precision"] = p_report["avg_precision"]
    results["profile_type_precision"] = p_report.get("type_avg_precision", {})

    # ---- Long-Horizon Chat ----
    logger.info(f"[{config_name}] Running longhorizon_chat...")
    task_lh = LongHorizonChatTask(num_samples=num_samples, seed=seed)
    lh_samples = task_lh.generate_samples()
    lh_predictions = _run_task_samples(agent, lh_samples, sample_type="longhorizon", keep_l3=False)
    lh_report = task_lh.evaluate_batch(lh_samples, lh_predictions)
    results["longhorizon_accuracy"] = lh_report["overall_accuracy"]
    results["longhorizon_type_accuracy"] = lh_report.get("type_accuracy", {})
    results["longhorizon_distance_accuracy"] = lh_report.get("distance_accuracy", {})

    # ---- 最终 agent 统计 ----
    results["agent_stats"] = agent.get_stats()

    return results


def _run_task_samples(
    agent: MemoryAgent,
    samples: list[Any],
    sample_type: str = "update",
    keep_l3: bool = False,
) -> list[str]:
    """通用的样本运行函数：将样本对话喂给 agent，返回预测列表。

    Args:
        agent: MemoryAgent 实例.
        samples: 任务样本列表.
        sample_type: 样本类型标识.
        keep_l3: 是否在 sample 间保留 L3 长期画像 (启用 L3 时应为 True).
    """
    predictions: list[str] = []

    for sample in samples:
        runner = SessionRunner(agent)

        # 提取用户消息
        if hasattr(sample, "conversation"):
            messages = [content for role, content in sample.conversation if role == "user"]
        else:
            messages = []

        # 添加查询
        if hasattr(sample, "query"):
            messages.append(sample.query)

        if not messages:
            predictions.append("")
            continue

        trace = runner.run_conversation(
            messages=messages,
            session_id=f"{sample_type}_{getattr(sample, 'sample_id', 'unknown')}",
        )

        if trace.agent_replies:
            predictions.append(trace.agent_replies[-1])
        else:
            predictions.append("")

        agent.reset(keep_l3=keep_l3)

    return predictions


def format_comparison_table(all_results: list[dict[str, Any]]) -> str:
    """格式化横向对比表 (Markdown)。"""
    if not all_results:
        return "No results."

    lines: list[str] = []

    # 表头
    config_names = [r["config"] for r in all_results]
    header = "| Metric | " + " | ".join(config_names) + " |"
    separator = "|" + "---|" * (len(config_names) + 1)
    lines.append(header)
    lines.append(separator)

    # 核心指标行
    metrics = [
        ("Update Accuracy", "update_accuracy"),
        ("Profile Precision", "profile_precision"),
        ("LongHorizon Accuracy", "longhorizon_accuracy"),
    ]

    for display_name, key in metrics:
        values = []
        for r in all_results:
            val = r.get(key, 0.0)
            values.append(f"{val:.2%}" if isinstance(val, float) else str(val))
        row = f"| {display_name} | " + " | ".join(values) + " |"
        lines.append(row)

    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MoM Ablation Study")
    parser.add_argument(
        "--configs", nargs="*", default=None,
        help="要运行的实验配置名列表 (默认运行全部)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=10, help="每个任务的样本数 (消融用小数据集)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    set_seed(args.seed)

    config_names = args.configs if args.configs else DEFAULT_CONFIGS

    all_results: list[dict[str, Any]] = []
    t_start = time.monotonic()

    for config_name in config_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"消融实验: {config_name}")
        logger.info(f"{'='*60}")

        result = run_single_config(
            config_name=config_name,
            seed=args.seed,
            num_samples=args.num_samples,
        )
        all_results.append(result)

    total_time = time.monotonic() - t_start

    # 输出对比表
    table = format_comparison_table(all_results)
    print(f"\n{'='*60}")
    print("📊 消融实验结果对比")
    print(f"{'='*60}")
    print(table)
    print(f"\n总耗时: {total_time:.1f}s")

    # 保存结果
    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "ablation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "configs": config_names,
            "results": all_results,
            "comparison_table": table,
            "total_time_s": total_time,
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"📁 结果已保存至: {report_path}")


if __name__ == "__main__":
    main()
