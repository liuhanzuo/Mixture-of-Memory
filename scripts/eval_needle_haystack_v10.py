"""
Needle-in-Haystack Evaluation for RMT v10.

Uses RMTv10Model with sandwich injection, optional L1/L2 memory.
"""

import os, sys, json, random, string, argparse
from pathlib import Path
from datetime import datetime
import logging

import torch
import torch.nn as nn
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_v10 import RMTv10Config, RMTv10Model, RMTv10Memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Haystack generation ────────────────────────────────────────────────────

RANDOM_TOPICS = [
    "The history of computing spans several centuries, from early mechanical calculators to modern quantum systems. Charles Babbage conceptualized the Analytical Engine in the 1830s.",
    "Photosynthesis is the process by which green plants transform light energy into chemical energy. During photosynthesis, light energy is captured and used to convert water and carbon dioxide into oxygen.",
    "The Mediterranean Sea is connected to the Atlantic Ocean, covering approximately 2.5 million square kilometers.",
    "Quantum mechanics provides a description of physical properties at the scale of atoms and subatomic particles.",
    "The Great Wall of China is a series of fortifications generally built along an east-to-west line across the historical northern borders.",
    "Machine learning algorithms build models based on training data to make predictions without being explicitly programmed.",
    "The water cycle describes the continuous movement of water on, above and below the surface of the Earth.",
    "Renaissance art marks a cultural rebirth with highly realistic linear perspective.",
    "The Amazon River is the largest river by discharge volume, with a basin covering over 7 million square kilometers.",
    "Volcanoes are ruptures in the crust that allow hot lava and gases to escape from a magma chamber.",
]

ZH_HAYSTACK_TOPICS = [
    "计算机科学的发展经历了数十年的演进，从早期的机械计算器到现代的量子计算系统。",
    "光合作用是绿色植物将光能转化为化学能的过程。",
    "量子力学是物理学的一个基本理论，描述原子和亚原子粒子的物理性质。",
    "长城是中国古代的军事防御工程，修筑历史可上溯到西周时期。",
    "机器学习算法基于训练数据构建模型，做出预测或决策。",
    "水循环描述地球表面、上方和下方水的持续运动。",
    "文艺复兴艺术标志着中世纪末期文化复兴和现代世界的崛起。",
    "亚马逊河是南美洲最大的河流，盆地覆盖面积超过700万平方公里。",
    "火山是地壳中的破裂口，允许热熔岩和气体逸出。",
    "人工智能致力于创建能执行通常需要人类智能的任务的系统。",
]

PROJECT_NAMES_ZH = [
    "阿尔法", "贝塔", "伽马", "德尔塔", "艾普西隆", "泽塔", "伊塔", "西塔",
    "卡帕", "拉姆达", "缪", "纽", "克西", "欧米克戎", "派", "柔",
    "星辰计划", "月光计划", "曙光计划", "银河计划", "极光计划",
    "凤凰计划", "雷霆计划", "天启计划", "深渊计划",
]


def generate_haystack_text(tokenizer, target_tokens, rng, lang="en"):
    topics = ZH_HAYSTACK_TOPICS if lang == "zh" else RANDOM_TOPICS
    tokens = []
    while len(tokens) < target_tokens:
        tokens.extend(tokenizer.encode(rng.choice(topics), add_special_tokens=False))
    return tokens[:target_tokens]


def generate_needle(rng, lang="en"):
    if lang == "zh":
        name = rng.choice(PROJECT_NAMES_ZH)
        code = "".join(rng.choices(string.digits, k=6))
        return f"记住这个信息：{name} 的编号是 {code}。", code, f"请问 {name} 的编号是什么？", name
    else:
        code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
        return f"The secret code is {code}.", code, "What is the secret code mentioned in the document?", None


# ─── Model loading ──────────────────────────────────────────────────────────

def load_base_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def load_rmt_v10_model(base_model_path, checkpoint_dir, device):
    # Auto-resolve: if checkpoint_dir has a final/ subdirectory, use it for weights
    # but look for rmt_config.json in parent first
    config_path = os.path.join(checkpoint_dir, "rmt_config.json")
    weight_dir = checkpoint_dir
    if not os.path.exists(config_path):
        # Maybe checkpoint_dir is the run dir, weights are in final/
        final_dir = os.path.join(checkpoint_dir, "final")
        if os.path.exists(os.path.join(checkpoint_dir, "rmt_config.json")):
            config_path = os.path.join(checkpoint_dir, "rmt_config.json")
        elif os.path.exists(os.path.join(final_dir, "config.json")):
            weight_dir = final_dir
            config_path = os.path.join(checkpoint_dir, "rmt_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No rmt_config.json in {checkpoint_dir}")
    with open(config_path) as f:
        rmt_cfg = json.load(f)

    # Build RMTv10Config from training config
    v10_cfg = RMTv10Config()
    v10_cfg.num_mem_tokens = rmt_cfg.get("num_mem_tokens", 16)
    v10_cfg.segment_length = rmt_cfg.get("segment_length", 1024)
    v10_cfg.max_n_segments = rmt_cfg.get("max_segments", 4)
    v10_cfg.use_l1 = rmt_cfg.get("use_l1", False)
    v10_cfg.l1_num_tokens = rmt_cfg.get("l1_num_tokens", 8)
    v10_cfg.l1_update_freq = rmt_cfg.get("l1_update_freq", 3)
    v10_cfg.l1_inject_layer = rmt_cfg.get("l1_inject_layer", -1)
    v10_cfg.use_l2 = rmt_cfg.get("use_l2", False)
    v10_cfg.l2_num_tokens = rmt_cfg.get("l2_num_tokens", 4)
    v10_cfg.l2_update_freq = rmt_cfg.get("l2_update_freq", 6)
    v10_cfg.l2_inject_layer = rmt_cfg.get("l2_inject_layer", -1)
    v10_cfg.use_importance_routing = rmt_cfg.get("use_importance_routing", True)

    # Tokenizer
    tok_dir = weight_dir if os.path.exists(os.path.join(weight_dir, "tokenizer_config.json")) else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model (merged or with adapter)
    merged_path = os.path.join(weight_dir, "model.safetensors")
    if os.path.exists(merged_path):
        model = AutoModelForCausalLM.from_pretrained(
            weight_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
        )
    else:
        from peft import PeftModel
        adapter_path = os.path.join(weight_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path) or os.path.exists(os.path.join(weight_dir, "adapter_model.bin")):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
            )
            model = PeftModel.from_pretrained(model, weight_dir)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
            )
    model.eval()

    # Wrap with RMT v10 (rmt_v10.py patched to handle both PeftModel and direct loading)
    rmt_model = RMTv10Memory(v10_cfg).wrap(model)
    rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)
    rmt_model.eval()

    # Load memory weights
    rmt_path = os.path.join(weight_dir, "rmt_memory.pt")
    if os.path.exists(rmt_path):
        state = torch.load(rmt_path, map_location=device)
        # Validate keys match v10 architecture before loading
        v10_expected = {"l0.", "recon_head."}
        actual_prefixes = set(k.split(".")[0] + "." for k in state.keys())
        has_v10_keys = any(k.startswith(p) for k in state.keys() for p in ["l0.", "recon_head.", "l1.", "l2."])
        has_legacy_keys = any(k.startswith(p) for k in state.keys() for p in ["memory_embeddings", "extractor.", "memory_predictor.", "segment_bias."])
        if has_legacy_keys and not has_v10_keys:
            raise RuntimeError(
                f"rmt_memory.pt keys are from an older RMT version (v5/v6/v8), not v10. "
                f"First key: {list(state.keys())[0]}. "
                f"load_state_dict would load NOTHING → randomly-initialized memory → 0% accuracy. "
                f"Fix: use a checkpoint trained with rmt_v10.py, or retrain."
            )
        result = rmt_model.load_state_dict(state, strict=False)
        loaded_keys = [k for k in result.unexpected_keys if not k.startswith("l0.") and not k.startswith("l1.") and not k.startswith("l2.") and not k.startswith("recon_head.")]
        if result.missing_keys:
            logger.warning(f"Missing RMT keys (using random init): {result.missing_keys}")
        if result.unexpected_keys:
            logger.info(f"Ignored keys from checkpoint: {result.unexpected_keys}")
        loaded_count = len(result.unexpected_keys)  # keys that matched
        logger.info(f"Loaded RMT v10 memory weights from {rmt_path} ({len(result.unexpected_keys)} keys matched)")
    else:
        logger.warning(f"No rmt_memory.pt in {weight_dir}")

    config = {
        "segment_length": v10_cfg.segment_length,
        "max_segments": v10_cfg.max_n_segments,
        "num_mem_tokens": v10_cfg.num_mem_tokens,
        "use_l1": v10_cfg.use_l1,
        "use_l2": v10_cfg.use_l2,
    }
    return rmt_model, tokenizer, config


# ─── Inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_base(model, tokenizer, input_ids, max_new_tokens=30):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=generated)
        next_tok = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        if next_tok.item() == tokenizer.eos_token_id:
            break
        generated = torch.cat([generated, next_tok], dim=1)
    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def generate_rmt_v10(rmt_model, tokenizer, input_ids, rmt_config, max_new_tokens=30, question_text=None):
    """
    Process document through RMT v10 segments, then generate answer using memory.
    
    v10 uses sandwich injection: [old_mem | content | placeholder_mem]
    After processing all segments, we get a final memory_state from L0.
    For generation, we prepend memory embeddings to the question and generate token-by-token.
    """
    cfg = rmt_model.config
    seg_len = cfg.segment_length
    K = cfg.num_mem_tokens
    B, L = input_ids.shape
    device = input_ids.device
    dtype = torch.bfloat16
    base_model = rmt_model.base_model

    # Determine number of segments (same logic as v10 forward)
    num_segments = min(cfg.max_n_segments, L // seg_len)
    num_segments = max(1, num_segments)

    # Process document segments
    segments = rmt_model._segment_input(input_ids, num_segments)
    memory_state = rmt_model.l0.get_initial_memory(B).to(dtype=dtype)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for seg_idx in range(num_segments):
            seg_ids = segments[seg_idx].to(device)
            content_embeds = base_model.get_input_embeddings()(seg_ids)
            inputs_embeds, attn_mask_2d = rmt_model.l0.build_sandwich_fast(content_embeds, memory_state)
            attn_mask_4d = rmt_model._make_4d_attn_mask(attn_mask_2d, dtype)
            total_len = inputs_embeds.shape[1]
            position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(B, -1)

            outputs = base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            new_memory = rmt_model.l0.extract_new_memory(outputs.hidden_states[-1])
            memory_state, _ = rmt_model.l0.apply_importance_routing(memory_state, new_memory)

    # Generate answer with memory prepended
    if question_text is None:
        question_text = "What is the secret code mentioned in the document?"
    q_ids = tokenizer.encode(question_text, add_special_tokens=False, return_tensors="pt").to(device)
    q_len = q_ids.shape[1]

    # Embed question
    q_embeds = base_model.get_input_embeddings()(q_ids)  # [B, q_len, D]

    # Prepend memory embeddings: [mem K | question q_len]
    # memory_state shape: [B, K, D]
    inputs_embeds = torch.cat([memory_state, q_embeds], dim=1)  # [B, K+q_len, D]

    generated_ids = []
    for step in range(max_new_tokens):
        cur_len = inputs_embeds.shape[1]  # K + q_len + already generated
        # Build attention mask: full attention for all tokens (no RMT causal mask during generation)
        attn_mask = torch.ones(1, 1, cur_len, cur_len, device=device, dtype=dtype)
        # Causal mask for text tokens; memory can attend to each other and be attended by all
        # Simple causal mask:
        causal = torch.tril(torch.ones(cur_len, cur_len, device=device, dtype=torch.bool))
        attn_mask = torch.zeros(1, 1, cur_len, cur_len, device=device, dtype=dtype)
        attn_mask[0, 0] = causal.float()
        attn_mask.masked_fill_(~causal, float('-inf'))

        position_ids = torch.arange(cur_len, device=device).unsqueeze(0).expand(B, -1)

        outputs = base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
        logits = base_model.lm_head(outputs.last_hidden_state[:, -1:, :])
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if next_tok.item() == tokenizer.eos_token_id:
            break
        generated_ids.append(next_tok.item())
        next_embed = base_model.get_input_embeddings()(next_tok)
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ─── Main evaluation ───────────────────────────────────────────────────────

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading base model ...")
    base_model, tokenizer = load_base_model(args.base_model, device)

    logger.info("Loading RMT v10 model ...")
    rmt_model, rmt_tokenizer, rmt_config = load_rmt_v10_model(args.base_model, args.checkpoint, device)

    segment_length = rmt_config["segment_length"]
    max_segments = rmt_config["max_segments"]
    base_max_len = getattr(base_model.config, "max_position_embeddings", 4096)

    model_label = "rmt_v10"
    if rmt_config.get("use_l2"):
        model_label += "_l2"

    all_results = []
    total_configs = len(args.lengths) * len(args.depths) * args.num_trials * 2
    done = 0

    for target_length in args.lengths:
        for depth in args.depths:
            for trial_idx in range(args.num_trials):
                needle_text, expected_answer, question, needle_name = generate_needle(rng, lang=args.lang)

                haystack_tokens = generate_haystack_text(tokenizer, target_length, rng, lang=args.lang)
                needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
                question_suffix = f"\n\n{question}"
                question_tokens = tokenizer.encode(question_suffix, add_special_tokens=False)

                insert_pos = int(len(haystack_tokens) * depth)

                # Base model
                full_tokens = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:] + question_tokens
                base_input = torch.tensor([full_tokens], dtype=torch.long, device=device)
                if base_input.shape[1] > base_max_len:
                    base_input = base_input[:, -base_max_len:]
                base_answer = generate_base(base_model, tokenizer, base_input)
                base_correct = expected_answer.lower() in base_answer.lower()
                all_results.append({"model_type": "base", "target_length": target_length, "depth": depth,
                                    "trial": trial_idx, "expected": expected_answer, "answer": base_answer,
                                    "is_correct": base_correct})

                # RMT v10 model (question NOT in document segments)
                rmt_doc_tokens = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
                rmt_tokens = list(rmt_doc_tokens)
                remainder = len(rmt_tokens) % segment_length
                if remainder:
                    rmt_tokens += [tokenizer.pad_token_id] * (segment_length - remainder)
                num_segs = len(rmt_tokens) // segment_length
                if num_segs > max_segments:
                    rmt_tokens = rmt_tokens[-(max_segments * segment_length):]

                rmt_input = torch.tensor([rmt_tokens], dtype=torch.long, device=device)
                rmt_answer = generate_rmt_v10(rmt_model, tokenizer, rmt_input, rmt_config, question_text=question)
                rmt_correct = expected_answer.lower() in rmt_answer.lower()
                all_results.append({"model_type": model_label, "target_length": target_length, "depth": depth,
                                    "trial": trial_idx, "expected": expected_answer, "answer": rmt_answer,
                                    "is_correct": rmt_correct})

                done += 2
                logger.info(f"[{done}/{total_configs}] len={target_length} depth={depth:.0%} "
                            f"base={'✓' if base_correct else '✗'} {model_label}={'✓' if rmt_correct else '✗'}")

    # ─── Summary ───
    summary = {}
    for mt in ["base", model_label]:
        subset = [r for r in all_results if r["model_type"] == mt]
        correct = sum(r["is_correct"] for r in subset)
        acc = correct / len(subset) if subset else 0
        summary[f"{mt}_accuracy"] = acc
        logger.info(f"\n{mt} accuracy: {acc:.2%} ({correct}/{len(subset)})")

    print("\n" + "=" * 70)
    print(f"NEEDLE-IN-A-HAYSTACK RESULTS — {model_label}")
    print("=" * 70)
    print(f"{'Length':>8} {'Depth':>8} | {'Base':>8} {model_label:>10}")
    print("-" * 70)
    for length in args.lengths:
        for depth in args.depths:
            bs = [r for r in all_results if r["model_type"] == "base" and r["target_length"] == length and abs(r["depth"] - depth) < 1e-6]
            rs = [r for r in all_results if r["model_type"] == model_label and r["target_length"] == length and abs(r["depth"] - depth) < 1e-6]
            b = sum(r["is_correct"] for r in bs) / len(bs) if bs else 0
            r = sum(r["is_correct"] for r in rs) / len(rs) if rs else 0
            print(f"{length:>8} {depth:>7.0%} | {b:>7.0%} {r:>9.0%}")
        print("-" * 70)
    print("=" * 70)

    output = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "rmt_config": rmt_config,
        "summary": summary,
        "results": all_results,
    }
    out_path = os.path.join(args.output_dir, "nih_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Needle-in-Haystack evaluation for RMT v10")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/nih_eval_v10/")
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"])
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
