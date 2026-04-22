"""
Needle-in-Haystack Evaluation for RMT v8 (Direct Injection).

Simplified from v5 eval: no dependency on legacy extractors.
Supports v8 (extractor_version=8) and falls back to v5 for older checkpoints.
"""

import os, sys, json, random, string, argparse
from pathlib import Path
from datetime import datetime
from typing import List
import logging

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_module import RMTMemory, RMTModel, build_rmt_attention_mask, build_rmt_position_ids

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


def load_rmt_model(base_model_path, checkpoint_dir, device):
    config_path = os.path.join(checkpoint_dir, "rmt_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No rmt_config.json in {checkpoint_dir}")
    with open(config_path) as f:
        rmt_cfg = json.load(f)

    segment_length = rmt_cfg.get("segment_length", 1024)
    max_segments = rmt_cfg.get("max_segments", 6)
    num_memory_tokens = rmt_cfg.get("num_memory_tokens", 16)
    bottleneck_dim = rmt_cfg.get("bottleneck_dim", 64)
    extractor_version = rmt_cfg.get("extractor_version", 8)

    tok_dir = checkpoint_dir if os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json")) else base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    merged_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(merged_path):
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
        )
    else:
        from peft import PeftModel
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path) or os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin")):
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
            )
            model = PeftModel.from_pretrained(model, checkpoint_dir)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"": device},
            )
    model.eval()

    hidden_dim = model.config.hidden_size
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=num_memory_tokens,
        max_segments=max_segments + 1,
        bottleneck_dim=bottleneck_dim,
        extractor_version=extractor_version,
    ).to(device=device, dtype=torch.bfloat16)

    rmt_path = os.path.join(checkpoint_dir, "rmt_memory.pt")
    if os.path.exists(rmt_path):
        rmt_memory.load_state_dict(torch.load(rmt_path, map_location=device))
        logger.info("Loaded RMT memory weights.")
    else:
        logger.warning(f"No rmt_memory.pt in {checkpoint_dir}")
    rmt_memory.eval()

    rmt_model = RMTModel(model, rmt_memory, segment_length=segment_length).to(device=device, dtype=torch.bfloat16)
    rmt_model.eval()

    config = {
        "segment_length": segment_length,
        "max_segments": max_segments,
        "num_memory_tokens": num_memory_tokens,
        "extractor_version": extractor_version,
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
def generate_rmt(rmt_model, tokenizer, input_ids, rmt_config, max_new_tokens=30, question_text=None):
    """Process haystack through RMT segments, then generate answer."""
    segment_length = rmt_config["segment_length"]
    B, L = input_ids.shape
    device = input_ids.device
    num_segments = max(L // segment_length, 1)

    if L < segment_length:
        pad = torch.full((B, segment_length - L), tokenizer.pad_token_id, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, pad], dim=1)
        L = segment_length

    old_memory = None
    n_mem = rmt_model.num_memory_tokens

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = start + segment_length
            seg_ids = input_ids[:, start:end]

            if old_memory is None:
                mem = rmt_model.rmt.get_initial_memory(seg_idx, B, device, torch.bfloat16)
            else:
                mem = old_memory

            _, seg_hidden = rmt_model._forward_single_segment(seg_ids, None, mem, seg_idx)

            mem_result = rmt_model.rmt.extract_memory(seg_hidden, old_memory)
            old_memory = mem_result[0] if isinstance(mem_result, tuple) else mem_result

        # Generate answer
        if question_text is None:
            question_text = "请问 文档中提到的编号是什么？"
        q_ids = tokenizer.encode(question_text, add_special_tokens=False, return_tensors="pt").to(device)
        inputs_embeds = rmt_model._embed_with_memory(q_ids, old_memory)

        q_len = q_ids.shape[1]
        last_seg_idx = num_segments - 1

        gen_attn_mask = build_rmt_attention_mask(q_len, n_mem, device)
        gen_attn_mask_4d = gen_attn_mask.unsqueeze(0).unsqueeze(0)
        gen_attn_float = torch.zeros_like(gen_attn_mask_4d, dtype=torch.bfloat16)
        gen_attn_float = gen_attn_float.masked_fill(~gen_attn_mask_4d, float('-inf'))

        gen_position_ids = build_rmt_position_ids(
            q_len + max_new_tokens, n_mem, last_seg_idx, device
        ).unsqueeze(0)
        init_position_ids = gen_position_ids[:, :n_mem + q_len]

        generated_ids = []
        for _ in range(max_new_tokens):
            cur_len = inputs_embeds.shape[1]
            if cur_len == n_mem + q_len:
                cur_mask = gen_attn_float
                cur_pos = init_position_ids
            else:
                cur_text_len = cur_len - n_mem
                cm = build_rmt_attention_mask(cur_text_len, n_mem, device)
                cm4d = cm.unsqueeze(0).unsqueeze(0)
                cur_mask = torch.zeros_like(cm4d, dtype=torch.bfloat16).masked_fill(~cm4d, float('-inf'))
                cur_pos = gen_position_ids[:, :cur_len]

            outputs = rmt_model.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask={"full_attention": cur_mask},
                position_ids=cur_pos,
                output_hidden_states=False,
            )
            logits = rmt_model.model.lm_head(outputs.last_hidden_state[:, -1:, :])
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if next_tok.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_tok.item())
            next_embed = rmt_model.model.get_input_embeddings()(next_tok)
            inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ─── Memory token verification (v8 specific) ───────────────────────────────

def verify_memory_tokens(rmt_model, input_ids, rmt_config):
    """
    Quick diagnostic: check that memory tokens carry non-trivial information.
    For v8, memory tokens should have higher norms than random noise.
    """
    segment_length = rmt_config["segment_length"]
    device = input_ids.device
    B, L = input_ids.shape
    num_segments = max(L // segment_length, 1)

    old_memory = None
    norms = []

    with torch.no_grad():
        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = start + segment_length
            seg_ids = input_ids[:, start:end]

            if old_memory is None:
                mem = rmt_model.rmt.get_initial_memory(seg_idx, B, device, torch.bfloat16)
            else:
                mem = old_memory

            _, seg_hidden = rmt_model._forward_single_segment(seg_ids, None, mem, seg_idx)

            mem_result = rmt_model.rmt.extract_memory(seg_hidden, old_memory)
            new_mem = mem_result[0] if isinstance(mem_result, tuple) else mem_result

            if old_memory is not None:
                # For v8: memory should be last-K hidden states, should have meaningful norms
                mem_norm = new_mem.norm(dim=-1).mean().item()
                norms.append(mem_norm)

            old_memory = new_mem.detach()

    return norms


# ─── Main evaluation ───────────────────────────────────────────────────────

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading base model ...")
    base_model, tokenizer = load_base_model(args.base_model, device)

    logger.info("Loading RMT model ...")
    rmt_model, rmt_tokenizer, rmt_config = load_rmt_model(args.base_model, args.checkpoint, device)

    segment_length = rmt_config["segment_length"]
    max_segments = rmt_config["max_segments"]
    base_max_len = getattr(base_model.config, "max_position_embeddings", 8192)

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

                # RMT model (question NOT in document segments)
                rmt_doc_tokens = haystack_tokens[:insert_pos] + needle_tokens + haystack_tokens[insert_pos:]
                rmt_tokens = list(rmt_doc_tokens)
                remainder = len(rmt_tokens) % segment_length
                if remainder:
                    rmt_tokens += [tokenizer.pad_token_id] * (segment_length - remainder)
                num_segs = len(rmt_tokens) // segment_length
                if num_segs > max_segments:
                    rmt_tokens = rmt_tokens[-(max_segments * segment_length):]

                rmt_input = torch.tensor([rmt_tokens], dtype=torch.long, device=device)
                rmt_answer = generate_rmt(rmt_model, tokenizer, rmt_input, rmt_config, question_text=question)
                rmt_correct = expected_answer.lower() in rmt_answer.lower()
                all_results.append({"model_type": "rmt_v8", "target_length": target_length, "depth": depth,
                                    "trial": trial_idx, "expected": expected_answer, "answer": rmt_answer,
                                    "is_correct": rmt_correct})

                done += 2
                logger.info(f"[{done}/{total_configs}] len={target_length} depth={depth:.0%} "
                            f"base={'✓' if base_correct else '✗'} rmt={'✓' if rmt_correct else '✗'}")

    # ─── Memory verification (v8 diagnostic) ───
    if rmt_config.get("extractor_version") == 8:
        logger.info("Running memory token verification ...")
        test_haystack = generate_haystack_text(tokenizer, segment_length * 3, rng, lang=args.lang)
        test_input = torch.tensor([test_haystack[:segment_length * 3]], dtype=torch.long, device=device)
        mem_norms = verify_memory_tokens(rmt_model, test_input, rmt_config)
        logger.info(f"Memory token norms across segments: {mem_norms}")

    # ─── Summary ───
    summary = {}
    for model_type in ["base", "rmt_v8"]:
        subset = [r for r in all_results if r["model_type"] == model_type]
        correct = sum(r["is_correct"] for r in subset)
        acc = correct / len(subset) if subset else 0
        summary[f"{model_type}_accuracy"] = acc
        logger.info(f"\n{model_type} accuracy: {acc:.2%} ({correct}/{len(subset)})")

    print("\n" + "=" * 70)
    print("NEEDLE-IN-A-HAYSTACK RESULTS")
    print("=" * 70)
    print(f"{'Length':>8} {'Depth':>8} | {'Base':>8} {'RMT v8':>8}")
    print("-" * 70)
    for length in args.lengths:
        for depth in args.depths:
            bs = [r for r in all_results if r["model_type"] == "base" and r["target_length"] == length and abs(r["depth"] - depth) < 1e-6]
            rs = [r for r in all_results if r["model_type"] == "rmt_v8" and r["target_length"] == length and abs(r["depth"] - depth) < 1e-6]
            b = sum(r["is_correct"] for r in bs) / len(bs) if bs else 0
            r = sum(r["is_correct"] for r in rs) / len(rs) if rs else 0
            print(f"{length:>8} {depth:>7.0%} | {b:>7.0%} {r:>7.0%}")
        print("-" * 70)
    print("=" * 70)

    output = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "rmt_config": rmt_config,
        "summary": summary,
        "results": all_results,
    }
    with open(os.path.join(args.output_dir, "nih_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {os.path.join(args.output_dir, 'nih_results.json')}")


def main():
    parser = argparse.ArgumentParser(description="Needle-in-Haystack evaluation for RMT v8")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/nih_eval_v8/")
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"])
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
