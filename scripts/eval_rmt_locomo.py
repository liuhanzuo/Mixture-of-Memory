#!/usr/bin/env python3
"""
RMT Evaluation Script for LoCoMo Benchmark.

Evaluates trained RMT models on LoCoMo QA tasks:
- Categories 1-4: Various QA types (multi-hop, single-hop, temporal, open-domain)
- Category 5: Adversarial (should answer "no information available")

Metrics: F1 Score, Exact Match, BERTScore

Usage:
    python scripts/eval_rmt_locomo.py \
        --checkpoint_dir outputs/rmt_v4_20260416_104930 \
        --locomo_data locomo/data/locomo10.json \
        --output_dir eval_results/locomo \
        --max_new_tokens 50
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_module import RMTMemory, RMTModel

# Import LoCoMo evaluation utilities
LOCOMO_ROOT = PROJECT_ROOT / "locomo"
sys.path.insert(0, str(LOCOMO_ROOT))
from task_eval.evaluation import f1_score, exact_match_score, normalize_answer

# BERTScore is optional
try:
    from task_eval.evaluation import bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    logger.warning("BERTScore not available. Install with: pip install bert-score")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoCoMoEvalConfig:
    """Configuration for LoCoMo evaluation."""
    checkpoint_dir: str
    locomo_data_path: str
    output_dir: str = "eval_results/locomo"
    
    # RMT config (loaded from checkpoint if available)
    num_memory_tokens: int = 16
    segment_length: int = 1024
    max_segments: int = 6
    num_memory_heads: int = 8
    bottleneck_dim: int = 64
    extractor_version: int = 3
    
    # Generation config
    max_new_tokens: int = 50
    temperature: float = 0.0
    top_p: float = 0.9
    do_sample: bool = False
    
    # Evaluation config
    categories: List[int] = None  # None = all categories
    max_qa_per_conv: int = None  # None = all QA pairs
    use_bertscore: bool = HAS_BERTSCORE  # BERTScore is slower
    
    # Data config
    max_context_tokens: int = 6144  # Limit context length (max_segments * segment_length)


def load_locomo_data(data_path: str, categories: Optional[List[int]] = None) -> List[Dict]:
    """Load and parse LoCoMo dataset.
    
    Returns list of conversations, each with:
    - sample_id
    - conversation: dict with sessions
    - qa: list of QA pairs with category, question, answer, evidence
    """
    logger.info(f"Loading LoCoMo data from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Filter by category if specified
    if categories:
        filtered = []
        for conv in data:
            filtered_qa = [q for q in conv.get('qa', []) if q.get('category') in categories]
            if filtered_qa:
                conv_copy = conv.copy()
                conv_copy['qa'] = filtered_qa
                filtered.append(conv_copy)
        data = filtered
    
    total_qa = sum(len(conv.get('qa', [])) for conv in data)
    logger.info(f"Loaded {len(data)} conversations with {total_qa} QA pairs")
    
    return data


def build_locomo_context(
    conversation: Dict,
    tokenizer,
    segment_length: int,
    max_segments: int,
    max_tokens: int,
) -> Tuple[List[int], List[Dict]]:
    """Build concatenated context from all sessions.
    
    Returns:
        token_ids: List of token IDs for the full conversation
        turn_info: List of turn metadata (session, speaker, dia_id)
    """
    conv_data = conversation.get('conversation', {})
    
    # Get all session keys in order
    session_pattern = re.compile(r'^session_(\d+)$')
    session_keys = [
        k for k in conv_data.keys()
        if session_pattern.match(k)
    ]
    session_keys.sort(key=lambda k: int(k.split('_')[1]))
    
    all_text = []
    turn_info = []
    
    speaker_a = conv_data.get('speaker_a', 'Speaker A')
    speaker_b = conv_data.get('speaker_b', 'Speaker B')
    
    for session_key in session_keys:
        turns = conv_data.get(session_key, [])
        if not isinstance(turns, list):
            continue
        
        for turn in turns:
            if not isinstance(turn, dict) or 'text' not in turn:
                continue
            
            text = turn['text'].strip()
            if not text:
                continue
            
            # Format: "Speaker: text"
            speaker = turn.get('speaker', '')
            formatted = f"{speaker}: {text}"
            
            all_text.append(formatted)
            turn_info.append({
                'session': session_key,
                'speaker': speaker,
                'dia_id': turn.get('dia_id', ''),
                'text': text,
                'formatted': formatted,
            })
    
    # Concatenate with separators
    full_text = " ".join(all_text)
    
    # Tokenize
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    
    # Truncate to max_tokens
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
        logger.info(f"  Truncated context from {len(token_ids)} to {max_tokens} tokens")
    
    return token_ids, turn_info


def load_rmt_model(
    checkpoint_dir: str,
    config: LoCoMoEvalConfig,
    device: torch.device,
) -> Tuple[RMTModel, AutoTokenizer]:
    """Load trained RMT model and tokenizer."""
    logger.info(f"Loading RMT model from {checkpoint_dir}...")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, "rmt_config.json")
    model_path = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            logger.info(f"Loaded saved config: {json.dumps(saved_config, indent=2)}")
            
            # Override with saved values
            config.num_memory_tokens = saved_config.get('num_memory_tokens', config.num_memory_tokens)
            config.segment_length = saved_config.get('segment_length', config.segment_length)
            config.max_segments = saved_config.get('max_segments', config.max_segments)
            config.num_memory_heads = saved_config.get('num_memory_heads', config.num_memory_heads)
            config.bottleneck_dim = saved_config.get('bottleneck_dim', config.bottleneck_dim)
            config.extractor_version = saved_config.get('extractor_version', config.extractor_version)
            model_path = saved_config.get('model_path')
    
    # Load tokenizer (use model_path from config if available)
    tokenizer_path = model_path if model_path else checkpoint_dir
    logger.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model from model_path (or checkpoint_dir if no model_path)
    base_model_path = model_path if model_path else checkpoint_dir
    logger.info(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    
    # Load LoRA adapters from checkpoint_dir if they exist
    # Check both checkpoint_dir and checkpoint_dir/final
    adapter_dir = checkpoint_dir
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        final_dir = os.path.join(checkpoint_dir, "final")
        if os.path.exists(os.path.join(final_dir, "adapter_config.json")):
            adapter_dir = final_dir
            adapter_config_path = os.path.join(final_dir, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        from peft import PeftModel
        logger.info(f"Loading LoRA adapters from {adapter_dir}...")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        logger.info("LoRA adapters merged")
    
    # Get hidden dimension
    hidden_dim = model.config.hidden_size
    
    # Initialize RMT memory
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=config.num_memory_tokens,
        num_heads=config.num_memory_heads,
        max_segments=config.max_segments + 1,
        bottleneck_dim=config.bottleneck_dim,
        extractor_version=config.extractor_version,
        use_reconstruction=False,  # No reconstruction loss during inference
    ).to(device=device, dtype=torch.bfloat16)
    
    # Load RMT weights (check final/ subdirectory too)
    rmt_path = os.path.join(checkpoint_dir, "rmt_memory.pt")
    if not os.path.exists(rmt_path):
        rmt_path_final = os.path.join(checkpoint_dir, "final", "rmt_memory.pt")
        if os.path.exists(rmt_path_final):
            rmt_path = rmt_path_final
    if os.path.exists(rmt_path):
        rmt_memory.load_state_dict(torch.load(rmt_path, map_location=device))
        logger.info(f"Loaded RMT memory weights from {rmt_path}")
    else:
        logger.warning(f"RMT memory weights not found at {rmt_path}, using random initialization")
    
    rmt_memory.eval()
    
    # Wrap in RMTModel
    rmt_model = RMTModel(model, rmt_memory, segment_length=config.segment_length)
    rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)
    rmt_model.eval()
    
    logger.info(f"Model loaded: num_memory_tokens={config.num_memory_tokens}, "
                f"segment_length={config.segment_length}, max_segments={config.max_segments}")
    
    return rmt_model, tokenizer


def generate_answer_with_rmt(
    model: RMTModel,
    tokenizer,
    context_ids: List[int],
    question: str,
    config: LoCoMoEvalConfig,
    device: torch.device,
) -> str:
    """Generate answer using RMT with context memory.
    
    Process the full context through RMT segments to build memory,
    then use that memory to answer the question.
    """
    segment_length = config.segment_length
    max_segments = config.max_segments
    num_memory_tokens = config.num_memory_tokens
    
    # Pad context to multiple of segment_length
    original_len = len(context_ids)
    num_segments = (original_len + segment_length - 1) // segment_length
    num_segments = min(num_segments, max_segments)
    padded_length = num_segments * segment_length
    
    if original_len < padded_length:
        context_ids = context_ids + [tokenizer.pad_token_id] * (padded_length - original_len)
    
    input_ids = torch.tensor([context_ids], dtype=torch.long).to(device)
    
    # Process context through RMT segments
    old_memory = None
    with torch.no_grad():
        B = input_ids.shape[0]
        
        for seg_idx in range(num_segments):
            start = seg_idx * segment_length
            end = start + segment_length
            seg_ids = input_ids[:, start:end]
            
            # Get memory for this segment
            if old_memory is None:
                mem = model.rmt.get_initial_memory(
                    seg_idx, B, device, torch.bfloat16
                )
            else:
                mem = old_memory
            
            # Forward segment (no labels, inference mode)
            loss, seg_hidden = model._forward_single_segment(seg_ids, None, mem, seg_idx)
            
            # Extract memory for next segment
            with torch.no_grad():
                extraction_result = model.rmt.extract_memory(seg_hidden.detach(), old_memory.detach() if old_memory is not None else None)
                # V3 extractor returns (new_memory, recon_loss) tuple
                if model.rmt.extractor_version == 3:
                    old_memory = extraction_result[0]  # Just use memory, ignore recon_loss
                else:
                    old_memory = extraction_result
    
    # Now generate answer with memory
    question_tokens = tokenizer.encode(
        f"\n\nQ: {question}\nA:",
        add_special_tokens=False
    )
    question_ids = torch.tensor([question_tokens], dtype=torch.long).to(device)
    
    generated_answer = ""
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Use last memory for generation
            inputs_embeds = model._embed_with_memory(question_ids, old_memory)
            
            # Simple greedy generation
            generated_ids = []
            
            for _ in range(config.max_new_tokens):
                # Forward
                outputs = model.model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=False,
                )
                logits = outputs.logits[:, -1, :]
                
                # Sample
                if config.do_sample:
                    logits = logits / config.temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                generated_ids.append(next_token.item())
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Continue generation
                next_embeds = model.model.get_input_embeddings()(next_token)
                inputs_embeds = torch.cat([inputs_embeds, next_embeds], dim=1)
            
            generated_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_answer


def evaluate_locomo_qa(
    model: RMTModel,
    tokenizer,
    locomo_data: List[Dict],
    config: LoCoMoEvalConfig,
    device: torch.device,
) -> Dict:
    """Evaluate RMT on LoCoMo QA tasks."""
    logger.info("=" * 70)
    logger.info("Starting LoCoMo QA Evaluation")
    logger.info("=" * 70)
    
    results = {
        'by_category': {},
        'by_sample': [],
        'overall': {},
    }
    
    all_f1_scores = []
    all_em_scores = []
    all_bert_scores = []
    
    category_f1 = {}
    category_em = {}
    
    # Process each conversation
    for conv_idx, conv in enumerate(tqdm(locomo_data, desc="Evaluating conversations")):
        sample_id = conv.get('sample_id', f'conv_{conv_idx}')
        qa_pairs = conv.get('qa', [])
        
        if config.max_qa_per_conv:
            qa_pairs = qa_pairs[:config.max_qa_per_conv]
        
        if not qa_pairs:
            continue
        
        # Build context once per conversation
        logger.info(f"\nProcessing conversation {conv_idx+1}/{len(locomo_data)}: {sample_id}")
        context_ids, turn_info = build_locomo_context(
            conv,
            tokenizer,
            config.segment_length,
            config.max_segments,
            config.max_context_tokens,
        )
        
        logger.info(f"  Context: {len(context_ids)} tokens, {len(turn_info)} turns")
        
        # Evaluate each QA pair
        for qa_idx, qa in enumerate(qa_pairs):
            question = qa.get('question', '').strip()
            ground_truth = str(qa.get('answer', ''))
            category = qa.get('category', 0)
            
            if not question or not ground_truth:
                continue
            
            # Generate answer
            try:
                generated = generate_answer_with_rmt(
                    model,
                    tokenizer,
                    context_ids,
                    question,
                    config,
                    device,
                )
            except Exception as e:
                logger.error(f"  Error generating answer for QA {qa_idx}: {e}")
                generated = ""
            
            # Compute metrics
            f1 = f1_score(generated, ground_truth)
            em = exact_match_score(generated, ground_truth)
            
            bert = 0.0
            if config.use_bertscore and HAS_BERTSCORE:
                try:
                    bert = bert_score(generated, ground_truth)
                except Exception as e:
                    logger.warning(f"  BERTScore computation failed: {e}")
            
            # Store results
            result = {
                'sample_id': sample_id,
                'qa_idx': qa_idx,
                'category': category,
                'question': question,
                'ground_truth': ground_truth,
                'generated': generated,
                'f1': f1,
                'exact_match': em,
                'bert_score': bert,
            }
            results['by_sample'].append(result)
            
            # Accumulate scores
            all_f1_scores.append(f1)
            all_em_scores.append(em)
            all_bert_scores.append(bert)
            
            if category not in category_f1:
                category_f1[category] = []
                category_em[category] = []
            category_f1[category].append(f1)
            category_em[category].append(em)
            
            # Log progress
            if (qa_idx + 1) % 10 == 0:
                logger.info(f"  Processed {qa_idx+1}/{len(qa_pairs)} QA pairs")
    
    # Compute overall statistics
    results['overall'] = {
        'num_samples': len(all_f1_scores),
        'avg_f1': sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0.0,
        'avg_em': sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0.0,
        'avg_bert': sum(all_bert_scores) / len(all_bert_scores) if all_bert_scores else 0.0,
    }
    
    # Compute per-category statistics
    for cat in sorted(category_f1.keys()):
        results['by_category'][cat] = {
            'num_samples': len(category_f1[cat]),
            'avg_f1': sum(category_f1[cat]) / len(category_f1[cat]),
            'avg_em': sum(category_em[cat]) / len(category_em[cat]),
        }
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Total QA pairs: {results['overall']['num_samples']}")
    logger.info(f"Overall F1: {results['overall']['avg_f1']:.4f}")
    logger.info(f"Overall EM: {results['overall']['avg_em']:.4f}")
    if config.use_bertscore:
        logger.info(f"Overall BERTScore: {results['overall']['avg_bert']:.4f}")
    
    logger.info("\n--- By Category ---")
    for cat in sorted(results['by_category'].keys()):
        cat_name = {
            1: 'Multi-hop',
            2: 'Single-hop',
            3: 'Temporal',
            4: 'Open-domain',
            5: 'Adversarial',
        }.get(cat, f'Cat{cat}')
        stats = results['by_category'][cat]
        logger.info(f"{cat_name} (cat={cat}): F1={stats['avg_f1']:.4f}, EM={stats['avg_em']:.4f}, n={stats['num_samples']}")
    
    logger.info("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RMT on LoCoMo benchmark")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to RMT checkpoint directory")
    parser.add_argument("--locomo_data", type=str,
                        default="locomo/data/locomo10.json",
                        help="Path to LoCoMo data file")
    parser.add_argument("--output_dir", type=str, default="eval_results/locomo",
                        help="Output directory for results")
    
    # RMT config (can be overridden)
    parser.add_argument("--num_memory_tokens", type=int, default=16)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=6)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--extractor_version", type=int, default=3)
    
    # Generation config
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    
    # Evaluation config
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category numbers (e.g., '1,2,3')")
    parser.add_argument("--max_qa_per_conv", type=int, default=None,
                        help="Max QA pairs per conversation (for debugging)")
    parser.add_argument("--use_bertscore", action="store_true",
                        help="Compute BERTScore (slower)")
    
    args = parser.parse_args()
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [int(c) for c in args.categories.split(',')]
    
    # Create config
    config = LoCoMoEvalConfig(
        checkpoint_dir=args.checkpoint_dir,
        locomo_data_path=args.locomo_data,
        output_dir=args.output_dir,
        num_memory_tokens=args.num_memory_tokens,
        segment_length=args.segment_length,
        max_segments=args.max_segments,
        bottleneck_dim=args.bottleneck_dim,
        extractor_version=args.extractor_version,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        categories=categories,
        max_qa_per_conv=args.max_qa_per_conv,
        use_bertscore=args.use_bertscore,
    )
    
    # Setup output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_rmt_model(args.checkpoint_dir, config, device)
    
    # Load LoCoMo data
    locomo_data = load_locomo_data(args.locomo_data, config.categories)
    
    # Run evaluation
    results = evaluate_locomo_qa(model, tokenizer, locomo_data, config, device)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.output_dir, f"locomo_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Also save a summary-only version
    summary_file = os.path.join(config.output_dir, f"locomo_summary_{timestamp}.json")
    summary = {
        'checkpoint_dir': args.checkpoint_dir,
        'timestamp': timestamp,
        'overall': results['overall'],
        'by_category': results['by_category'],
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
