"""
RMT Evaluation Script for Qwen3-8B.

Evaluates trained RMT models on:
1. Perplexity (PPL) - Language modeling quality
2. Needle-in-Haystack (NiH) - Long-context retrieval
3. Memory Utilization - Memory token analysis
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.rmt.rmt_module import RMTMemory, RMTModel


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for RMT evaluation."""
    checkpoint_dir: str
    data_path: str
    eval_type: str = "all"  # all, ppl, nih, memory
    output_dir: str = "eval_results"
    
    # RMT config
    num_memory_tokens: int = 64
    segment_length: int = 2048
    max_segments: int = 6
    num_memory_heads: int = 8
    bottleneck_dim: int = 32
    
    # Evaluation config
    batch_size: int = 1
    max_docs: int = None  # Limit number of documents for quick testing
    
    # NiH config
    nih_needle_positions: List[float] = None
    nih_document_lengths: List[int] = None
    nih_num_trials: int = 5


class RMTEvalDataset(Dataset):
    """Dataset for RMT evaluation."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        segment_length: int = 2048,
        max_segments: int = 6,
        max_docs: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.segment_length = segment_length
        self.max_segments = max_segments
        self.max_total_tokens = segment_length * max_segments
        
        logger.info(f"Loading evaluation data from {data_path}...")
        self.tokenized_docs = []
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_docs is not None and len(self.tokenized_docs) >= max_docs:
                    break
                
                doc = json.loads(line)
                tokens = tokenizer.encode(doc['text'], add_special_tokens=False)
                
                if len(tokens) >= segment_length:
                    # Pad to fixed length
                    num_segs = min(len(tokens) // segment_length, max_segments)
                    tokens = tokens[:num_segs * segment_length]
                    target_len = max_segments * segment_length
                    
                    if len(tokens) < target_len:
                        tokens = tokens + [tokenizer.pad_token_id] * (target_len - len(tokens))
                    
                    self.tokenized_docs.append(tokens)
        
        logger.info(f"Evaluation dataset: {len(self.tokenized_docs)} documents")
    
    def __len__(self):
        return len(self.tokenized_docs)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_docs[idx]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        num_segments = len(tokens) // self.segment_length
        return {"input_ids": input_ids, "labels": labels, "num_segments": num_segments}


def collate_fn(batch, segment_length: int, pad_token_id: int):
    """Collate function for batching."""
    max_segs = max(item["num_segments"] for item in batch)
    max_len = max_segs * segment_length
    
    input_ids_list = []
    labels_list = []
    num_segments_list = []
    
    for item in batch:
        ids = item["input_ids"]
        labs = item["labels"]
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)])
        input_ids_list.append(ids)
        labels_list.append(labs)
        num_segments_list.append(item["num_segments"])
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "num_segments": num_segments_list,
    }


def load_rmt_model(checkpoint_dir: str, config: EvalConfig, device: torch.device) -> Tuple[RMTModel, AutoTokenizer]:
    """Load trained RMT model and tokenizer."""
    logger.info(f"Loading RMT model from {checkpoint_dir}...")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, "rmt_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
            # Override with command line args
            config.num_memory_tokens = saved_config.get('num_memory_tokens', config.num_memory_tokens)
            config.segment_length = saved_config.get('segment_length', config.segment_length)
            config.max_segments = saved_config.get('max_segments', config.max_segments)
            config.num_memory_heads = saved_config.get('num_memory_heads', config.num_memory_heads)
            config.bottleneck_dim = saved_config.get('bottleneck_dim', 32)
            config.extractor_version = saved_config.get('extractor_version', 2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    
    # Get hidden dimension
    hidden_dim = model.config.hidden_size
    
    # Initialize RMT memory
    extractor_version = getattr(config, 'extractor_version', 2)
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=config.num_memory_tokens,
        num_heads=config.num_memory_heads,
        max_segments=config.max_segments + 1,
        bottleneck_dim=getattr(config, 'bottleneck_dim', 32),
        extractor_version=extractor_version,
    ).to(device=device, dtype=torch.bfloat16)
    
    # Load RMT weights
    rmt_path = os.path.join(checkpoint_dir, "rmt_memory.pt")
    if os.path.exists(rmt_path):
        rmt_memory.load_state_dict(torch.load(rmt_path, map_location=device))
        logger.info("Loaded RMT memory weights")
    
    rmt_memory.eval()
    
    # Wrap in RMTModel
    rmt_model = RMTModel(model, rmt_memory, segment_length=config.segment_length)
    rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)
    rmt_model.eval()
    
    return rmt_model, tokenizer


class RMTEvaluator:
    """Evaluator for RMT models."""
    
    def __init__(self, model: RMTModel, tokenizer, config: EvalConfig, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
    
    def evaluate_perplexity(self, dataloader: DataLoader) -> Dict:
        """Evaluate perplexity on held-out documents."""
        logger.info("Evaluating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        all_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass with RMT
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = self.model(input_ids, labels, training=False)
                
                # Count valid tokens (not -100)
                valid_mask = (labels != -100)
                num_valid = valid_mask.sum().item()
                
                total_loss += loss.item() * num_valid
                total_tokens += num_valid
                all_losses.append(loss.item())
        
        # Calculate PPL
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        
        results = {
            "perplexity": ppl,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "num_docs": len(dataloader.dataset),
        }
        
        logger.info(f"Perplexity: {ppl:.4f}")
        return results
    
    def needle_in_haystack_test(self, num_trials: int = 5) -> Dict:
        """Run Needle-in-Haystack tests."""
        logger.info("Running Needle-in-Haystack tests...")
        
        results = {
            "tests": [],
            "summary": {}
        }
        
        # Default test configurations
        if self.config.nih_needle_positions is None:
            needle_positions = [0.1, 0.3, 0.5, 0.7, 0.9]  # Beginning to end
        else:
            needle_positions = self.config.nih_needle_positions
        
        if self.config.nih_document_lengths is None:
            document_lengths = [4096, 8192, 12288]  # 2, 4, 6 segments
        else:
            document_lengths = self.config.nih_document_lengths
        
        # Sample documents for haystack
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
        docs = [doc['text'] for doc in dataset if len(doc['text']) > 1000]
        
        for doc_length in document_lengths:
            for needle_pos in needle_positions:
                for trial in range(num_trials):
                    result = self._run_single_nih_test(doc_length, needle_pos, docs, trial)
                    results["tests"].append(result)
        
        # Calculate summary statistics
        accuracies = [r['accuracy'] for r in results['tests']]
        results['summary'] = {
            'total_tests': len(accuracies),
            'correct': sum(accuracies),
            'accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
        }
        
        # By needle position
        for pos in needle_positions:
            pos_tests = [r for r in results['tests'] if abs(r['needle_position'] - pos) < 0.05]
            pos_acc = sum(r['accuracy'] for r in pos_tests) / len(pos_tests) if pos_tests else 0
            results['summary'][f'accuracy_pos_{int(pos*100)}'] = pos_acc
        
        # By document length
        for length in document_lengths:
            len_tests = [r for r in results['tests'] if r['document_length'] == length]
            len_acc = sum(r['accuracy'] for r in len_tests) / len(len_tests) if len_tests else 0
            results['summary'][f'accuracy_len_{length}'] = len_acc
        
        logger.info(f"NiH Accuracy: {results['summary']['accuracy']:.2%}")
        return results
    
    def _run_single_nih_test(self, document_length: int, needle_pos: float, docs: List[str], trial: int) -> Dict:
        """Run a single Needle-in-Haystack test."""
        
        # Generate needle fact
        needles = [
            "The secret code is X7K9.",
            "The answer is 42.",
            "The password is password123.",
            "The magic number is 7.",
            "The hidden key is ALPHA.",
        ]
        needle_text = needles[trial % len(needles)]
        
        # Extract question from needle
        if "code" in needle_text.lower():
            question = "What is the secret code?"
            expected_answer = needle_text.split("is ")[-1].rstrip(".")
        elif "answer" in needle_text.lower():
            question = "What is the answer?"
            expected_answer = needle_text.split("is ")[-1].rstrip(".")
        elif "password" in needle_text.lower():
            question = "What is the password?"
            expected_answer = needle_text.split("is ")[-1].rstrip(".")
        elif "number" in needle_text.lower():
            question = "What is the magic number?"
            expected_answer = needle_text.split("is ")[-1].rstrip(".")
        else:
            question = "What is the hidden key?"
            expected_answer = needle_text.split("is ")[-1].rstrip(".")
        
        # Build haystack document
        needle_idx = int(document_length * needle_pos)
        # Tokenize to approximate length
        haystack_tokens = []
        doc_idx = 0
        
        while len(haystack_tokens) < document_length and doc_idx < len(docs):
            tokens = self.tokenizer.encode(docs[doc_idx], add_special_tokens=False)
            haystack_tokens.extend(tokens)
            doc_idx += 1
        
        haystack_tokens = haystack_tokens[:document_length]
        
        # Insert needle at position
        needle_tokens = self.tokenizer.encode(needle_text, add_special_tokens=False)
        haystack_tokens[needle_idx:needle_idx] = needle_tokens
        
        # Pad to multiple of segment_length
        num_segments = (len(haystack_tokens) + self.config.segment_length - 1) // self.config.segment_length
        padded_length = num_segments * self.config.segment_length
        haystack_tokens.extend([self.tokenizer.pad_token_id] * (padded_length - len(haystack_tokens)))
        
        input_ids = torch.tensor([haystack_tokens], dtype=torch.long).to(self.device)
        
        # Process through RMT segments (no loss calculation)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward through all segments to build memory
                B, L = input_ids.shape
                num_segments = L // self.config.segment_length
                
                old_memory = None
                for seg_idx in range(num_segments):
                    start = seg_idx * self.config.segment_length
                    end = start + self.config.segment_length
                    seg_ids = input_ids[:, start:end]
                    
                    if old_memory is None:
                        mem = self.model.rmt.get_initial_memory(seg_idx, B, self.device, torch.bfloat16)
                    else:
                        mem = old_memory
                    
                    # Forward without labels (inference mode)
                    _, seg_hidden = self.model._forward_single_segment(seg_ids, None, mem, seg_idx)
                    
                    # Update memory
                    with torch.no_grad():
                        extracted = self.model.rmt.extract_memory(seg_hidden, old_memory)
                        # extract_memory returns (new_memory, recon_loss) tuple for v3/v5
                        old_memory = extracted[0] if isinstance(extracted, tuple) else extracted
        
        # Generate answer
        question_tokens = self.tokenizer.encode(f"\n\nQ: {question}\nA:", add_special_tokens=False)
        question_ids = torch.tensor([question_tokens], dtype=torch.long).to(self.device)
        
        # Generate with memory from last segment (using correct attention mask + position_ids)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_tensor = self.model.generate_with_memory(
                question_ids, old_memory, segment_idx=num_segments - 1, max_new_tokens=20,
            )
            answer = self.tokenizer.decode(generated_tensor[0], skip_special_tokens=True)
        
        # Check accuracy
        accuracy = expected_answer.lower() in answer.lower()
        
        return {
            "trial": trial,
            "document_length": document_length,
            "needle_position": needle_pos,
            "needle_text": needle_text,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "accuracy": accuracy,
        }
    
    def analyze_memory_utilization(self, dataloader: DataLoader) -> Dict:
        """Analyze memory token utilization."""
        logger.info("Analyzing memory utilization...")
        
        all_memory_tokens = []
        segment_memories = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 10:  # Analyze first 10 batches
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                B, L = input_ids.shape
                num_segments = L // self.config.segment_length
                
                old_memory = None
                batch_memories = []
                
                for seg_idx in range(num_segments):
                    start = seg_idx * self.config.segment_length
                    end = start + self.config.segment_length
                    seg_ids = input_ids[:, start:end]
                    
                    if old_memory is None:
                        mem = self.model.rmt.get_initial_memory(seg_idx, B, self.device, torch.bfloat16)
                    else:
                        mem = old_memory
                    
                    # Forward
                    _, seg_hidden = self.model._forward_single_segment(seg_ids, None, mem, seg_idx)
                    
                    # Extract memory
                    new_memory = self.model.rmt.extract_memory(seg_hidden, old_memory)
                    batch_memories.append(new_memory.detach().cpu())
                    
                    if old_memory is None:
                        old_memory = new_memory
                    else:
                        old_memory = new_memory
                
                segment_memories.append(batch_memories)
        
        # Calculate cosine similarities between consecutive segments
        similarities = []
        for batch_memories in segment_memories:
            for i in range(len(batch_memories) - 1):
                mem1 = batch_memories[i]  # [B, num_mem, hidden_dim]
                mem2 = batch_memories[i + 1]
                
                # Flatten for comparison
                mem1_flat = mem1.view(-1, mem1.shape[-1])  # [B*num_mem, hidden_dim]
                mem2_flat = mem2.view(-1, mem2.shape[-1])
                
                # Cosine similarity
                sim = F.cosine_similarity(mem1_flat, mem2_flat, dim=-1)
                similarities.append(sim.mean().item())
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        results = {
            "avg_memory_similarity": avg_similarity,
            "num_segments_analyzed": len(similarities),
            "num_memory_tokens": self.config.num_memory_tokens,
        }
        
        logger.info(f"Average memory similarity: {avg_similarity:.4f}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RMT model")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--eval_type", type=str, default="all", choices=["all", "ppl", "nih", "memory"])
    
    # RMT config
    parser.add_argument("--num_memory_tokens", type=int, default=64)
    parser.add_argument("--segment_length", type=int, default=2048)
    parser.add_argument("--max_segments", type=int, default=6)
    parser.add_argument("--num_memory_heads", type=int, default=8)
    parser.add_argument("--bottleneck_dim", type=int, default=32)
    
    # Evaluation config
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_docs", type=int, default=None)
    
    # NiH config
    parser.add_argument("--nih_num_trials", type=int, default=5)
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create config
    config = EvalConfig(
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        eval_type=args.eval_type,
        output_dir=args.output_dir,
        num_memory_tokens=args.num_memory_tokens,
        segment_length=args.segment_length,
        max_segments=args.max_segments,
        num_memory_heads=args.num_memory_heads,
        bottleneck_dim=args.bottleneck_dim,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        nih_num_trials=args.nih_num_trials,
    )
    
    # Load model
    model, tokenizer = load_rmt_model(args.checkpoint_dir, config, device)
    
    # Create evaluator
    evaluator = RMTEvaluator(model, tokenizer, config, device)
    
    # Run evaluations
    all_results = {
        "checkpoint_dir": args.checkpoint_dir,
        "eval_type": args.eval_type,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Perplexity
    if args.eval_type in ["all", "ppl"]:
        logger.info("=" * 50)
        logger.info("Running Perplexity Evaluation")
        logger.info("=" * 50)
        
        dataset = RMTEvalDataset(
            args.data_path,
            tokenizer,
            segment_length=args.segment_length,
            max_segments=args.max_segments,
            max_docs=args.max_docs,
        )
        
        def collate_with_config(batch):
            return collate_fn(batch, args.segment_length, tokenizer.pad_token_id)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_with_config,
            num_workers=0,
        )
        
        ppl_results = evaluator.evaluate_perplexity(dataloader)
        all_results["perplexity"] = ppl_results
    
    # Needle-in-Haystack
    if args.eval_type in ["all", "nih"]:
        logger.info("=" * 50)
        logger.info("Running Needle-in-Haystack Tests")
        logger.info("=" * 50)
        
        nih_results = evaluator.needle_in_haystack_test(args.nih_num_trials)
        all_results["needle_in_haystack"] = nih_results
    
    # Memory utilization
    if args.eval_type in ["all", "memory"]:
        logger.info("=" * 50)
        logger.info("Running Memory Utilization Analysis")
        logger.info("=" * 50)
        
        dataset = RMTEvalDataset(
            args.data_path,
            tokenizer,
            segment_length=args.segment_length,
            max_segments=args.max_segments,
            max_docs=min(args.max_docs or 100, 10),  # Use fewer docs for memory analysis
        )
        
        def collate_with_config(batch):
            return collate_fn(batch, args.segment_length, tokenizer.pad_token_id)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_with_config,
            num_workers=0,
        )
        
        memory_results = evaluator.analyze_memory_utilization(dataloader)
        all_results["memory_utilization"] = memory_results
    
    # Save results
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("=" * 50)
    logger.info(f"Results saved to {results_path}")
    logger.info("=" * 50)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    if "perplexity" in all_results:
        print(f"Perplexity: {all_results['perplexity']['perplexity']:.4f}")
    if "needle_in_haystack" in all_results:
        print(f"NiH Accuracy: {all_results['needle_in_haystack']['summary']['accuracy']:.2%}")
    if "memory_utilization" in all_results:
        print(f"Memory Similarity: {all_results['memory_utilization']['avg_memory_similarity']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
