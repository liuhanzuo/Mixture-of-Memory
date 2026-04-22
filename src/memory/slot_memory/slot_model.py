"""SlotMemoryModel — wraps Qwen3-8B + SlotMemoryCompressor + LoRA.

Processes documents in segments. On each segment boundary:
1. Extract segment hidden states
2. Compress into slots via SlotMemoryCompressor
3. Inject slots as prefix for the next segment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class SlotMemoryModel(nn.Module):
    """Wraps a causal LM with slot memory injection."""

    def __init__(
        self,
        model: nn.Module,
        compressor: nn.Module,
        segment_length: int = 1024,
        max_segments: int = 4,
        bptt_depth: int = 2,
    ):
        super().__init__()
        self.model = model  # PeftModel (LoRA-wrapped)
        self.compressor = compressor  # SlotMemoryCompressor
        self.segment_length = segment_length
        self.max_segments = max_segments
        self.bptt_depth = bptt_depth
        self.num_slots = compressor.num_slots
        self.hidden_dim = compressor.hidden_dim

    def _build_attention_mask(self, seq_len: int, device) -> torch.Tensor:
        """Build 2D attention mask. Slot tokens attend bidirectionally."""
        total_len = self.num_slots + seq_len
        causal = torch.tril(torch.ones(total_len, total_len, device=device)).bool()
        causal[:self.num_slots, :] = True  # slots: full bidirectional
        return causal

    def _build_position_ids(self, seq_len: int, device) -> torch.Tensor:
        """Position IDs: slots get 0..S-1, content gets S..S+T-1."""
        slot_pos = torch.arange(self.num_slots, device=device)
        seg_pos = torch.arange(seq_len, device=device) + self.num_slots
        return torch.cat([slot_pos, seg_pos])

    def _embed_with_slots(self, input_ids: torch.Tensor, slot_embeddings: torch.Tensor) -> torch.Tensor:
        """Concatenate slot embeddings with token embeddings."""
        token_embeds = self.model.get_input_embeddings()(input_ids)
        return torch.cat([slot_embeddings, token_embeds], dim=1)

    def _forward_single_segment(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        slot_embeddings: torch.Tensor,
        segment_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single segment with slot prefix.

        Returns:
            loss: scalar CE loss
            segment_hidden: hidden states of the content (excluding slots)
        """
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = slot_embeddings.dtype
        actual_seg_len = input_ids.shape[1]

        inputs_embeds = self._embed_with_slots(input_ids, slot_embeddings)

        attn_mask = self._build_attention_mask(actual_seg_len, device)
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)
        position_ids = self._build_position_ids(actual_seg_len, device).unsqueeze(0).expand(B, -1)

        if labels is not None:
            slot_labels = torch.full((B, self.num_slots), -100, device=device, dtype=labels.dtype)
            full_labels = torch.cat([slot_labels, labels], dim=1)
        else:
            full_labels = None

        # Build 4D attention mask
        bool_mask_4d = attn_mask.unsqueeze(1)
        attn_mask_4d = torch.zeros_like(bool_mask_4d, dtype=dtype)
        attn_mask_4d = attn_mask_4d.masked_fill(~bool_mask_4d, torch.tensor(float('-inf'), dtype=dtype))

        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attn_mask_4d},
            position_ids=position_ids,
            output_hidden_states=False,
        )

        hidden = outputs.last_hidden_state
        logits = self.model.lm_head(hidden)

        loss = torch.tensor(0.0, device=device)
        if full_labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # Extract segment hidden states (exclude slot positions)
        segment_hidden = hidden[:, self.num_slots:, :]
        return loss, segment_hidden

    def _generate_single_segment(
        self,
        input_ids: torch.Tensor,
        slot_embeddings: torch.Tensor,
        segment_idx: int,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """Generate tokens for a single segment (for NiH eval)."""
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = slot_embeddings.dtype
        q_len = input_ids.shape[1]

        inputs_embeds = self._embed_with_slots(input_ids, slot_embeddings)
        attn_mask = self._build_attention_mask(q_len, device)
        attn_mask_3d = attn_mask.unsqueeze(0).expand(B, -1, -1)
        bool_mask_4d = attn_mask_3d.unsqueeze(1)
        attn_mask_4d = torch.zeros_like(bool_mask_4d, dtype=dtype)
        attn_mask_4d = attn_mask_4d.masked_fill(~bool_mask_4d, float('-inf'))
        position_ids = self._build_position_ids(q_len, device).unsqueeze(0).expand(B, -1)

        outputs = self.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attn_mask_4d},
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=True,
        )
        logits = self.model.lm_head(outputs.last_hidden_state)
        past_kv = outputs.past_key_values

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]

        for step in range(max_new_tokens - 1):
            token_embeds = self.model.get_input_embeddings()(next_token)
            next_pos = position_ids[:, -1:] + 1 + step
            outputs = self.model.model(
                inputs_embeds=token_embeds,
                attention_mask={"full_attention": torch.zeros(B, 1, 1, outputs.last_hidden_state.shape[1] + 1, device=device, dtype=dtype)},
                position_ids=next_pos,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = self.model.lm_head(outputs.last_hidden_state)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_kv = outputs.past_key_values
            generated.append(next_token)

        return torch.cat(generated, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        lambda_recon: float = 0.5,
        lambda_ce: float = 1.0,
    ) -> Dict[str, Any]:
        """Process document in segments with slot memory.

        Returns dict with 'loss', 'ce_loss', 'recon_loss', 'num_segments'.
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Determine number of segments
        num_segments = min(L // self.segment_length, self.max_segments)
        if num_segments == 0:
            num_segments = 1

        total_ce_loss = torch.tensor(0.0, device=device)
        total_recon_loss = torch.tensor(0.0, device=device)

        # Track gradients for truncated BPTT
        # BPTT depth: how many segments back to propagate
        # segment_losses stores (ce_loss, recon_loss, slot_states) per segment
        segment_cache = []  # list of (ce_loss, recon_loss, slots, gate)

        prev_slots = None
        prev_gate = None

        for seg_idx in range(num_segments):
            start = seg_idx * self.segment_length
            end = start + self.segment_length
            seg_ids = input_ids[:, start:end]
            seg_labels = labels[:, start:end] if labels is not None else None

            # Prepare slot embeddings for this segment
            if prev_slots is None:
                # First segment: use learnable slot embeddings as prefix
                slot_embeds = self.compressor.slot_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype=input_ids.dtype)
            else:
                slot_embeds = prev_slots

            # Forward through LM
            ce_loss, segment_hidden = self._forward_single_segment(
                seg_ids, seg_labels, slot_embeds, seg_idx
            )

            # Compress segment into slots
            updated_slots, recon_loss, gate = self.compressor(
                segment_hidden, prev_slots=prev_slots, prev_gate=prev_gate
            )

            # Store losses for BPTT
            segment_cache.append({
                'ce_loss': ce_loss,
                'recon_loss': recon_loss,
                'slots': updated_slots,
                'gate': gate,
            })

            prev_slots = updated_slots.detach()  # detach for next segment (truncated BPTT)
            prev_gate = gate.detach() if gate is not None else None

            # Truncated BPTT: backward through oldest segment in window
            if self.bptt_depth > 0 and len(segment_cache) > self.bptt_depth:
                old = segment_cache.pop(0)
                old_loss = lambda_ce * old['ce_loss'] + lambda_recon * old['recon_loss']
                old_loss.backward(retain_graph=False)

            total_ce_loss = total_ce_loss + ce_loss.detach()
            total_recon_loss = total_recon_loss + recon_loss.detach()

        # Backward remaining segments in cache
        remaining_loss = torch.tensor(0.0, device=device)
        for entry in segment_cache:
            remaining_loss = remaining_loss + lambda_ce * entry['ce_loss'] + lambda_recon * entry['recon_loss']
        remaining_loss.backward(retain_graph=False)

        total_loss = total_ce_loss + lambda_recon * total_recon_loss
        avg_ce = (total_ce_loss / num_segments).item()
        avg_recon = (total_recon_loss / num_segments).item()

        return {
            'loss': total_loss,
            'ce_loss': avg_ce,
            'recon_loss': avg_recon,
            'num_segments': num_segments,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        question_segment: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """Inference: process document segments, then generate on question.

        Args:
            input_ids: full document [B, L]
            question_segment: optional question tokens [B, Q]
            max_new_tokens: tokens to generate
        """
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = input_ids.dtype

        num_segments = min(input_ids.shape[1] // self.segment_length, self.max_segments)
        prev_slots = None

        # Process all content segments
        with torch.no_grad():
            for seg_idx in range(num_segments):
                start = seg_idx * self.segment_length
                end = start + self.segment_length
                seg_ids = input_ids[:, start:end]

                if prev_slots is None:
                    slot_embeds = self.compressor.slot_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)
                else:
                    slot_embeds = prev_slots

                _, segment_hidden = self._forward_single_segment(seg_ids, None, slot_embeds, seg_idx)
                updated_slots, _, _ = self.compressor(segment_hidden)
                prev_slots = updated_slots

            # Generate on question segment (or last segment)
            if question_segment is not None:
                return self._generate_single_segment(question_segment, prev_slots, num_segments, max_new_tokens)
            else:
                return self._generate_single_segment(input_ids[:, -self.segment_length:], prev_slots, num_segments, max_new_tokens)
