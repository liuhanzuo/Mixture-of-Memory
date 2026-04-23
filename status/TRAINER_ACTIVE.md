# TRAINER_ACTIVE.md - Active Training Runs

No active training runs.

## Recent Completed Run

### DMS 8x Compression (completed 2026-04-23 11:12)
- **Status**: COMPLETED (800/800 steps)
- **Config**: Qwen3-8B, 8x compression, bf16, 8 GPUs
  - per_device_train_batch_size: 1
  - gradient_accumulation_steps: 8
  - max_seq_length: 2048
  - sliding_window: 256
  - num_train_steps: 800
- **Training time**: ~2h 26m 51s
- **Final metrics**:
  - Final loss: 68.25
  - Final grad_norm: 0 (converged)
  - Final learning_rate: ~4.27e-10
  - DMS parameters: 147,492
- **Checkpoint**: `outputs/dms_8x/final`
- **Log**: `outputs/dms_8x/train.log`
