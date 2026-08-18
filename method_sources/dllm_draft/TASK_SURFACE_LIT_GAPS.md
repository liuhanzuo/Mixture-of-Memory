## Task-surface lens: verified literature gaps (2026-08-07)

KEY: arXiv 2510.04605 'Exploring the Power of DLLMs for SE' is the ONLY paper claiming
diffusion>AR on repair. It is NOT a credible preemption:
  - CoRR-only (OpenReview/DBLP: venueid=dblp.org/journals/CORR/2025, no conference venue)
  - compares commercial Mercury-Diffusion-7B vs AR-Llama3-8B (a GENERAL AR model, not a code
    model) => catastrophic confound; no Qwen2.5-Coder control
  - two garbled [?] citations for its own PR/DDF1 metric definitions
  - cites LLaDA as arXiv:2309.12401, which is actually 'The dependence of assembly bias on
    the cosmic web' (astrophysics) => fabricated reference

Zero-hit arXiv queries (repair/edit surface is empty for diffusion):
  diffusion+program repair; diffusion+HumanEvalFix; +CanItEdit; +Defects4J; +QuixBugs;
  +HumanEvalPack; +multi-site edit; +refactoring; diffusion LM+execution feedback

DreamOn (2602.01326, ICLR 2026 Poster) ALREADY did HumanEval-Infilling single/multi-line +
SantaCoder-FIM(Python) vs Qwen2.5-Coder-7B/Deepseek-Coder-6.7B/Seed-Coder-8B.
  Dream-Coder-7B single-line 55.5 -> +DreamOn 92.1 ; Qwen2.5-Coder-7B 92.6
  multi-line 43.2 -> 63.8 ; Qwen 58.7    SantaCoder-FIM 59.3 -> 79.0 ; Qwen 79.8
  NOT covered: any compute/NFE accounting, RandomSpanInfilling, multi-span, non-Python.

Confidence-based remasking is already known weak (2606.12232 null result); execution-grounded
remasking is untouched.

HARD LOCAL BLOCKERS: no docker/podman => SWE-bench* infeasible. no java/javac/go/rustc =>
Defects4J + HumanEvalPack{java,cpp,go,rust} infeasible. Qwen2.5-Coder-7B NOT on either disk.
