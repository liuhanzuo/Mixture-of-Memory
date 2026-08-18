# Trained Write diagnostic snapshot

This compact anonymous artifact records the completed task-150 result integrated
as an appendix diagnostic in Paper A. It contains aggregate values only and no
private paths, node identifiers, model weights, benchmark text, or credentials.

The trained Write LoRA acts on layers 0--11; the frozen flagship Read LoRA acts
on layers 12--35. The evaluated cohort is RULER `niah_multikey_1` at 8k and 16k,
100 examples per length. The result is intentionally scoped as a synthetic
paired diagnostic rather than a natural-task or production-serving claim.
