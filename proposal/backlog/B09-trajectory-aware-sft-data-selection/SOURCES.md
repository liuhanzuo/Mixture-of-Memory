# Sources

## Internal research records

- `../../../status/RESEARCHER_REPORTS.jsonl`
  - 2026-08-08 report on training-free / low-training-cost SFT data selection.
- Conversation research synthesis on:
  - trajectory-aware selection;
  - experimental design and leakage control;
  - training-free, forward-only, and low-training method taxonomy.

## Primary literature

### Closest agent and multi-turn collisions

- Yang et al., **TopoCurate: Modeling Interaction Topology for Tool-Use Agent
  Training**, arXiv:2603.01714.
  - whole-trajectory tool-use selection using Reflective Recovery, Semantic
    Efficiency and Strategic Diversity;
  - strongest collision for broad “agent trajectory-aware selection” claims.
- Li et al., **Data Selection for Multi-turn Dialogue Instruction Tuning**,
  Findings of ACL 2026, arXiv:2604.07892.
  - MDS: whole-dialogue global semantic coverage plus local structural quality;
  - strongest collision for “multi-turn rows are not independent” claims.
- Chen et al., **ATLAS: Agent Tuning via Learning Critical Steps**,
  Findings of ACL 2025, arXiv:2503.02197.
  - critical-step loss masking over plan, observation, action and correction.
- Li et al., **Verified Critical Step Optimization for LLM Agents**,
  arXiv:2602.03412.
  - outcome-flip verified decision points and targeted preference optimization.
- Han et al., **SWE-TRACE: Optimizing Long-Horizon SWE Agents through Rubric
  Process Reward Models and Heuristic Test-Time Scaling**,
  arXiv:2604.14820.
  - step-wise oracle synthesis of token-efficient, shortest-path SWE trajectories.
- **A Systematic Evaluation of Trajectory Data Curation for LoRA Fine-Tuning
  of Code Agents**, arXiv:2607.17205.
  - systematic efficiency/style filtering of code-agent trajectories.

### Large-scale instruction selection and representation matching

- Ivison et al., **Large-Scale Data Selection for Instruction Tuning**,
  arXiv:2503.01807.
  - RDS+, weighted hidden-state pooling, target-query round-robin selection;
  - shows many methods fail to beat random as pool size grows.
- Xia et al., **LESS: Selecting Influential Data for Targeted Instruction
  Tuning**, ICML 2024, arXiv:2402.04333.
  - short LoRA warmup, projected optimizer-aware gradient features.
- Agarwal et al., **DELIFT: Data Efficient Language Model Instruction
  Fine-Tuning**, ICLR 2025, arXiv:2411.04425.
  - ICL-based pairwise utility with facility-location and mutual-information
    submodular objectives, including target-specific selection.
- Renduchintala et al., **SMART: Submodular Data Mixture Strategy for
  Instruction Tuning**, Findings of ACL 2024, arXiv:2403.08370.
  - submodular task selection, task-budget allocation and within-task subset
    selection.

### Quality, difficulty, and diversity

- Li et al., **From Quantity to Quality: Boosting LLM Performance with
  Self-Guided Data Selection for Instruction Tuning**, NAACL 2024,
  arXiv:2308.12032.
  - Instruction-Following Difficulty (IFD).
- Li et al., **Superfiltering: Weak-to-Strong Data Filtering for Fast
  Instruction-Tuning**, ACL 2024, arXiv:2402.00530.
  - weak proxy models for efficient difficulty filtering.
- Liu et al., **What Makes Good Data for Alignment? A Comprehensive Study of
  Automatic Data Selection in Instruction Tuning**, ICLR 2024,
  arXiv:2312.15685.
  - DEITA: complexity × quality followed by representation diversity filtering.
- Chen et al., **AlpaGasus: Training a Better Alpaca with Fewer Data**,
  ICLR 2024, arXiv:2307.08701.
  - frozen strong-LLM quality filtering.
- Zhao et al., **Long Is More for Alignment: A Simple but Tough-to-Beat
  Baseline for Instruction Fine-Tuning**, ICML 2024, arXiv:2402.04833.
  - longest-response selection as a mandatory simple baseline.
- Chen et al., **MIG: Automatic Data Selection for Instruction Tuning by
  Maximizing Information Gain in Semantic Space**, 2025,
  arXiv:2504.13835.
  - quality-weighted label-graph information gain and submodular selection.
- Wang et al., **Data Whisperer: Efficient Data Selection for Task-Specific
  LLM Fine-Tuning via Few-Shot In-Context Learning**, 2025,
  arXiv:2505.12212.
  - training-free, attention-weighted few-shot ICL selector.
- Chen et al., **Agent-FLAN: Designing Data and Methods of Effective Agent
  Tuning for Large Language Models**, Findings of ACL 2024,
  arXiv:2403.12881.
  - capability decomposition and rebalancing across reasoning, tool retrieval,
    parameter understanding and instruction/format following.
- Ye et al., **Disentangling Reasoning Tokens and Boilerplate Tokens For
  Language Model Fine-tuning**, Findings of ACL 2025, arXiv:2412.14780.
  - learned discrimination and differential weighting of reasoning versus
    boilerplate tokens in agent data.

### Agent trajectory and critical-step supervision

- See the closest-collision section above and `NOVELTY.md`.

## Classical subset-selection background

- Facility-location, k-center and monotone submodular coreset selection.
- Partition/laminar constraints for parent trajectory and benchmark caps.
- Knapsack-style token-budget constraints.
- Lazy/stochastic greedy for scalable approximate maximization.

## Claims requiring further source audit before publication

1. Whether an existing work already combines:
   - parent-trajectory caps;
   - critical-step credit;
   - target facility coverage;
   - token-budgeted submodular selection.
2. Official implementations and license compatibility for TopoCurate, MDS,
   ATLAS, CSO, RDS+, LESS, DELIFT, SMART, IFD,
   DEITA/MIG scorers and any selected embedding model.
3. Exact venue status for recent 2026 arXiv works that are not yet formally
   published.
4. Additional search for parent-child/group-constrained coreset selection in
   imitation learning, offline RL and sequential decision-making literature.

The broad novelty claim has already been narrowed in `NOVELTY.md`. Do not promote
to an algorithmic novelty claim until the remaining four items are audited and a
non-compositional technical core is implemented.
