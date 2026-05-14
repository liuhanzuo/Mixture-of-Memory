# Mixture-of-Memory Project — Complete Exploration Index

**Generated:** 2026-05-05  
**Purpose:** Comprehensive reference for Git + CI/CD + LLM-powered code review system planning

---

## 📚 Generated Documentation (3 Files)

### 1. **EXPLORATION_REPORT.md** (23 KB, 446 lines)
   **Comprehensive technical deep-dive**
   - Full directory structure with 27 root entries
   - Complete src/memory hierarchy (15 submodules)
   - scripts/ organization (200+ files across 5 categories)
   - Status file semantics and workflows
   - State file operational patterns
   - Key findings for CI/CD integration
   - Project statistics and conclusions
   - **Best for:** Planning your full CI/CD system

### 2. **QUICK_REFERENCE.txt** (13 KB, 162 lines)
   **One-page lookup guide**
   - Executive summary
   - Critical files checklist
   - State file categories (append-only vs write-only)
   - Code structure visualization
   - Agent architecture diagram
   - Multi-node orchestration rules
   - Code review checklist
   - Implementation milestones
   - **Best for:** Quick lookups while implementing

### 3. **COMMAND_REFERENCE.md** (11 KB, 200 lines)
   **Executable commands and code snippets**
   - How to read each manual file
   - Commands to understand current state
   - Git validation commands
   - GPU monitoring commands
   - Experiment history queries
   - Python code patterns for CI/CD
   - State file validation examples
   - **Best for:** Implementation and scripting

---

## 📖 Original Project Files (Must Read)

### Essential Manuals

1. **CLAUDE.md** (18 KB, 336 lines)
   - **Lines 3-14**: Auto-dispatch rules
   - **Lines 18-31**: PENDING_TASKS system ⭐
   - **Lines 66-69**: Git rules (NO Claude trailers) ⚠️
   - **Lines 210-237**: Parallel algorithm isolation
   - **Lines 305-315**: Version management requirements
   - **READ THIS FIRST for:** Understanding autonomy rules

2. **HEARTBEAT.md** (17 KB, 440 lines)
   - **Lines 7-50**: Architecture explanation
   - **Lines 70-200**: 6-step mandatory checks ⭐
   - **Lines 301-400**: Experiment lifecycle automation
   - **Lines 415-439**: Response format
   - **READ THIS FIRST for:** Understanding heartbeat operation

### Task & Status Files

3. **status/PENDING_TASKS.md** (Current state)
   - `[RUNNING]` unleashed_cross_attn (b200-2/3, ratio ~0.998)
   - `[PENDING]` long_context_eval (needs user confirm)
   - `[PENDING]` direction_decision (research + user confirm)
   - Node availability table

4. **status/** (9 state files total)
   - TRAINER_ACTIVE.md (write-only)
   - TRAINER_REQUESTS.jsonl (append-only)
   - TRAINER_APPROVALS.jsonl (append-only)
   - RESEARCHER_REPORTS.jsonl (append-only)
   - ISSUES.jsonl (append-only)
   - gpu_runs.jsonl (append-only)
   - TRAINER_ACTIVITY.jsonl (append-only)

### Architecture & Literature

5. **versions/vN_*.md** (3 version docs)
   - v2_cross_attention.md
   - v3_infini_attention.md
   - v4_chunk_last_hidden_memory.md

6. **UPDATELOG.md** (277 KB)
   - 5000+ timestamped entries
   - Recent: Infini-attention recommendation

7. **RESEARCH_LITERATURE.md** (277 KB)
   - Continuously updated literature survey

---

## 🎯 Key Findings Summary

### Critical Constraints
- ⚠️ **NO "Co-Authored-By: Claude" in commits** (CLAUDE.md Line 66-69)
- ⚠️ **Every src/memory/ change requires versions/vN_*.md** (CLAUDE.md Line 305-315)
- ⚠️ **PENDING_TASKS.md MUST be checked every heartbeat cycle** (HEARTBEAT.md Line 70-86)

### Operational Patterns
- **Append-Only Logs**: Use correction entries on error, never edit
- **Write-Only State**: TRAINER_ACTIVE.md must be fully replaced, never edited
- **Confidence-Based Autonomy**: researcher high/very_high confidence → auto-execute
- **Forced Loop Closure**: No waiting for user after experiment completion

### Multi-Node Orchestration
- **5 compute nodes**: 4 B200 (8×L20A @ 183 GiB) + 1 local (8×H20 @ 97.8 GiB)
- **Algorithm isolation**: Each gets unique config names, output dirs, log files
- **Scheduling rule**: No idle GPU + pending task allowed

### Code Review Checkpoints
- Numerical precision (float32 vs float64)
- Tensor shape mismatches
- Gradient clipping bugs
- Memory semantics vs design doc
- Eval symmetry verification

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Python modules | ~70 files |
| Training scripts | 40+ |
| Evaluation scripts | 50+ |
| Launcher scripts | 80+ |
| Test files | 16 |
| Architecture versions | 3 (v2, v3, v4) |
| State files | 9 (7 append-only, 1 write-only, 1 regular) |
| Remote B200 nodes | 4 |
| Local GPU count | 8 H20 cards |
| Total GPU memory | 6024 GiB |
| Current PPL best | 2.5-5.8 |
| Current PPL baseline | 1200-2000 |

---

## ✅ Implementation Roadmap

### Week 1: Learn the System
- [ ] Read CLAUDE.md (main manual)
- [ ] Read HEARTBEAT.md (heartbeat operation)
- [ ] Understand status/ file semantics
- [ ] Map state files workflow
- [ ] Read EXPLORATION_REPORT.md (full reference)

### Week 2: Design CI/CD Integration
- [ ] Implement Claude trailer detection + rejection
- [ ] Implement versions/vN_*.md validation
- [ ] Implement PENDING_TASKS.md format checking
- [ ] Implement append-only log safety
- [ ] Map researcher/coder dispatch triggers

### Week 3: Implement Core Features
- [ ] Git pre-commit hooks (Claude trailer check)
- [ ] GitHub Actions (mirror heartbeat checks)
- [ ] Structured researcher dispatch (checklist-based)
- [ ] Structured coder dispatch (isolation-aware)
- [ ] PPL anomaly detection (>100 escalation)

### Month 1: Advanced Features
- [ ] LLM root-cause analysis on failures
- [ ] Semantic code change detection
- [ ] Multi-branch experiment matrix
- [ ] Checkpoint validation pipeline
- [ ] Auto-commit from UPDATELOG.md entries

---

## 🔗 Navigation Guide

### If you need to understand...

**Git rules & commit constraints**
→ CLAUDE.md (lines 66-69) + EXPLORATION_REPORT.md

**How heartbeat decides actions**
→ HEARTBEAT.md (lines 55-259) + CLAUDE.md (lines 44-61)

**Task queue structure**
→ status/PENDING_TASKS.md + EXPLORATION_REPORT.md

**Multi-node orchestration**
→ CLAUDE.md (lines 71-79, 210-237) + QUICK_REFERENCE.txt

**Code review checklist**
→ HEARTBEAT.md (lines 171-199) + QUICK_REFERENCE.txt

**State file patterns**
→ CLAUDE.md (lines 286-301) + COMMAND_REFERENCE.md

**Experimental results**
→ status/gpu_runs.jsonl + RESEARCHER_REPORTS.jsonl + UPDATELOG.md

**Architecture design**
→ versions/vN_*.md + RESEARCH_LITERATURE.md

---

## 🎓 Key Design Lessons

1. **State as System of Record**
   - No background processes; each session is stateless
   - All state in version-controlled files (JSONL + Markdown)
   - Append-only for safety; write-only for critical updates

2. **Autonomous But Accountable**
   - Heartbeat can auto-execute with researcher confirmation
   - But MUST record every action (UPDATELOG.md)
   - Forced loop closure prevents stuck experiments

3. **Isolation by Convention**
   - Parallel algorithms don't require different branches
   - Instead: config name → output dir → log file (1:1:1 mapping)
   - Coordination via PENDING_TASKS.md (not Git)

4. **Code Review as LLM Task**
   - Not just syntax/style; includes semantic checks
   - Dispatches researcher on failures (not manual review)
   - Auto-fix for high-confidence issues

5. **PPL as Diagnostic Signal**
   - PPL > 100 = model broken (not just suboptimal)
   - PPL > 1000 = random output (check basics)
   - This signals architectural issues, not hyperparameter tweaking

---

## 💾 All Files at a Glance

```
Generated for you:
├── EXPLORATION_REPORT.md      (23 KB) — Full technical deep-dive
├── QUICK_REFERENCE.txt        (13 KB) — One-page lookup
├── COMMAND_REFERENCE.md       (11 KB) — Executable commands
└── INDEX.md                   (this file) — Navigation guide

Original project (critical reads):
├── CLAUDE.md                  (18 KB) — Main manual ⭐
├── HEARTBEAT.md              (17 KB) — Heartbeat operation ⭐
├── status/PENDING_TASKS.md   — Current state ⭐
├── .gitignore                (1 KB)  — Git rules
└── versions/vN_*.md          (30 KB) — Architecture docs

Logs & History:
├── UPDATELOG.md              (277 KB) — Timestamped log
├── RESEARCH_LITERATURE.md    (277 KB) — Literature survey
└── status/*.jsonl            — State files
```

---

## 🚀 Next Steps

1. **Start here**: Read QUICK_REFERENCE.txt (5 min)
2. **Then read**: CLAUDE.md (20 min) + HEARTBEAT.md (20 min)
3. **Deep dive**: EXPLORATION_REPORT.md (30 min)
4. **Execute**: COMMAND_REFERENCE.md patterns for your CI/CD

---

**Last Updated:** 2026-05-05 11:38 UTC
**Status:** Complete exploration ready for implementation
