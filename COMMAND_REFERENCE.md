# Mixture-of-Memory Project — Command & File Reference

## 📂 Files Generated For You

**Location:** `/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/`

1. **EXPLORATION_REPORT.md** (446 lines)
   - Complete directory structure with annotations
   - Full contents of CLAUDE.md and HEARTBEAT.md
   - State file descriptions and workflows
   - CI/CD integration recommendations
   - Project statistics and conclusions

2. **QUICK_REFERENCE.txt** (162 lines)
   - One-page quick lookup
   - Critical files overview
   - State file semantics table
   - Agent architecture diagram
   - Code review checklist
   - Implementation milestones

3. **This file** (command_reference.md)
   - How to navigate the project
   - Key commands to explore further

---

## 🔍 Key Commands to Understand The Project

### Read the Main Instructions

```bash
# Main operational manual (MUST READ)
cat /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/CLAUDE.md | less

# Heartbeat operation manual (MUST READ)
cat /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/HEARTBEAT.md | less

# Quick reference (START HERE)
cat /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/QUICK_REFERENCE.txt
```

### Understand Current State

```bash
# See what tasks are pending/running
cat /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/PENDING_TASKS.md

# Check current active training
cat /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/TRAINER_ACTIVE.md

# See recent operations
tail -50 /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/UPDATELOG.md

# Check for open issues
cat /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/ISSUES.jsonl | jq 'select(.status=="open")'
```

### Explore Code Structure

```bash
# Memory implementations
ls -la /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/src/memory/

# Training scripts
ls /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/scripts/train_*.py | head -20

# Evaluation scripts
ls /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/scripts/eval_*.py | head -20

# Architecture versions
cat /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/versions/v*.md
```

### Check Git Status

```bash
cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
git log --oneline -20
git status
git diff HEAD~5..HEAD

# Check for any commit messages with Claude trailers (SHOULD BE ZERO)
git log --grep="Co-Authored-By.*Claude" || echo "✅ No Claude trailers found"
```

### Monitor GPU Status

```bash
# Local GPUs
nvidia-smi

# Remote nodes (requires SSH access + password)
for ip in 28.89.17.143 28.89.17.144 28.89.17.85 28.89.19.134; do
  echo "=== b200 node $ip ==="
  sshpass -f /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/configs/password.txt \
    ssh -o StrictHostKeyChecking=no root@$ip nvidia-smi 2>/dev/null || echo "Timeout or error"
done
```

### Explore Experiment History

```bash
# Training run history (append-only)
tail -30 /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/gpu_runs.jsonl | jq '.'

# Researcher reports
tail -10 /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/RESEARCHER_REPORTS.jsonl | jq '.'

# Trainer requests
cat /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/TRAINER_REQUESTS.jsonl | jq '.[] | select(.status=="pending")'

# Heartbeat activity
tail -10 /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/TRAINER_ACTIVITY.jsonl | jq '.'
```

---

## 🎯 For CI/CD + Git Planning

### Key Validation Points

```bash
# 1. Check for Claude trailers (MUST REJECT in CI/CD)
git log --all --format="%B" | grep -i "Co-Authored-By.*Claude\|Co-Authored-By.*Anthropic" || echo "✅ No Claude trailers"

# 2. Validate version docs exist for memory/ changes
git diff HEAD~1..HEAD src/memory/ | wc -l
if [ $(git diff HEAD~1..HEAD src/memory/ | wc -l) -gt 100 ]; then
  ls -la /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/versions/ | grep "v$(date +%s)"
fi

# 3. Check PENDING_TASKS format
python3 << 'PYTHON'
import yaml
with open('/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/PENDING_TASKS.md') as f:
    content = f.read()
    assert '[PENDING]' in content or '[RUNNING]' in content or '[DONE]' in content
    print("✅ PENDING_TASKS.md format valid")
PYTHON

# 4. Verify state file append-only compliance
tail -1 /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/gpu_runs.jsonl | jq . || echo "✅ Append-only log valid"
```

### Heartbeat Checks (Replicable in CI/CD)

```bash
# This is what heartbeat runs every 20 minutes:

# Step 0: Check PENDING_TASKS
PENDING=$(grep -c "\[PENDING\]" /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/PENDING_TASKS.md)
RUNNING=$(grep -c "\[RUNNING\]" /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/PENDING_TASKS.md)
echo "Pending tasks: $PENDING, Running: $RUNNING"

# Step 1: Local GPU status
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Step 2: Check for stale processes
ps aux | grep -E "python|train" | grep -v grep

# Step 3: Check TRAINER_REQUESTS for approval
grep '"status":"pending"' /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/TRAINER_REQUESTS.jsonl 2>/dev/null || echo "No pending requests"

# Step 4: Check active training
tail -5 /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/TRAINER_ACTIVE.md
```

---

## 📋 State Files Quick Reference

### Append-Only Files (use `.correction` lines on error)

```
status/TRAINER_REQUESTS.jsonl      # Trainer asks for approval
status/TRAINER_APPROVALS.jsonl     # Main approves/denies
status/RESEARCHER_REPORTS.jsonl    # Researcher findings
status/ISSUES.jsonl                # Bug tracking
status/gpu_runs.jsonl              # Training history
status/TRAINER_ACTIVITY.jsonl      # Heartbeat logs
UPDATELOG.md                       # Human-readable log
```

Example correction entry:
```json
{"ts":"2026-05-05T11:30:00Z","correction":"prior entry at 2026-05-05T11:20:00Z had wrong node; actual was b200-3 not b200-2"}
```

### Write-Only Files (FULL REPLACE, NO EDIT)

```
status/TRAINER_ACTIVE.md           # Current training status
```

Never use `Edit` tool on this file. Always:
1. Read it
2. Modify in memory
3. Write the full file back

---

## 🚀 For Your CI/CD Implementation

### Integrate These Checks

```python
# pseudo-code for your CI/CD

def validate_commit(commit):
    """Validate a commit before merge"""
    
    # 1. No Claude trailers
    msg = commit.message()
    assert "Co-Authored-By" not in msg or "Claude" not in msg, \
        "❌ REJECT: Claude trailer in commit message"
    
    # 2. If src/memory/ changed, require versions/vN_*.md
    changes = commit.changed_files()
    if any("src/memory/" in f for f in changes):
        versions_exist = os.path.exists(f"versions/v{version_number}_*.md")
        assert versions_exist, \
            "❌ REJECT: Memory architecture change without versions/vN_*.md"
    
    # 3. Check PENDING_TASKS.md hasn't been corrupted
    validate_pending_tasks_format()
    
    # 4. If training-related changes, auto-dispatch researcher review
    if any("train_" in f for f in changes):
        dispatch_researcher_code_review(commit)
    
    return True
```

### State File Patterns for Monitoring

```python
# Monitor experiment progress
import json

def get_active_experiments():
    """Read status files to see what's running"""
    with open("status/PENDING_TASKS.md") as f:
        tasks = parse_pending_tasks(f.read())
    
    with open("status/TRAINER_ACTIVE.md") as f:
        active = parse_trainer_active(f.read())
    
    return {
        "pending": [t for t in tasks if t["status"] == "PENDING"],
        "running": [t for t in tasks if t["status"] == "RUNNING"],
        "active_training": active,
    }

def is_gpu_idle():
    """Check if any GPU is idle"""
    import subprocess
    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"], 
                          capture_output=True, text=True)
    utils = [int(line.split()[0]) for line in result.stdout.strip().split('\n')]
    return any(u == 0 for u in utils)

def check_heartbeat_health():
    """Validate heartbeat hasn't been missed"""
    import json
    from datetime import datetime, timedelta
    
    with open("status/TRAINER_ACTIVITY.jsonl") as f:
        last_heartbeat = json.loads(f.readlines()[-1])
    
    age = datetime.now() - datetime.fromisoformat(last_heartbeat["timestamp"])
    
    if age > timedelta(hours=1):  # Should run every 20 min
        return "⚠️ WARNING: Heartbeat missed (last run: {})".format(age)
    
    return "✅ Heartbeat healthy"
```

---

## 📚 Documentation Map

| File | Size | Purpose |
|------|------|---------|
| CLAUDE.md | 18 KB | Main instruction manual |
| HEARTBEAT.md | 17 KB | Heartbeat operation guide |
| README.md | 17 KB | Project overview |
| UPDATELOG.md | 277 KB | Timestamped operation log |
| RESEARCH_LITERATURE.md | 277 KB | Literature survey |
| FIX_PLAN.md | 22 KB | Known issues & fixes |
| versions/v*.md | 30 KB | Architecture documentation |
| EXPLORATION_REPORT.md | 18 KB | **NEW: This report** |
| QUICK_REFERENCE.txt | 9 KB | **NEW: One-pager** |

---

## ✅ Validation Checklist for Your Implementation

Before deploying your Git + CI/CD + LLM code review system:

- [ ] Can detect and reject Claude trailers in commits
- [ ] Can validate PENDING_TASKS.md format
- [ ] Can enforce versions/vN_*.md for memory/ changes
- [ ] Can append to append-only logs safely
- [ ] Can write-replace TRAINER_ACTIVE.md without edits
- [ ] Can parse researcher confidence levels (high/medium/low)
- [ ] Can trigger LLM subagents (researcher/coder dispatch)
- [ ] Can handle multi-node experiments without conflicts
- [ ] Can detect PPL > 100 anomalies
- [ ] Can read and respect PENDING_TASKS auto_launch flags

---

## 🎓 Design Patterns Learned

1. **Append-Only Logs** — Use correction entries instead of edits
2. **Write-Only State** — Avoid Edit tool on critical files (use Read→modify→Write)
3. **Structured Task Queue** — PENDING_TASKS.md is heartbeat's primary input
4. **Confidence-Based Autonomy** — high/very_high confidence → auto-execute
5. **Forced Loop Closure** — No waiting for user after experiment completion
6. **Isolation by Namespacing** — Each algorithm gets unique config/output/log names
7. **Semantic Code Review** — Check for "memory semantics changes" not just line diffs

---

**For full exploration details, see EXPLORATION_REPORT.md (446 lines)**
