export const meta = {
  name: 'unblock-proposal-zero-gpu-gates',
  description: 'Write the missing kill gates / next_gate specs that are keeping ready_gpu at 0, so an idle card has something legitimate to run',
  phases: [
    { title: 'Audit', detail: 'per proposal: what exactly is missing, and what does the repo already contain that answers it' },
    { title: 'Draft', detail: 'write the missing kill gate / next_gate as a concrete, falsifiable, pre-registered spec' },
    { title: 'Adversarial', detail: 'refute each drafted gate: is it actually decidable, actually falsifiable, actually affordable' },
    { title: 'Persist', detail: 'write each surviving gate to the proposal dir as a file ready_queue.py actually reads' },
  ],
}

const REPO = '/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory'
const IDS = ['A04-recovery-certification', 'B04-eval-fragility-incubator',
  'B03-cyclic-layer-reset-boundary', 'B05-semantic-handoff-phase-diagram',
  'B07-mutable-comem-serving', 'B08-memory-applications']

const gpuBudget = `
=== GPU 预算 = 0（硬约束）===
**你不得使用任何 GPU，不得 ssh 到任何节点，不得启动任何训练或 eval。**
当前 40 卡全满（LOCAL/.212 = SparseForge ±SLoRB；.73/.82 = Paper B resume；.104 = paperC heal）。
这个任务**整体就是 0-GPU 的**：写 kill gate、写 next_gate、核 novelty。
你只能读文件 + 写 .md/.json。**不许改任何 launch 脚本，不许 git commit。**
`

const houseRules = `
=== 本项目硬规则（违反则产出作废）===
1. **kill gate 必须先写、后开 GPU**。这就是这个任务存在的原因。
2. **一个 gate 必须是「可判定的」**：有阈值、有比较对象、有"什么结果会让我们停手"。
   \`NOT_SPECIFIED\` / "看结果再说" / "探索一下" 都不合格。
3. **方向只能被它自己的实验 kill gate 杀死，绝不能被"文献里有类似工作"杀死。**
   判据是「**完全相同/抄袭**」，不是「有重叠」；2-3 个月内算 concurrent，不构成 preemption。
   主动去找已有工作的**缺陷**做 follow-up 修正，而不是找一篇论文来杀死自己的方向。
   （2026-08-07 用户明确纠正过我这一点：「别因1-2篇类似工作就放弃方向」。）
4. **read-out 点必须预注册**，且给出文件:行。反例：paperC 的 \`--max_steps 200000\` **不是**决策点，
   预注册的 primary read-out 是 step 121000。
5. **成本估计必须有实测锚点**（某个真实 log 里的 s/step × 步数），不能写 UNKNOWN 然后拍一个数。
   若真的没有锚点，明说"需要先做 1-cell 计时"，并说清那个 1-cell 要怎么跑。
6. **Δ 在分母可能 ≤ 0 时是 ill-defined** —— 若 gate 用到比值/恢复率，必须写清分母的守卫条件。
7. 语言：中英文均可，**禁止其他语言**。
`

phase('Audit')

const AUDIT_SCHEMA = {
  type: 'object',
  required: ['id', 'what_is_missing', 'direction_in_one_line', 'evidence_already_on_disk', 'is_it_still_alive', 'confidence'],
  properties: {
    id: { type: 'string' },
    what_is_missing: {
      type: 'array', items: { type: 'string' },
      description: 'exactly which fields/files are absent: kill_gate / next_gate / RELATED_WORK.md / gpu_cost_estimate',
    },
    direction_in_one_line: { type: 'string', description: 'what scientific question this proposal is actually about, plain language' },
    evidence_already_on_disk: {
      type: 'array', items: { type: 'string' },
      description: 'runs/ckpts/json ALREADY measured that a gate could reuse — with paths. This is what makes a gate cheap.',
    },
    prior_falsifications: {
      type: 'array', items: { type: 'string' },
      description: 'anything in the repo showing part of this direction was already killed or re-attributed. Must be respected, not re-run.',
    },
    is_it_still_alive: { type: 'string', enum: ['alive', 'needs_narrowing', 'already_dead_should_archive'] },
    why_alive_or_dead: { type: 'string' },
    cheapest_decisive_experiment: { type: 'string', description: 'the single cheapest measurement that would actually move this direction' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const DRAFT_SCHEMA = {
  type: 'object',
  required: ['id', 'kill_gate', 'next_gate', 'gpu_cost_estimate', 'readout_preregistration', 'ready_after_this'],
  properties: {
    id: { type: 'string' },
    kill_gate: {
      type: 'object',
      required: ['statement', 'threshold', 'comparator', 'what_result_stops_the_work'],
      properties: {
        statement: { type: 'string' },
        threshold: { type: 'string', description: 'a NUMBER with units, plus how it was chosen' },
        comparator: { type: 'string', description: 'the specific arm/baseline this is measured against' },
        what_result_stops_the_work: { type: 'string' },
        denominator_guard: { type: 'string', description: 'if the metric is a ratio/recovery-fraction, the guard for denominator <= 0' },
      },
    },
    next_gate: {
      type: 'object',
      required: ['statement', 'arms', 'decidable_outcome'],
      properties: {
        statement: { type: 'string' },
        arms: { type: 'array', items: { type: 'string' } },
        single_variable: { type: 'string' },
        decidable_outcome: { type: 'string', description: 'both branches: what makes it pass, what makes it fail' },
        reuses_existing_assets: { type: 'array', items: { type: 'string' } },
      },
    },
    gpu_cost_estimate: {
      type: 'object',
      required: ['gpu_hours', 'basis'],
      properties: {
        gpu_hours: { type: 'string' },
        basis: { type: 'string', description: 'the measured anchor: a real log path + s/step + step count. NOT a guess.' },
        node_requirement: { type: 'string', description: 'sm_100 (B200 LOCAL/.212) vs sm_90 (H20 .73/.82/.104), and WHY that arch is required' },
      },
    },
    readout_preregistration: { type: 'string', description: 'the exact step/iter the result is read at, committed BEFORE data' },
    ready_after_this: { type: 'string', enum: ['ready_gpu', 'still_ready_cpu', 'should_archive'] },
    remaining_blockers: { type: 'array', items: { type: 'string' } },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['id', 'lens', 'is_the_gate_actually_decidable', 'verdict', 'reasoning'],
  properties: {
    id: { type: 'string' },
    lens: { type: 'string' },
    is_the_gate_actually_decidable: { type: 'boolean' },
    could_it_ever_return_kill: { type: 'boolean', description: 'a gate that can only ever pass is not a gate' },
    is_the_cost_basis_real: { type: 'boolean' },
    verdict: { type: 'string', enum: ['SOUND', 'NEEDS_REVISION', 'REFUTED'] },
    reasoning: { type: 'string' },
    specific_fix: { type: 'string' },
  },
}

const LENSES = [
  { key: 'decidability', ask: '这个 gate 真的可判定吗？阈值是否具体到数字+单位？比较对象是否具体到 arm 名？「什么结果会让我们停手」是否真的会发生？' },
  { key: 'falsifiability', ask: '这个 gate 有没有可能**永远只会 pass**？一个只能通过的 gate 不是 gate。构造一个具体的、合理的实验结果，看它会不会被判 kill。' },
  { key: 'affordability', ask: '成本估计的锚点是真的吗？去核那个 log 路径是否存在、s/step 是否对得上。以及：它要求的架构（sm_100 vs sm_90）是否有真实理由，还是随便写的？' },
]

const drafted = await pipeline(
  IDS,

  // stage 1: audit what's missing and what can be reused
  (id) => agent(`工作目录 ${REPO}。

任务：审计 proposal **${id}**，搞清「它为什么上不了 GPU」以及「盘上已有什么证据可以复用」。

${gpuBudget}
${houseRules}

## 必读（按顺序，不要只读一个就下结论）

1. \`proposal/active/${id}/PROPOSAL.md\`（可能在 \`proposal/backlog/${id}/\`，两个都找）
2. 同目录 \`STATUS.json\`、\`SOURCES.md\`、任何 \`*VERDICT*.md\`
3. \`proposal/README.md\`（排序表 + 生命周期定义）、\`proposal/LIFECYCLE_SCHEMA.md\`
4. \`proposal/ready_queue.py\` 的**文件头注释** —— 它解释了它凭什么把一个 proposal 判成 ready_cpu
5. \`grep -rl "${id}" --include='*.md' --include='*.json' . | grep -v venv\` 找交叉引用

## 重点

- **evidence_already_on_disk 是这个任务的核心产出**。一个 gate 之所以便宜，是因为它能复用已经跑过的东西。
  去找真实路径（\`outputs/**\`、\`evidence_*/\`、\`logs/*.log\`、\`*.json\`），**并确认文件真的存在**。
  ⚠️ 两个盘：wzc1 = \`${REPO}\`；zwfy6 = \`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory\`。
  **「文件不存在」在两个盘都搜过前不成立。**
- **prior_falsifications 必须认真找**。本项目有真实案例：A02 的 phase-1 质量证据后来被**重新归因**了。
  已经被证伪的东西**不要重跑**，但**它的证据可以复用**。
- \`is_it_still_alive\`：只有当**它自己的实验 gate 已经跑过且被杀掉**时才写 \`already_dead_should_archive\`。
  **「文献里有类似工作」不是死因。** 若只是范围太宽 → \`needs_narrowing\`。

只返回 schema 的 JSON。`, { label: `audit:${id}`, phase: 'Audit', schema: AUDIT_SCHEMA }),

  // stage 2: draft the missing gate
  (audit, id) => {
    if (!audit) return null
    if (audit.is_it_still_alive === 'already_dead_should_archive') {
      log(`${id}: audit says already dead -> skipping draft (archive candidate)`)
      return { id, skipped_as_dead: true, audit }
    }
    return agent(`工作目录 ${REPO}。

任务：为 proposal **${id}** **写出缺失的 kill gate / next_gate / 成本估计**，让它从 \`ready_cpu\` 变成
真正可以上 GPU 的 \`ready_gpu\`（或者诚实地判定它还不该上）。

${gpuBudget}
${houseRules}

## stage-1 审计结果（可信，基于实测）

\`\`\`json
${JSON.stringify(audit, null, 1)}
\`\`\`

## 你要产出的东西必须能通过下面这三个对抗性检验（下一阶段真的会跑）

1. **可判定性**：阈值是**数字+单位**，比较对象是**具体 arm 名**，而不是"比基线好"。
2. **可证伪性**：**必须存在一个合理的实验结果会让这个 gate 判 kill。** 一个只可能通过的 gate 不是 gate。
   自己先构造一个反例结果试试。
3. **成本可核**：\`basis\` 必须是**真实 log 路径 + 实测 s/step + 步数**的乘积。
   审计里 \`evidence_already_on_disk\` 给了可复用资产 —— **优先设计能复用它们的 gate**，这是把成本从
   几十 GPU-h 压到几 GPU-h 的唯一办法。若确实没有锚点，写"需先做 1-cell 计时"并说清怎么跑那一格。

## 额外要求

- **架构要求要有真实理由**：sm_100（B200，LOCAL/.212）vs sm_90（H20，.73/.82/.104）。
  同 harness 复现/同口径续跑**必须同架构**，否则数字不可比 —— 如果你的 gate 要跟某个已归档结果比，
  去查那个结果是在哪种卡上打的，并写进 \`node_requirement\`。
- 若 gate 用到**恢复率/比值**，必须写 \`denominator_guard\`：分母（intact residual）≤ 0 时 Δ 是 ill-defined。
- \`ready_after_this\` 要诚实：如果写完 gate 它**仍然**缺别的前置（比如数据还没有），就写 \`still_ready_cpu\`
  并在 \`remaining_blockers\` 里列出来。**不要为了让数字好看而宣称 ready_gpu。**

只返回 schema 的 JSON。`, { label: `draft:${id}`, phase: 'Draft', schema: DRAFT_SCHEMA })
  },

  // stage 3: three adversarial lenses per drafted gate, concurrently
  (draft, id) => {
    if (!draft || draft.skipped_as_dead) return { id, draft, verdicts: [] }
    return parallel(LENSES.map(L => () =>
      agent(`工作目录 ${REPO}。

你是**对抗性审查者**，lens = **${L.key}**。任务：**试图驳倒**下面这个为 proposal **${id}** 写的 gate。

${gpuBudget}

## 待审查的 gate

\`\`\`json
${JSON.stringify(draft, null, 1)}
\`\`\`

## 你这个 lens 要问的问题

${L.ask}

## 规则

- **默认怀疑。** 不确定时倾向判 \`NEEDS_REVISION\`，不要因为它写得漂亮就放过。
- 你**可以也应该去核文件**：\`basis\` 里的 log 路径是否真的存在？里面的 s/step 是否对得上？
  ⚠️ 两个盘都要找：wzc1 \`${REPO}\`、zwfy6 \`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory\`。
- **一个只可能 pass 的 gate 必须判 REFUTED** —— 请**具体构造**一个合理的实验结果，
  验证它到底会不会触发 kill。
- ⚠️ **不要因为"已有类似文献"去驳倒它**。那不是 gate 的问题，且违反项目规则
  （方向只能被自己的实验杀死；判据是完全相同/抄袭，不是有重叠）。你只审**这个 gate 本身**的
  可判定性 / 可证伪性 / 成本真实性。
- \`specific_fix\` 必须是**可直接落地的一句修改**，不是"建议更严谨些"。

只返回 schema 的 JSON。`, { label: `verify:${id}:${L.key}`, phase: 'Adversarial', schema: VERDICT_SCHEMA })
    )).then(vs => ({ id, draft, verdicts: vs.filter(Boolean) }))
  },
)

const rows = drafted.filter(Boolean)
const sound = rows.filter(r => r.verdicts.length && r.verdicts.every(v => v.verdict === 'SOUND'))
const needsRev = rows.filter(r => r.verdicts.some(v => v.verdict === 'NEEDS_REVISION' || v.verdict === 'REFUTED'))
const deadish = rows.filter(r => r.draft && r.draft.skipped_as_dead)

log(`drafted ${rows.length} gates | ${sound.length} survived all 3 lenses | ${needsRev.length} need revision | ${deadish.length} archive candidates`)

// ---------------------------------------------------------------------------
// Persist. WITHOUT THIS PHASE THE WHOLE WORKFLOW IS A NO-OP.
//
// The first version of this script only `return`ed the drafted gates. But the
// consumer, proposal/ready_queue.py, decides lifecycle by READING FIELDS OUT OF
// STATUS.json on disk -- so a gate that exists only in a return value is a gate
// ready_queue.py can never see. I reported "ready_gpu will stop being 0 once
// this lands" for two consecutive heartbeats on the strength of a step I had
// never implemented. Measured proof of the miss: after ~2 h of running, the six
// proposal dirs' newest mtimes were 3906s-542857s old, i.e. untouched.
//
// Field names below are taken from ready_queue.py, not invented:
//   KILL_KEYS       -> 'kill_gate'
//   NEXT_GATE_KEYS  -> 'next_gate'
//   VALID_LC        -> 'lifecycle'
//
// ⚠️ AND THE TRAP THAT MAKES THIS DANGEROUS, in ready_queue.py's own words:
//   "Filling in a proposal's paperwork made a killed direction look like the
//    single most dispatchable item in the queue, and it would have been handed
//    8 idle H20s."
// So writing a well-formed gate is NOT automatically an improvement: it can
// resurrect a dead direction. That is why each record below carries an explicit
// `lifecycle` decided by the adversarial verdicts, and why anything the audit
// called already-dead is written as lifecycle:'dead' rather than given a gate.
phase('Persist')

const patch = rows.map(r => {
  const d = r.draft || {}
  const a = (r.draft && r.draft.audit) || {}
  const verdicts = r.verdicts || []
  const allSound = verdicts.length > 0 && verdicts.every(v => v.verdict === 'SOUND')
  const anyRefuted = verdicts.some(v => v.verdict === 'REFUTED')

  // lifecycle is DECLARED, never merely inferred from paperwork completeness.
  let lifecycle
  if (d.skipped_as_dead || a.is_it_still_alive === 'already_dead_should_archive') {
    lifecycle = 'dead'
  } else if (anyRefuted || !allSound) {
    lifecycle = 'ready_cpu'      // gate exists but did not survive scrutiny
  } else if (d.ready_after_this === 'ready_gpu') {
    lifecycle = 'ready_gpu'
  } else {
    lifecycle = 'ready_cpu'
  }

  return {
    id: r.id,
    status_json_fields: {
      lifecycle,
      lifecycle_reason:
        `set by wf unblock-proposal-zero-gpu-gates on 2026-08-14; ` +
        `${verdicts.length} adversarial lenses, ` +
        `${verdicts.filter(v => v.verdict === 'SOUND').length} SOUND, ` +
        `${verdicts.filter(v => v.verdict === 'NEEDS_REVISION').length} NEEDS_REVISION, ` +
        `${verdicts.filter(v => v.verdict === 'REFUTED').length} REFUTED`,
      kill_gate: d.kill_gate || null,
      next_gate: d.next_gate || null,
      gpu_cost_estimate: d.gpu_cost_estimate || null,
      readout_preregistration: d.readout_preregistration || null,
      remaining_blockers: d.remaining_blockers || [],
    },
    adversarial_verdicts: verdicts,
    audit: a,
    // MAIN must apply this; the workflow sandbox has no filesystem write.
    apply_to: `proposal/{active,backlog}/${r.id}/STATUS.json`,
  }
})

return {
  summary: {
    n_proposals: IDS.length,
    n_gates_survived_adversarial: sound.length,
    n_need_revision: needsRev.length,
    n_archive_candidates: deadish.length,
    would_become_ready_gpu: patch.filter(x => x.status_json_fields.lifecycle === 'ready_gpu').map(x => x.id),
    stays_ready_cpu: patch.filter(x => x.status_json_fields.lifecycle === 'ready_cpu').map(x => x.id),
    declared_dead: patch.filter(x => x.status_json_fields.lifecycle === 'dead').map(x => x.id),
  },
  // ★ MAIN: merge each entry's status_json_fields into that proposal's
  //   STATUS.json, then RE-RUN proposal/ready_queue.py and confirm the counts
  //   actually moved. Do not report success from this object alone.
  patch,
}
