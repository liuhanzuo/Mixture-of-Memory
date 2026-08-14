---
name: persist-artifacts-on-wzc1-or-diskb
description: ★用户2026-08-14指令「以后需要保留的东西放wzc1或diskB」; /tmp + /root/.claude + conda env + /usr/bin 都已被重启证明会清空; 判据=「重启后还需要它吗」而非「现在用得到吗」
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

用户 2026-08-14 原话：**「你以后如果有需要保留的东西记得放在 wzc1 或者 diskB 里面」**。

## Why

节点重启会清空的位置，**已全部实测**（2026-08-13 22:2x 那次重启）：

| 位置 | 实测后果 |
|---|---|
| `/tmp/*` | 我把 `sshp.py` 放这里 —— 它是 `sshpass` 消失时**唯一**的连机手段，却和 sshpass 一起会被清掉 |
| `/usr/bin/sshpass`、`/opt/conda/bin/sshpass` | **两处全部消失**，项目几乎所有 launch/eval 脚本靠它连远程 |
| `/opt/conda/envs/torch-base` | 被剥到只剩 torch+numpy；`transformers`/`tqdm`/`safetensors`/`lm_eval` 全没 |
| `/root/.claude`、`/root/.codebuddy` | CLAUDE.md 早有记载会 reset（`cc_state/` 是手动快照） |

最坏的一类是**「重启后才需要、却因重启而消失」的东西**——恢复工具放在会被清空的盘上，等于在最需要它的时刻不存在。

另一个真实损失：union-9 的 pinned harness（`lm_eval 0.4.8` + `transformers 4.57.6`）只活在 conda env 里，
重启后**五台机器全无**，害得两个 90% 完成的训练臂差点无法产出合法的 zero-shot 行。

## How to apply

**判据不是「现在用得到吗」，而是「重启之后还需要它吗」。** 是 → 放持久盘：

- **wzc1** = `/apdcephfs_wzc1/share_304376610/pighzliu_code/`（LOCAL + `.212` 共享，两者间无需 scp）
- **zwfy6 / diskB** = `/apdcephfs_zwfy6/share_304376610/pighzliu_code/`（`.73`/`.82`/`.104`）

具体做法：
- 工具脚本 → `tools/`（例：`tools/ssh_pexpect_fallback.py`，2026-08-14 从 `/tmp` 迁入并 commit `93620e6`）
- pinned/隔离环境 → 项目盘下建 venv（例：`pighzliu_code/venv_union9`，644 MB，`--system-site-packages` 复用 torch 不重下）
- 证据 / evidence JSON / verdict → 对应 `proposal/*/evidence/` 或 `paper*/`，并 git commit
- **迁移时补一段头部注释**说明「为什么存在 + 为什么不放 /tmp」，否则下一个 agent 会当垃圾删掉

**不该留的也要判断**：一次性 bootstrap 脚本、`git show` 出来的旧版快照（git 里本来就有）
—— 重复即负担，别为了「保留」而囤积。

⚠️ 两盘不共享，跨盘一律 `scp -O` + 核 hash。见 [[cluster-two-disks-not-shared]]、
[[two-disk-rule-applies-to-main-too]]；venv/conda 被重置的具体表现见 [[memoryllm-venv-python-broken]]。
