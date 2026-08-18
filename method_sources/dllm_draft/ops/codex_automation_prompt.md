# Scaffold-Coder 30 分钟自动推进

这是定时触发的同一 Codex thread heartbeat。请直接执行，不要只汇报计划：

1. 只巡检 `.104`：`28.83.24.104:36000`，远端项目根目录固定为
   `/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104`。
   密码文件固定为
   `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_h20_24104.txt`，
   必须用 `sshpass -f`，不得输出密码。
2. 不要运行当前 `scaffold-server-heartbeat` skill 的默认 inspect 脚本，
   因为它仍绑定旧服务器。通过 SSH 直接检查 `.104` 的
   `ops/state/active_run.json`、`ops/queue.tsv`、`ops/history.tsv`、
   `ops/logs/`、GPU、checkpoint 和 success artifact。
3. 严禁终止未知、外部或未注册进程；只能处理
   `ops/state/active_run.json` 中登记的进程组。
4. 若注册任务健康，记录实际进度、loss、吞吐、显存和 ETA，不干预。
5. 若任务完成，验证 artifact，并推进
   `ELASTIC_SCAFFOLD_EXPERIMENT_TODO.md` 中第一个可执行实验。
6. 若任务失败，读取完整注册日志，做最小修复，运行测试，Git commit，
   同步 `.104`，然后以新的可审计 run ID 重跑。
7. 优先推进语义保真训练主线：
   `SEMTRAIN-001 → SEMTRAIN-002 → SEMTRAIN-003`，但必须遵守文档中的
   门槛和止损条件。
8. 所有实验使用注册 queue 和 heartbeat；保存 config、command、
   checkpoint、metrics、NFE、cumulative tokens、wall-clock、failure
   reason 和 Git commit。
9. 若 GPU 被未知任务占用，只读巡检并做本地 CPU 工作，不得杀进程。
10. 每次自动推进结束，在 `ops/logs/codex_automation.log` 对应 run 日志和
   Git 中留下可审计记录。不要等待用户确认，除非涉及未授权资源或
   不可逆外部操作。
