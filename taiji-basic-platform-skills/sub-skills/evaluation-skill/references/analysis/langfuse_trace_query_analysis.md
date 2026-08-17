# 按 task_id 查询 Langfuse Trace 数据

<!-- ROUTE_KEYWORDS
langfuse, trace, 追踪, 调用链, agent轨迹, 拉取trace, 查trace, 查看trace, 下载trace, 下载trace数据,
get_taiji_eval_task_detail, langfuse_env, enable_traj
-->

---

## 适用场景

用户提供一个评估任务的 `task_id`，希望**下载**该任务对应的 Langfuse trace 数据（调用链/agent 轨迹）到本地文件（JSON），而非在对话里展示原始内容。

> ⛔ 本流程不需要查阅 `evaluation_api.md` 确认 `get_taiji_eval_task_detail` 的参数/签名——下方Step 1
> 已给出完整调用示例，直接按此调用即可，不要为了"验证工具是否存在"而额外去搜索其他文档。

## 背景要点

- **字段名均为 snake_case**：`get_taiji_eval_task_detail` 响应经后端 `OpenApiResponseUtil.toSnake()`
  深度转换（含全大写缩写词，如 `LANGFUSE_PUBLIC_KEY`→`langfuse_public_key`）。⛔ 不要按驼峰/全大写
  格式（`langfuseEnv`）取值，下方 Step 1 示例已是转换后的真实字段名。
- **无需查询 project_id**：Langfuse 鉴权是"一组 public/secret key唯一绑定一个 project"，
  trace 查询接口不接受也不需要 project_id。`langfuse_project_id` 只用于拼
  Web UI 链接，跟拉取 trace 数据无关。
- **直连 API，不走 npx**：下载脚本 `scripts/download_langfuse_traces.py` 直接用 Python
  `requests` 调 Langfuse Public API（`Authorization: Basic base64(public_key:secret_key)`）。

## 完整流程（2 步）

### Step 1：查任务详情，取Langfuse key三元组 + create_time

```
call get_taiji_eval_task_detail {task_id: <用户提供>}
→ 返回 data.extra_info（字段已是 snake_case）：
  {
    "traj_project_name": "hy3.0测试",
    "enable_traj": true,
    "langfuse_env": {
      "langfuse_base_url": "<langfuse-host>",
      "langfuse_public_key": "pk-lf-...",
      "langfuse_secret_key": "sk-lf-...",
      "enable_langfuse": "TRUE"
    }
  }
→ 同时取 data.create_time（如 "2026-07-14 22:57:10"），用作下载脚本的
  --from-timestamp 时间下界（见Step 2）
```

**前置校验（不得跳过）**：若 `extra_info` 为空、或 `enable_traj` 不为 `true`、或 `langfuse_env`
缺失 → 告知用户"该任务未开启 Langfuse 追踪，无法查询 trace"，流程终止，不继续 Step 2。

### Step 2：调用下载脚本，把 trace 数据落盘为本地文件

> 🎯 使用配套脚本 `scripts/download_langfuse_traces.py` 完成下载，**不得**把命令输出/trace
> 内容直接粘贴回显给用户，也不要手写裸 shell 循环或裸 `requests.get` 代替它——该脚本内置并发
> 控制、断点续传、失败重试，手写版本没有这些保障，大批量场景下既慢又可能产出损坏数据。
> 依赖 `requests` 库（未安装先 `pip install requests`）。

```bash
python3 scripts/download_langfuse_traces.py \
  --task-id <task_id> \
  --output-dir ./langfuse_traces/<task_id> \
  --public-key "<langfuse_env.langfuse_public_key>" \
  --secret-key "<langfuse_env.langfuse_secret_key>" \
  --host "<langfuse_env.langfuse_base_url>" \
  --fs-access <direct 或 download-only，必填，见下表> \
  --from-timestamp "<create_time 原样传入，如 2026-07-14 22:57:10>"
```

**关键参数**：

| 参数 | 必填 | 说明 |
|---|---|---|
| `--fs-access` | ✅ | **Agent必须先自我判断**：能直接访问用户文件系统（本地桌面场景）→ `direct`；只能靠生成文件给用户点击下载（网页端场景，如 knot）→ `download-only`。拿不准按 `download-only` 处理（更保守）。这是 Agent 对自身产品环境的认知，脚本不做任何运行时探测。 |
| `--from-timestamp` | 建议传| 直接传Step 1 的 `create_time` **原始裸字符串**，不要手动拼 `T`/`Z`。脚本按 GMT+8 自动换算成 UTC 再发给 Langfuse；用于收窄服务端全表扫描范围，明显提速。语义是"下界"，早于真实产生时间不会漏数据。 |
| `--to-timestamp` | 一般不传 | 不传则脚本在启动时自动冻结当前 UTC 时刻，全程分页复用同一值，避免仍在运行中的任务导致分页漂移。同样接受裸字符串，规则与 `--from-timestamp` 一致。 |
| `--archive-format` | 可选 | 不传则由 `--fs-access` 推导：`direct`→`none`（不打包）、`download-only`→`zip`（打包）。可显式传 `zip`/`jsonl`/`none` 覆盖，但 `download-only` 下传`none` 会被脚本拒绝（报错退出码 2），防止用户要点 N 次下载按钮。 |

## 关键行为

- **并发**（两个独立池，均固定上限，不无限爬升）：列表分页固定 `--list-concurrency`（默认 6）；
  trace 详情初始/上限并发 `--initial-concurrency`/`--max-concurrency`（默认 16），遇限流/高失败率
  自动降并发（下限 `--min-concurrency`，默认 1）
- **超大数据量封顶**（`--max-pages` 默认 300 页 × 每页固定 50 条 = 15000 条上限，每页条数
  为脚本内部固定值，不可配置）：达到上限仅打印警告、不阻断，按已拉到的部分继续；需要完整
  数据可分段 `--from-timestamp` 多次运行或加大 `--max-pages`。默认不需要主动向用户解释这个上限。
- **断点续传**：每条 trace 落盘为 `<output_dir>/<trace_id>.json`，`manifest.json` 记录状态
  （`ok`/`fail`/`pending`）。中断或失败后**原样重跑同一条命令**即可，自动跳过已成功的，只重试
  `pending`/`fail` 部分。
- **打包交付**：全部下载完成后（含"断点续传发现已全部完成"），脚本按最终生效的
  `--archive-format` 决定是否把散文件合并成一个文件（`zip`=打包成单文件，解压后仍是逐条 JSON；
  `jsonl`=合并成一个文件每行一条，免解压；`none`=保留散文件不打包）。**脚本会打印一行
  `[archive] 下载路径：<绝对路径>`**（打包时指向打包文件，不打包时指向 `output_dir`）。
- 数据量较大时（如 2000+ 条）告知用户是异步/耗时过程，建议放到后台执行完成后再汇报。

**退出码**：0=全部成功；1=仍有失败条目（终端已含失败 `trace_id` 列表，详情见 `manifest.json`）；
2=参数校验失败（缺 `--fs-access` 或 `download-only`+`none` 非法组合，需修正命令重跑，非下载失败）。

### 交付要求（强制）

下载完成后的交付动作取决于 Agent 自身运行环境（即 Step 2 `--fs-access` 的判断结果）：

| 环境判断 | `--fs-access` 取值 | 交付时**必须**包含的内容 | 禁止行为 |
|---|---|---|---|
| **direct**（可访问用户文件系统，如本地桌面客户端） | `direct` | ① 下载条数 N；② 脚本打印的 `下载路径：` 绝对路径原样带出；③ 明确告知用户该路径即为落盘地址，用户可直接在文件系统打开。 | 严禁只回复"下载完成"而不带绝对路径 |
| **download-only**（网页端场景，如 knot，只能靠生成文件链接给用户点击下载） | `download-only` | ① 下载条数 N；② **必须生成 zip 文件的下载链接**并在回复中展示给用户点击；③ 即使数据量不大也必须打包成 zip 后给出链接，不得只告知"已下载完成"或只给散文件目录路径。 | 严禁只回复"已下载完成"而不附 zip 下载链接；严禁让用户自行去文件系统找散文件 |

> 🎯 判断规则同 `--fs-access`：能直接访问用户文件系统 → `direct`；只能靠生成文件给用户点击下载 → `download-only`；拿不准按 `download-only` 处理。
> ⛔ **网页端场景下，即使最终产物只有少量文件，也必须打包成 zip 并在回复里给出可点击的下载链接**，不允许只回复"下载完成"四个字。

## ⚠️ 已知风险（暂不处理，当前为求快版本）

- `get_taiji_eval_task_detail` 现已把 `extra_info` 整块透出，其中 `langfuse_secret_key` 是
  **明文项目级共享密钥**（非任务级隔离凭据），任何能调用该工具的调用方都能拿到。
- 若后续需要收紧：① 后端对该字段脱敏/单独鉴权；② 改为后端代理端点直接返回 trace 数据，密钥
  不出后端。当前暂不实施，仅记录风险。

## ⛔ 严禁

- 严禁在 `enable_traj` 非 `true` 时继续执行 Step 2
- 严禁手动给 `--from-timestamp`/`--to-timestamp` 拼接 `T`/`Z` 或做时区换算——直接传原始裸
  字符串，手动拼 `Z` 会把 GMT+8 误当UTC，导致时间偏早 8 小时
- 严禁把 `langfuse_secret_key` 原样展示给用户在聊天记录里（可用于命令执行，但不主动回显）
- 严禁对下载结果做摘要/截断/格式化后回显——产物是落盘文件，不是聊天内容
- 严禁跳过 Step 1 或去其他文档确认 `get_taiji_eval_task_detail` 的参数
- 严禁省略必填参数 `--fs-access`，也严禁不加判断地固定传某个值——必须先判断 Agent 自身当前
  所处环境再传参
- 严禁下载完成后只回复"下载完成"而不交付产物路径：
  - `direct` 场景必须带出脚本打印的 `下载路径：` 绝对路径
  - `download-only`（网页端）场景必须生成 zip 下载链接并展示给用户，严禁只回复"下载完成"
