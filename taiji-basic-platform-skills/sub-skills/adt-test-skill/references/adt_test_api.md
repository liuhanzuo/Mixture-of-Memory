## create_adt_test_task

创建一个 ADT 模型基础测试任务，向 ADT 平台发起对指定模型服务组的四项基础能力检测。

> ⚠️ **无需 api_token 参数**：ADT 接口使用内置固定认证，发起人 RTX 从当前用户上下文（`X-Auth-Username`）自动获取。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call create_adt_test_task '{"model": "hunyuan-standard", "concurrency": 5}'
```

**参数：**

| 参数 | 类型 | 必填 | 默认 | 描述 |
|------|------|:---:|------|------|
| `model` | string | ✅ | — | 模型服务组名称（**是已部署的服务组名，不是模型名/ID**），例如 `"hunyuan-standard"` |
| `concurrency` | int | ❌ | 5 | 测试支持的并发数（≥1） |
| `model_hyper_params` | object | ❌ | `{}` | 请求超参 JSON 对象，可含 `temperature`/`top_p`/`top_k`/`repetition_penalty`/`max_tokens`/`seed`/`openai_infer`(bool)/`chat_template_kwargs`(含 `reasoning_effort`/`interleaved_thinking` 等) |

**返回（`data`）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 新建任务 ID |
| `model` | string | 模型服务组名称 |
| `concurrency` | int | 并发数 |
| `staff_name` | string | 发起人 RTX |

示例：
```json
{ "code": 0, "message": "success",
  "data": { "task_id": "123456", "model": "hunyuan-standard", "concurrency": 5, "staff_name": "zhangsan" } }
```

---

## get_adt_test_task_detail

根据 `task_id` 查询 ADT 测试任务的当前状态与结果。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_adt_test_task_detail '{"task_id": "123456"}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|:---:|------|
| `task_id` | string | ✅ | ADT 测试任务 ID（由 `create_adt_test_task` 返回） |

**返回（`data`）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |
| `status` | int/null | 状态码（见 SKILL.md §4.1 状态码字典） |
| `status_text` | string/null | 状态中文释义 |
| `is_terminal` | bool/null | 是否已进入终态（3/4/5/7/8/10/11） |
| `result` | object | 各测试项结果，含 `name_cn`（中文名）/`passed`/`failed`/`total`/`report_url` |
| `detail` | object | ADT 后端完整原始返回 |

示例：
```json
{ "code": 0, "message": "success",
  "data": {
    "task_id": "123456", "status": 3, "status_text": "执行成功", "is_terminal": true,
    "result": { "HTEXTGARBLE": { "name_cn": "乱码检测", "passed": 100, "failed": 0, "total": 100 } },
    "detail": {}
  } }
```

> `task_id` 不存在时：`{ "code": 40401, "message": "资源不存在：...", "data": null }`（HTTP 404）。

**展示建议**：回显 `status_text` + `is_terminal`；已终态时列出四项检测的 `通过 passed/total, 失败 failed`；`report_url` 若有则原样给出（严禁自造）。

---

## stop_adt_test_task

根据 `task_id` 向 ADT 平台发送停止指令，中止正在执行的测试任务。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call stop_adt_test_task '{"task_id": "123456"}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|:---:|------|
| `task_id` | string | ✅ | 要停止的 ADT 测试任务 ID |

**返回（`data`）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |
| `stopped` | bool | 是否已成功发送停止指令 |
| `detail` | object | ADT 后端返回 |

> ⚠️ **后端不做 owner 校验**：传错 `task_id` 会真把别人的任务停掉。停止前必须确认 `task_id` 归属，操作他人任务需二次确认。

---

## run_adt_test_pipeline

工程链路评估（**第 1 步**）：创建一个 ADT 模型基础测试任务，并由服务端后台**每 5 分钟自动轮询**任务状态，进入终态或超时后通过**企微（TOF4）通知**发起人。

> ⚠️ 本工具属**写操作**，会创建 ADT 任务并启动最长约 6 小时的后台轮询协程；调用前建议向用户复述 `model` 参数与轮询/通知行为。
>
> ℹ️ 本工具**后端只做 ADT 测试 + 轮询 + 通知，不复制评测任务**。完整的「工程链路评估」还包含**第 2 步复制基准评测任务**，由本 skill 在 Agent 层组合调用 `evaluation-skill` 的 `clone_taiji_eval_task` 完成（见 SKILL.md §3/§4 编排契约）。纯评测任务操作（与工程链路评估无关）请走 `evaluation-skill`。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call run_adt_test_pipeline '{"model": "hunyuan-standard", "concurrency": 5}'
```

**参数：**（同 `create_adt_test_task`）

| 参数 | 类型 | 必填 | 默认 | 描述 |
|------|------|:---:|------|------|
| `model` | string | ✅ | — | 模型服务组名称 |
| `concurrency` | int | ❌ | 5 | 并发数 |
| `model_hyper_params` | object | ❌ | `{}` | 请求超参（同上） |

**返回（`data`）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 新建任务 ID |
| `model` | string | 模型服务组名称 |
| `concurrency` | int | 并发数 |
| `staff_name` | string | 发起人 RTX |
| `poll_started` | bool | 后台轮询是否已启动 |
| `poll_interval_seconds` | int | 轮询间隔（秒，默认 300） |

示例：
```json
{ "code": 0, "message": "success",
  "data": { "task_id": "123456", "model": "hunyuan-standard", "concurrency": 5,
            "staff_name": "zhangsan", "poll_started": true, "poll_interval_seconds": 300 } }
```

**后续查询**：任务完成会企微通知；用户也可随时用 `get_adt_test_task_detail` 手动查进度。
