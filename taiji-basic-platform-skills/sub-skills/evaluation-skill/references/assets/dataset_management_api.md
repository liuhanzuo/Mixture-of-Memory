## upload_taiji_eval_dataset_file

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `file_name` | string | ✅ | 文件名，例如 `my_dataset.jsonl` |
| `file_content_base64` | string | ✅ | 文件内容的 base64 编码字符串 |

> ⚠️ **Agent 操作前置**：Agent 需先将文件内容读取并进行 base64 编码，再传入 `file_content_base64`。
> ```python
> import base64
> with open("/path/to/file.jsonl", "rb") as f:
>     content_b64 = base64.b64encode(f.read()).decode("utf-8")
> ```

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `file_id` | number | 文件记录 ID |
| `file_path` | string | 文件在 Ceph 上的完整路径（即后续 `dataset_version_ceph_path` 所需的值） |
| `file_name` | string | 原始文件名 |
| `creator` | string | 上传用户 |
| `create_time` | string | 上传时间，格式 `yyyy-MM-dd HH:mm:ss` |

#### 示例

```bash
# 先 base64 编码文件，再作为 JSON 参数传入（不要用 --stdin 管道）
python3 scripts/connect_mcp.py call upload_taiji_eval_dataset_file "{\"file_name\":\"my_eval_data.jsonl\",\"file_content_base64\":\"$(python3 -c 'import base64,sys;print(base64.b64encode(open(sys.argv[1],\"rb\").read()).decode())' my_eval_data.jsonl)\"}"
```

> 🚫 禁止使用 `--stdin` 或管道传入参数。connect_mcp.py 只接受 JSON 字符串参数。

返回示例：
```json
{
  "file_id": 123,
  "file_path": "/apdcephfs/share_xxx/eval_upload/my_eval_data.jsonl",
  "file_name": "my_eval_data.jsonl",
  "creator": "zhangsan",
  "create_time": "2026-05-18 20:00:00"
}
```

---

## create_taiji_eval_dataset

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `dataset_name` | string | ✅ | 数据集名称（全局唯一） |
| `dataset_version_name` | string | ✅ | 数据集版本名称（如 `v1.0`） |
| `dataset_version_ceph_path` | string | ✅ | 文件的 Ceph 路径，即 `upload_taiji_eval_dataset_file` 返回的 `filePath` |
| `dataset_version_desc` | string | ❌ | 版本描述，可选 |
| `admin` | string | ❌ | 数据集管理员（默认为上传用户） |
| `dataset_tags` | string | ❌ | 数据集标签，多个用逗号分隔 |
| `blackbox` | boolean | ❌ | 是否黑盒数据集，默认 `false` |
| `upload_type` | string | ❌ | 上传类型标识，可选 |
| `visibility` | string | ❌ | 可见范围，可选值：`CURRENT_WORKSPACE`（默认，仅当前空间）、`ALL_PLATFORM`（全平台）、`SPECIFIED_WORKSPACES`（指定空间） |
| `visible_ws_ids` | string | ❌ | 当 `visibility=SPECIFIED_WORKSPACES` 时，填写可见空间 ID 列表（逗号分隔）；全平台时填 `"all"` |

> ⚠️ **必填字段**：`dataset_name`、`dataset_version_name`、`dataset_version_ceph_path` 三项缺一不可，否则接口会返回 400 校验失败。

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `dataset_id` | number | 数据集 ID |
| `dataset_name` | string | 数据集名称 |
| `dataset_desc` | string | 数据集描述 |
| `dataset_version_id` | number | 数据集版本 ID |
| `dataset_version_name` | string | 数据集版本名称 |
| `dataset_version_ceph_path` | string | 数据集文件 Ceph 路径（回显） |
| `upload_type` | string | 上传类型标识（回显） |

#### 示例

```bash
python3 scripts/connect_mcp.py call create_taiji_eval_dataset '{
  "dataset_name": "my-eval-dataset-2026",
  "dataset_version_name": "v1.0",
  "dataset_version_ceph_path": "/apdcephfs/share_xxx/eval_upload/my_eval_data.jsonl",
  "dataset_version_desc": "首版测试数据集",
  "visibility": "CURRENT_WORKSPACE"
}'
```

返回示例：
```json
{
  "dataset_id": 456,
  "dataset_name": "my-eval-dataset-2026",
  "dataset_desc": null,
  "dataset_version_id": 789,
  "dataset_version_name": "v1.0",
  "dataset_version_ceph_path": "/apdcephfs/share_xxx/eval_upload/my_eval_data.jsonl",
  "upload_type": null
}
```

---

### Dataset 完整两步上传流程

```
用户提供 JSONL 文件，或要求 Agent “创建一个 .jsonl 文件并上传”
       ↓
① Agent 将文件写入本地磁盘（如 /tmp/xxx.jsonl），并确保至少一行合法 JSONL
       ↓
② 调用 upload_taiji_eval_dataset_file
   → 拿到 file_path（Ceph 路径）
       ↓
③ 询问用户：数据集名称 / 版本名 / 描述（如未提供则追问）
       ↓
④ 调用 dataset/create
   → dataset_version_ceph_path = 上一步的 filePath
   → 返回 datasetId + datasetVersionId
       ↓
✅ 告知用户数据集创建成功，id 和 dataset_version_id 可用于创建评测任务
```

> ⚠️ **强制链路**：只要是“上传/创建 JSONL 评测数据集”，必须先 `upload_taiji_eval_dataset_file` 再 `create_taiji_eval_dataset`。严禁跳过上传、凭空填写 `dataset_version_ceph_path` 后直接创建。若当前对话中没有刚刚由 `upload_taiji_eval_dataset_file` 返回的 `file_path`，禁止调用 `create_taiji_eval_dataset`。
>
> 💡 **browsecomp/格式咨询**：用户问“数据格式改成 browsecomp 后怎么评估/怎么正常评估”时，先调用 `list_taiji_eval_datasets`（带 `wsid`，必要时 `keyword="browsecomp"` 或用户给的名称）查看当前空间已有数据集/版本，再基于真实返回给出后续创建评测集、集合或评测任务建议；不要只给泛化流程。

---

### 数据集（Dataset）管理——查询/修改/删除

## list_taiji_eval_datasets

分页查询数据集列表，支持关键词搜索。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_datasets '{"keyword": "评测", "page_index": 1, "page_size": 20}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `keyword` | string | ❌ | 按数据集**名称**模糊匹配；传入纯数字时按 ID 精确匹配 |
| `tags` | string | ❌ | 按标签过滤 |
| `my_dataset` | boolean | ❌ | `true` = 只返回当前用户创建的数据集；用户说"我的/我创建的"时传 `true`（默认 false） |
| `page_index` | number | ❌ | 页码（1-based，默认 1） |
| `page_size` | number | ❌ | 每页数量（默认 10） |
| `order_by` | string | ❌ | 排序字段（默认 `id` 降序） |

---

## get_taiji_eval_dataset_detail

获取数据集详情，含所有版本信息。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_dataset_detail '{"dataset_id": 456}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_id` | number | ✅ | 数据集 ID |

---

## update_taiji_eval_dataset

修改数据集名称、描述或可见范围。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call update_taiji_eval_dataset '{"dataset_id": 456, "dataset_name": "新名称", "dataset_desc": "新描述"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_id` | number | ✅ | 数据集 ID |
| `dataset_name` | string | ❌ | 新名称 |
| `dataset_desc` | string | ❌ | 新描述 |
| `dataset_tags` | string | ❌ | 数据集标签（可选） |
| `admin` | string | ❌ | 管理员（可选） |
| `visibility` | string | ❌ | 可见范围：`CURRENT_WORKSPACE` / `ALL_PLATFORM` / `SPECIFIED_WORKSPACES` |
| `visible_ws_ids` | string | ❌ | 可见空间 ID 列表（`visibility=SPECIFIED_WORKSPACES` 时必填，多个用逗号分隔） |

---

## delete_taiji_eval_dataset

删除数据集及所有版本。🔴 **不可逆操作**，执行前必须二次确认。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call delete_taiji_eval_dataset '{"dataset_id": 456}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_id` | number | ✅ | 数据集 ID |

---

## list_taiji_eval_dataset_versions

查询数据集的所有版本。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_dataset_versions '{"dataset_id": 456}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_id` | number | ✅ | 数据集 ID |
| `keyword` | string | ❌ | 关键词搜索 |
| `page_index` | number | ❌ | 页码（默认 1） |
| `page_size` | number | ❌ | 每页数量（默认 10） |
| `order_by` | string | ❌ | 排序字段（默认 `id` 降序） |

---

## update_taiji_eval_dataset_version

修改数据集版本名称或描述。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call update_taiji_eval_dataset_version '{"dataset_version_id": 789, "version_name": "v2.0", "version_desc": "新版描述"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_version_id` | number | ✅ | 版本 ID |
| `version_name` | string | ❌ | 新版本名 |
| `version_desc` | string | ❌ | 新描述 |
| `admin` | string | ❌ | 管理员（可选） |
| `blackbox` | boolean | ❌ | 是否黑盒数据集（可选） |

---

## delete_taiji_eval_dataset_version

删除数据集的某个版本。🔴 **不可逆操作**，执行前必须二次确认。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call delete_taiji_eval_dataset_version '{"dataset_version_id": 789}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_version_id` | number | ✅ | 版本 ID |

---

## clone_taiji_eval_dataset_version

复制数据集版本。复制后新版本名称为原名称加 `_copy` 后缀，状态为 `COPYING`（后台异步复制底层数据，完成后自动变为 `RELEASED`）。注意：源版本必须为 `RELEASED` 状态。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call clone_taiji_eval_dataset_version '{"dataset_version_id": 789}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_version_id` | number | ✅ | 要复制的源数据集版本 ID |

**返回字段**：`id`（新版本 ID）、`version_name`（加 `_copy` 后缀）、`dataset_id`、`data_cnt`、`status`（`COPYING`）、`creator`、`create_time` 等。

---

## download_taiji_eval_dataset_version_file

下载数据集版本文件。返回内容为 JSON（`file_content_base64`），不是二进制流，Agent 需自行 base64 解码后写入本地文件。仅支持中小文件。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call download_taiji_eval_dataset_version_file '{"dataset_version_id": 789}'
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_version_id` | number | ✅ | 数据集版本 ID。用户通常只知道版本名（如 `lamictest`），看不到 id，需先用 `list_taiji_eval_dataset_versions` 按 `version_name` 精确匹配获取 |

**返回字段**：`dataset_version_id`、`file_name`、`file_size`（字节数）、`file_content_base64`（文件内容 base64 编码）。

> 🛑 **错误与权限处理规则（最高优先级）**：下载失败（权限不足、HTTP 400/500、文件不存在等）时，**立即停止**，把错误原文告知用户，**不得**：
> - 重复调用本工具重试
> - 换工具名/参数反复尝试
> - 调用 storage 域工具（`list_storage_dir`、`get_storage_file_content` 等）绕过
> - 写脚本或搜索其他替代方案
>
> 正确做法："下载失败：您不是该版本的负责人，无权下载文件内容。请联系版本创建者或管理员。"
>
> Python 解码示例：
> ```python
> import base64
> content = base64.b64decode(result["file_content_base64"])
> with open("/tmp/xxx.jsonl", "wb") as f:
>     f.write(content)
> ```
>
> 📥 **下载内容一次性读取**：base64 解码后的内容直接打印或写入文件后即处理完毕，**不要再次调用本工具重新下载**。如需处理内容，使用已解码的本地文件。正确流程：`download（1次）→ Python 解码 → 写入文件 → 处理文件`。

---

## create_taiji_eval_dataset_version

在**已有数据集**下新建版本。⚠️ 区别于 `create_taiji_eval_dataset`（那个是建全新数据集），本接口不会创建新的 dataset，只在指定 `dataset_id` 下新增一个版本。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call create_taiji_eval_dataset_version '{
  "dataset_id": 1984,
  "version_name": "lamictest_modified",
  "upload_type": "local",
  "ceph_path": "/apdcephfs/share_xxx/eval_upload/training_data.jsonl",
  "version_desc": "基于 lamictest 修改"
}'
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `dataset_id` | number | ✅ | 数据集 ID，决定在哪个数据集下新建版本；用户一开始就应明确指定 |
| `version_name` | string | ✅ | 新版本名称，需在该数据集下唯一。**调用前建议先用 `list_taiji_eval_dataset_versions` 核对是否重名**，重复会创建失败 |
| `upload_type` | string | ✅ | 枚举 `local` / `ceph`。⛔ **不可为空**，传空值后端会报错 |
| `ceph_path` | string | ✅ | `upload_type=local` 时填 `upload_taiji_eval_dataset_file` 返回的 `file_path`；`upload_type=ceph` 时填用户提供的 Ceph 路径 |
| `version_desc` | string | ❌ | 版本描述，默认空字符串 |

> 🔒 **系统自动填充，不向用户询问**：`blackbox`（固定 `false`）、`copy`（固定 `false`，仅做权限校验不复制数据）、`admin`（固定为当前调用用户）。这三个字段不接受也不需要用户传入。

**返回字段**：新建的 `MultimodalEvaluationDatasetVersion` 实体（snake_case），含 `id`（新版本 ID）、`dataset_id`、`version_name`、`ceph_path`、`upload_type`、`status`（初始为 `PENDING` 或 `IO_PENDING`，异步解析中）等。

---

### 数据集版本导出-修改-重传工作流

> 场景：用户想"导出一个已有数据集版本，让 Agent 按要求修改后，作为**新版本**重新上传"，不覆盖原版本。
> 用户通常**只知道版本名（如 `lamictest`），不知道也看不到 `dataset_version_id`**，Agent 必须自己反查。

```
① 收集必填信息
   向用户确认：dataset_id（在哪个数据集下操作）
             source_version_name（要改的现有版本名，如 "lamictest"）
             version_name（重传后的新版本名，如 "lamictest_modified"）
             upload_type（local / ceph；ceph 时还需 ceph_path）
        ↓
② 反查源版本 id（用户看不到 id，只能给名字）
   调用 list_taiji_eval_dataset_versions({"dataset_id": ...})
   在返回列表中按 version_name 精确匹配 source_version_name
   → 匹配到 → 取其 dataset_version_id
   → 未匹配到 → 报错「数据集 {dataset_id} 下找不到名为 {source_version_name} 的版本」，终止流程
   同时检查列表中是否已存在与 version_name（新名）相同的条目
   → 存在 → 提前提醒用户「新版本名已存在，创建会失败」，请其更换 version_name
        ↓
③ 下载源文件
   调用 download_taiji_eval_dataset_version_file({"dataset_version_id": 上一步取到的 id})
   → base64 解码，写入本地临时文件（如 /tmp/taiji_dataset_ops/{dataset_version_id}/{file_name}）
   → 若报错（无权限/非负责人）：告知用户并终止
        ↓
④ 按用户要求修改文件
   读取 JSONL → 按需变换（改字段、过滤行、批量替换等）→ 逐行 json.loads 校验仍为合法 JSONL → 写回
        ↓
⑤ 重传为新版本
   分支 A（upload_type=local）：
     1. 读取修改后文件，base64 编码
     2. 调用 upload_taiji_eval_dataset_file({"file_name": ..., "file_content_base64": ...}) → 取 file_path
     3. 调用 create_taiji_eval_dataset_version({
          "dataset_id": ..., "version_name": ..., "upload_type": "local",
          "ceph_path": <上一步 file_path>, "version_desc": ...
        })
   分支 B（upload_type=ceph）：
     1. 直接调用 create_taiji_eval_dataset_version({
          "dataset_id": ..., "version_name": ..., "upload_type": "ceph",
          "ceph_path": <用户提供的路径>, "version_desc": ...
        })
   → 若返回 version_name 重复错误：提醒用户更换新版本名，回到步骤①重新收集
        ↓
⑥ 告知用户新版本已创建成功
   输出新版本 id / version_name / status，并说明原版本未被覆盖
```

> ⚠️ **强制约束**：
> - `source_version_name` 与 `version_name` 不能相同（同一 dataset_id 下版本名唯一）。
> - `upload_type` 必须显式为 `local` 或 `ceph`，不可留空。
> - 仅支持中小文件；大文件（GB 级以上）暂不支持此工作流。
> - `blackbox`/`copy`/`admin` 由 Agent 调用 `create_taiji_eval_dataset_version` 时无需传入，系统自动处理。

---

