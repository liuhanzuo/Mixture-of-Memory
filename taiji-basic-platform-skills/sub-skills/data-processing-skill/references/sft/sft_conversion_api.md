## list_hunyuan_data_sft_tokenizers

**功能**：获取当前系统支持的所有分词器（tokenizer）类型列表。**建议在创建转 bin 任务前先调用此工具**，确认可用的分词器。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|

> 本工具无业务参数（认证 Token 由 Header 自动附加，无需传参）。调用时传 `{}`，**不要传 `wsid`**。

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_hunyuan_data_sft_tokenizers '{}'
```

### 返回字段说明

返回 Markdown 格式的分词器列表，包含：

| 字段 | 说明 |
|------|------|
| 分词器名称 | 如 `HY3.0_SFT_Tokenizer`，用于创建转 bin 任务时的 `tokenizer` 参数 |

### 返回示例

```
# 可用分词器列表

共 1 个分词器：

1. `HY3.0_SFT_Tokenizer`
```

### 单工具 SOP

- `tokenizer` 缺失时，**先调用本工具拿到全列表**，然后把列表里的所有分词器名称展示给用户挑选；不得直接让用户"自己输入一个 tokenizer 名"，**也不得把文档示例里的 tokenizer 名（如 `HY3.0_SFT_Tokenizer`）当默认值直接用**——示例值仅为演示，一切以本工具返回的实际列表为准。
- 用户已显式给出 tokenizer 时直接使用，**不要**再调用本工具。

---

## create_hunyuan_data_sft_conversion

**功能**：创建一个 SFT 数据转 bin 任务并自动启动。将后训练的文本数据通过指定的分词器和序列长度进行二进制编码，生成可直接用于模型训练的 `.bin` 文件。

> ⚠️ **单步 vs 端到端 pipeline 的判别（最高优先级）**：**两端都已在南京** → 直接调用本工具（单步）；**任一端不在南京** / 用户提了"自动中转 / pipeline / 流水线 / 跨地域" → 走 `sft_bin_auto_transfer_flow.md` 的跨地域 Pipeline。
> ⚠️ **南京地域强校验是单步转 bin 的硬约束**：调用本工具**之前**必须按下方「路径南京地域强校验」做地域识别，三分支处理（南京 ✅ → 直接执行；非南京 ❌ → **阻断并抛出错误 + 结束**；无法判断 ⚠️ → 警告后继续）。

**必须确认的信息（最高优先级）：**
1. **seq_len（序列长度）**：必填。用户未提供时**必须先向用户列出可选项 `4096 / 8192 / 32768`** 让用户选，不得直接默默套默认；用户明确说"用默认"才回落到 `4096`。
2. **tokenizer（分词器）**：必填。用户未提供时**必须先调用 `list_hunyuan_data_sft_tokenizers` 获取可选列表，再把全部可选项列给用户挑选**；用户明确说"用默认"才回落到列表第一个。**严禁**让用户自己输入 tokenizer 名。
3. **input_path（输入路径）**：必填，Ceph 路径，系统会校验存在性和读权限。⚠️ **`input_path` 必须是单个数据文件，不能是目录**（例如 `/apdcephfs_jn2/share/leslizhang/sft/example.jsonl`）。
   - **必须以 `.json` / `.jsonl` / `.txt` / `.parquet` 等文件扩展名结尾（指向单个文件）**；若用户给的是以 `/` 结尾的目录（或自述是目录），Agent 必须先向用户索取该目录下的**具体文件路径**，未拿到文件路径前不进入后续的南京地域强校验和任务创建。
4. **storage_path（输出路径）**：⚠️ **硬必填，无默认值，Agent 严禁自行构造或推测**。必须由**用户显式提供**；用户未提供时必须直接向用户索取，不得根据 `input_path` 拼接、不得用 `<input_path>_bin/` 之类的猜测路径、不得复用任何历史任务的输出路径。拿到用户提供的路径后，路径必须是 Ceph 路径、空目录且有写权限。
   - **向用户索取 / 展示该字段含义时，必须使用简洁话术**：
     > `storage_path`：转 bin 后的产出路径（例如：`/apdcephfs_jn/share_302316223/xxxx/`）
   - 不要在向用户的描述里塞入"必须是您工作空间下的南京 ceph 目录"、"Agent 不会替您拼接路径"等冗余约束话术；地域校验由 Agent 内部按下方「路径南京地域强校验」静默执行。
5. **⚠️ 南京地域强校验（最高优先级，调用前置）**：当前转 bin 任务的计算资源**只在南京地域**，`input_path` 和 `storage_path` **都必须是南京地域的 Ceph 路径**。识别规则见下方「路径南京地域强校验」。
   - **非南京地域 ❌ → 必须阻断 + 抛错结束**：不能调用本工具；向用户明确返回"输入/输出路径不在计算资源所在地域（南京）"的错误，并列出 `input_path` / `storage_path` 以及其判定结果。**Agent 此处必须结束，不得主动提议走 pipeline 或其他回退路径。** 如果用户自行在后续对话中表达"改走 pipeline"或"我给南京路径"，再由用户显式触发下一轮流程。
   - **无法判断 ⚠️ → 继续执行前先警告**：必须输出一段提示"请确认输入或输出路径是在南京地域，否则可能出现转bin任务失败的风险"，再继续调用工具。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `seq_len` | int | ✅ 必填 | - | 序列长度，如 4096、8192、32768 |
| `tokenizer` | str | ✅ 必填 | - | 分词器类型，可通过 `list_hunyuan_data_sft_tokenizers` 获取可选值 |
| `input_path` | str | ✅ 必填 | - | 输入数据的 Ceph 路径。⚠️ **必须是单个数据文件，不能是目录**（例如 `/apdcephfs_jn2/share/leslizhang/sft/example.jsonl`）——若用户给的是目录，须先向用户索取具体文件路径再继续。系统会校验路径存在性和读权限。**必须为南京地域（路径前缀需匹配 `/apdcephfs_nj*/` 或 `/apdcephfs_jn*/`）** |
| `storage_path` | str | ✅ 必填（**无默认**） | - | 输出 bin 数据的 Ceph 路径（需为空目录且有写权限）。**硬必填、无默认值、Agent 严禁自行构造或推测**，必须由用户显式提供；用户未提供时直接向用户索取，不得用 `<input_path>_bin/` 之类拼接路径。**必须为南京地域（路径前缀需匹配 `/apdcephfs_nj*/` 或 `/apdcephfs_jn*/`）** |
| `name` | str | 选填（默认值） | `"create_by_bin_auto_transfer_ceph_skill"` | 任务名称。**不传时 MCP server 默认填充** `create_by_bin_auto_transfer_ceph_skill`，便于运维统一检索。Agent **无须**主动传该字段；仅在用户显式要求"换名字"时才传用户指定的值 |
| `owners` | list[str] | ❌ 可选 | `null` | 责任人 RTX 列表 |
| `wsid` | int | ❌ 可选 | `null` | 工作空间 ID。仅在未传 `modality` 时作空间默认模态兜底；**不能**再仅靠 wsid 判定多模态/语言任务路由 |
| `modality` | str | ⚠️ 强烈建议 | `null` | 数据模态：`TEXT` / `MULTIMODAL`。决定走 `postTrainTokenization` 还是 `multimodalDataPostTrainBin`。10103/12290 打通空间**必须向用户确认并显式传入**；未传且无可靠默认时按语言模型处理 |

> ⚠️ **注意**：`source` 参数由系统自动填充为 `TAIJI_MCP`，`start` 固定为 `true`（创建即启动），这两个参数不暴露给用户。

### 入参白名单（反幻觉）

本文文件的工具入参**已在各参数表中穷举完毕**，**严禁**向用户索取或自行附加任何"参数表中没有"的字段。最常见的幻觉错例：
- ❌ **`stage` / `thinkingType` / `thinking_type` 与本模块完全无关**：这两个字段属于 🗂️ **后训练数据链路管理**（`posttrain-data-skill`）的 `create_hunyuan_data_topic_dataset` 工具，挂在 **Topic Dataset** 层；**SFT 转 bin（单步 + 跨地域 Pipeline）的所有工具都不接收这两个参数**，**严禁**列入"参数核对清单"、**严禁**向用户索取。
- ❌ **`enable_inspection` / `version` / `key` / `dataset_id` / `topic_id` / `topic_data_id`** 等同样属于后训练数据链路管理 / 后训练数据质检模块，**与 SFT 转 bin 无关**，严禁混入参数核对清单。
- ✅ **单步本工具完整必填集**：`input_path`、`storage_path`、`seq_len`、`tokenizer` 四项；`name` 选填，不传时 MCP server 默认填充 `create_by_bin_auto_transfer_ceph_skill`。
- 跨地域 Pipeline 完整必填集见 `sft_bin_auto_transfer_flow.md`。

### 调用示例

**最简调用**（仅必填参数）：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_sft_conversion '{
  "seq_len": 8192,
  "tokenizer": "HY3.0_SFT_Tokenizer",
  "input_path": "/apdcephfs_jn2/share/leslizhang/sft/example.jsonl",
  "storage_path": "/apdcephfs_jn2/share/leslizhang/sft_bin/",
  "name": "create_by_bin_auto_transfer_ceph_skill"
}'
```

**完整调用**（含可选参数；ℹ️ `name` 选填，不传时 MCP server 默认填充 `create_by_bin_auto_transfer_ceph_skill`）：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_sft_conversion '{
  "seq_len": 8192,
  "tokenizer": "HY3.0_SFT_Tokenizer",
  "input_path": "/apdcephfs_jn2/share/leslizhang/sft/example.jsonl",
  "storage_path": "/apdcephfs_jn2/share/leslizhang/sft_bin/",
  "name": "create_by_bin_auto_transfer_ceph_skill",
  "owners": ["user1", "user2"],
  "wsid": 12290,
  "modality": "MULTIMODAL"
}'
```

### 返回字段说明

返回 Markdown 格式的任务详情，包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 任务 ID，用于后续查询和重试 |
| `name` | str | 任务名称 |
| `pipelineStatus` | str | Pipeline 整体状态（PENDING / RUNNING / SUCCEEDED / FAILED / STOPPED） |
| `taskStatus` | str | 子任务状态 |
| `tokenizer` | str | 使用的分词器 |
| `seqLen` | int | 序列长度 |
| `inputPath` | str | 输入数据路径 |
| `storagePath` | str | 输出 bin 数据路径。⚠️ **MCP server 已对外暴露"剥离 subPath 后"的版本**——后端原始返回是"用户原始路径 + subPath"拼接而成（例如 `/apdcephfs_nj2/.../skill/3/bin_data/HY3.0_SFT_Tokenizer_32768`），MCP server 在格式化"输出路径"前会主动剥离尾部的 `subPath` 后缀，使 Agent 看到的"输出路径"恰好等于用户原始的 `storage_path`。Agent 直接展示即可，**严禁**再拼回 `subPath` |
| `subPath` | str | 后端的内部存储约定（如 `bin_data/HY3.0_SFT_Tokenizer_8192`），仅作 debug 暴露。⚠️ **严禁**把它拼到"输出路径"后面再展示给用户——给用户的最终落地路径**始终是 MCP server 已剥离过的"输出路径"原值** |
| `ownerList` | list | 责任人列表 |
| `creator` | str | 创建人 |
| `createTime` | str | 创建时间 |

### 返回示例

```
# 转bin任务详情 (ID: 12345)

- **任务名称**: create_by_bin_auto_transfer_ceph_skill
- **Pipeline 状态**: 🔄 RUNNING
- **子任务状态**: PENDING
- **分词器**: HY3.0_SFT_Tokenizer
- **序列长度**: 8192
- **输入路径**: `/apdcephfs_jn2/share/leslizhang/sft/`
- **输出路径**: `/apdcephfs_jn2/share/leslizhang/sft_bin/`
- **责任人**: user1, user2
- **创建人**: your_rtx_name
- **创建时间**: 2026-04-10 16:00:00
```

> ⚠️ 后端会在内部把数据写到 `<用户 storage_path>/<subPath>` 下（subPath 形如 `bin_data/HY3.0_SFT_Tokenizer_8192`）。MCP server 已经在工具响应里把 `storagePath` 还原为用户原始 `storage_path`（剥离尾部 `subPath`），Agent 直接展示返回里的"输出路径"即可；**严禁**自行把 `subPath` 拼回"输出路径"。

---

### 路径南京地域强校验（调用 create_hunyuan_data_sft_conversion 前置）

> 🌏 **为什么必须校验**：当前 SFT 转 bin 任务的计算资源**只部署在南京地域**，如果 `input_path` 或 `storage_path` 不在南京地域，会出现 I/O 跨地域、读/写失败或性能极差等问题，必须在调用 `create_hunyuan_data_sft_conversion` **之前**完成校验。

### 校验的触发时机

对以下所有调用都必须在**调用工具之前**执行本小节的校验逻辑：
- `create_hunyuan_data_sft_conversion`（任何一次创建转 bin 任务的调用）
- `retry_hunyuan_data_sft_conversion` 之前的重新规划（若用户显式表示更换输入/输出路径）
- 跨地域 pipeline 的 Step 2（用 `create_hunyuan_data_sft_conversion`）— 不过 Pipeline 内部的 Step 0 + Step 1 已经把数据搬到南京，Step 2 调用前**仅做兜底校验**即可

对 `get_hunyuan_data_sft_conversion` / `list_hunyuan_data_sft_tokenizers` 这类纯查询工具不需要校验。

### 路径地域识别算法

**输入**：一个 Ceph 路径字符串 `path`
**输出**：枚举值之一 `{南京, 非南京, 无法判断}`

**算法伪代码**：

```
规则: 判断 Ceph 路径是否位于南京地域
前置: 去掉路径首尾空白，取第一级目录名 first_segment
      （例：/apdcephfs_nj7/xxx/yyy → first_segment = "apdcephfs_nj7"）
      （例：/apdcephfs_jn/xxx/yyy   → first_segment = "apdcephfs_jn"）

1. 若 path 不以 "/" 开头，或找不到第一级目录名
   → 返回 "无法判断"
2. 若 first_segment 不以 "apdcephfs_" 开头（大小写不敏感）
   → 返回 "无法判断"
3. 取 first_segment 去掉前缀 "apdcephfs_" 之后的剩余部分 xxxxx（大小写不敏感比较）
4. 若 xxxxx 属于 {"nj", "jn"}，或 xxxxx 以 "nj" / "jn" 开头
   → 返回 "南京"
5. 否则
   → 返回 "非南京"
```

**等价正则**（用于快速判别"南京"）：`^/apdcephfs_(nj|jn)[^/]*/` （大小写不敏感）

### 典型路径判别示例

> 提示：南京地域在太极 Ceph 上同时存在两种历史命名前缀 `apdcephfs_nj*` 与 `apdcephfs_jn*`（新老命名并存），**两者都判定为南京 ✅**。

| 路径 | 第一级目录 | xxxxx | 判定 |
|------|-----------|-------|------|
| `/apdcephfs_nj/share/leslizhang/train/` | `apdcephfs_nj` | `nj` | ✅ 南京 |
| `/apdcephfs_nj7/share/...` | `apdcephfs_nj7` | `nj7` | ✅ 南京 |
| `/apdcephfs_nj11/share/ckpt/global_step/` | `apdcephfs_nj11` | `nj11` | ✅ 南京 |
| `/apdcephfs_jn/share/leslizhang/train/` | `apdcephfs_jn` | `jn` | ✅ 南京 |
| `/apdcephfs_jn2/share_303438049/...` | `apdcephfs_jn2` | `jn2` | ✅ 南京 |
| `/apdcephfs_jn_123/data/` | `apdcephfs_jn_123` | `jn_123` | ✅ 南京 |
| `/apdcephfs_zwfy5/share/...` | `apdcephfs_zwfy5` | `zwfy5` | ❌ 非南京（中卫 zw 命名） |
| `/apdcephfs_cq10/data/` | `apdcephfs_cq10` | `cq10` | ❌ 非南京（重庆 cq 命名） |
| `/apdcephfs_sh8/share/...` | `apdcephfs_sh8` | `sh8` | ❌ 非南京（上海 sh 命名） |
| `/ceph/bucket-a/sft-data/` | `ceph` | — | ⚠️ 无法判断 |
| `/data/user/train.json` | `data` | — | ⚠️ 无法判断 |

### 三分支处理策略

**分支 1：两个路径都判定为 ✅ 南京**
- 直接进入下一步，继续调用 `create_hunyuan_data_sft_conversion`。

**分支 2：任一路径判定为 ❌ 非南京**
- **必须阻断单步执行**，**严禁**带着非南京路径去调用 `create_hunyuan_data_sft_conversion`。
- 向用户**直接抛出错误并结束**：列出命中非南京的路径 + 地域判定结果，并明确告知"计算资源仅部署在南京（`/apdcephfs_nj*` / `/apdcephfs_jn*`）"。
- **严禁**主动建议走 pipeline 或其他回退方案；**严禁**在同一轮对话中接着做其他动作；Agent 直接结束本次 skill。
- 如果两条路径都是非南京，同时列出两条。
- 若用户在后续对话中主动要求"换路径"或"改走 pipeline"，由用户显式触发新一轮流程。

**分支 3：任一路径判定为 ⚠️ 无法判断**
- 不阻断执行，但必须在调用工具**之前**向用户输出一段警告：
  > ⚠️ 您提供的 `input_path` / `storage_path` 未匹配到标准的 `/apdcephfs_xxxxx/` 前缀，无法自动判断地域。**请确认输入或输出路径是在南京地域，否则可能出现转bin任务失败的风险。**
- 警告输出后继续调用 `create_hunyuan_data_sft_conversion`。
- 若既有 ⚠️ 又有 ❌，以 ❌ 的阻断语义为准（即仍然阻断）。

### 对话话术模板

**模板 A（两个路径都是非南京时）：**
> ❌ 检测到 `input_path` 和 `storage_path` 都不在**南京地域**：
> - `input_path`：`{用户提供的 input_path}` → 地域：非南京
> - `storage_path`：`{用户提供的 storage_path}` → 地域：非南京
>
> 当前转 bin 任务的计算资源仅部署在南京（`/apdcephfs_nj*` / `/apdcephfs_jn*`）。**本次调用已阻断，skill 结束。**

**模板 B（仅其中一个是非南京时）：**
> ❌ 检测到 `{input_path 或 storage_path}` 不在**南京地域**：`{用户提供的路径}`。另一路径 `{另一参数名}`：`{另一路径}` 已通过南京地域校验。
>
> 当前转 bin 任务的计算资源仅部署在南京（`/apdcephfs_nj*` / `/apdcephfs_jn*`）。**本次调用已阻断，skill 结束。**

**模板 C（存在无法判断的路径，继续前警告）：**
> ⚠️ 风险提示：您提供的路径未匹配到标准的 `/apdcephfs_xxxxx/` 前缀，无法自动判断所在地域：
> - `{参数名}`：`{路径}`
>
> 转 bin 任务的计算资源仅部署在**南京**，请确认上述路径是在南京地域，否则可能出现转 bin 任务失败的风险。如需继续，我将按当前路径调用转 bin 工具。

---

## get_hunyuan_data_sft_conversion

**功能**：根据任务 ID 查询 SFT 转 bin 任务的详细信息和执行状态。用于在创建任务后轮询任务进度，判断任务是否完成；也用于查询 bin 文件大小（`size` 字段）。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 转bin任务 ID（创建任务时返回的 `id`）。⚠️ **本工具是平铺签名**，`task_id` 直接作为顶层参数传入，**不要**包在 `params` 对象里 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_sft_conversion '{"task_id": 12345}'
```

### 状态说明

| 状态值 | Emoji | 说明 |
|--------|-------|------|
| `PENDING` | ⏳ | 待执行，任务已创建等待执行 |
| `RUNNING` | 🔄 | 执行中，任务正在执行 |
| `SUCCEEDED` | ✅ | **终态**：执行成功，查看 `storagePath` 获取产出路径 |
| `FAILED` | ❌ | **终态**：执行失败，查看 `message` 获取错误详情 |
| `STOPPED` | ⏹️ | **终态**：任务被手动停止 |

### 返回字段说明

返回 Markdown 格式的任务详情，字段与创建任务返回相同，额外包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `message` | str | 失败时的错误详情（仅 FAILED 状态有值） |
| `endTime` | str | 任务结束时间（仅终态有值） |
| `size` | long | 产出数据大小（字节，仅 SUCCEEDED 有值） |
| `rangeStart` | int | 数据范围起始（仅 SUCCEEDED 有值） |
| `rangeEnd` | int | 数据范围结束（仅 SUCCEEDED 有值） |

### 返回示例

**执行中：**

```
# 转bin任务详情 (ID: 12345)

- **任务名称**: create_by_bin_auto_transfer_ceph_skill
- **Pipeline 状态**: 🔄 RUNNING
- **子任务状态**: RUNNING
- **分词器**: HY3.0_SFT_Tokenizer
- **序列长度**: 8192
- **输入路径**: `/apdcephfs_jn2/share/leslizhang/sft/`
- **输出路径**: `/apdcephfs_jn2/share/leslizhang/sft_bin/`
```

**执行成功：**

```
# 转bin任务详情 (ID: 12345)

- **任务名称**: create_by_bin_auto_transfer_ceph_skill
- **Pipeline 状态**: ✅ SUCCEEDED
- **子任务状态**: SUCCEEDED
- **分词器**: HY3.0_SFT_Tokenizer
- **序列长度**: 8192
- **输入路径**: `/apdcephfs_jn2/share/leslizhang/sft/`
- **输出路径**: `/apdcephfs_jn2/share/leslizhang/sft_bin/`
- **产出大小 (size)**: 1.00 GB （精确值 1073741824 字节）
- **数据范围**: 0 ~ 127
- **结束时间**: 2026-04-10 18:30:00

### ✅ 任务已完成
产出路径: `/apdcephfs_jn2/share/leslizhang/sft_bin/`
```

**执行失败：**

```
# 转bin任务详情 (ID: 12345)

- **任务名称**: create_by_bin_auto_transfer_ceph_skill
- **Pipeline 状态**: ❌ FAILED
- **子任务状态**: FAILED
- **分词器**: HY3.0_SFT_Tokenizer
- **序列长度**: 8192
- **输入路径**: `/apdcephfs_jn2/share/leslizhang/sft/`
- **输出路径**: `/apdcephfs_jn2/share/leslizhang/sft_bin/`
- **结束时间**: 2026-04-10 17:15:00

### ❌ 错误详情
```
转bin失败: pipeline=67890, error=WeData UC任务执行失败：tokenizer配置错误
```

> 💡 返回详情里已包含 `message`；Agent 必须**直接把该错误透传给用户并结束本轮回复**。**严禁**主动调用 `retry_hunyuan_data_sft_conversion` 或追问用户"是否要重试"。如果用户在后续对话中显式要求"帮我重试"，再进入重试工具章节。
```

### 单工具 SOP：查询 bin 文件大小（size）

> ⚠️ **前提约束：必须明确是后训练 SFT 转 bin 上下文**。当用户提示词中出现"bin 文件大小 / bin 文件 size / 产出多大 / 转 bin 生成了多少数据 / 产出 size"等关键词时，**必须先确认用户说的是后训练 SFT 转 bin 任务**（即 `create_hunyuan_data_sft_conversion` 或 Pipeline Step 2 产出的转 bin 任务），才能路由到本工具。原因：后续可能存在预训练转 bin 等其他类型的转 bin 任务，它们的 size 查询走不同的工具；在上下文不明确时，**必须先追问用户是哪类转 bin 任务**。

**触发词示例**：
- "查一下这个转 bin 任务生成的 bin 文件大小 / size"
- "这个后训练转 bin 任务产出了多少数据 / 产出多大"
- "转 bin 完成了，bin 文件有多大"
- "SFT 转 bin 的产出 size 是多少"
- "task_id=12345 的转 bin 任务，bin 文件大小是多少"

**Agent 行为规范：**
1. **参数校验**：需要 `task_id`（转 bin 任务 ID），与普通查询状态完全相同，无额外参数。
2. **调用工具**：直接调用 `get_hunyuan_data_sft_conversion(task_id=xxx)`，返回里的 `产出大小 (size)` 字段即为 bin 文件大小。
3. **size 字段说明**：
   - 单位：字节（long 类型），工具返回时已自动转为人可读格式（如 `1.00 GB`），同时附上精确字节数。
   - **仅 `pipelineStatus=SUCCEEDED` 时有值**；任务未完成时 `size` 为 null，工具返回里不会显示该行。
4. **非 SUCCEEDED 时的兜底话术**：若调用后发现任务状态不是 SUCCEEDED（如 RUNNING / FAILED），应如实告知用户：
   - RUNNING：「任务仍在执行中，bin 文件尚未生成，size 暂无数据。可稍后再次查询。」
   - FAILED：「任务已失败，未产出 bin 文件，size 无数据（失败原因：{原样透传 message}）。」**不要主动提议重试**。
   - PENDING：「任务尚未开始执行，bin 文件尚未生成，size 暂无数据。」
5. **禁止编造 size**：若 size 字段为 null 或任务未 SUCCEEDED，**严禁向用户返回 0 或任何估算值**，必须如实说明"尚无数据"。

---

## retry_hunyuan_data_sft_conversion

**功能**：对失败的 SFT 转 bin 任务进行重试。仅当任务处于 `FAILED` 状态时可以重试。需要当前用户是任务的 owner 或管理员。

> ⚠️ **只有在用户显式要求重试时才进入本工具**。Agent 自身绝不主动发起重试，也不主动提议"是否要重试"——任何一步 FAILED 都必须"直接抛出原因并结束"。仅当用户在后续对话中显式要求重试时才调用。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 转bin任务 ID。⚠️ **本工具是平铺签名**，`task_id` 直接作为顶层参数传入，**不要**包在 `params` 对象里 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call retry_hunyuan_data_sft_conversion '{"task_id": 12345}'
```

### 返回字段说明

返回 Markdown 格式的重试结果，包含任务最新状态。重试成功后 `pipelineStatus` 会变为 `RUNNING`。

### 返回示例

```
🔄 任务 12345 已重新启动！

# 转bin任务详情 (ID: 12345)

- **任务名称**: create_by_bin_auto_transfer_ceph_skill
- **Pipeline 状态**: 🔄 RUNNING
- **子任务状态**: PENDING
- **分词器**: HY3.0_SFT_Tokenizer
- **序列长度**: 8192
- **输入路径**: `/apdcephfs_jn2/share/leslizhang/sft/`
- **输出路径**: `/apdcephfs_jn2/share/leslizhang/sft_bin/`
```

---

### 常见错误

| 阶段 | 错误类型 | 错误信息示例 | 处理建议 |
|------|---------|-------------|---------|
| 单步转 bin | 路径不存在 / 无读权限 | 后端返回路径校验失败 | 提示用户检查 `input_path` 是否存在且有读权限（`input_path` 必须是单个数据文件，不能是目录） |
| 单步转 bin | 输出路径非空 | 后端返回路径校验失败 | 提示用户 `storage_path` 需为空目录且有写权限 |
| 单步转 bin | **当前地域不支持数据** | 后端返回 `当前地域不支持数据` | 把 `message` 原样透传；明确告知"转 bin 任务的计算资源仅部署在**南京**，`input_path` / `storage_path` 都必须以 `/apdcephfs_nj*/` 或 `/apdcephfs_jn*/` 开头"；**直接结束 skill**，**严禁**主动建议"改走 pipeline"或其他回退 |
| 单步转 bin | **地域不一致** | 计算资源与输入数据所在地域不匹配 | 把 `message` 原样透传；**直接结束 skill**，**严禁**主动建议走 Pipeline 或任何回退路径 |
| 单步转 bin | 任务不存在 | `未找到 ID 为 xxx 的转 bin 任务` | 提示用户检查任务 ID 是否正确 |
| 单步转 bin | tokenizer 错误 | `pipelineStatus=FAILED`，`message` 含 tokenizer 配置错误 | 把 `message` 原样透传；**直接结束 skill**。**严禁**主动发起 `retry_hunyuan_data_sft_conversion`；仅当用户后续显式要求"重试"时再进入重试章节，且重试前可先建议用户通过 `list_hunyuan_data_sft_tokenizers` 确认 tokenizer 是否合法 |
