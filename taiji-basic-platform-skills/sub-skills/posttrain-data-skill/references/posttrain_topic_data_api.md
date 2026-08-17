## create_hunyuan_data_topic

**用途**：创建一个后训练 Topic（主题）。Topic 是后训练数据链路的最顶层容器，下面挂数据集（Dataset），数据集再挂具体的数据版本（TopicData）。

### 触发条件

- 用户提示词**明确包含**以下关键词之一 → 路由到本工具：
  - 「建后训练 topic」「新建后训练主题」「创建 posttrain topic」
  - 「我要建个后训练数据的 topic，名字叫 xxx」
  - 「顶层容器 / 主题 / 后训练数据管理的 topic」
- 用户只说"建 topic"但语义模糊 → 必须追问"是后训练数据管理的 Topic（→本工具）还是其它系统里的 topic（如 Kafka/MQ/评测话题，本 Skill 不支持）"；不得直接调工具。
- 用户说"建评测 topic / 评测话题"→ ❌ 不走本工具（本 Skill 暂不支持建评测 Topic）。
- 用户说"建 Kafka topic"→ ❌ 不在本 Skill 支持范围。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为 0；未知时先走 `references/helper_api.md`（`list_user_workspaces`） |
| `key` | str | ✅ 必填 | - | Topic 全局唯一标识符（**全库唯一**）；建议英文小写 + 下划线 |
| `name` | str | ✅ 必填 | - | Topic 可读名称（**全局唯一**） |
| `modality` | str | ⚠️ 强烈建议 | 空间默认 | `TEXT` / `MULTIMODAL`。Topic 是模态源头；10103/12290 打通空间应显式传入。**用户未明确 modality 时必须追问「文本还是多模态？」，不得凭猜测自行赋值** |
| `desc` | str | 否 | `null` | Topic 描述 |
| `owners` | list[str] | 否 | `null` | 责任人 RTX 列表；MCP 层未传时自动填入当前 Token 用户 |

### 参数前置校验（调用前必做）

1. `wsid` 非 0；否则反馈"wsid（工作空间 ID）不能为空"，并提示先看 `references/helper_api.md`。
2. `key` / `name` 非空白字符串；两者都**全局唯一**；已存在直接引导换新值而不是重试。
3. `owners` 未传 → MCP 自动用 Token 用户；传了必须是字符串列表。

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_topic '{
  "wsid": 10103,
  "key": "math_reasoning",
  "name": "数学推理 Topic",
  "desc": "用于数学推理类后训练数据管理",
  "owners": ["leslizhang"]
}'
```

### 返回字段说明

返回 Markdown 文本；关键字段如下：

| 字段 | 说明 |
|------|------|
| Topic ID | 后端生成的 `PostTrainTopic.id`；**下一步**作为 `create_hunyuan_data_topic_dataset` 的 `topic_id` |
| Key | Topic 全局唯一 key |
| Name | Topic 可读名称 |
| 责任人 | owners 列表 |
| 后续操作建议 | 明确给出下一步应调用哪个工具、以及 ID 应如何传递 |

### 返回示例

```
# 后训练 Topic 已创建 (ID: 101)

- **Topic ID**: 101
- **Key**: `math_reasoning`
- **Name**: 数学推理 Topic
- **描述**: 用于数学推理类后训练数据管理
- **工作空间 (wsid)**: 10103
- **责任人**: leslizhang
- **创建人**: leslizhang
- **创建时间**: 2026-04-27 17:08:34

### 🔁 后续操作建议
1. 在本 Topic 下建数据集：`create_hunyuan_data_topic_dataset` (topic_id=101)
2. 数据集建好后再建具体数据版本：`create_hunyuan_data_topic_data` (dataset_id=新数据集 ID，SFT 默认 `include_baseline=true`)
3. 创建时已开质检开关则无需再调旧 `create_hunyuan_data_quality_inspection`；用 get/preview/download 查看结果即可
```

### 典型错误与处理建议

| 触发条件 | 后端 message | 处理建议 |
|------|------|------|
| `key` 已存在 | `Topic已经存在：key=xxx` | 让用户换一个新 key，或复用已存在的 Topic（无需新建） |
| `name` 已存在 | `Topic名称已经存在：key=xxx` | 让用户换一个新 name |
| 只读账号 | HTTP 403 | 提示用户切换为有写权限的账号 |

---

## create_hunyuan_data_topic_dataset

**用途**：在某个后训练 Topic 下创建一个数据集。数据集是 TopicData（数据版本）的直接上游，同时承载了 `stage`（训练阶段）与 `thinkingType`（思考类型）两个关键属性，下游 TopicData 会自动继承。

### 触发条件

- 用户提示词**明确包含**以下关键词之一 → 路由到本工具：
  - 「在 topic XX 下建数据集」「挂在后训练 topic 下的数据集」「新建 posttrain dataset」
  - 「建 SFT 数据集」「建 DPO 数据集」「建 GRPO 数据集」「建 REWARD 数据集」（带明确 stage 的）
- 用户只说"建数据集"但语义模糊 → 必须追问"是后训练 Topic 下的数据集（承载 stage/thinkingType）→ 请提供 topic_id 与 key，还是评测数据集（本 Skill 暂不支持创建）"；不得直接调工具。
- 用户说"建评测数据集"→ ❌ 不走本工具。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为 0 |
| `topic_id` | int | ✅ 必填 | - | 所属 Topic 的 ID；必须已存在；若无 → 先走 `create_hunyuan_data_topic` |
| `key` | str | ✅ 必填 | - | 数据集全局唯一 key；可能受 `keyPattern` 正则约束 |
| `desc` | str | 否 | `null` | 数据集描述 |
| `stage` | str | 否 | `null` | 训练阶段，取值：`SFT` / `GRPO` / `DPO` / `REWARD`。**会被下游 TopicData 自动继承** |
| `thinking_type` | str | 否 | `null` | 思考类型，取值：`FAST_THINKING`（快思考-短链）/ `SLOW_THINKING`（慢思考-长链）/ `HOLISTIC_THINKING`（全思考）。⚠️ 传枚举名而非 value。**会被下游 TopicData 自动继承** |
| `owners` | list[str] | 否 | `null` | 责任人列表；未传时 MCP 自动用 Token 用户 |

### 参数前置校验

1. `wsid` 非 0。
2. `topic_id` 正整数且必须已存在；若用户只给了 Topic 名称，应先引导用 `create_hunyuan_data_topic` 新建。
3. `key` 非空白；**全局唯一**；可能需要匹配 `keyPattern` 正则。
4. `stage` 必须是 `{SFT, GRPO, DPO, REWARD}` 之一；用户传了"sft"小写等也需归一化或反馈（非法值由**后端**按 enum name 严格反序列化时拒绝，返回业务错误）。
5. `thinking_type` 必须是 `{FAST_THINKING, SLOW_THINKING, HOLISTIC_THINKING}` 之一；**不要用 FAST/SLOW/FULL 这类 value**（后端按 enum name 反序列化，传 value 会报错）。

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_topic_dataset '{
  "wsid": 10103,
  "topic_id": 101,
  "key": "math_sft_v1",
  "desc": "SFT 阶段快思考数据集",
  "stage": "SFT",
  "thinking_type": "FAST_THINKING"
}'
```

### 返回字段说明

| 字段 | 说明 |
|------|------|
| Dataset ID | 后端生成的 `PosttrainTopicDataset.id`；**下一步**作为 `create_hunyuan_data_topic_data` 的 `dataset_id` |
| 所属 Topic ID / 名称 | 关联的 Topic 信息 |
| 训练阶段 (stage) | 下游 TopicData 自动继承 |
| 思考类型 (thinkingType) | 下游 TopicData 自动继承 |

### 返回示例

```
# 后训练数据集已创建 (ID: 7001)

- **Dataset ID**: 7001
- **所属 Topic ID**: 101
- **所属 Topic 名称**: 数学推理 Topic
- **Key**: `math_sft_v1`
- **描述**: SFT 阶段快思考数据集
- **训练阶段 (stage)**: SFT
- **思考类型 (thinkingType)**: FAST_THINKING
- **责任人**: leslizhang
- **创建人**: leslizhang
- **创建时间**: 2026-04-27 17:09:00

### 🔁 后续操作建议
1. 在本数据集下建数据版本：`create_hunyuan_data_topic_data` (dataset_id=7001)
2. 本数据集的 stage / thinkingType 会**自动继承**到其下 TopicData，因此建数据版本时无需再传 stage / thinking_type。
```

### 典型错误

| 触发条件 | 后端 message | 处理建议 |
|------|------|------|
| `topic_id` 不存在 | `topic不存在：id=xxx` | 让用户先用 `create_hunyuan_data_topic` 建 Topic |
| `key` 已存在 | `Topic数据集已经存在：key=xxx` | 换 key 或直接复用已有数据集 |
| `key` 正则不匹配 | `数据名称必须符合正则表达式：...` | 按后端给出的正则调整 key |
| `stage` / `thinking_type` 取值非法 | 后端按 enum name 严格反序列化拒绝（返回业务错误） | 按枚举表选一个合法值（大小写/字面量须完全匹配） |

---

## create_hunyuan_data_topic_data

**用途**：在某个后训练数据集下注册一个具体的数据版本（TopicData），即把一个 Ceph 上的单文件 JSONL 数据源登记到 datax 后训练数据管理平台，后续可基于此做 SFT 转 bin、质检、消融实验等。

### 触发条件

- 用户提示词**明确包含**以下关键词之一 → 路由到本工具：
  - 「在数据集 XX 下建数据版本 / 注册数据 / 挂一条数据」
  - 「把 /apdcephfs_xxx/train.jsonl 注册为数据集 xxx 的 v1 版本」
  - 「建一条 posttrain TopicData」「新建后训练数据版本」
- 用户说"把数据拷贝到某个路径"→ ❌ 走 `create_hunyuan_data_export_task` / `create_hunyuan_data_cudofs_copy_task`。
- 用户说"把数据转成 bin"→ ❌ 走 `create_hunyuan_data_sft_conversion`。
- 用户意图模糊（只说"建数据"）→ 必须追问区分"登记为后训练数据版本 / 数据拷贝 / 转 bin"。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为 0 |
| `dataset_id` | int | ✅ 必填 | - | 所属数据集 ID；**stage / thinkingType 自动从此数据集继承** |
| `version` | str | ✅ 必填 | - | 数据版本；`(dataset_id, version)` 唯一；可能受 `versionPattern` 正则约束 |
| `source_path` | str | ✅ 必填 | - | 源文件 Ceph 路径；**必须单文件 JSONL**（非目录） |
| `desc` | str | 否 | `null` | 数据描述 |
| `app_group` | str | 否 | `null` | 应用组（business_flag）；与 location / source_path 一致性后端强校验 |
| `location` | str | 否 | `null` | 地域（如 `nj` / `jn` / `zwfy`）；为空时后端按 source_path 前缀推断 |
| `epochs` | float | 否 | `null` | 采样轮数 |
| `priority` | int | 否 | `null` | 优先级（越大越优先） |
| `include_baseline` | bool | 否* | 协议 `false`；**Agent 对 SFT 默认传 `true`** | 是否对**底线质检**算子发起质检（创建时立即批量发起）。前端 SFT 新建默认开启；**编辑态禁用，创建后无法补开**。*Agent 侧视为 SFT 准必填默认 |
| `include_content` | bool | 否 | `false` | 是否对**Topic 内容质检**算子发起质检；前端默认关闭，用户明示才开 |
| `include_general_content` | bool | 否 | `false` | 是否对**通用内容质检**算子发起质检；三者**至少开启一项**即视为开启质检并在创建时触发，无需再调 `create_hunyuan_data_quality_inspection`。旧参数 `enable_inspection` 已废弃 |
| `owners` | list[str] | 否 | `null` | 责任人；MCP 未传时自动填 Token 用户 |
| `tag_list` | list[str] | 否 | `null` | 标签名列表 |
| `data_source` | str | 否 | `null` | `TopicDataConfig.dataSource`，数据来源说明；RL 场景常用 |
| `manual_check_report` | str | 否 | `null` | `TopicDataConfig.manualCheckReport`，人工抽检报告链接或路径 |
| `quality_score` | float | 否 | `null` | `TopicDataConfig.qualityScore`，质量分（0~1 浮点数） |

### 参数前置校验

1. `wsid` 非 0、`dataset_id` 正整数、`version` / `source_path` 非空白。
2. **`source_path` 硬约束**（违反会在质检 / 转 bin 阶段大面积失败）：
   - **必须是单个文件**，不能是目录（后端 `FileType.FILE` 强校验）；
   - **内容必须是 JSONL**（每行一个独立 JSON 对象）；后缀 `.json` 也可但不能是整文档单一 JSON 数组或跨行格式化；
   - 当前用户对该路径**有读权限**；
   - path / app_group / location 三者一致（后端 `ensureAppGroupInfoValid` 校验）。
3. `stage` 与 `thinking_type` **本工具不暴露也不传**；后端自动从 Dataset 继承。若用户希望用不同的 stage，应先 `create_hunyuan_data_topic_dataset` 新建一个数据集。
4. **质检开关对齐前端（强制，创建时唯一窗口）**：
   - OpenAPI 协议层三开关默认均为 `false`；**Agent 不得因「可选」而省略**，否则页面【底线质检】【内容质检】会一直显示「未质检」，且前端编辑态无法补触发。
   - **SFT**（用户未说不做质检）：必须显式传 `include_baseline=true`；`include_content` / `include_general_content` 默认 `false`。回复中回显「已按前端默认开启底线质检」。
   - **GRPO**：默认三者 `false`（与前端一致）；用户明示要开再传。
   - 用户明确「先不做质检」→ 三者全关。
   - 用户明示要内容质检 → `include_content=true`（可与底线同时开）。
   - 开启任一项时提醒：「创建时已对所选质检算子发起 V2 批量质检，无需再调 `create_hunyuan_data_quality_inspection`」（该旧工具走 V1 记录，填不满页面两列）。

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_topic_data '{
  "wsid": 10103,
  "dataset_id": 7001,
  "version": "2026-04-27-v1",
  "source_path": "/apdcephfs_jn2/share_xxx/math/train.jsonl",
  "desc": "数学推理 SFT 数据 v1",
  "include_baseline": true,
  "include_content": true
}'
```

### 返回字段说明（节选自 `PostTrainTopicDataInfo`）

| 字段 | 说明 |
|------|------|
| TopicData ID | 后端生成的 `PostTrainTopicData.id`；**是 `create_hunyuan_data_quality_inspection` 的 `topic_data_id` 输入** |
| 状态 (status) | `PENDING / RUNNING / SUCCEEDED / FAILED`，带 emoji 可视化；**只有 SUCCEEDED 才可下游使用** |
| 所属 Topic / Dataset | 链路关联信息（含名称） |
| 全局 Key | `<datasetKey>-<stage>-<thinkingType>-<version>` 拼出的全局唯一 key |
| stage / thinkingType | 从数据集继承 |
| sourcePath / storagePath | 原路径与后端搬到的统一存储路径；非 owner 时会脱敏为 `***` |
| appGroup / location | 路径所在应用组与地域 |
| includeBaseline / includeContent / includeGeneralContent | 三个顶层质检开关，分别表示是否对底线 / Topic 内容 / 通用内容质检算子自动触发了质检 🆕 |
| isNew | 是否为本次新建（布尔）；已存在同 (dataset_id, version) 时可能为 false 🆕 |
| inspectionId | 若已触发质检，关联的质检记录 ID |

### 返回示例

```
# 后训练数据版本已创建 (TopicData ID: 3021)

- **TopicData ID**: 3021
- **状态**: ⏳ PENDING
- **所属 Topic ID**: 101
- **所属 Topic 名称**: 数学推理 Topic
- **所属数据集 ID**: 7001
- **数据集 Key**: `math_sft_v1`
- **全局 Key**: `math_sft_v1-SFT-FAST_THINKING-2026-04-27-v1`
- **版本**: 2026-04-27-v1
- **训练阶段**: SFT
- **思考类型**: FAST_THINKING
- **描述**: 数学推理 SFT 数据 v1
- **源路径 (sourcePath)**: `/apdcephfs_jn2/share_xxx/math/train.jsonl`
- **应用组**: share_xxx
- **地域**: jn
- **底线质检 (includeBaseline)**: True
- **内容质检 (includeContent)**: True
- **通用内容质检 (includeGeneralContent)**: False
- **是否新建 (isNew)**: True
- **责任人**: leslizhang
- **创建人**: leslizhang
- **创建时间**: 2026-04-27 17:10:00

### 🔁 后续操作建议
1. ⏳ 当前数据尚未 SUCCEEDED，请稍后通过 MCP 或前端查询 TopicData 详情，等 status 变为 SUCCEEDED 后再进行下一步。
2. 创建时已开启质检开关（include_baseline / include_content 等），后端会对所选算子自动触发一次质检；稍后可用 `get_hunyuan_data_quality_inspection` 查看 inspectionId 关联的结果。
3. 若要做 SFT 转 bin：可用本 TopicData 的 sourcePath 作为 `create_hunyuan_data_sft_conversion` 的 input_path（注意地域要求南京）。
```

### 典型错误

| 触发条件 | 后端 message | 处理建议 |
|------|------|------|
| `dataset_id` 不存在 | `topic数据集不存在：id=xxx` | 让用户先用 `create_hunyuan_data_topic_dataset` 建数据集 |
| `(dataset_id, version)` 重复 | `topic数据已经存在：datasetId=xxx, version=yyy` | 若用户明确指定了该 `version`，必须停止并报告重复，不能擅自改成 `v2_日期` / `version_时间戳` 继续注册；若返回里没有 `topic_data_id`，不要猜测/扫描附近 ID，也不要把 `dataset_id` 当 `topic_data_id` 去查，应请用户提供现有 `topic_data_id` 或明确授权换唯一 version 后再注册 |
| `version` 正则不匹配 | `版本必须符合正则表达式：...` | 按后端正则调整 |
| `source_path` 不是单文件 | `FileType 不匹配：path=xxx, expected=FILE` | 让用户提供**单文件**路径，不是目录 |
| `source_path` 路径地域无法识别 | `未找到Ceph路径对应的地域，可用:[...]` | 路径前缀不在后端可识别的地域/应用组容器白名单内；从报错列出的「可用」前缀里选，或改用应用组允许的 ceph 前缀 |
| `source_path` 路径不存在 | `请确保当前路径存在:xxx` | 让用户确认该 ceph 路径真实存在且可读 |
| app_group / location / path 不一致 | `appGroup 与 location 不匹配` | 让用户核对 app_group 与路径所在地域 |
| 路径无读权限 | `no permission: path=xxx` | 让用户先申请读权限或换有权限的路径 |

---

## get_hunyuan_data_topic_data

**用途**：根据 `topic_data_id` 查询后训练 TopicData 的详情与当前状态。

这是 `create_hunyuan_data_topic_data` 的**必然搭档**：创建是异步的，后端返回时 `status` 多为 `PENDING`/`RUNNING`，真正可用（`SUCCEEDED`）必须通过本工具**显式轮询**才能确认。

### 触发条件

- 用户提示词**明确包含**以下关键词之一 → 路由到本工具：
  - 「查 TopicData 状态 / 进度 / 搬运完了吗」
  - 「这条后训练数据好了吗 / 可以跑质检了吗」
  - 「看一下 topic_data_id=XXX 的详情 / sourcePath / inspectionId」
- 用户说"查质检"→ ❌ 走 `get_hunyuan_data_quality_inspection`（不是本工具）。
- 用户说"查任务状态"但 ID 没上下文（不是 `basic_train_` 开头、也没说 `conversion` / `export` / `cudofs` / `inspection` / `topic_data`）→ 必须**先追问**是哪类任务再路由，不得默认走本工具。
- 用户只给 `topic_id` / `dataset_id` / `inspection_id` 但说"查 topic 数据"→ 必须追问用户是否把 ID 搞混了；**本工具只接受 `topic_data_id`**。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `topic_data_id` | int | ✅ 必填 | - | TopicData 主键 ID（`PostTrainTopicData.id`）；**不是** `topic_id` / `dataset_id` / `inspection_id` |
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为 0；后端 `workspaceService.validate(wsid, data)` 会做工作空间归属强校验 |

### 参数前置校验

1. `topic_data_id` 正整数；为空 / 0 / 负数 → 直接反馈"topic_data_id 不能为空或非正整数"，不调工具。
2. `wsid` 非 0；否则提示先按 `references/helper_api.md`（`list_user_workspaces`）获取。
3. ID 混淆兜底：若用户给的看起来是 `topic_id` / `dataset_id` / `inspection_id`（例如上一步刚做了 `create_hunyuan_data_topic`），必须追问"您是要查 TopicData 详情吗？请提供 `topic_data_id`（即 `create_hunyuan_data_topic_data` 返回的 ID）"。

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_topic_data '{
  "topic_data_id": 3021,
  "wsid": 10103
}'
```

> ⚠️ **注意**：本工具的签名是**位置参数**（`topic_data_id`、`wsid` 直接平铺），不是 `params` 对象；与 `create_hunyuan_data_topic_data` 的 `{...}` 结构不同。

### 返回字段说明（基于 `PostTrainTopicDataInfo`）

| 字段 | 说明 |
|------|------|
| 状态 (`status`) | `PENDING` / `RUNNING` / `SUCCEEDED` / `FAILED`，带 emoji 可视化；**下游可用的唯一充分条件是 `SUCCEEDED`** |
| 所属 Topic / Dataset | 链路关联信息（含 topicName、datasetKey） |
| 全局 Key | `<datasetKey>-<stage>-<thinkingType>-<version>` 拼出的全局唯一 key |
| stage / thinkingType | 从数据集继承（不可在 TopicData 层覆盖） |
| sourcePath / storagePath | 原路径与统一存储路径；**非 owner / admin 时会被后端脱敏为 `***`**，此时返回里会带提示 |
| appGroup / location | 路径所在应用组与地域 |
| includeBaseline / includeContent / includeGeneralContent | 三个顶层质检开关，分别表示是否对底线 / Topic 内容 / 通用内容质检算子在就绪时自动触发质检 🆕 |
| isNew | 是否为本次新建（布尔） 🆕 |
| inspectionId | 若已触发质检，关联的质检记录 ID；可直接传给 `get_hunyuan_data_quality_inspection` |

### 返回示例（SUCCEEDED 状态）

```
# 后训练数据版本详情 (TopicData ID: 3021)

- **TopicData ID**: 3021
- **状态**: ✅ SUCCEEDED
- **所属 Topic ID**: 101
- **所属 Topic 名称**: 数学推理 Topic
- **所属数据集 ID**: 7001
- **数据集 Key**: `math_sft_v1`
- **全局 Key**: `math_sft_v1-SFT-FAST_THINKING-2026-04-27-v1`
- **版本**: 2026-04-27-v1
- **训练阶段**: SFT
- **思考类型**: FAST_THINKING
- **源路径 (sourcePath)**: `/apdcephfs_jn2/share_xxx/math/train.jsonl`
- **统一存储路径 (storagePath)**: `/apdcephfs_jn2/share_xxx/unified/posttrain_topic_data_3021/`
- **应用组**: share_xxx
- **地域**: jn
- **底线质检 (includeBaseline)**: True
- **内容质检 (includeContent)**: False
- **通用内容质检 (includeGeneralContent)**: False
- **是否新建 (isNew)**: True
- **已关联质检记录 ID**: 88
- **责任人**: leslizhang
- **创建人**: leslizhang
- **创建时间**: 2026-04-27 17:10:00

### 🔁 后续操作建议
1. ✅ 数据状态已是 SUCCEEDED，可进行下一步。
2. 已开启质检开关（include_baseline 等）且关联质检 inspectionId=88；可用 `get_hunyuan_data_quality_inspection` (inspection_id=88) 查看质检进度与结果。
3. 若要做 SFT 转 bin：可用本 TopicData 的 sourcePath 作为 `create_hunyuan_data_sft_conversion` 的 input_path（注意地域要求南京）。
```

### 返回示例（PENDING 状态，需要继续轮询）

```
# 后训练数据版本详情 (TopicData ID: 3021)

- **TopicData ID**: 3021
- **状态**: ⏳ PENDING
- **所属 Topic ID**: 101
- **所属数据集 ID**: 7001
- **版本**: 2026-04-27-v1
- **源路径 (sourcePath)**: `/apdcephfs_jn2/share_xxx/math/train.jsonl`
- **创建时间**: 2026-04-27 17:10:00

### 🔁 后续操作建议
1. ⏳ 当前数据仍为 PENDING，后台搬运尚未完成。建议间隔 30 秒~ 数分钟再次调用 `get_hunyuan_data_topic_data` 继续轮询，直到 status 变为 SUCCEEDED（或 FAILED）。
2. 若创建时未开 `include_*` 导致页面【底线/内容质检】为「未质检」：须**新建 version** 并显式传开关（SFT 默认 `include_baseline=true`）；不要指望 `create_hunyuan_data_quality_inspection` 补开 V2 列。
3. 若要做 SFT 转 bin：可用本 TopicData 的 sourcePath 作为 `create_hunyuan_data_sft_conversion` 的 input_path（注意地域要求南京）。
```

### 典型错误

| 触发条件 | 后端 message / 本地反馈 | 处理建议 |
|------|------|------|
| `topic_data_id` 不存在 | `topic数据不存在：id=xxx` | 让用户确认 ID 来源；常见错是把 `topic_id` / `dataset_id` / `inspection_id` 传进来 |
| `wsid` 与 TopicData 工作空间不匹配 | `workspace not match` | 让用户核对 wsid，或按 `references/helper_api.md`（`list_user_workspaces`）重新选一个 |
| 非 owner 查询 | 路径字段显示 `***` | 本地提示"非 owner/admin，路径已脱敏"；如需完整路径请让 owner 查询 |
| `status=FAILED` | 详情里 `message` 含失败原因（若有） | 到太极前端查看详细失败原因；必要时 `create_hunyuan_data_topic_data` 用新的 version 重建，不要盲目重试同一条 |

---

## query_hunyuan_data_topic_datas

**用途**：按工作空间（`wsid`）分页查询后训练 TopicData（数据版本）列表，并支持按数据版本名称、创建人、主状态、创建时间范围做可选筛选。复用 `PostTrainTopicDataService#list` 及其工作空间过滤逻辑。

这是 `get_hunyuan_data_topic_data` 的**列表版补充**：`get` 用于已知 `topic_data_id` 查单条；本工具用于**用户手上没有确切 ID、只知道筛选条件**（名称片段 / 创建人 / 状态 / 创建时间范围）时批量检索，拿到列表后再定位到具体的 `topic_data_id`。

### 触发条件

- 用户提示词**明确包含**以下关键词之一 → 路由到本工具：
  - 「查一下 wsid=XXX 下有哪些后训练数据版本 / 列一下 TopicData」
  - 「谁在最近创建了后训练数据 / 按创建人 leslizhang 查数据版本」
  - 「查一下失败/成功的数据版本 / 按状态筛选 TopicData」
  - 「按名称片段 / 创建时间范围找数据版本」
- 用户**已经给出确切 `topic_data_id`**、只想看那一条 → ❌ 走 `get_hunyuan_data_topic_data`，不是本工具。
- 用户说"查质检列表 / 有哪些质检"→ ❌ 不在本工具（本模块暂无质检列表工具，见质检模块）。
- 用户说"查 Topic 列表 / Dataset 列表"→ ❌ 本模块暂未提供 `list_posttrain_topics` / `list_posttrain_topic_datasets`（见下方 Q7）。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为空；未知时先走 `references/helper_api.md`（`list_user_workspaces`） |
| `name` | str | 否 | `null` | 数据版本名称（**模糊匹配**） |
| `creator` | str | 否 | `null` | 创建人（**模糊匹配** creator 列） |
| `status` | str | 否 | `null` | 主状态筛选，取值：`PENDING` / `RUNNING` / `FAILED` / `SUCCEEDED`（枚举名，大写） |
| `create_start_time` | str | 否 | `null` | 创建时间范围起始（**闭区间**），格式 `yyyy-MM-dd HH:mm:ss` |
| `create_end_time` | str | 否 | `null` | 创建时间范围结束（**闭区间**），格式 `yyyy-MM-dd HH:mm:ss` |
| `page` | int | 否 | `1` | 页码（**1-based**）；<=0 时后端归一为 1 |
| `page_size` | int | 否 | `20` | 每页数量；<=0 时归一为 20，**最大 1000**（超出按 1000 截断） |

### 参数前置校验

1. `wsid` 非空；否则反馈"wsid（工作空间 ID）不能为空"，并提示先看 `references/helper_api.md`。
2. `status` 若传，必须是 `{PENDING, RUNNING, FAILED, SUCCEEDED}` 之一（枚举名，不接受 `success` / `失败` 等别名）。
3. `create_start_time` / `create_end_time` 若传，必须是 `yyyy-MM-dd HH:mm:ss` 格式；建议成对提供以构成闭区间。
4. `name` / `creator` 为模糊匹配，无需精确全名。
5. `page` / `page_size` 越界时后端自动归一，无需额外校验，但要注意 `page_size` 上限 1000。

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_topic_datas '{
  "wsid": 10103,
  "creator": "leslizhang",
  "status": "SUCCEEDED",
  "create_start_time": "2026-04-01 00:00:00",
  "create_end_time": "2026-04-30 23:59:59",
  "page": 1,
  "page_size": 20
}'
```

### 返回字段说明

返回统一分页结构（`OpenApiPageResult`），关键字段：

| 字段 | 说明 |
|------|------|
| `items` | TopicData 详情列表（每项为 `PostTrainTopicDataInfo`），含 `id`、`topic_id`、`dataset_id`、`dataset_key`、`key`、`version`、`name`、`topic_name`、`stage`、`thinking_type`、`creator`、`status`、`create_time`、`update_time` 等 |
| `page` | 当前页码（1-based） |
| `page_size` | 每页数量 |
| `total` | 满足条件的总条数 |
| `has_more` | 是否还有下一页（`page * page_size < total`） |

> 💡 `items[i].id` 即 `topic_data_id`，可直接喂给 `get_hunyuan_data_topic_data` / `create_hunyuan_data_quality_inspection` 等下游工具。非 owner / admin 查询时，列表项的 `source_path` / `storage_path` 同样会被后端脱敏为 `***`。

### 返回示例

```
# 后训练数据版本列表（wsid=10103，第 1/2 页，共 23 条）

| # | topic_data_id | 版本 (version) | 名称 | 所属 Topic | stage | thinkingType | 创建人 | 状态 | 创建时间 |
|---|---------------|----------------|------|-----------|-------|--------------|--------|------|----------|
| 1 | 3025 | 2026-04-28-v1 | math_sft_v1-... | 数学推理 Topic | SFT | FAST_THINKING | leslizhang | ✅ SUCCEEDED | 2026-04-28 10:12:03 |
| 2 | 3021 | 2026-04-27-v1 | math_sft_v1-... | 数学推理 Topic | SFT | FAST_THINKING | leslizhang | ✅ SUCCEEDED | 2026-04-27 17:10:00 |

---

- **总条数**: 23
- **当前页**: 1 / 2（page_size=20）
- **has_more**: true

### 🔁 后续操作建议
1. 用某条的 `topic_data_id` 调 `get_hunyuan_data_topic_data` 查看完整详情/搬运状态。
2. 需要看下一页时，把 `page` 加 1 再次调用本工具。
```

> ⚠️ 展示时**严格按 `items` 原始顺序**（后端已排序），禁止重排 / 截断；表格末行与后续汇总之间必须留空行（见 `SKILL.md` §1.3 输出规范）。

### 典型错误

| 触发条件 | 后端 message / 本地反馈 | 处理建议 |
|------|------|------|
| `wsid` 缺失 | `wsid 不能为空` | 让用户提供 wsid，或按 `references/helper_api.md`（`list_user_workspaces`）获取 |
| `status` 取值非法 | 后端按 enum name 反序列化拒绝 | 从 `PENDING / RUNNING / FAILED / SUCCEEDED` 中选一个（大写枚举名） |
| 时间格式不对 | 反序列化失败 | 改为 `yyyy-MM-dd HH:mm:ss` |
| `page_size` 过大 | 后端自动按 1000 截断 | 无需处理，注意结果最多 1000 条/页 |

---
