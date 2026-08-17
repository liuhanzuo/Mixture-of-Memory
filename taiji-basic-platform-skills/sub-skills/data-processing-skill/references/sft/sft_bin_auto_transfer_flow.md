# 跨地域 SFT 转 bin 自动搬运 Pipeline（流程文档）

> 本文档是「跨地域 SFT 转 bin 自动搬运 Pipeline」的**流程编排规范**。它解决"源 ceph 不在南京 / 目标 ceph 不在南京"的端到端转 bin 场景：把任意 ceph 上的 SFT 数据 →（必要时中转南京）→ 转 bin →（必要时再搬到目标 ceph）。
> **本文档是流程文档**（多步 SOP），命中时必须**先完整阅读再执行**；单步 SFT 转 bin 工具的手册见 `sft_conversion_api.md`；跨地域复用 `data_processing_api.md` 的 `create_hunyuan_data_export_task`。

> 🔧 **本次重构要点**：历史上曾存在 5 个 `bin_pipeline_*` 封装工具，**现已全部下线，MCP server 不再暴露它们**。跨地域 Pipeline 现由 **Agent 在会话内编排底层工具**实现，映射关系如下：

| 已下线的封装工具 | 现在怎么做 |
|---|---|
| `bin_pipeline_check_path_location` | Agent **内联**按"路径地域识别算法"判断 `src_in_nj` / `tgt_in_nj`，不调任何工具 |
| `bin_pipeline_get_default_nj_staging_root` | Agent 直接使用 skill 内**硬编码的默认南京中转根目录**（见下方常量），不调任何工具 |
| `bin_pipeline_step1_export_to_nanjing` | Agent 调 `create_hunyuan_data_export_task`（`source_path` → `<nj_staging_root>/json/<staging_timestamp>`），再用 `get_hunyuan_data_export_task` 轮询 |
| `bin_pipeline_step2_convert_to_bin` | Agent 调 `create_hunyuan_data_sft_conversion`（`input_path`/`storage_path` 按场景派生），再用 `get_hunyuan_data_sft_conversion` 轮询 |
| `bin_pipeline_step3_export_to_target` | Agent 调 `create_hunyuan_data_export_task`（`<nj_staging_root>/bin/<staging_timestamp>` → `target_path`），再用 `get_hunyuan_data_export_task` 轮询 |

📌 **平台默认南京中转根目录（硬编码常量）**：当跨地域 Pipeline 需要南京 staging 而用户未提供 `nj_staging_root` 时，Agent 使用以下默认值：

```
DEFAULT_NJ_STAGING_ROOT = /apdcephfs_nj2/share_303722668/datax-pre/skill/ceph_pipeline_bin_default_address
```

使用该默认值前，Agent **必须**显式提示用户"使用平台默认中转 ceph 地址带宽会比较慢，建议提供您工作空间下的南京 ceph 路径"，取得继续意愿后再使用。

---

## 一、Pipeline 硬约束（最高优先级）

1. **`source_path` 必须是单个数据文件，不能是目录**（跨地域中转按单文件搬运）。与两端都在南京的单步转 bin 一致——单步 `create_hunyuan_data_sft_conversion` 的 `input_path` **也必须是单个数据文件**。若用户给的是目录，Agent 必须先向用户索取该目录下的**具体文件路径**，未拿到文件路径前不启动 Step 1 / Step 2 / Step 3。
2. **南京地域强校验**：SFT 转 bin 只能在南京等支持地域执行，非南京的输入 / 输出会直接被拒。整条 pipeline 最多分 3 步来跑：

```
[非南京源 ceph] --(Step 1: create_hunyuan_data_export_task)--> [<nj_staging_root>/json/<staging_timestamp>]
                --(Step 2: create_hunyuan_data_sft_conversion)--> [<nj_staging_root>/bin/<staging_timestamp>]
                --(Step 3: create_hunyuan_data_export_task)--> [目标 ceph]
```

3. **`<staging_timestamp>` 由 Agent 在 Step 0 根据当前时间生成（格式 `YYYYMMDDHHmmss`），整条 pipeline 共用一份**。两条 staging 路径（`/json/<ts>` 与 `/bin/<ts>`）由 Agent 在内部静默拼接，**严禁**让用户提供。
4. **`storage_path` / `target_path` 是硬必填、无默认**：Agent 严禁自行构造（不得用 `<input_path>_bin/`、不得复用其他任务的产出目录）；用户未提供 → 直接向用户索取，收到答复前不得启动任何工具。
5. **`nj_staging_root` 是按需选填**：仅在 Agent 内联地域识别判定任一端非南京时才询问；询问后若用户拒绝提供、又不同意用平台默认常量，直接结束 skill；**未拿到有效 `nj_staging_root` 前禁止发起 Step 1 / Step 2 / Step 3 的任何创建工具**。
6. **恢复时禁止重复创建任务**：用户贴回恢复清单后，应基于 `*_task_id` 直接走 `get_hunyuan_data_export_task` / `get_hunyuan_data_sft_conversion` 路径，**绝不能**再次调用 `create_hunyuan_data_export_task`（Step 1/3）/ `create_hunyuan_data_sft_conversion`（Step 2）去重新创建（会导致同一份数据搬两次 / bin 写两遍）。
7. **任务命名统一约束**：Pipeline 编排时创建的**所有** SFT 转 bin / ceph2ceph 拷贝任务，`name` 字段**必须显式传 `create_by_bin_auto_transfer_ceph_skill`**（由于 Step 1 / Step 3 复用通用 `create_hunyuan_data_export_task`，其默认任务名按后端实体生成，并非本 skill 固定名，因此必须显式传）。单步 `create_hunyuan_data_sft_conversion` 若 server 端已默认填充则无须显式传，但显式传也无副作用。
8. **产出路径展示规则**：向用户展示最终 bin 文件落地路径时，**必须直接展示用户提供的 `storage_path` / `target_path` 原值**，**严禁**在路径后面追加任何子路径（包括但不限于 `bin_data/<tokenizer>_<seq_len>/`、`/output/`、`/<run_id>/`）。
9. **失败即抛出原因并结束，不换策略**：任何一步（单步 `create_hunyuan_data_sft_conversion` / Step 1 / Step 2 / Step 3）一旦进入 FAILED / STOPPED 终态：
   - 只有在创建/查询结果里已经拿到有效 `task_id` 时，才调用对应日志工具（Step 1 / Step 3 / 单步导出 → `get_hunyuan_data_export_task_log`，必须传 `task_id` + `wsid`）；如果创建请求在生成 task_id 前直接返回错误，直接透传该错误，**不要**调用日志工具。Step 2 / 单步转 bin 在返回的 `message` 字段里已含错误详情，直接透传。
   - 把 `task_id` + 原始错误原因**原样**透传给用户（不做改写 / 不做总结）。
   - 🧾 输出 `current_phase = "failed"` 的恢复清单。
   - **直接结束 skill**，**绝不**主动发起重试 / 切换路径 / 切换工具 / 改走 pipeline 回退 / 提议其他方案。
   - **严禁**调用 `retry_hunyuan_data_sft_conversion`，**严禁**询问用户"是否要重试"；如果用户后续主动要求重试，再由用户在新的对话轮次里显式触发。

---

## 二、4 种场景的执行编排（由 Step 0 地域识别结果决定）

| 场景 | `src_in_nj` | `tgt_in_nj` | 执行步骤 | 说明 |
|---|---|---|---|---|
| 1 | ✅ 南京 | ✅ 南京 | 直接走单步 `create_hunyuan_data_sft_conversion`（**不进入 pipeline**） | `input_path = source_path`、`storage_path = target_path` |
| 2 | ❌ 非南京 | ✅ 南京 | Step 1 → Step 2 | Step 1：`source_path` → `<nj_staging_root>/json/<staging_timestamp>`；Step 2：以 `<nj_staging_root>/json/<staging_timestamp>` 为输入、以**用户提供的 `target_path`** 为输出（直接落地） |
| 3 | ✅ 南京 | ❌ 非南京 | Step 2 → Step 3 | Step 2：以 `source_path` 为输入、以 `<nj_staging_root>/bin/<staging_timestamp>` 为输出；Step 3：`<nj_staging_root>/bin/<staging_timestamp>` → `target_path` |
| 4 | ❌ 非南京 | ❌ 非南京 | Step 1 → Step 2 → Step 3 | Step 1：`source_path` → `<nj_staging_root>/json/<staging_timestamp>`；Step 2：以 `<nj_staging_root>/json/<staging_timestamp>` 为输入、以 `<nj_staging_root>/bin/<staging_timestamp>` 为输出；Step 3：`<nj_staging_root>/bin/<staging_timestamp>` → `target_path` |

> ⚠️ **场景 1 不走 Pipeline**：两端都在南京时直接走单步转 bin（`sft_conversion_api.md`），无需 staging、无需任何中转步骤。

---

## 三、Pipeline 必备输入与默认值

### 输入参数表

| 字段 | 类型 | 必填 | 默认行为 / 说明 |
|---|---|---|---|
| `wsid` | int | ✅ | 工作空间 ID；缺失时按 `helper_api.md` 反查 |
| `source_path` | str | ✅ | 源 SFT 数据 ceph 路径（任意地域）。⚠️ **必须是单个数据文件，不能是目录** |
| `target_path` | str | ✅ 必填（**无默认**） | 最终 bin 要交付的 ceph 路径（任意地域）。⚠️ **硬必填、无默认值、Agent 严禁自行构造或推测** |
| `seq_len` | int | ⚠️ | 序列长度。**用户未提供时必须先列出可选项 `4096 / 8192 / 32768`** 让用户选；用户明确说"用默认"才回落到 `4096` |
| `tokenizer` | str | ⚠️ | 分词器。**用户未提供时必须先调 `list_hunyuan_data_sft_tokenizers` 拿全列表后列给用户挑选**；用户明确说"用默认"才回落到列表第一个。**严禁**让用户自己输入 tokenizer 名 |
| `nj_staging_root` | str | ✋ 选填（按需触发） | 南京 staging 根目录（必须南京 ceph）。**收集阶段不主动询问**；只有 Agent 内联地域识别判定**任一端非南京**时才询问 |
| `staging_timestamp` | str | 🤖 Agent 内部生成 | **由 Agent 在 Step 0 根据当前时间生成**，格式 `YYYYMMDDHHmmss`（如 `20260514153045`），整条 pipeline 共用一份。**严禁**让用户提供，**严禁**复用历史时戳 |
| ~~`run_id`~~ | ~~str~~ | ❌ 已废弃 | **已废弃**：staging 路径规范改为 `<nj_staging_root>/json|bin/<staging_timestamp>`，不再使用 `run_id` 子目录 |
| ~~`task_name_prefix`~~ | ~~str~~ | ❌ 已废弃 | **已废弃**：本 skill 创建的所有子任务 `name` 一律固定为 `create_by_bin_auto_transfer_ceph_skill` |

**约定的 staging 路径规范（由 Agent 在 Step 0 静默拼接，不让用户提供）**：
- json 子目录（用于 Step 1 落地源数据 + Step 2 输入）：`<nj_staging_root>/json/<staging_timestamp>`
- bin 子目录（用于 Step 2 输出 + Step 3 源）：`<nj_staging_root>/bin/<staging_timestamp>`

### 默认值处理流程（按需触发）

> 任何"用默认值"的分支都必须**显式告知用户**，得到用户认可后再继续；不要静默套默认。

**`nj_staging_root` 按需触发处理流程：**
1. **询问用户在两种来源里二选一**（用户没主动提供 `nj_staging_root` 时）：
   > ℹ️ 您的源/目标路径需要先中转到南京 staging。请选择中转 ceph 地址：
   > 1. **使用平台默认中转 ceph 地址**：我会使用平台内置的默认南京中转目录 `/apdcephfs_nj2/share_303722668/datax-pre/skill/ceph_pipeline_bin_default_address`。⚠️ 平台默认地址**带宽较慢**，跨地域拷贝可能耗时较久。
   > 2. **您自己提供**（**推荐**）：给我您工作空间下的一个南京 ceph 路径（以 `/apdcephfs_nj` 或 `/apdcephfs_jn` 开头），用作中转 staging 根目录。
   >
   > 请告诉我您的选择，或直接给出您的 `nj_staging_root` 路径。

2. **用户选择"自己提供"** → 等用户给出有效路径，校验路径以 `/apdcephfs_nj` 或 `/apdcephfs_jn` 开头，校验通过后用作 `nj_staging_root` 进入下一步；不通过则继续向用户索取，**未拿到合法路径前禁止发起 Step 1 / Step 2 / Step 3 的任何创建工具**。
3. **用户选择"使用平台默认"** → 直接采用 skill 内置常量 `DEFAULT_NJ_STAGING_ROOT` 作为 `nj_staging_root`，并**再次显式向用户提示带宽风险**：
   > ⚠️ 将使用平台默认中转 ceph 地址：`/apdcephfs_nj2/share_303722668/datax-pre/skill/ceph_pipeline_bin_default_address`。请注意：**使用平台默认中转 ceph 地址带宽会比较慢**，可能拖慢整条 pipeline。要继续使用平台默认值吗？

   拿到用户继续意愿后才进入 Step 1。
4. **用户拒绝提供、也不同意用平台默认** → 直接结束 skill，并告知用户"未提供 `nj_staging_root`，跨地域 pipeline 无法启动"。

**`seq_len` 缺失（必须先列出可选项再让用户选）：**
1. **必须**向用户输出明确的可选项询问，**不得**只默默套默认值：
   > ℹ️ `seq_len` 你没指定。常见可选项有 **4096 / 8192 / 32768**，请选择一个值或告诉我您要的具体序列长度。
2. 用户答复后用其指定的值；用户明确说"用默认 / 你帮我选"时，再使用默认值 `4096`，并告知用户"已采用默认 4096"。
3. **严禁**在没列出可选项的情况下直接套默认 4096。

**`tokenizer` 缺失（必须先列出可选项再让用户选）：**
1. **先**调用 `list_hunyuan_data_sft_tokenizers` 拿到全列表。
2. **必须**把全部分词器名称列给用户挑选。
3. 用户挑了哪个就用哪个；用户明确说"用默认 / 你帮我选"时，再使用列表里的第一个作为默认值。
4. **严禁**在没列出可选项的情况下直接挑列表第一个；也**严禁**让用户"自己输入一个 tokenizer 名"。

---

## 四、Step 0 路径地域识别（Agent 内联，无工具）

**功能**：对 `source_path` 和 `target_path` 同时执行南京地域识别，作为 Step 0 决策"是否跳过 Step 1 / Step 3"的依据。**由 Agent 在会话内按下方算法自行判断，不调用任何 MCP 工具**。

### 识别算法

识别逻辑与 `sft_conversion_api.md` 的"路径南京地域强校验"完全相同，对 `source_path` / `target_path` 各算一次：

```
判定一个 ceph 路径是否在南京：
1. 去掉首尾空白，取第一级目录名 first_segment
   （/apdcephfs_nj7/xxx → apdcephfs_nj7；/apdcephfs_jn/xxx → apdcephfs_jn）
2. first_segment 不以 apdcephfs_ 开头（大小写不敏感）→ 无法判断（按非南京保守处理需询问用户）
3. 取 apdcephfs_ 之后的剩余部分 xxxxx：
   - xxxxx ∈ {nj, jn} 或以 nj / jn 开头 → 南京 ✅（src_in_nj / tgt_in_nj = True）
   - 否则 → 非南京 ❌（= False）
```

**等价正则**（快速判别"南京"）：`^/apdcephfs_(nj|jn)[^/]*/`（大小写不敏感）。

### 判定结果用途

| `src_in_nj` | `tgt_in_nj` | 场景 | 结论 |
|---|---|---|---|
| True | True | 场景 1 | 两端都在南京 → **不进 pipeline**，直接走单步 `create_hunyuan_data_sft_conversion` |
| False | True | 场景 2 | 跑 Step 1 → Step 2（跳过 Step 3） |
| True | False | 场景 3 | 跑 Step 2 → Step 3（跳过 Step 1） |
| False | False | 场景 4 | 跑完整 Step 1 → Step 2 → Step 3 |

> ℹ️ **特别说明**：本识别是**启发式**的，仅供决策"是否跳过 Step 1 / Step 3"。如果判错（例如某个不带 `_nj`/`_jn` 关键字但实际在南京的特殊 ceph 路径），用户可以**显式告诉 Agent**直接跳过 Step 1 / Step 3。

---

## 五、Step 1：源 → 南京 staging /json/<staging_timestamp>（用 `create_hunyuan_data_export_task`）

**功能**：把非南京源 ceph 数据搬到 `<nj_staging_root>/json/<staging_timestamp>`。**直接调用底层 `create_hunyuan_data_export_task`**（普通 ceph2ceph 拷贝，详见 `data_processing_api.md`），返回 `task_id` 后由 Agent 轮询 `get_hunyuan_data_export_task`。

**前置条件**：
1. Step 0 已完成参数收集 + 地域识别（`src_in_nj == False`，即场景 2 或场景 4）。
2. `nj_staging_root` 已经过用户认可（用户提供 / 用户同意使用平台默认常量）。
3. `staging_timestamp` 已由 Agent 在 Step 0 生成，整条 pipeline 共用。
4. **`source_path` 已确认是单个数据文件（不能是目录）**。

### 入参映射（`create_hunyuan_data_export_task`）

| `create_hunyuan_data_export_task` 参数 | Pipeline Step 1 取值 |
|---|---|
| `wsid` | 用户工作空间 ID（不能为 0） |
| `data_source` | 固定 `TAIJI_WEB`（按路径导出，可省略默认即为该值） |
| `source_path` | 用户的源 ceph 路径（非南京）。⚠️ **必须是单个文件，不能是目录** |
| `target_path` | `<nj_staging_root>/json/<staging_timestamp>`（即 `nj_json_path`，由 Agent 静默拼接，不让用户提供） |
| `name` | **必须显式传** `create_by_bin_auto_transfer_ceph_skill` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_export_task '{
  "wsid": 10103,
  "data_source": "TAIJI_WEB",
  "source_path": "/apdcephfs_cq10/share/leslizhang/sft_raw/",
  "target_path": "/apdcephfs_jn2/share_302316223/leslizhang/staging/json/20260514153045/",
  "name": "create_by_bin_auto_transfer_ceph_skill"
}'
```

### 轮询规则（Agent 必须遵守，**自动驱动、严禁让用户催**）

> **Step 1 创建任务后，Agent 必须自己周期性调用 `get_hunyuan_data_export_task` 直到拿到终态**，**严禁**告知用户"过几分钟告诉我再查一次"或类似让用户手动催的话术。

- 轮询工具：`get_hunyuan_data_export_task(task_id=step1_task_id, wsid=wsid)`。
- **间隔**：30 秒；**最大次数**：90 次（约 45 分钟）；这是 Agent 自身循环的节奏，**由 Agent 自动执行**。
- 状态分支：
  - `SUCCEEDED` → 🧾 输出恢复清单（`current_phase = "step1_done"`）→ **立即自动调起 Step 2**，**无须**询问用户"要不要继续"。
  - `FAILED` → **立即**调用 `get_hunyuan_data_export_task_log(task_id=step1_task_id)` 取到原始错误日志，把 `task_id` + 日志内容**原样**透传给用户；🧾 输出恢复清单（`current_phase = "failed"`）；**直接结束 pipeline**，**严禁**自动重试、**严禁**改走备用路径、**严禁**跳到 Step 2。
  - `RUNNING` / `PENDING` → 按 30s 间隔继续等，**Agent 内部循环**，不向用户索要"再查一次"。
- 超过最大次数 → 🧾 输出恢复清单（保留 `current_phase = "step1_polling"`），把当前最新状态、`task_id` 与日志信息透传给用户。

> 如果 Step 0 决定**跳过** Step 1（`src_in_nj == True`），则在内部直接把 `nj_input_path = source_path`，🧾 输出恢复清单（`current_phase = "step1_done"`，`step1_task_id = null`，备注"skipped"），进入 Step 2。

---

## 六、Step 2：南京 SFT 转 bin（用 `create_hunyuan_data_sft_conversion`）

**功能**：在南京地域内做 SFT 转 bin（`nj_input_path` → `nj_storage_path`）。**直接调用底层 `create_hunyuan_data_sft_conversion`**（详见 `sft_conversion_api.md`），返回 `task_id` 后由 Agent 用 `get_hunyuan_data_sft_conversion` 轮询。

**前置条件**：
1. Step 1 已 SUCCEEDED（或被合法跳过，源就在南京）。
2. `nj_input_path` 与 `nj_storage_path` 都已落在南京（Agent 在 Step 0 已保证；这里再做一次兜底判断即可）。

### 入参映射（`create_hunyuan_data_sft_conversion`）

| `create_hunyuan_data_sft_conversion` 参数 | Pipeline Step 2 取值 |
|---|---|
| `seq_len` | 用户指定（缺省按"默认值处理流程"处理） |
| `tokenizer` | 用户指定（缺省调 `list_hunyuan_data_sft_tokenizers` 选） |
| `input_path` | Step 2 的输入：<br>① 场景 2 / 4（src 非南京，已过 Step 1）→ `<nj_staging_root>/json/<staging_timestamp>`（即 `nj_json_path`）<br>② 场景 3（src 在南京，跳过 Step 1）→ 直接 = `source_path` |
| `storage_path` | Step 2 的输出：<br>① 场景 2（tgt 在南京，无需 Step 3）→ 直接 = **用户提供的 `target_path`**<br>② 场景 3 / 4（tgt 非南京，需 Step 3 拉走）→ `<nj_staging_root>/bin/<staging_timestamp>`（即 `nj_bin_path`） |
| `name` | **必须显式传** `create_by_bin_auto_transfer_ceph_skill` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_sft_conversion '{
  "seq_len": 4096,
  "tokenizer": "HY3.0_SFT_Tokenizer",
  "input_path": "/apdcephfs_jn2/share_302316223/leslizhang/staging/json/20260514153045/",
  "storage_path": "/apdcephfs_jn2/share_302316223/leslizhang/staging/bin/20260514153045/",
  "name": "create_by_bin_auto_transfer_ceph_skill"
}'
```

### 轮询规则（Agent 必须遵守，**自动驱动、严禁让用户催**）

> **Step 2 创建任务后，Agent 必须自己周期性调用 `get_hunyuan_data_sft_conversion` 直到拿到终态**。

- 轮询工具：`get_hunyuan_data_sft_conversion(task_id=step2_task_id)`。
- **间隔**：60 秒；**最大次数**：120 次（约 2 小时）；由 Agent 自动执行。
- 状态分支：
  - `SUCCEEDED` → 🧾 输出恢复清单（`current_phase = "step2_done"`）→ **立即自动调起 Step 3**（场景 3 / 4）或**直接结束并输出最终结果**（场景 2，target 已在南京），**无须**询问用户"要不要继续"。
  - `FAILED` → 把返回的 `message` 原始错误详情**原样**透传给用户；🧾 输出恢复清单（`current_phase = "failed"`）；**直接结束 pipeline**，**严禁**自动发起 `retry_hunyuan_data_sft_conversion`、**严禁**询问用户"是否要重试"、**严禁**切换路径或改走其他工具。
  - `STOPPED` → 视为终态失败，🧾 输出恢复清单（`current_phase = "failed"`），停止 pipeline。
  - `PENDING` / `RUNNING` → 按 60s 间隔继续等，**Agent 内部循环**。
- 超过最大次数 → 🧾 输出恢复清单（保留 `current_phase = "step2_polling"`），透传最新状态给用户。

---

## 七、Step 3：南京 staging /bin/<staging_timestamp> → 目标 ceph（用 `create_hunyuan_data_export_task`）

**功能**：把南京 `<nj_staging_root>/bin/<staging_timestamp>` 的产出搬到用户的 `target_path`（非南京目标）。**直接调用底层 `create_hunyuan_data_export_task`**。

**前置条件**：
1. Step 2 已 SUCCEEDED（且场景为 3 或 4，即 `tgt_in_nj == False`，Step 2 输出落在了 `<nj_staging_root>/bin/<staging_timestamp>`）。
2. `target_path` 不在南京（场景 2 不需要 Step 3，因为 Step 2 已直接把 bin 写到 `target_path`）。

### 入参映射（`create_hunyuan_data_export_task`）

| `create_hunyuan_data_export_task` 参数 | Pipeline Step 3 取值 |
|---|---|
| `wsid` | 用户工作空间 ID（不能为 0） |
| `data_source` | 固定 `TAIJI_WEB` |
| `source_path` | `<nj_staging_root>/bin/<staging_timestamp>`（即 `nj_bin_path`，= Step 2 的 `storage_path`） |
| `target_path` | 用户提供的最终目标 ceph 路径（非南京） |
| `name` | **必须显式传** `create_by_bin_auto_transfer_ceph_skill` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_export_task '{
  "wsid": 10103,
  "data_source": "TAIJI_WEB",
  "source_path": "/apdcephfs_jn2/share_302316223/leslizhang/staging/bin/20260514153045/",
  "target_path": "/apdcephfs_zwfy5/share/leslizhang/sft_bin_20260514/",
  "name": "create_by_bin_auto_transfer_ceph_skill"
}'
```

### 轮询规则（Agent 必须遵守，**自动驱动、严禁让用户催**）

- 轮询工具：`get_hunyuan_data_export_task(task_id=step3_task_id, wsid=wsid)`。
- **间隔**：30 秒；**最大次数**：90 次（约 45 分钟）；由 Agent 自动执行。
- 状态分支：
  - `SUCCEEDED` → 🧾 输出恢复清单（`current_phase = "done"`），pipeline 完成 ✅，**立即自动输出最终结果**，**无须**等待用户回复。
  - `FAILED` → **立即**调用 `get_hunyuan_data_export_task_log(task_id=step3_task_id)` 取到原始错误日志，把 `task_id` + 日志内容**原样**透传给用户；🧾 输出恢复清单（`current_phase = "failed"`）；**直接结束 pipeline**，**严禁**自动重试、**严禁**改走备用路径。
  - 其他（`RUNNING` / `PENDING`） → 按 30s 间隔继续等，**Agent 内部循环**。
- 超过最大次数 → 🧾 输出当前态恢复清单 + 把任务 ID / 最新状态透传给用户。

> 如果 Step 0 决定**跳过** Step 3（`tgt_in_nj == True`），则 🧾 输出恢复清单（`current_phase = "done"`，`step3_task_id = null`，备注"skipped"）后直接结束（bin 已在 `target_path`）。

---

## 八、🧾 BIN Pipeline 恢复清单（中断恢复关键产物，必须输出）

> **本模块没有跨会话自动恢复能力**：LLM 一旦关闭，TodoWrite 状态、内存里的 `task_id` / `staging_timestamp` / 参数都会丢失。但**后端 datax 任务和 ceph staging 数据是独立异步执行的**，关 LLM 不会让它们消失。因此，**只要用户手里始终握有一份"恢复清单"**，下次重新打开 LLM 时把清单贴回来，Agent 就能用 `get_hunyuan_data_export_task` / `get_hunyuan_data_sft_conversion` 接着轮询、推进剩余步骤。**这是本模块跨会话恢复的唯一凭据。**

### 清单格式（机器可读 + 人可读）

每次输出恢复清单都使用**完全一致的标题与代码块格式**，方便用户在历史中检索 / 复制：

````markdown
## 🧾 BIN Pipeline 恢复清单（请保留以备中断恢复）

```json
{
  "skill": "posttrain-bin-auto-transfer",
  "scenario": 1 | 2 | 3 | 4,
  "staging_timestamp": "<YYYYMMDDHHmmss>",
  "wsid": <wsid>,
  "source_path": "<source_path>",
  "target_path": "<target_path>",
  "nj_staging_root": "<nj_staging_root>",
  "nj_staging_root_origin": "user_provided | platform_default",
  "seq_len": 4096,
  "seq_len_origin": "user_provided | default",
  "tokenizer": "<tokenizer>",
  "tokenizer_origin": "user_provided | default",
  "nj_json_path": "<nj_staging_root>/json/<staging_timestamp>",
  "nj_bin_path": "<nj_staging_root>/bin/<staging_timestamp>",
  "skip_step1": false,
  "skip_step3": false,
  "step1_task_id": null,
  "step1_status": null,
  "step2_task_id": null,
  "step2_status": null,
  "step3_task_id": null,
  "step3_status": null,
  "current_phase": "step0_planned | step1_polling | step1_done | step2_polling | step2_done | step3_polling | done | failed | aborted",
  "updated_at": "<ISO8601 时间戳>"
}
```

> 💡 **若 LLM 中途被关闭**：下次重新打开 LLM，直接把上面这段 JSON 贴回去，并说"继续这条 bin pipeline"，Agent 会按 JSON 中的 `current_phase` 与 `*_task_id` 接着轮询 / 推进剩余步骤；不会重复创建任务。
````

### 输出时机（任何一处都不能漏）

| 触发点 | 必须输出恢复清单 | `current_phase` 取值 |
|---|---|---|
| Step 0 完成（参数已确定、TodoWrite 已写好） | ✅ | `step0_planned` |
| Step 1 创建任务后（拿到 `step1_task_id`） | ✅ | `step1_polling` |
| Step 1 SUCCEEDED / 跳过 | ✅ | `step1_done` |
| Step 2 创建任务后（拿到 `step2_task_id`） | ✅ | `step2_polling` |
| Step 2 SUCCEEDED | ✅ | `step2_done` |
| Step 3 创建任务后（拿到 `step3_task_id`） | ✅ | `step3_polling` |
| Step 3 SUCCEEDED / 跳过 | ✅ | `done` |
| **任何步骤 FAILED 或轮询超时** | ✅ | `failed`（或保留为对应 polling 阶段，并在备注中说明） |
| **用户主动放弃 / pipeline 异常终止** | ✅ | `aborted` |

> **每次输出都必须是"完整的清单"**（不要只贴 diff），便于用户复制保存。每次输出都更新 `updated_at`、`current_phase` 与已知的 `*_task_id` / `*_status`。

### 中断恢复流程（用户重新打开 LLM 时）

1. 用户把上次保留的 **🧾 BIN Pipeline 恢复清单** JSON 贴给 Agent。
2. Agent 读取 JSON，然后：
   - 校验关键字段：`scenario`、`staging_timestamp`、`wsid`、`nj_staging_root`、`source_path`、`target_path`。
   - **不要重新创建任务**！按 `current_phase` 决定从哪一步接续：
     - `step0_planned` → 没有任何 task_id，从 Step 1 / Step 2 重新开始（按 skip 标志判断）。
     - `step1_polling` → 直接 `get_hunyuan_data_export_task(step1_task_id, wsid)` 接着轮询。
     - `step1_done` → 跳过 Step 1，进入 Step 2。
     - `step2_polling` → 直接 `get_hunyuan_data_sft_conversion(step2_task_id)` 接着轮询。
     - `step2_done` → 进入 Step 3（或结束，若 `skip_step3=true`）。
     - `step3_polling` → 直接 `get_hunyuan_data_export_task(step3_task_id, wsid)` 接着轮询。
     - `done` → 直接告知用户已完成，最终 bin 在 `target_path`。
     - `failed` / `aborted` → 把上次失败 / 中止信息复述给用户，问是否要重试 / 续跑。
3. 恢复期间同样按"每个关键节点输出恢复清单"的规则继续打印，保证用户**任何时刻都能再次复制最新清单**。

> ⚠️ 如果用户没有保留清单、也记不起 `staging_timestamp` / `task_id` —— 引导用户去太极平台 Web "导出任务列表" / "SFT 转 bin 任务列表" 中按任务名称 `create_by_bin_auto_transfer_ceph_skill` + 时间筛选，拿到 task_id 后再回到本模块接续。

---

## 九、Pipeline 执行步骤（Agent 严格按顺序执行）

整个 pipeline 严格按照 **TodoWrite** 维护一份任务列表，每完成一项立刻更新进度。

### Step 0：前置识别与路径规划

> 🆕 如果用户在开场就贴了一份"🧾 BIN Pipeline 恢复清单"JSON，**先走"中断恢复流程"**（见上），不要从零跑 Step 0。

1. **解决缺省参数**（不含 `nj_staging_root`，它是按需触发）：
   - `seq_len` 缺失：**必须先**向用户列出可选项 `4096 / 8192 / 32768` 让其挑选；用户明确说"用默认"才回落到 4096。
   - `tokenizer` 缺失：**必须先**调 `list_hunyuan_data_sft_tokenizers` 拿全列表后**完整列给用户挑选**；用户明确说"用默认"才回落到列表第一个。
   - 注意：**此时不要主动询问 `nj_staging_root`**，留到地域识别之后按需触发。
2. **Agent 内联按"路径地域识别算法"判断** `source_path` / `target_path`，得到 `src_in_nj` / `tgt_in_nj`（不调工具）。
3. **校验 `source_path` 是单个文件（输入必须是文件的硬约束）**：`source_path` **必须是单个数据文件、不能是目录**。若用户给的 `source_path` 明显是目录，Agent 必须先向用户索取该目录下的**具体文件路径**，未拿到文件路径前**禁止**启动任何 Pipeline 步骤。
4. **按 (src_in_nj, tgt_in_nj) 组合识别"四种场景"**（决定后续步骤的具体路径派生），见上文"4 种场景"表。
5. **判断是否需要 `nj_staging_root`**（场景 2 / 3 / 4 都需要）：
   - 用户已显式提供 → 校验路径以 `/apdcephfs_nj` 或 `/apdcephfs_jn` 开头后用之。
   - 用户未提供 → 按"默认值处理流程"询问"用平台默认（带宽慢）还是自己提供（推荐）"→ 处理用户选择 → 取到合法的 `nj_staging_root`；用户拒绝或默认未命中且用户也不打算提供 → 直接结束 skill。
6. **生成 `staging_timestamp`**（场景 2 / 3 / 4 都需要）：用**当前时间**生成 `YYYYMMDDHHmmss`；整条 pipeline **共用同一份**，Step 1 / 2 / 3 全部沿用，**严禁**每步重新生成。
7. **派生本次 pipeline 的 staging 子目录**（基于 `nj_staging_root` + `staging_timestamp` 静默拼接，不暴露给用户）：
   - `nj_json_path = <nj_staging_root>/json/<staging_timestamp>`（Step 1 落地路径 / Step 2 输入路径）
   - `nj_bin_path  = <nj_staging_root>/bin/<staging_timestamp>`（Step 2 输出路径 / Step 3 源路径）
8. **按场景具体派生 Step 1/2/3 的入参**：

   | 场景 | Step 1 | Step 2 | Step 3 |
   |---|---|---|---|
   | 场景 2（src 非南京 / tgt 南京） | `source_path → nj_json_path` | `nj_input_path = nj_json_path`，`nj_storage_path = target_path`（**直接落到用户 target**，不写 `nj_bin_path`） | — 跳过 |
   | 场景 3（src 南京 / tgt 非南京） | — 跳过 | `nj_input_path = source_path`，`nj_storage_path = nj_bin_path` | `nj_bin_path → target_path` |
   | 场景 4（两端非南京） | `source_path → nj_json_path` | `nj_input_path = nj_json_path`，`nj_storage_path = nj_bin_path` | `nj_bin_path → target_path` |

9. **TodoWrite** 写出此次 pipeline 的待办列表（按场景动态包含被执行的 Step 1 / Step 2 / Step 3 三项中实际要跑的子集）。
10. **显式确认**：列出本次场景编号、每一步要跑 / 跳过的状态、派生出的 `staging_timestamp` / `nj_json_path` / `nj_bin_path`（如有）、本次使用的 `seq_len` / `tokenizer` / `nj_staging_root` 的来源（用户提供 vs 平台默认 + 已警告带宽风险）、估算的中转开销，等用户确认后再继续；如果用户立刻就要执行可以跳过该确认。
11. 🧾 **输出恢复清单**（`current_phase = "step0_planned"`，`scenario` 填本次场景编号，task_id 字段全为 `null`），并在末尾提示用户："**请保留这份清单。如果中途 LLM 关闭，下次贴回 JSON 即可恢复。**"

### Step 1：源 → 南京 staging（场景 2 / 4 执行；场景 3 跳过）

详见上文"五、Step 1"。

### Step 2：南京 SFT 转 bin（场景 2 / 3 / 4 都执行）

详见上文"六、Step 2"。

### Step 3：南京 staging → 目标 ceph（场景 3 / 4 执行；场景 2 跳过）

详见上文"七、Step 3"。

### 最终输出（成功 / 失败都要走）

最终给用户的总结消息要包含：
1. 是否成功 ✅ / ❌。
2. 每一步的 `task_id` 与终态（Step 1 / Step 2 / Step 3 各执行 / 跳过）。
3. **最终 bin 文件落地路径**：直接展示用户原始的 `target_path` 原值，**严禁**在后面追加 `bin_data/<tokenizer>_<seq_len>/` 等子路径。
4. 用到的关键参数（seq_len、tokenizer、staging 目录）。
5. 如失败：错误日志摘要 + 排查建议。
6. **🧾 最终恢复清单**（`current_phase` ∈ `done` / `failed` / `aborted`）——成功也照样输出。

输出格式建议（Markdown）：

```markdown
# 跨地域 SFT-bin Pipeline 执行结果

- 状态: ✅ SUCCEEDED / ❌ FAILED
- 用时: 约 X 分钟
- 最终 bin 路径: `<target_path>`

## 步骤明细
| 步骤 | 任务 ID | 状态 | 备注 |
|---|---|---|---|
| Step 1 (export → 南京) | <id 或 -> | SUCCEEDED / SKIPPED | ... |
| Step 2 (SFT 转 bin)   | <id>     | SUCCEEDED / FAILED  | ... |
| Step 3 (export → 目标) | <id 或 -> | SUCCEEDED / SKIPPED | ... |

## 关键参数
- seq_len = ...   <!-- 标注是用户指定还是默认值 4096 -->
- tokenizer = ... <!-- 标注是用户指定还是 list_hunyuan_data_sft_tokenizers 默认 -->
- staging 根 = ... <!-- 标注是用户提供还是来自平台默认（带宽风险） -->
- staging_timestamp = ... <!-- 本次 pipeline 在 Step 0 生成的时间戳 YYYYMMDDHHmmss -->
- 场景 = ... <!-- 1 / 2 / 3 / 4 -->
```

---

## 十、失败处理策略

| 阶段 | 处理 |
|---|---|
| 单步 `create_hunyuan_data_sft_conversion` FAILED | 把返回的 `message` 错误详情**原样**透传；**直接结束 skill**，**严禁**自动调用 `retry_hunyuan_data_sft_conversion`、**严禁**询问"是否要重试" |
| Step 1 FAILED | 立即调 `get_hunyuan_data_export_task_log` 取日志；把 `task_id` + 日志**原样**透传；🧾 输出失败恢复清单；**直接结束 pipeline**，**严禁**自动重跑、**严禁**切换路径 |
| Step 2 FAILED | 把返回的 `message` **原样**透传；🧾 输出失败恢复清单；**直接结束 pipeline**，**严禁**自动发起 `retry_hunyuan_data_sft_conversion` |
| Step 3 FAILED | 立即调 `get_hunyuan_data_export_task_log` 取日志；把 `task_id` + 日志**原样**透传；🧾 输出失败恢复清单；**直接结束 pipeline**，**严禁**自动重跑 |
| 任意步骤 SUCCEEDED 但下一步前置条件丢失 | 显式停下来报错，不要硬塞下一步 |
| 任何 401 / 403 错误 | 透传错误，提示用户检查是否任务 owner |

**重要原则：**
- ❌ 不要在内存里同时跑多条 pipeline（本模块一次只跑一条）。
- ❌ 不要在前一阶段没 SUCCEEDED 时启动下一阶段。
- ❌ 不要静默吞掉 FAILED；任何失败都必须把 `task_id` + 错误信息原样展示。
- ❌ **⛔ 失败即抛出原因并结束，不换策略**：任何一步 FAILED → 透传原始错误 + 🧾 输出失败恢复清单 → **直接结束 pipeline / skill**。**严禁**自动发起 `retry_hunyuan_data_sft_conversion`、**严禁**询问"是否要重试"、**严禁**主动建议"改走 pipeline / 换路径"等回退策略。
- ❌ **`storage_path` / `target_path` 是硬必填、无默认**：Agent 严禁自行构造；用户未提供 → 直接向用户索取，收到答复前不得启动任何工具。
- ❌ **`nj_staging_root` 是按需选填**：仅在 Agent 内联地域识别判定任一端非南京时才询问；询问后若用户拒绝提供、又不同意用平台默认常量，直接结束 skill。
- ❌ **输入必须是单个文件、不能是目录**：无论单步转 bin（`input_path`）还是跨地域 Pipeline（`source_path`），若用户给的是目录，必须先向用户索取具体文件路径。
- ❌ **恢复时禁止重复创建任务**：用户贴回恢复清单后，应基于 `*_task_id` 直接走 `get_hunyuan_data_export_task` / `get_hunyuan_data_sft_conversion` 路径。
- ✅ 每一步完成（包括跳过）后立即调用 `TodoWrite` 把对应 todo 设为 done。
- ✅ **每个关键节点（创建任务后、SUCCEEDED、FAILED、超时、跳过）都必须打印一份完整的 🧾 恢复清单**，并显式提醒用户保留。
