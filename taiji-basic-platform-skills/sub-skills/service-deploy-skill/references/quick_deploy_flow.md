## 快速部署

复制已有服务做"骨架"，只换模型和资源参数创建新服务。流程：搜模型 → 查模型地域 → 定应用组 → 交叉匹配 → 模板+骨架 → 确认部署。**严禁**并行执行任何步骤，**严禁**跳步执行。

> 🚨 **交叉匹配是部署的核心前提，不可跳过或自行决定卡型！** 部署前**必须**先执行 Step 4 交叉匹配（模型地域 ∩ 应用组地域 × 所有卡型），按空余降序生成候选列表后严格串行遍历。**严禁**凭经验猜测卡型、**严禁**跳过交叉匹配直接指定 GPU、**严禁**按卡型总计聚合排序。
> 💡 **部署方式**：必须使用 `deploy_from_template.py` 脚本（一键提取模板参数，自动传入全部必填字段）。

### 前置输入

| 输入 | 必填 | 说明 |
|------|------|------|
| 模型名/关键词 | 是 | 用 `search_hunyuan_models_cards` 搜索获取 mould_id |
| 应用组名称 | 是 | 用户指定；未指定则调 `query_user_app_groups` 让用户选 |
| Token 状态 | 否 | 不作为业务入参；由 `connect_mcp.py` 自动读取。若首次调用返回 `NO_TOKEN_CONFIGURED` / 401 / 403，再按 SKILL.md §1.1 处理 |
| wsid | 否 | 优先用户提供；不知道时调 `list_user_workspaces` |
| GPU 卡型 | 否 | 不提供时自动按空余降序选取 |

服务名自动生成：`{用户名}-{模型简称}-{序号}`，如 `uricornli-dsv4-001`

### 执行步骤（串行，逐步推进）

**Step 1 — 搜模型**
`search_hunyuan_models_cards(keyword=关键词)` → 得到 `mould_id`
- 搜不到就换关键词（空格敏感）

**Step 2 — 查模型地域**
`get_hunyuan_models_card_detail(**model_id**=上一步的mould_id)` → `location_infos`
- 注意：搜索返回 `mould_id`，详情接口参数名是 `model_id`，不是 `mould_id`
- **同时提取 `scene_type` 字段**，按下方映射表转为 `get_deploy_template_detail` 的 `service_scene` 参数（来源：aide `SCENE_TYPE_CHOICES` 第三个字段）：
  | scene_type | service_scene |
  |-----------|---------------|
  | `text` / `text-moe` / `text2sql` / `text-code` / `roleplay` / `translation` / `function-call` / `deepseek` | `text_to_text` |
  | `multimodal` / `multimodal-MoE` / `video_to_text` / `audio_video_image_to_text` | `multimodal` |
  | `image` / `video` / `text-3D` / `image-oteam3.6` | `text_to_image` |
  | `audio` / `audio_to_text` | `audio` |
  | `embedding` / `music` / `image-dit-1.9` / `open-source` 或其他空 service_scene | `text_to_text`（兜底） |
  - **默认兜底**：scene_type 不在表中或 service_scene 为空时，用 `text_to_text`

**Step 3 — 确定应用组（必须由用户确认，严禁自行选择）**
  - ⚠️ **硬约束：不得自行挑选应用组**。即使只有一个有权限的应用组，也必须展示给用户并等待确认
- 无权限 → 列出用户有权限的应用组供重选

**Step 4 — GPU 配额与交叉匹配**
`query_app_group_gpu_info(app_group_id=应用组)` → Markdown 表格
- 取第 5 列「空余」做交叉匹配（不是「申请中」也不是「排队」）
- **交叉匹配 = (模型地域 ∩ 应用组地域) × 该地域所有卡型**，对每个交集地域-卡型组合列出 `[卡型, 地域, 空余数]`，卡型可用性由 Step 5 模板检查验证
- **必须遍历所有交集地域的所有卡型行**（A100PRO/A800/L40/L40S/ZXC200/H20 等全部），不能只取某一种卡型
- **所有卡型-地域对混合按空余降序排列**，不得按卡型分组后每组取前几行
- ⚠️ **严禁按卡型总计聚合排序**：必须按 `[卡型, 地域, 空余数]` 三元组排序，不能先按卡型汇总再排序。例：ZXC200/sz:109 应排在 H20/sh:64 前面（109>64），即使 H20 总计更多
- 零候选提示换应用组

> 示例：应用组 GPU 表含 A100PRO/tj:71余, L40S/gy:51余, H20/sh:32余, H20/bj:32余, ZXC200/sz:109余, L40/qy:5余 ...
> → 交集地域所有卡型 → 混合降序: ZXC200/sz:109, A100PRO/tj:71, L40S/gy:51, H20/sh:32, H20/bj:32, L40/qy:5, ...

**Step 5 — 模板 + 骨架 + 部署（严格串行）**
🚫 **严禁并行！一次只处理一个卡型，A→B 完整走完才看下一个！**

对 Step 4 候选列表（空余多→少），**逐个卡型完整串行**：先验模板，再找骨架，都通过就进入 Step 6 等待用户确认后部署。一个卡型任一环节失败才换下一个：

```
对每个候选（空余多→少，串行）:
  A. get_deploy_template_detail(model_id=mould_id, wsid, gpu_name, service_scene=Step2映射值)
     → image_name 为空 → 换下一个卡型
     → image_name 非空 → 继续 B

  B. 找骨架（不限卡型，骨架 GPU 会被模板覆盖）：
     ① list_deploy_inferences(wsid, keyword=模型全名) → 有则进入 Step 6 确认
     ② 无 → list_deploy_inferences(wsid, keyword=模型简称, only_mine=true) → 有则进入 Step 6 确认
     ③ 仍无 → list_deploy_inferences(wsid, only_mine=true) → 有则进入 Step 6 确认
     → ③ 也无 → 换下一个卡型

全部卡型走完仍未部署 → 提示换应用组
```

🔴 **部署命令**（唯一方式）：
```bash
python3 scripts/deploy_from_template.py \
  --model-id <model_id> --gpu-name <卡型> --service-scene <sc> \
  --skeleton <骨架服务名> --new-name <新服务名> \
  --wsid <wsid> --app-group-id <应用组 ID> --location <地域>
```

**Step 6 — 展示配置摘要，等待用户确认（🔴 强制，不可跳过）**
将以下完整配置展示给用户，**用户明确同意后才能执行部署命令**：

| 项目 | 来源 |
|------|------|
| 模型名 + mould_id | Step 1 |
| GPU 卡型 + 地域 | Step 4 |
| 应用组 | Step 3 |
| 模板参数（host_gpu_num/host_num/pp/tp/镜像/框架） | Step 5-A |
| 骨架服务名 | Step 5-B |
| 新服务名 | 自动生成或用户指定 |

🚫 **严禁**在用户确认前执行部署。

**硬约束（违反即错）：**
1. **严格按空余降序遍历**，不得跳过或调整顺序
2. 🚫 **禁止并行查询多卡型**：一次只能处理一个卡型，等 A+B 全部完成（无论成败）才能处理下一个
3. 同一卡型模板+骨架都通过才能部署，一环失败就换下一个
4. 部署失败换骨架不换卡型，骨架全换完才换下一个卡型
5. **⭐ 必须使用 `deploy_from_template.py` 部署（最高优先级）**：模板参数的「全量原封不动透传」由脚本自动完成。**严禁**绕过脚本自己拼 `clone_deploy_inference` 调用、**严禁**用骨架字段替换模板字段、**严禁**因部署失败而修改任何模板字段。失败排障请走 Step 6 失败处理 SOP（换骨架/换卡型），不是改字段。
6. **🔴 部署前必须展示配置摘要并等待用户确认（不可跳过）**：参考上方 Step 6 的配置摘要表格，**严禁**在用户明确确认前执行部署命令。

**部署失败处理（Step 6 之后）：**
- **模型无权限**（`无权限 / 编辑模型设置分享人`）→ 报告错误信息（含模型管理员 RTX），列出其他同系列可能有权限的模型供用户选择
- **名称冲突** → 自动追加序号重试（如 `xxx-001` → `xxx-002`）
- **脚本输出 `❌ get_deploy_template_detail 失败: ... 无推理模板`** → 按 Step 5 SOP 换下一个候选卡型，不要换模型
- **脚本输出 `❌ clone_deploy_inference 失败: ...` 含 `hint`** → 把 hint 内容透传给用户（如「请通过 Web 界面操作」等指引）
- **脚本输出含 `NO_TOKEN_CONFIGURED / 401 / 403`** → 走 SKILL.md §1.1 token 配置流程，配置后最多重试 1 次

### clone_deploy_inference 参数映射

| 含义 | 来源 | **参数名** |
|------|------|-----------|
| 骨架服务名 | `list_deploy_inferences` → **`name`** 字段（不带 `_AIDE` 后缀） | `source_inference_name` |
| 新服务名 | 自动生成或用户指定 | `target_inference_name` |
| 模型 ID | `search_hunyuan_models_cards` → `mould_id` | `model_ids`（数组 `[185120]`） |
| 应用组 ID | `query_app_group_detail` → `id`（或应用组 ID 字段） | `app_group_id`（字符串） |
| 工作空间 | 用户 | `wsid`(int) |
| 部署地域 | Step 4 交叉匹配结果 | `location`（如 `"sh"`） |
| 模型地域 | Step 2 结果 | `model_location`（如 `"gy"`） |
| GPU 卡型 | `get_deploy_template_detail` → `gpu_name` | `gpu_name`（如 `"H20"`） |
| 单机GPU卡数 | `get_deploy_template_detail` → `host_gpu_num` | `gpu_per_host` |
| GPU机器数 | `get_deploy_template_detail` → `host_num` | `host_count` |
| 流水线并行 | `get_deploy_template_detail` → `trans.config.INFERENCE_PP_SIZE` | `pipeline_parallel_size` |
| 张量并行 | `get_deploy_template_detail` → `trans.config.INFERENCE_TP_SIZE` | `tensor_parallel_size` |
| 推理框架 | `get_deploy_template_detail` → `framework_type` | `framework_type` |
| 服务场景 | Step 2 `scene_type` 映射（⚠️ 不是模板返回的默认值） | `service_scene` |
| 启动命令 | `get_deploy_template_detail` → `start_command` | `start_command` |
| 环境变量 | `get_deploy_template_detail` → `envs` | `envs`（对象，不是字符串） |
| 北极星配置复制开关 | 用户明确指示（默认关闭） | `copy_polaris_config`（bool，默认 false） |

> `list_deploy_inferences` 返回 `name`（短名）和 `service`（带 `_AIDE` 后缀）两个字段。**clone_deploy_inference 必须用 `name`**，用 `service` 会报 400。
> `copy_polaris_config` 默认 false，仅在用户明确要求「保留/复制源服务的北极星配置」时才传 true；且源服务 `polaris_token` 必须非空，否则接口返回 HTTP 400。**⚠️ 严格 bool 类型，不接受字符串。**
> 骨架服务的地域和应用组与最终部署无关——location/model_location/app_group_id 由 Step 4 决定。

### 场景示例

```
用户：帮我在 TaiJi_HYAide_HYapp_EXTRA 部署 deepseek-v4-flash，wsid=11331

AI: search_hunyuan_models_cards(keyword="flash") → mould_id=185120
AI: get_hunyuan_models_card_detail(model_id=185120) → 地域: [sz,nj,zw,bj,...]
AI: query_app_group_detail("TaiJi_HYAide_HYapp_EXTRA") → 地域: [zw,bj,...], 权限OK
AI: query_app_group_gpu_info(...) → H20/sh:32余, H20/bj:9余, A800/zw:5余
    → 交叉匹配按空余降序: [H20/sh:32, H20/bj:9, A800/zw:5]
AI: Step5: get_deploy_template_detail(model_id=185120, wsid=11331, gpu_name="H20")→✅ | get_deploy_template_detail(model_id=185120, wsid=11331, gpu_name="A800")→❌
    → 支持: H20(sh:32>bj:9)
AI: list_deploy_inferences(11331, keyword="DeepSeek-V4-Flash-平台") → 有结果 → 选为骨架

确认: DeepSeek-V4-Flash | H20×8 | 2机 | sh | HYapp_EXTRA | 名: uricornli-dsv4-001
  ↓ 用户确认
AI: python3 scripts/deploy_from_template.py \
      --model-id 185120 --gpu-name H20 --service-scene text_to_text \
      --skeleton "<骨架服务name>" --new-name uricornli-dsv4-001 \
      --wsid 11331 --app-group-id "12345" --location sh
  ↓ ✅ service_id=xxx | name=uricornli-dsv4-001
```

### 异常情况处理

| 场景 | 处理 |
|------|------|
| 模型与应用组无共同地域 | 提示："模型在[A,B]，应用组在[C,D]，无交集，请换应用组" |
| 所有卡型配额为零 | 提示换应用组或等资源释放 |
| 无匹配骨架 | Step 5 ② 两步搜不到 → 弹性池兜底 |
| clone_deploy_inference 成功但 pods 为 0 | `get_deploy_inference_detail` 查状态 → 排队则等资源，否则 `list_deploy_instances` + `get_deploy_instance_logs` 查日志 |

---
