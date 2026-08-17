## query_storage_clusters


查询指定应用组的存储集群列表，包含各集群的冷文件大小统计。

### 参数

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| app_group_id | string | ✅ | 应用组标识（businessFlag） | `TaiJi_TB_storage_opt` |
| storage_type | string | ❌ | 存储类型，默认 `filestore`。`filestore`=文件存储，`objectstore`=对象存储；未显式提 COS/对象存储/桶时不要传 `objectstore` | `filestore` / `objectstore` |
| location | string | ❌ | 地域筛选，不传则查全部地域 | `sz` / `nj` / `gy` / `cq` |
| merge_hifs | boolean | ❌ | 文件存储是否合并 HIFS 集群，默认 true；用户明确说“不要算 HIFS”时传 `false` 或结果中过滤 HIFS | `true` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_storage_clusters '{"app_group_id":"TaiJi_TB_storage_opt", "storage_type":"filestore"}'
```

### 返回字段

> 接口返回 **snake_case** 字段名。

| 字段 | 说明 |
|------|------|
| cluster_name | 集群名称 |
| storage_type | 存储类型（如 ceph / hifs） |
| location | 地域 |
| total_storage_gb | 总配额（GB） |
| used_storage_gb | 已使用容量（GB） |
| total_files | 文件总数 |
| cold_size_gb | 冷文件总大小（GB），无冷热数据时为 `null` |
| container_path | 容器挂载路径 |
| tag | 存储标签 |
| allow_dir_manage | 是否允许目录管理（布尔） |
| is_security_governed | 是否已做安全治理 |

### 查询对象存储（COS）

`query_storage_clusters` 通过 `storage_type=objectstore` 即可查询应用组的**对象存储（COS）**集群信息，无需新增工具。

```bash
python3 scripts/connect_mcp.py call query_storage_clusters '{"app_group_id":"TaiJi_TB_storage_opt", "storage_type":"objectstore", "merge_hifs":true}'
```

> 💡 用户口语中的「COS / 对象存储 / 桶 / bucket」均对应 `storage_type=objectstore`。

对象存储场景下各字段含义与注意事项：

| 字段 | objectstore 场景说明 |
|------|------|
| storage_type | 对象存储类型 |
| location | 地域（如 `gy` / `nj` / `sh`） |
| used_storage_gb | 已使用容量（GB） |
| total_storage_gb | 对象存储通常为 `0`（无固定配额，非异常） |
| cold_size_gb | 对象存储通常为 `null`（无冷热分析，非异常） |
| container_path | 容器挂载路径，如 `/cos_sh1/share_xxx` |
| tag | 存储标签 |
| allow_dir_manage | 是否允许目录管理（布尔） |

> ⚠️ **对象存储字段差异提醒**：objectstore 与 filestore 不同，`total_storage_gb` 多为 `0`、`cold_size_gb` 多为 `null`，这是对象存储的正常特征，**不要误判为"无数据"或"查询失败"**，应正常展示已用容量、地域、容器路径等有效字段。

---


## query_app_group_ceph_locations


查询指定应用组的 **ceph 地域信息**：返回去重后的地域（英文+中文逗号串）+ 各集群精简明细。

> 💡 与 `query_storage_clusters` 同源但**返回更轻量**：只给地域列表 + 集群精简字段，不含冷文件分析。

### 参数

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| app_group_id | string | ✅ | 应用组标识 | `TaiJi_HYAide_GZZY` |
| storage_type | string | ❌ | 存储类型，默认 `filestore`。`filestore`=文件存储，`objectstore`=对象存储 | `filestore` / `objectstore` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_app_group_ceph_locations '{"app_group_id":"TaiJi_TB_storage_opt"}'
```

### 返回字段

> 接口返回 **snake_case** 字段名。顶层：`location`（英文地域逗号串）+ `ch_location`（中文地域逗号串）+ `clusters`（集群明细数组）。

| 字段 | 说明 |
|------|------|
| location | 去重后的**英文**地域逗号串（如 `"nj,gy"`，保持出现顺序） |
| ch_location | 去重后的**中文**地域逗号串（如 `"南京,贵阳"`，与 `location` 顺序一一对应） |
| clusters | 集群明细数组，每项字段见下 |

`clusters[]` 每项：

| 字段 | 说明 |
|------|------|
| location | 地域（英文缩写） |
| cluster_name | 集群名称 |
| container_path | 容器挂载路径 |
| total_storage_gb | 总配额（GB） |
| used_storage_gb | 已使用容量（GB） |
| storage_type | 存储类型（如 ceph） |
| tag | 存储标签 |

> 展示建议：先给地域概览（可用 `ch_location` 中文更友好，如"南京、贵阳"），再按需列 `clusters` 明细。`location` 与 `ch_location` 均为逗号分隔字符串（非数组），顺序一一对应。

---


## query_storage_dir_permission


查询指定集群下某目录的权限信息（读写用户、只读用户、配额、子目录权限）。

### 参数

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| app_group_id | string | ✅ | 应用组标识 | `TaiJi_HYAide_DopTest` |
| cluster_name | string | ✅ | 集群名称 | `jp_jn5_cephfs` |
| storage_type | string | ❌ | 存储类型，默认且仅支持 `filestore`（目录权限为文件存储专属概念，对象存储不适用） | `filestore` |
| location | string | ❌ | 地域 | `nj` |
| dir | string | ❌ | 目录层级，默认 `/` | `/hunyuan` |

### dir 参数说明

- 查询根目录（share_xxx 目录下）填 `/`
- 查询 share_xxx/somedir 下的内容填 `/somedir`
- `query_storage_dir_permission` 的目录参数名是 `dir`，不是 `path`；也不需要 `container_path`
- 目录配额/目录已用量来自返回的 `quota`、`used` 和 `dir_info_list`，用户问“各目录配额用了多少/谁有读写权限”时应使用本工具，而不是只看集群级 `total_storage_gb/used_storage_gb`

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_storage_dir_permission '{"app_group_id":"TaiJi_HYAide_DopTest", "cluster_name":"jp_jn5_cephfs", "location":"nj", "dir":"/hunyuan"}'
```

### 返回字段

### 返回字段

> 接口返回 **snake_case** 字段名。

| 字段 | 说明 |
|------|------|
| read_write_users | 读写用户列表（`all_users` 表示所有人） |
| read_only_users | 只读用户列表 |
| quota | 配额 |
| used | 已使用量 |
| dir_info_list | 子目录权限列表 |

---


## query_storage_dir_governance_detail


治理目录明细。合并目录文件列表与冷热分析结果，返回与前端存储治理页面一致的综合表格。

内部调用两个接口：
1. `queryDirFromDataServer` — 获取目录/文件列表
2. `getDirColdWarmInfo` — 获取冷热分析

### 参数

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| app_group_id | string | ✅ | 应用组标识 | `TaiJi_HYAide_DopTest` |
| cluster_name | string | ✅ | 集群名称 | `jp_jn5_cephfs` |
| container_path | string | ❌ | 容器挂载路径（从 query_storage_clusters 的 `container_path` 获取；不传时可能无法定位数据） | `/apdcephfs_jn5/share_305546123` |
| path | string | ❌ | 相对目录路径（相对于 container_path），默认根目录 `/` | `/` |
| location | string | ❌ | 地域 | `nj` |

> 💡 虽然 `container_path`、`location`、`path` 按契约为选填，但为准确定位目录数据，**强烈建议先调 `query_storage_clusters` 获取 `container_path`/`location` 后再串联调用**。
>
> ⚠️ `query_storage_dir_governance_detail` 仅用于文件存储目录明细/冷热分析，不支持 COS/objectstore 桶文件列表。用户问 COS 桶里有哪些文件时，不要用本工具反复试错，应说明当前存储治理工具不支持 COS 目录级列表。

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_storage_dir_governance_detail '{"app_group_id":"TaiJi_HYAide_DopTest", "cluster_name":"jp_jn5_cephfs", "path":"/", "location":"nj", "container_path":"/apdcephfs_jn5/share_305546123"}'
```

### 返回字段与展示列

> 接口返回 **snake_case** 字段名（每个目录/文件一条记录）。展示列与字段对应如下：

| 展示列名 | 返回字段 | 说明 |
|------|------|------|
| 目录 | name | 目录或文件名 |
| （是否目录） | is_dir | true=目录，false=文件 |
| 说明&建议 | suggestion | 冷热分析建议（如"立即归档"、"建议清理"、"活跃使用"），无冷热数据时为 `null` |
| 文件总数 | file_count | 目录下文件数量 |
| 大小 | size_bytes | 目录/文件总大小（字节） |
| 冷文件大小 | cold_file_size_bytes | 冷文件占用空间（字节），无冷热数据时为 `null` |
| 冷文件占比 | cold_ratio | 冷文件大小 / 总大小，无冷热数据时为 `null` |
| 距最后访问天数 | days_since_last_access | 距今多少天未被访问 |
| 最后访问日期 | last_access_time | 最近一次访问时间 |
| 创建日期 | ctime | 目录/文件创建时间（Unix 时间戳，秒） |

---


## apply_storage_quota


⚠️ **写操作** — 提交存储配额扩容/缩容申请单，会创建真实审批单并通知审批人。

### ⚠️ 调用前必须二次确认

调用本工具前，**必须**先向用户展示以下摘要并获得明确确认：

```
📋 存储扩容申请摘要：
- 申请人：{applicant}
- 应用组：{app_group_id}
- 集群：{cluster}
- 地域：{location}
- 当前配额 → 目标配额：{quota - quota_diff} GB → {quota} GB（变更 {quota_diff} GB）
- 审批人：{platform_auditor}
- 申请理由：{apply_reason}

确认提交？（是/否）
```

用户回复"确认"/"是"/"提交"后才可调用。

### 参数

> 必填性：仅 `app_group_id`/`applicant`/`task_type`/`apply_reason` 为必填；其余参数后端均接收，按业务场景补齐。
>
> ⚠️ **本工具仅用于文件存储的配额扩缩容**：`task_type` 固定 `extend_quota`（含扩容与缩容），`storage_type` 固定 `filestore`。新增集群、异地挂载白名单等走其他流程，不在本工具范围。

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| app_group_id | string | ✅ | 应用组标识 | `TaiJi_TB_storage_opt` |
| applicant | string | ✅ | 申请人（企业微信/RTX 账号） | `haodedu` |
| task_type | string | ✅ | 任务类型，固定 `extend_quota`（配额扩缩容） | `extend_quota` |
| apply_reason | string | ✅ | 申请理由 | `模型训练数据增长需扩容` |
| cluster | string | ❌ | 目标集群名称（扩缩容场景建议提供） | `jp_gy4_cephfs` |
| quota | integer | ❌ | 目标配额（GB），变更后总配额 | `201` |
| quota_diff | integer | ❌ | 本次增减配额（GB），正数扩容，负数缩容 | `1` / `-50` |
| location | string | ❌ | 地域 | `gy` |
| storage_type | string | ❌ | 存储类型，固定 `filestore`（本工具仅支持文件存储） | `filestore` |
| platform_auditor | string | ❌ | 平台审批人（逗号分隔多人），默认 `joefang`。**审批人有固定候选集（后端 ceph_quota admin 权限表），非任意值**，无特殊要求时用默认即可，不要自行编造 | `joefang` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call apply_storage_quota '{"applicant":"haodedu", "app_group_id":"TaiJi_TB_storage_opt", "cluster":"jp_gy4_cephfs", "task_type":"extend_quota", "platform_auditor":"joefang", "quota":201, "quota_diff":1, "apply_reason":"测试接口", "storage_type":"filestore", "location":"gy"}'
```

### 返回说明

成功时返回申请摘要和状态（pending 待审批）。审批人将收到企业微信通知。

### 审批流程状态

| 状态值 | 含义 |
|--------|------|
| pending | 待审批 |
| platform_approve | 平台已通过 |
| platform_reject | 平台已拒绝 |

---


## list_storage_dir


列出指定 **ceph 完整路径**下的文件/子目录元数据，类似 `ll` / `ls -l`。用户给出一条 ceph 挂载路径、想知道“这个路径下有哪些文件/目录、各占多大”时用本工具。

### 参数

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| path | string | ✅ | 完整 ceph 路径（含 `/apdcephfs_xxx/share_xxx` 或 `/taijifs_xxx/share_xxx` 前缀） | `/apdcephfs_jn5/share_305546123/hunyuan` |
| location | string | ❌ | 地域，不传由后端从 `path` 自动解析 | `nj` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_storage_dir '{"path":"/apdcephfs_jn5/share_305546123/hunyuan"}'
```

### 返回字段

> 接口返回 **snake_case** 字段名。顶层 `items` 为数组，每项一个文件/子目录；按 `name` 升序，**最多 2000 条**。

| 字段 | 说明 |
|------|------|
| name | 文件/目录名 |
| is_dir | true=目录，false=文件 |
| size_bytes | 大小（字节） |
| file_count | 目录下文件数（文件条目通常为 0/1） |
| atime | 最近访问时间（Unix 时间戳，秒；可能带小数） |
| mtime | 最近修改时间（Unix 时间戳，秒） |
| ctime | 创建时间（Unix 时间戳，秒） |

> 💡 展示时把 `size_bytes` 换算成 KiB/MiB/GiB、把时间戳格式化为可读日期更友好。条目达 2000 上限时提醒用户进入更深子目录 `path` 再查。

---


## get_storage_file_content


读取指定 **ceph 完整路径**的**小文件**内容（≤1MB），用于查看配置、日志、脚本等文本文件。

### 参数

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| path | string | ✅ | 完整 ceph **文件**路径 | `/apdcephfs_jn5/share_305546123/hunyuan/config.yaml` |
| location | string | ❌ | 地域，不传由后端从 `path` 自动解析 | `nj` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_storage_file_content '{"path":"/apdcephfs_jn5/share_305546123/hunyuan/config.yaml"}'
```

### 返回字段

> 接口返回 **snake_case** 字段名。内容按 **UTF-8** 解码为文本。

| 字段 | 说明 |
|------|------|
| path | 文件路径（回显） |
| size_bytes | 文件大小（字节） |
| content | 文件文本内容（UTF-8） |

> ⚠️ **1MB 上限**：文件超过 1MB 后端直接拒绝（返回 `file too large` 错误），这是防超时保护，不是故障。遇大文件应提示用户该文件过大、无法整文件读取，可改用 `list_storage_dir` 看元数据，或下钻更小的目标文件。非 UTF-8 二进制文件解码后可能是乱码，属正常现象。

---


## query_storage_cluster_free_space


查询某个 **ceph 完整路径所在集群（share 根）** 的水位：配额、已用、剩余、使用率。用户问“这个路径/这块盘还剩多少空间、水位多高、用了百分之多少”时用本工具。

### 参数

| 参数 | 类型 | 必填 | 说明 | 示例 |
|------|------|------|------|------|
| path | string | ✅ | 完整 ceph 路径（后端自动规约到 share 根 `/<prefix>/share_xxx` 计算水位） | `/apdcephfs_jn5/share_305546123/hunyuan` |
| location | string | ❌ | 地域，不传由后端从 `path` 自动解析 | `nj` |

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_storage_cluster_free_space '{"path":"/apdcephfs_jn5/share_305546123/hunyuan"}'
```

### 返回字段

> 接口返回 **snake_case** 字段名。水位按 **share 根**（`/<prefix>/share_xxx`）统计，`path` 会回显规约后的 share 路径。

| 字段 | 说明 |
|------|------|
| path | 规约到的 share 根路径（回显） |
| quota_size_bytes | 配额总量（字节） |
| used_size_bytes | 已用量（字节） |
| free_bytes | 剩余量（字节） |
| usage_ratio | 使用率（0~1 的小数；配额为 0 或缺失时为 `null`） |

> 💡 展示时把字节换算成 GiB/TiB，`usage_ratio` 乘 100 显示为百分比。`usage_ratio=null` 表示该 share 无固定配额（非异常），此时只展示已用/剩余即可。

---
