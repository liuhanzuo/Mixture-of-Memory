#!/usr/bin/env python3
"""从 get_deploy_template_detail 取参数，原封不动传 clone_deploy_inference。

用法:
  python3 deploy_from_template.py --model-id 185120 --gpu-name H20 \
    --service-scene text_to_text --skeleton "骨架名" --new-name "新名" \
    --wsid 11331 --app-group-id "12345" --location sh

退出码:
  0  部署成功（stdout 输出 service_id / name）
  非 0  失败（stderr 输出 ❌ 错误信息 + 💡 修复提示，便于 agent 解析）
"""
import json, subprocess, sys, os

MCP = os.path.join(os.path.dirname(__file__), "connect_mcp.py")

# token 相关关键字（命中时给 agent 明确的"按 SKILL.md Step 0 配置"提示）
TOKEN_ERR_KEYS = ("NO_TOKEN_CONFIGURED", "401", "403", "invalid token", "unauthorized")


def _build_error_msg(tool, msg, hint=""):
    """拼装人类可读的错误信息（纯文本拼接，可复用/可测试，不退出）。"""
    parts = [f"❌ {tool} 失败: {msg}"]
    if hint:
        parts.append(f"💡 {hint}")
    # token 失效自动追加 Step 0 引导（如果 hint 中尚未提及）
    if any(k.lower() in msg.lower() for k in TOKEN_ERR_KEYS) and "Step 0" not in (hint or ""):
        parts.append('💡 token 缺失或失效，请按 SKILL.md Step 0 协议配置：python3 scripts/connect_mcp.py save-token "<token>"')
    return "\n".join(parts)


def _fail(tool, msg, hint=""):
    """统一失败出口：拼装错误信息 → 输出到 stderr → 非零退出。"""
    print(_build_error_msg(tool, msg, hint), file=sys.stderr)
    sys.exit(1)


def _mcp_tail(raw_output):
    """提取输出尾部最后 5 行非空内容，用于错误诊断。"""
    return "\n".join([ln for ln in raw_output.strip().splitlines() if ln.strip()][-5:])


def _mcp(tool, **kwargs):
    try:
        args_json = json.dumps(kwargs, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        _fail(tool, f"参数序列化失败: {e}", hint="请检查传入参数是否包含不可 JSON 序列化的类型")
    try:
        result = subprocess.run(
            ["python3", MCP, "call", tool, args_json],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        _fail(tool, "python3 未找到，请确认 Python 环境已安装")
    except subprocess.TimeoutExpired:
        _fail(tool, f"{tool} 调用超时（120s），请稍后重试")
    except OSError as e:
        _fail(tool, f"执行失败: {e}")
    if result.returncode != 0:
        _fail(tool, f"connect_mcp.py 返回非零退出码 ({result.returncode}): {result.stderr.strip() or result.stdout.strip()}")
    raw_output = result.stdout
    marker = raw_output.find("📤 Result:")
    if marker >= 0:
        raw_output = raw_output[marker:]
    json_start = raw_output.find("{")
    if json_start < 0:
        _fail(tool, _mcp_tail(raw_output) or "(MCP 无响应)")
    stack, start = 0, json_start
    for pos in range(json_start, len(raw_output)):
        if raw_output[pos] == "{": stack += 1
        elif raw_output[pos] == "}":
            stack -= 1
            if stack == 0:
                try:
                    return json.loads(raw_output[start:pos+1])
                except (json.JSONDecodeError, ValueError):
                    continue
    _fail(tool, f"返回内容非合法 JSON: {_mcp_tail(raw_output)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=int, required=True, help="模型 ID（v3 新参数名，原 --mould-id）")
    p.add_argument("--gpu-name", required=True)
    p.add_argument("--service-scene", required=True)
    p.add_argument("--skeleton", required=True, help="骨架服务名（source_inference_name），来自 list_deploy_inferences 返回的 name 字段")
    p.add_argument("--new-name", required=True, help="新建的目标服务名（target_inference_name）")
    p.add_argument("--wsid", type=int, required=True)
    p.add_argument("--app-group-id", required=True, help="应用组 ID（v3 新参数名，原 --queue-name 传应用组名，v3 改传应用组 ID）")
    p.add_argument("--location", required=True)
    p.add_argument("--model-location", default="", help="模型挂载位置（v3 新参数名，原 --mould-location）")
    a = p.parse_args()

    tmpl = _mcp("get_deploy_template_detail",
                model_id=a.model_id, wsid=a.wsid, gpu_name=a.gpu_name,
                service_scene=a.service_scene, inference_type="inference")
    if not tmpl.get("image_name"):
        _fail(
            "get_deploy_template_detail",
            f"model_id={a.model_id} 在 {a.gpu_name} 卡型下无推理模板（image_name 为空）",
            hint=f"按 service_deploy_api.md 快速部署 Step 5 SOP：换下一个候选卡型重试；若全部卡型都无模板，则该模型不支持模板部署",
        )

    config = tmpl.get("trans", {}).get("config", {})
    body = {
        "source_inference_name": a.skeleton,
        "target_inference_name": a.new_name,
        "wsid": a.wsid,
        "app_group_id": a.app_group_id,
        "location": a.location,
        "model_ids": [a.model_id],
        "image_name": tmpl["image_name"],
        "gpu_per_host": tmpl.get("host_gpu_num") or 0,
        "host_count": tmpl.get("host_num") or 1,
        "pipeline_parallel_size": int(config.get("INFERENCE_PP_SIZE") or 0),
        "tensor_parallel_size": int(config.get("INFERENCE_TP_SIZE") or 0),
        "framework_type": tmpl.get("framework_type", ""),
        "gpu_name": tmpl.get("gpu_name", a.gpu_name),
        "service_scene": a.service_scene,
    }
    if tmpl.get("start_command") is not None:
        body["start_command"] = tmpl["start_command"]
    if tmpl.get("envs") is not None:
        # v3: envs 从 string 类型改为 object 类型；模板返回值本身即为 object，直接透传
        body["envs"] = tmpl["envs"]
    if a.model_location:
        body["model_location"] = a.model_location

    deploy_result = _mcp("clone_deploy_inference", **body)
    if "error" in deploy_result:
        _fail(
            "clone_deploy_inference",
            deploy_result["error"],
            hint=deploy_result.get("hint", ""),
        )

    # 成功输出尽可能多的可用信息，便于 agent 直接透传给用户
    parts = [f"✅ service_id={deploy_result.get('id')}"]
    if deploy_result.get("name"):
        parts.append(f"name={deploy_result['name']}")
    for url_key in ("url", "link", "pageUrl", "page_url"):
        if deploy_result.get(url_key):
            parts.append(f"url={deploy_result[url_key]}")
            break
    print(" | ".join(parts))
