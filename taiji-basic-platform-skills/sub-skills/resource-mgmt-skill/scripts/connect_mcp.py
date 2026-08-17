#!/usr/bin/env python3
"""
Taiji MCP Server - Self-contained client script for Skill.

Connects to Taiji MCP Server via Streamable HTTP transport
without requiring MCP registration or pip install. This is the single
entry point for the taiji-official-skills skill.

Usage:
  python connect_mcp.py list                                          # List all tools
  python connect_mcp.py call <tool_name> [args_json]                  # Call a tool
  python connect_mcp.py call <tool_name> --file <json_file>           # Call with args from file
  python connect_mcp.py call <tool_name> --stdin                      # Call with args from stdin
  python connect_mcp.py info <tool_name>                              # Show tool details
  python connect_mcp.py save-token <pat_token> [mcp_url]              # Save token to local credentials
  python connect_mcp.py --interactive                                 # Interactive mode (explicit only)
  python connect_mcp.py --pretty <command>                            # Pretty JSON for manual inspection

Environment Variables:
  TAIJI_MCP_URL     MCP Server base URL (default: http://taiji-openapi.woa.com)
  TAIJI_PAT_TOKEN   Auth token (env var > ~/.config/taiji/credentials.json)

MCP Server URL: configurable via TAIJI_MCP_URL env var
"""

import os
import sys
import json
import hashlib
import io
import time
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# Ensure stdout/stderr use UTF-8 encoding.
# When launched as a subprocess (e.g. by Claude Code / CodeBuddy Skill runner),
# sys.stdout.encoding may be None / 'ascii' / 'ANSI_X3.4-1968', which would
# otherwise mangle Chinese characters in the output.
def _force_utf8_stream(stream):
    enc = getattr(stream, "encoding", None)
    if enc is None or enc.lower() not in ("utf-8", "utf8"):
        try:
            return io.TextIOWrapper(
                stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except (AttributeError, ValueError):
            return stream
    return stream


sys.stdout = _force_utf8_stream(sys.stdout)
sys.stderr = _force_utf8_stream(sys.stderr)

# Default MCP Server configuration (base URL without path)
DEFAULT_MCP_URL = "http://taiji-openapi.woa.com"

# Local credentials file path: ~/.config/taiji/credentials.json
CREDENTIALS_PATH = Path.home() / ".config" / "taiji" / "credentials.json"


def _load_local_credentials() -> dict:
    """Load token from local credentials file (~/.config/taiji/credentials.json).

    File format:
        {
            "pat_token": "your_token_here",
            "mcp_url": "http://taiji-openapi.woa.com"  // optional
        }

    Returns:
        dict with 'pat_token' and optionally 'mcp_url', or empty dict if not found.
    """
    if not CREDENTIALS_PATH.exists():
        return {}
    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_credentials(pat_token: str, mcp_url: str = None) -> str:
    """Save token to local credentials file (~/.config/taiji/credentials.json).

    Creates directory and file with secure permissions (600).
    Preserves existing fields (e.g. mcp_url) when only updating pat_token.

    Args:
        pat_token: PAT token string to persist.
        mcp_url: Optional MCP server URL override.

    Returns:
        The absolute path of the credentials file on success.

    Raises:
        ValueError: if pat_token is empty.
        OSError: if file/directory creation fails.
    """
    if not pat_token or not pat_token.strip():
        raise ValueError("pat_token must not be empty")

    # Load existing data to preserve other fields
    existing = _load_local_credentials()

    existing["pat_token"] = pat_token.strip()
    if mcp_url:
        existing["mcp_url"] = mcp_url.strip()

    # Record save timestamp for diagnostics
    from datetime import datetime
    existing["updated_at"] = datetime.now().isoformat()

    # Ensure directory exists
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write atomically (write to same path; small file, acceptable)
    with open(CREDENTIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    # Set file permission to 600 (owner read/write only)
    os.chmod(CREDENTIALS_PATH, 0o600)

    return str(CREDENTIALS_PATH)


# =============================================================================
# Hot-reload enforcement (aligned with do_ai's CLI-layer gate)
# =============================================================================
#
# connect_mcp.py is the actual chokepoint every skill invocation flows through:
# the Agent runs `python3 connect_mcp.py call <tool>` for every MCP operation.
# SKILL.md tells the Agent to run hot_reload.py first, but that's advisory — the
# Agent may skip it. So we enforce it here in code, the same way do_ai enforces
# hot_reload in its CLI entrypoint.
#
# Design (mirrors do_ai's cheap-gate / heavy-action split):
#   1. TTL gate is inline & in-process: read the cached timestamp, and if we're
#      inside the TTL window (5 min, matching hot_reload.py) return immediately
#      — zero network, zero subprocess. This is the 99% path and adds ~microseconds.
#   2. Only when the TTL has expired/missing do we spawn hot_reload.py.
#
# Difference from do_ai: do_ai *blocks* the command when the TTL is expired and
# forces the Agent to re-run. taiji never blocks the user's task — hot_reload.py
# is already fully best-effort (silently no-ops on any network/service failure),
# so we run it and proceed regardless of the outcome. A transient network blip
# must never break a tool call.

# TTL must match hot_reload.py's TTL_HOURS (5 minutes).
_HOT_RELOAD_TTL_SECONDS = 300

# State files live outside the skill tree so a read-only / shared install can
# still record them. Must stay in sync with hot_reload.py's _state_dir().
_DEFAULT_MANAGER_HOME = "~/.taiji-skill-manager"


def _find_basic_skill_root() -> Optional[Path]:
    """Return a valid enclosing basic package, if this script is bundled in it."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (
            parent.name == "taiji-basic-platform-skills"
            and (parent / "SKILL.md").is_file()
            and (parent / "hot_reload.py").is_file()
            and (parent / "sub-skills").is_dir()
        ):
            return parent
    return None


def _find_skill_root() -> Optional[Path]:
    """Select the actual update target root.

    An enclosing basic package always wins, even though a bundled sub-skill
    also contains its own ``hot_reload.py``.  Without the explicit basic-root
    test, adding standalone reloaders would accidentally update only the child
    inside a basic installation.
    """
    basic_root = _find_basic_skill_root()
    if basic_root is not None:
        return basic_root
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hot_reload.py").is_file():
            return parent
    return None


def _local_skill_commit() -> str:
    """Read the git_commit persisted by hot_reload.py (``.skill_commit``).

    Returns '' when the skill root or the file is absent — the caller then
    simply omits the version header (best-effort, matches hot-reload's silent
    degradation on network failure).
    """
    try:
        root = _find_skill_root()
        if root is None:
            return ""
        f = root / ".skill_commit"
        if f.is_file():
            return f.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def _hot_reload_state_dir(skill_root: Path) -> Path:
    """Mirror of hot_reload.py's _state_dir() — keep both in sync."""
    home = os.environ.get("TAIJI_SKILL_MANAGER_HOME") or _DEFAULT_MANAGER_HOME
    base = Path(os.path.expandvars(os.path.expanduser(str(home))))
    key = hashlib.sha1(str(skill_root).encode("utf-8")).hexdigest()[:16]
    return base / "hot_reload" / key


def _read_ttl_stamp(path: Path) -> Optional[float]:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _hot_reload_ttl_valid(skill_root: Path) -> bool:
    """Inline TTL check, equivalent to hot_reload.py's _ttl_valid().

    Reading the cache here lets us skip the subprocess entirely on the hot path.
    Reads skill dir first (normal writable installs write there); falls back to
    manager home for read-only installs that had to write there instead.
    """
    last = _read_ttl_stamp(skill_root / ".update_cache")
    if last is None:
        last = _read_ttl_stamp(_hot_reload_state_dir(skill_root) / ".update_cache")
    if last is None:
        return False
    return (time.time() - last) < _HOT_RELOAD_TTL_SECONDS


def enforce_hot_reload() -> None:
    """Ensure skills are up to date before serving any MCP request.

    Fully best-effort: any failure (no skill root, TTL parse error, subprocess
    crash, timeout) is swallowed so the user's task always proceeds.
    """
    if os.environ.get("TAIJI_NO_HOT_RELOAD"):
        return

    try:
        skill_root = _find_skill_root()
        if skill_root is None:
            return  # Not running inside the skill tree; nothing to reload.

        # Hot path: inside TTL window → skip network + subprocess entirely.
        if _hot_reload_ttl_valid(skill_root):
            return

        # TTL expired/missing → run hot_reload.py (it self-refreshes the TTL).
        # stdout and stderr are dropped: update chatter must not pollute tool
        # output, and surface-level errors must not disrupt the user's task.
        subprocess.run(
            [sys.executable, str(skill_root / "hot_reload.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
    except Exception:
        # Never let hot-reload housekeeping break an actual tool call.
        pass


# =============================================================================
# MCPClient - Core client with Streamable HTTP transport (self-contained)
# =============================================================================


class MCPClient:
    """
    Taiji MCP Client - connects to MCP Server via Streamable HTTP transport.

    Streamable HTTP transport flow:
      1. POST /mcp with JSON-RPC payload → receive response (JSON or SSE)
      2. Each request is independent (stateless)

    Token is resolved automatically via a two-level fallback:
      1. Environment variable TAIJI_PAT_TOKEN
      2. Local credentials file ~/.config/taiji/credentials.json

    Token is passed via HTTP Header 'X-Auth-Token', NOT as a tool argument.

    Usage:
        # Token auto-resolved from env var or local credentials file
        client = MCPClient(url="http://taiji-openapi.woa.com")
        client.initialize()
        tools = client.list_tools()
        result = client.call_tool("example_tool", {"resource_id": "<resource_id>"})

        # Dynamic method dispatch
        result = client.example_tool(page=1, page_size=20)

        # Always close when done
        client.close()
    """

    def __init__(self, url: str = None):
        """
        Initialize MCPClient.

        Token resolution order (first non-empty wins):
          1. Environment variable TAIJI_PAT_TOKEN
          2. Local credentials file ~/.config/taiji/credentials.json → pat_token

        URL resolution order:
          1. Explicit `url` parameter
          2. Environment variable TAIJI_MCP_URL
          3. Local credentials file → mcp_url
          4. Default: http://taiji-openapi.woa.com

        Args:
            url: MCP Server base URL (optional, auto-resolved if not provided).
        """
        credentials = _load_local_credentials()

        # Resolve token: env var > local credentials file
        self.api_token = os.environ.get("TAIJI_PAT_TOKEN", "") or credentials.get("pat_token", "")

        # Resolve URL: explicit param > env var > local credentials file > default
        resolved_url = (
            url
            or os.environ.get("TAIJI_MCP_URL")
            or credentials.get("mcp_url")
            or DEFAULT_MCP_URL
        )
        self.base_url = resolved_url.rstrip("/")
        self.mcp_endpoint = f"{self.base_url}/mcp"
        self.request_id = 0
        self.tools_cache: Optional[list] = None
        self.last_tools_error: Optional[dict] = None
        self.initialized = False

    def _next_request_id(self) -> int:
        """Generate next request ID."""
        self.request_id += 1
        return self.request_id

    def _build_headers(self) -> dict:
        """
        Build HTTP headers for MCP requests.

        Includes Content-Type, Accept, X-Auth-Token header for tool
        authentication, platform-client-type header for access log tracking,
        and x-skill-git-commit (skill 包的 git commit 短 SHA，供 mcp_server 把 bad
        case 归属到具体版本；仅当本地已通过热更新落盘了 commit 时才携带).
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "platform-client-type": "basic_skills",
        }
        if self.api_token:
            headers["X-Auth-Token"] = self.api_token
        git_commit = _local_skill_commit()
        if git_commit:
            headers["x-skill-git-commit"] = git_commit[:12]
        return headers

    def _parse_sse_response(self, text: str) -> dict:
        """
        Parse SSE-formatted response body to extract JSON-RPC result.

        Streamable HTTP may return responses in SSE format
        (Content-Type: text/event-stream). This method extracts
        the JSON-RPC message from SSE data lines.

        Args:
            text: raw SSE response body

        Returns:
            parsed JSON-RPC response dict
        """
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    continue
        return {"error": {"code": -4, "message": "Failed to parse SSE response"}}

    def _send_request(self, method: str, params: dict = None) -> dict:
        """
        Send MCP JSON-RPC request via Streamable HTTP transport.

        Posts JSON-RPC to POST /mcp and parses the response.
        Response may be JSON (application/json) or SSE (text/event-stream).

        Uses Python standard library urllib (zero external dependencies).

        Args:
            method: MCP method name (e.g. "initialize", "tools/list", "tools/call")
            params: request parameters

        Returns:
            response data dict
        """
        if params is None:
            params = {}

        req_id = self._next_request_id()

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.mcp_endpoint,
                data=data,
                headers=self._build_headers(),
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                status_code = resp.status
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read().decode("utf-8")

            if status_code not in (200, 202):
                return {
                    "error": {
                        "code": status_code,
                        "message": f"HTTP {status_code}: {body[:500]}",
                    }
                }

            # Handle empty response (e.g. 202 Accepted for notifications)
            if not body.strip():
                return {"result": {}}

            # Parse based on Content-Type
            if "text/event-stream" in content_type:
                return self._parse_sse_response(body)
            else:
                return json.loads(body)

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            return {
                "error": {
                    "code": e.code,
                    "message": f"HTTP {e.code}: {error_body or e.reason}",
                }
            }
        except (urllib.error.URLError, OSError) as e:
            return {"error": {"code": -3, "message": f"Request failed: {e}"}}

    def _send_notification(self, method: str, params: dict = None):
        """
        Send a JSON-RPC notification (no id, no response expected).

        Args:
            method: notification method name
            params: notification parameters
        """
        if params is None:
            params = {}

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.mcp_endpoint,
                data=data,
                headers=self._build_headers(),
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except (urllib.error.URLError, OSError):
            pass  # Notifications are fire-and-forget

    def initialize(self) -> dict:
        """
        Initialize MCP connection via Streamable HTTP.

        1. Send initialize JSON-RPC request (POST /mcp)
        2. Send initialized notification

        Returns:
            initialization response dict
        """
        # Step 1: Send initialize request
        response = self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "taiji-mcp-client",
                    "version": "0.2.0",
                },
            },
        )

        if "error" not in response:
            self.initialized = True
            # Step 2: Send initialized notification
            self._send_notification("notifications/initialized", {})

        return response

    def list_tools(self, force_refresh: bool = False) -> list:
        """
        Get all available tools.

        Args:
            force_refresh: force refresh the cache

        Returns:
            list of tool definitions
        """
        if self.tools_cache and not force_refresh:
            return self.tools_cache

        response = self._send_request("tools/list", {})

        if "result" in response and "tools" in response["result"]:
            self.last_tools_error = None
            self.tools_cache = response["result"]["tools"]
            return self.tools_cache

        self.last_tools_error = response if isinstance(response, dict) else {"error": {"code": 1, "message": "工具列表响应无效"}}
        return []

    def get_tool_info(self, tool_name: str) -> Optional[dict]:
        """
        Get info for a specific tool.

        Args:
            tool_name: tool name

        Returns:
            tool info dict, or None if not found
        """
        tools = self.list_tools()
        for tool in tools:
            if tool["name"] == tool_name:
                return tool
        return None

    def call_tool(self, tool_name: str, arguments: dict = None) -> dict:
        """
        Call a specific tool.

        Token is always sent via HTTP Header (resolved at init time from
        env var or local credentials file). Do NOT pass api_token in arguments.

        If no token is configured, returns an error immediately without
        making a network request (local-first token check).

        Args:
            tool_name:  tool name
            arguments:  tool arguments

        Returns:
            tool call result
        """
        # Local token check: fail fast without network if no token configured
        if not self.api_token:
            return {
                "error": {
                    "code": 401,
                    "message": "NO_TOKEN_CONFIGURED: No PAT token found. "
                               "Checked: env TAIJI_PAT_TOKEN, "
                               f"file {CREDENTIALS_PATH}. "
                               "Please run: python3 connect_mcp.py save-token <your_token>",
                }
            }

        if arguments is None:
            arguments = {}
        else:
            arguments = dict(arguments)  # Shallow copy to avoid mutating caller's dict

        # Strip api_token from arguments if accidentally passed (silent cleanup)
        arguments.pop("api_token", None)

        return self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )

    def close(self):
        """Close client (no-op for stateless HTTP transport)."""
        pass

    def __getattr__(self, name: str):
        """
        Dynamic method dispatch - supports client.tool_name(**kwargs) syntax.

        Examples:
            client.example_tool(resource_id="<resource_id>")
            client.example_tool(page=1, page_size=20)
        """
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        def tool_caller(**kwargs):
            return self.call_tool(name, kwargs)

        return tool_caller

    def __repr__(self) -> str:
        status = "connected" if self.initialized else "not initialized"
        return f"MCPClient(url='{self.base_url}', status='{status}')"


# =============================================================================
# CLI Helpers
# =============================================================================

def parse_tool_args(argv_remaining: list) -> dict:
    """解析 CLI、文件或 stdin 中的单个 JSON 对象参数。"""
    if not argv_remaining:
        return {}

    flag = argv_remaining[0]
    try:
        if flag == "--file":
            if len(argv_remaining) != 2:
                raise ValueError("--file 需要且仅接受一个 JSON 文件路径")
            with open(argv_remaining[1], "r", encoding="utf-8") as file:
                value = json.load(file)
        elif flag == "--stdin":
            if len(argv_remaining) != 1:
                raise ValueError("--stdin 不能与其他参数同时使用")
            raw = sys.stdin.read()
            if not raw.strip():
                raise ValueError("stdin 未提供 JSON")
            value = json.loads(raw)
        else:
            value = json.loads(" ".join(argv_remaining))
    except FileNotFoundError as exc:
        raise ValueError(f"参数文件不存在: {exc.filename}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"参数 JSON 无法解析: {exc.msg}") from exc

    if not isinstance(value, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    return value


def print_tools(tools: list):
    """Format and print tool list."""
    print(f"\n📋 Available Tools ({len(tools)} total):")
    print("=" * 80)

    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")[:80]
        print(f"  • {name}")
        print(f"    {desc}")

        if "inputSchema" in tool and "properties" in tool["inputSchema"]:
            props = tool["inputSchema"]["properties"]
            required = tool["inputSchema"].get("required", [])
            if props:
                params = []
                for pname in props:
                    req_mark = "*" if pname in required else ""
                    params.append(f"{pname}{req_mark}")
                print(f"    Params: {', '.join(params)}")
        print()


def print_tool_detail(tool: dict):
    """Print detailed tool info."""
    print(f"\n📝 Tool Detail: {tool['name']}")
    print("=" * 80)
    print(f"Description: {tool.get('description', 'N/A')}")

    if "inputSchema" in tool and "properties" in tool["inputSchema"]:
        print("\nParameters:")
        props = tool["inputSchema"]["properties"]
        required = tool["inputSchema"].get("required", [])

        for pname, pinfo in props.items():
            req_mark = " (required)" if pname in required else " (optional)"
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            print(f"  • {pname}{req_mark}")
            print(f"    Type: {ptype}")
            if pdesc:
                print(f"    Desc: {pdesc}")


def print_result(result: dict):
    """Format and print tool call result."""
    print("\n📤 Result:")
    print("=" * 80)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
    elif "result" in result:
        res = result["result"]
        if "content" in res and isinstance(res["content"], list):
            for item in res["content"]:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    try:
                        data = json.loads(text)
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError:
                        print(text)

        else:
            print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


# =============================================================================
# Interactive REPL Mode
# =============================================================================

def interactive_mode(client: MCPClient):
    """Interactive REPL mode."""
    print("\n🎮 Interactive Mode (type 'help' for commands, 'quit' to exit)")
    print("=" * 80)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Bye!")
                break

            if user_input.lower() == "help":
                print(
                    """
Available commands:
  list                              - List all tools
  info <tool_name>                  - Show tool details
  call <tool_name> [args_json]      - Call a tool
  call <tool_name> --file <path>    - Call with args from file
  <tool_name> [args_json]           - Call a tool (shorthand)
  help                              - Show this help
  quit                              - Exit

Examples:
  list
  info example_tool
  call example_tool {"resource_id": "<resource_id>"}
  example_tool {"page": 1, "page_size": 20}
"""
                )
                continue

            if user_input.lower() == "list":
                tools = client.list_tools(force_refresh=True)
                print_tools(tools)
                continue

            if user_input.lower().startswith("info "):
                tool_name = user_input[5:].strip()
                tool = client.get_tool_info(tool_name)
                if tool:
                    print_tool_detail(tool)
                else:
                    print(f"❌ Tool '{tool_name}' not found")
                continue

            # Parse call command
            if user_input.lower().startswith("call "):
                remainder = user_input[5:].strip()
            else:
                remainder = user_input

            parts = remainder.split(None, 1)
            tool_name = parts[0]
            args = {}

            if len(parts) > 1:
                arg_str = parts[1].strip()
                if arg_str.startswith("--file"):
                    file_parts = arg_str.split(None, 1)
                    if len(file_parts) < 2:
                        print("❌ --file requires a JSON file path")
                        continue
                    try:
                        with open(file_parts[1].strip(), "r", encoding="utf-8") as f:
                            args = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"❌ File error: {e}")
                        continue
                else:
                    try:
                        args = json.loads(arg_str)
                    except json.JSONDecodeError as e:
                        print(f"❌ Argument JSON parse error: {e}")
                        continue

            # Validate tool exists
            tool = client.get_tool_info(tool_name)
            if not tool:
                print(
                    f"❌ Tool '{tool_name}' not found. Type 'list' to see available tools."
                )
                continue

            print(f"🔧 Calling tool: {tool_name}")
            if args:
                print(f"   Args: {json.dumps(args, ensure_ascii=False)}")

            result = client.call_tool(tool_name, args)
            print_result(result)

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def _error_payload(code: int, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _emit_json(payload: object, pretty: bool = False) -> None:
    if pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _result_payload(result: dict) -> object:
    """解包单段 JSON text 返回；其他情况完整保留 MCP 原始包络。"""
    response = result.get("result") if isinstance(result, dict) else None
    content = response.get("content") if isinstance(response, dict) else None
    if isinstance(content, list) and len(content) == 1 and content[0].get("type") == "text":
        try:
            return json.loads(content[0].get("text", ""))
        except (json.JSONDecodeError, TypeError):
            pass
    return result


def _has_result_error(result: dict) -> bool:
    if not isinstance(result, dict) or "error" in result:
        return True
    response = result.get("result")
    if not isinstance(response, dict):
        return False
    if response.get("isError"):
        return True
    for item in response.get("content", []):
        if item.get("type") != "text":
            continue
        try:
            payload = json.loads(item.get("text", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            if "error" in payload:
                return True
            # 业务接口常用 {"code": 40001, "message": "..."} 表示失败，
            # 即使 MCP transport 层本身成功，也必须返回非零退出码。
            code = payload.get("code")
            if code is not None and code != 0:
                return True
    return False


def _split_output_flags(argv: list[str]) -> tuple[list[str], bool, bool]:
    pretty = "--pretty" in argv
    json_only = "--json-only" in argv
    interactive = "--interactive" in argv
    if pretty and json_only:
        raise ValueError("--pretty 与 --json-only 不能同时使用")
    if interactive and (pretty or json_only):
        raise ValueError("--interactive 不能与 --pretty 或 --json-only 同时使用")
    flags = {"--pretty", "--json-only", "--interactive"}
    return [value for value in argv if value not in flags], pretty, interactive


def main() -> int:
    """CLI 入口：非交互命令 stdout 始终只输出一个 JSON 值。"""
    try:
        argv, pretty, interactive = _split_output_flags(sys.argv[1:])
    except ValueError as exc:
        _emit_json(_error_payload(2, str(exc)))
        return 2

    if interactive:
        if argv:
            _emit_json(_error_payload(2, "--interactive 不接受命令参数"))
            return 2
        enforce_hot_reload()
        interactive_mode(MCPClient())
        return 0
    if not argv:
        _emit_json(_error_payload(2, "请提供命令；交互模式请显式使用 --interactive"), pretty)
        return 2

    command = argv[0].lower()
    if command == "save-token":
        if len(argv) not in (2, 3):
            _emit_json(_error_payload(2, "用法: save-token <pat_token> [mcp_url]"), pretty)
            return 2
        try:
            path = save_credentials(argv[1], argv[2] if len(argv) == 3 else None)
        except (ValueError, OSError) as exc:
            _emit_json(_error_payload(1, f"保存凭证失败: {exc}"), pretty)
            return 1
        _emit_json({"ok": True, "credentials_path": path}, pretty)
        return 0

    # save-token 是凭证引导路径，不触发更新；其余实际 MCP 操作先执行门禁。
    enforce_hot_reload()

    client = MCPClient()
    try:
        if not client.api_token:
            _emit_json(
                _error_payload(
                    401,
                    "NO_TOKEN_CONFIGURED: 请通过 TAIJI_PAT_TOKEN 或 save-token 配置凭证。",
                ),
                pretty,
            )
            return 1

        initialized = client.initialize()
        if "error" in initialized:
            _emit_json(initialized, pretty)
            return 1

        tools = client.list_tools()
        if client.last_tools_error is not None:
            _emit_json(client.last_tools_error, pretty)
            return 1
        if command == "list":
            if len(argv) != 1:
                _emit_json(_error_payload(2, "list 不接受额外参数"), pretty)
                return 2
            _emit_json({"tools": tools}, pretty)
            return 0

        if command == "info":
            if len(argv) != 2:
                _emit_json(_error_payload(2, "用法: info <tool_name>"), pretty)
                return 2
            tool = client.get_tool_info(argv[1])
            if tool is None:
                _emit_json(_error_payload(404, f"工具不存在: {argv[1]}"), pretty)
                return 1
            _emit_json(tool, pretty)
            return 0

        if command == "call":
            if len(argv) < 2:
                _emit_json(_error_payload(2, "用法: call <tool_name> [args_json]"), pretty)
                return 2
            tool_name, raw_args = argv[1], argv[2:]
        else:
            tool_name, raw_args = argv[0], argv[1:]

        tool = client.get_tool_info(tool_name)
        if tool is None:
            _emit_json(_error_payload(404, f"工具不存在: {tool_name}"), pretty)
            return 1
        try:
            arguments = parse_tool_args(raw_args)
        except ValueError as exc:
            _emit_json(_error_payload(2, str(exc)), pretty)
            return 2

        result = client.call_tool(tool_name, arguments)
        _emit_json(_result_payload(result), pretty)
        return 1 if _has_result_error(result) else 0
    except (urllib.error.URLError, OSError) as exc:
        _emit_json(_error_payload(1, f"请求失败: {exc}"), pretty)
        return 1
    except Exception as exc:
        _emit_json(_error_payload(1, f"客户端异常: {exc}"), pretty)
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
