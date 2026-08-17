#!/usr/bin/env python3
"""启动 Insight Case 对比可视化服务.
用法: python3 serve_insight_compare.py --input ./cases.jsonl [--port 8765]
"""
import argparse, http.server, json, os, sys, urllib.parse

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "insight_compare.html")
DEFAULT_PORT = 8765


def load_data(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def inject_data(html, data_json):
    """将数据注入 HTML 模板的 DATA_PLACEHOLDER."""
    return html.replace("__DATA_PLACEHOLDER__", data_json)


def make_server(input_path):
    """返回一个带有预加载数据的 HTTP handler."""
    data = load_data(input_path)
    data_json = json.dumps(data, ensure_ascii=False)
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()
    html = inject_data(template, data_json)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode())
                return
            # 代理 JSONL 文件请求
            parsed = urllib.parse.urlparse(self.path)
            requested = parsed.path.lstrip("/")
            if requested and os.path.isfile(requested):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
                return
            super().do_GET()

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def log_message(self, fmt, *args):
            pass  # 静默日志

    return Handler


def main():
    p = argparse.ArgumentParser(description="启动 Insight Case 对比可视化服务")
    p.add_argument("--input", "-i", required=True, help="JSONL 数据文件")
    p.add_argument("--port", "-p", type=int, default=DEFAULT_PORT)
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] 文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)

    handler = make_server(args.input)
    server = http.server.HTTPServer(("", args.port), handler)
    print(f"✅ 服务已启动: http://localhost:{args.port}", file=sys.stderr)
    print(f"   数据来源: {args.input}", file=sys.stderr)
    sys.stderr.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
