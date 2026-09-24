import http.server, functools, os
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()
os.chdir("/home/user/lvllm/monitoring/web")
http.server.ThreadingHTTPServer(("127.0.0.1", 8787), H).serve_forever()
