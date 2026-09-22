"""
Web chat UI for the GGUF model.

Usage (inside a Codespace):
  python web.py

Then open http://localhost:8000 in your browser.
GitHub Codespaces auto-forwards the port and shows an "Open in Browser"
notification; click it (or the Ports tab > open the 8000 port).
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from llama_cpp import Llama

REPO_ID = "mradermacher/Qwen2.5-Coder-1.5B-Unsensored-DPO-i1-GGUF"
FILENAME = "Qwen2.5-Coder-1.5B-Unsensored-DPO.i1-Q4_K_M.gguf"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, FILENAME)
PORT = 8000

SYSTEM_PROMPT = (
    "You are Qwen2.5-Coder, a helpful AI coding assistant. "
    "Answer the user's question directly and write working code."
)

MODEL_CTX = 8192
MAX_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

_lm_lock = Lock()
_llm = None

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen2.5-Coder Unsensored DPO</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --bubble:#222736; --accent:#4f8cff; --text:#e6e9ef; --muted:#9aa2b1; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  header { padding:14px 20px; background:var(--panel); border-bottom:1px solid #262b36; display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:16px; margin:0; }
  header span { color:var(--muted); font-size:12px; }
  main { max-width:860px; margin:0 auto; padding:20px; display:flex; flex-direction:column; height:calc(100vh - 110px); }
  #messages { flex:1; overflow-y:auto; padding:8px 0; }
  .msg { margin:10px 0; display:flex; flex-direction:column; }
  .msg .tag { font-size:11px; color:var(--muted); margin-bottom:4px; }
  .msg .bubble { padding:10px 14px; border-radius:12px; line-height:1.5; white-space:pre-wrap; word-break:break-word; }
  .user { align-items:flex-end; }
  .user .bubble { background:var(--accent); color:#fff; border-bottom-right-radius:3px; max-width:80%; }
  .assistant .bubble { background:var(--bubble); border:1px solid #2a3040; border-bottom-left-radius:3px; max-width:100%; }
  footer { display:flex; gap:10px; padding-top:10px; }
  textarea { flex:1; resize:none; background:var(--panel); color:var(--text); border:1px solid #2a3040; border-radius:10px; padding:10px; font:inherit; min-height:52px; max-height:140px; }
  textarea:focus { outline:none; border-color:var(--accent); }
  button { background:var(--accent); color:#fff; border:none; border-radius:10px; padding:0 18px; font:inherit; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .status { color:var(--muted); font-size:12px; padding:6px 2px; }
</style>
</head>
<body>
<header>
  <h1>Qwen2.5-Coder-1.5B-Unsensored-DPO</h1>
  <span>llama.cpp running on CPU</span>
</header>
<main>
  <div id="messages"></div>
  <div id="status" class="status"></div>
  <footer>
    <textarea id="input" rows="1" placeholder="Ask anything... (Enter to send, Shift+Enter for new line)"></textarea>
    <button id="send">Send</button>
  </footer>
</main>
<script>
const history = [];
const $msg = document.getElementById("messages");
const $status = document.getElementById("status");
const $input = document.getElementById("input");
const $send = document.getElementById("send");

function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  const tag = role === "user" ? "You" : "Model";
  div.innerHTML = '<span class="tag">' + tag + '</span><div class="bubble">' + esc(text) + '</div>';
  $msg.appendChild(div);
  $msg.scrollTop = $msg.scrollHeight;
}
async function send() {
  const text = $input.value.trim();
  if (!text) return;
  $input.value = "";
  addMsg("user", text);
  history.push({ role: "user", content: text });
  $send.disabled = true;
  $status.textContent = "Thinking…";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history })
    });
    const data = await res.json();
    const reply = data.reply || "No response.";
    history.push({ role: "assistant", content: reply });
    addMsg("assistant", reply);
  } catch (e) {
    addMsg("assistant", "Error: " + e + ". Is python web.py still running?");
  } finally {
    $status.textContent = "";
    $send.disabled = false;
    $input.focus();
  }
}
$send.addEventListener("click", send);
$input.addEventListener("keydown", function(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$input.focus();
</script>
</body>
</html>
"""


def ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print("Model not found. Downloading it first...")
    from huggingface_hub import hf_hub_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=MODEL_DIR)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _respond(self, data: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in ("", "index.html"):
            self._respond(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._respond(b"Not found", "text/plain", 404)

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._respond(b"Not found", "text/plain", 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in body.get("messages", []):
            m = m if isinstance(m, dict) else {}
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            if role in ("user", "assistant", "system") and content:
                messages.append({"role": role, "content": content})

        with _lm_lock:
            response = _llm.create_chat_completion(
                messages=messages,
                max_tokens=int(body.get("max_tokens", MAX_TOKENS)),
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        text = response["choices"][0]["message"]["content"].strip()
        self._respond(json.dumps({"reply": text}).encode("utf-8"), "application/json")


def main() -> None:
    global _llm
    ensure_model()

    print(f"Loading model: {MODEL_PATH}")
    _llm = Llama(model_path=MODEL_PATH, n_ctx=MODEL_CTX, n_gpu_layers=0, verbose=False)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Chat page running at http://0.0.0.0:{PORT}")
    print("In Codespaces: a notification 'Open in Browser' will appear, or use the Ports tab.")
    server.serve_forever()


if __name__ == "__main__":
    main()