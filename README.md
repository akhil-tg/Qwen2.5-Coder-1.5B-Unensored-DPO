# Qwen2.5-Coder-1.5B-Unsensored-DPO (via GitHub Codespaces)

Run the **DPO-unjailed** coding model
**[mradermacher/Qwen2.5-Coder-1.5B-Unsensored-DPO-i1-GGUF](https://huggingface.co/mradermacher/Qwen2.5-Coder-1.5B-Unsensored-DPO-i1-GGUF)**
in the cloud with **GitHub Codespaces** — no local GPU or powerful PC needed.
The model is **public** — no approvals, no tokens required.

## What you need
Just a free **GitHub account**. Nothing else. (No Hugging Face account or token
needed — the model is public.)

## How to run (cloud, ~5 min)

### 1. Start GitHub Codespaces
1. Open this repository on github.com.
2. Click the green **Code** button → **Codespaces** → **Create codespace on main**.
3. Wait ~2 minutes for the container to build (installs prebuilt `llama-cpp-python`).

### 2. Download the model (Q4_K_M, ~1.1 GB)
```bash
python download_model.py
```

### 3. Chat
```bash
python chat.py
```

Commands: type `/quit` to exit, `/reset` to clear conversation history.
Optional: `python chat.py --max-new-tokens 1024` for longer responses.

### (Better) Web page instead of the terminal
Same model, but a real chat page in your browser:
```bash
python web.py
```
GitHub Codespaces shows an **"Open in Browser"** notification for port
`8000` — click it. (Or use the **Ports** tab → open port 8000.)
Type in the box; Enter sends, Shift+Enter makes a new line.

## Files
| File | Purpose |
|------|---------|
| `download_model.py` | Downloads the Q4_K_M GGUF into `./models` |
| `chat.py` | Interactive chat CLI (llama.cpp, runs on CPU) |
| `web.py` | Browser chat page (port 8000, Codespaces auto-forwards it) |
| `.devcontainer.json` | Codespaces container config |
| `requirements.txt` | `llama-cpp-python` + download helpers |
| `.env.example` | Optional HF token template (not required) |

## Local use (optional, runs on weak PCs too)
The GGUF is CPU-friendly. Any machine with ~4-8 GB RAM and ~2 GB free disk can
run this — this includes most ordinary laptops:
```bash
pip install -r requirements.txt
python download_model.py
python chat.py
```

## About the model
- 1.5B parameter coding model from the Qwen2.5-Coder family, DPO-trained
  to follow instructions without the usual refusals ("unsensored").
- Quantized by mradermacher with imatrix to Q4_K_M for efficient CPU inference.
- More quantizations (Q2–Q6) available on the model page if you prefer
  lower size / higher quality.