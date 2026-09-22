# Qwen2.5-Coder-1.5B-Unensored-DPO (via GitHub Codespaces)

Run the gated model **[timhoek/Qwen2.5-Coder-1.5B-Unensored-DPO](https://huggingface.co/timhoek/Qwen2.5-Coder-1.5B-Unensored-DPO)**
in the cloud with **GitHub Codespaces** — no local GPU or powerful PC needed.

## Prerequisites

1. A GitHub account (free).
2. A Hugging Face account **with access to this gated model**
   (request access on the model page, and confirm the owner approved you).
3. A Hugging Face **access token** (read): https://huggingface.co/settings/tokens

## How to run (cloud, ~5 min)

### 1. Start GitHub Codespaces
1. Open this repository on github.com.
2. Click the green **Code** button → **Codespaces** → **Create codespace on main**.
3. Wait ~2 minutes for the container to build (it auto-installs dependencies).

### 2. Give the Codespace your Hugging Face token
The model is **gated**, so the cloud machine needs your token. Safest way — a secret:

1. On github.com repo: **Settings → Secrets and variables → Codespaces** → **New repository secret**.
2. Name: `HF_TOKEN`, value: `hf_...` (your token).
3. Rebuild/restart the Codespace so the secret is picked up.

Alternative (quick but less secure): open a terminal inside the Codespace and run:
```bash
echo "HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx" > .env
```

### 3. Download the model
```bash
python download_model.py
```

### 4. Chat
```bash
python chat.py
```

Commands: type `/quit` to exit, `/reset` to clear conversation history.

## Tip: keep the model downloaded across sessions
By default the model stays in `./models` inside the Codespace (kept for a few days).
To turn it into a persistent Codespace secret so it never re-downloads, that is not
necessary — a fresh `python download_model.py` will pull it again in a couple of minutes.

## Files
| File | Purpose |
|------|---------|
| `download_model.py` | Downloads the gated model into `./models` |
| `chat.py` | Interactive chat CLI |
| `.devcontainer.json` | Codespaces container config |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for `HF_TOKEN` |

## Local use (optional)
If you ever want to run on your own machine, a decent CPU with ~8 GB free RAM is
enough for this 1.5B model:
```bash
pip install -r requirements.txt
echo "HF_TOKEN=hf_..." > .env
python download_model.py
python chat.py
```