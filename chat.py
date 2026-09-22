"""
Interactive chat CLI for
  mradermacher/Qwen2.5-Coder-1.5B-Unsensored-DPO-i1-GGUF (Q4_K_M)

Usage:
  python chat.py

  - Auto-downloads the model on first run if ./models is empty.
  - Uses llama.cpp on CPU (works on small/weak machines).
  - /quit to exit, /reset to clear conversation history.
"""
import argparse
import os

from env_util import load_dotenv
from llama_cpp import Llama

REPO_ID = "mradermacher/Qwen2.5-Coder-1.5B-Unsensored-DPO-i1-GGUF"
FILENAME = "Qwen2.5-Coder-1.5B-Unsensored-DPO.i1-Q4_K_M.gguf"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, FILENAME)

SYSTEM_PROMPT = (
    "You are Qwen2.5-Coder, a helpful AI coding assistant. "
    "Answer the user's question directly and write working code."
)


def ensure_model() -> None:
    if os.path.exists(MODEL_PATH):
        return
    print("Model not found. Downloading it first...")
    from huggingface_hub import hf_hub_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=MODEL_DIR)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-size", type=int, default=8192)
    args = parser.parse_args()

    ensure_model()

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=args.context_size,
        n_gpu_layers=0,
        verbose=False,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("\nChat started (llama.cpp, CPU). Type /quit to exit, /reset to clear history.\n")

    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if user.lower() in ("/quit", "/exit"):
            break
        if user.lower() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("History cleared.\n")
            continue
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        text = response["choices"][0]["message"]["content"].strip()

        messages.append({"role": "assistant", "content": text})
        print(f"\nModel: {text}\n")


if __name__ == "__main__":
    main()