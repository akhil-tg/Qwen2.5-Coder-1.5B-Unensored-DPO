"""
Interactive chat CLI for timhoek/Qwen2.5-Coder-1.5B-Unensored-DPO.

Usage (inside a GitHub Codespace):
  python chat.py

  - Model auto-downloads on first run if ./models is empty.
  - Uses ./models if present, otherwise pulls straight from the Hub.
  - Requires HF_TOKEN for the gated model (see .env.example).
  - Use --max-new-tokens to cap response length, /quit to exit, /reset to clear history.
"""
import argparse
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "timhoek/Qwen2.5-Coder-1.5B-Unensored-DPO"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

CHAT_TEMPLATE = "{role}: {content}\n"

SYSTEM_PROMPT = (
    "You are Qwen2.5-Coder, a helpful AI coding assistant. "
    "Answer the user's question directly and write working code."
)


def load_model(max_new_tokens_default: int):
    token_flag = os.getenv("HF_TOKEN")
    token = {"token": token_flag} if token_flag else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source = LOCAL_DIR if os.path.isdir(LOCAL_DIR) and os.listdir(LOCAL_DIR) else MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(source, **token)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        **token,
    ).to(device)

    print(f"Loaded from: {source} (device={device}, params={sum(p.numel() for p in model.parameters())/1e9:.2f}B)")
    return tokenizer, model, device, max_new_tokens_default


def build_prompt(messages: list[dict[str, str]]) -> str:
    lines = [f"[INST] {SYSTEM_PROMPT} [/INST]"]
    for msg in messages:
        lines.append(CHAT_TEMPLATE.format(**msg))
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        raise SystemExit("No HF_TOKEN found. See .env.example — this model is gated.")

    tokenizer, model, device, max_new_tokens = load_model(args.max_new_tokens)
    messages: list[dict[str, str]] = []

    print("\nChat started. Type /quit to exit, /reset to clear history.\n")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if user.lower() in ("/quit", "/exit"):
            break
        if user.lower() == "/reset":
            messages.clear()
            print("History cleared.\n")
            continue
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        prompt = build_prompt(messages)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        messages.append({"role": "assistant", "content": response})
        print(f"\nModel: {response}\n")


if __name__ == "__main__":
    main()