"""
Download the gated model timhoek/Qwen2.5-Coder-1.5B-Unensored-DPO into ./models.

Requires:
  - A Hugging Face access token (READ) with access granted to the model
  - The token provided as:
      an environment variable  -> HF_TOKEN
      or a .env file           -> HF_TOKEN=...
"""
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

MODEL_ID = "timhoek/Qwen2.5-Coder-1.5B-Unensored-DPO"


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "No HF_TOKEN found.\n"
            "Set it as an environment variable or create a .env file:\n"
            "  HF_TOKEN=hf_xxxx\n"
            "\nGet a token at https://huggingface.co/settings/tokens"
        )

    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {MODEL_ID} -> {local_dir}")
    snapshot_download(
        repo_id=MODEL_ID,
        token=token,
        local_dir=local_dir,
    )
    print("Download complete.")


if __name__ == "__main__":
    main()