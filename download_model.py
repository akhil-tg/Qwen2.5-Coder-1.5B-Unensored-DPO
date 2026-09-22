"""
Download the public GGUF model
  mradermacher/Qwen2.5-Coder-1.5B-Unsensored-DPO-i1-GGUF
into ./models.

The model is public (no approval needed). A HF_TOKEN is optional
and only used if you have one set.
"""
import os

from env_util import load_dotenv
from huggingface_hub import hf_hub_download

REPO_ID = "mradermacher/Qwen2.5-Coder-1.5B-Unsensored-DPO-i1-GGUF"
FILENAME = "Qwen2.5-Coder-1.5B-Unsensored-DPO.i1-Q4_K_M.gguf"

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN") or None

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading {REPO_ID} ({FILENAME}) -> {MODEL_DIR}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=token,
        local_dir=MODEL_DIR,
    )
    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()