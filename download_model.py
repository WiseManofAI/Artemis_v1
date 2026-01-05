import os
import requests

MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "tinyllama.gguf")

def download():
    if os.path.exists(MODEL_PATH):
        print("Model already exists.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading LLM (one-time)...")

    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Download complete.")

if __name__ == "__main__":
    download()
