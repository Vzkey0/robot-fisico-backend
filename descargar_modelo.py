from huggingface_hub import snapshot_download

snapshot_download("TinyLlama/TinyLlama-1.1B-Chat-v1.0", local_dir="tinyllama")
