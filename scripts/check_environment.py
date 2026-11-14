"""Report the environment without downloading a model or requiring ML imports."""

import importlib.metadata
import platform


def main():
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    for package in ("torch", "transformers", "datasets", "tokenizers",
                    "safetensors", "huggingface-hub", "jupyterlab", "ipykernel"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        print(f"{package}: {version}")
    try:
        import torch
    except (ImportError, OSError) as exc:
        print(f"PyTorch unavailable: {exc}")
        return
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
