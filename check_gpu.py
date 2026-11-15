"""Confirms that PyTorch sees the GPU inside WSL2 and that BGE-M3 can run on it."""
import platform
import time

import torch

print(f"platform      : {platform.machine()} / {platform.platform()}")
print(f"torch         : {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA device. Causes: (1) CPU-only torch wheel installed, re-run "
        "pip install torch --index-url https://download.pytorch.org/whl/cu130 ; "
        "(2) Windows NVIDIA driver too old; (3) a Linux NVIDIA driver was installed "
        "inside WSL and broke the stub, remove it."
    )
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"gpu           : {name}, {total:.1f} GB")

t0 = time.time()
a = torch.randn(4096, 4096, device="cuda")
b = a @ a
torch.cuda.synchronize()
print(f"matmul 4096^2 : {time.time() - t0:.3f} s")

try:
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    m = SentenceTransformer("BAAI/bge-m3", device="cuda")
    v = m.encode(["PUCCH carries HARQ-ACK, SR and CSI in 5G NR."] * 64, batch_size=32)
    print(f"bge-m3        : {v.shape} in {time.time() - t0:.1f} s (includes model download on first run)")
except Exception as e:  # noqa: BLE001
    print(f"bge-m3 check skipped: {e}")
