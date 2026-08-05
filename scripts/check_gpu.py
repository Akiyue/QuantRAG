"""Verify the inference backends actually reach the GPU.

    python scripts/check_gpu.py

A CPU-only llama-cpp-python installs cleanly and runs correctly - just fifty
times slower. That failure is silent, and on a 96,000-cell grid it is the
difference between an overnight run and a fortnight. Check before committing
GPU time, not after.
"""

from __future__ import annotations

import sys


def check_llamacpp() -> bool:
    print("tier A - llama-cpp-python")
    try:
        import llama_cpp
    except ImportError:
        print("  not installed")
        print("  pip install llama-cpp-python \\")
        print("    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
        return False

    print(f"  version         : {getattr(llama_cpp, '__version__', 'unknown')}")
    try:
        offload = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:  # noqa: BLE001 - old builds lack the symbol
        print(f"  gpu offload     : could not query ({exc})")
        return False

    print(f"  gpu offload     : {offload}")
    if not offload:
        print()
        print("  This build is CPU-only. It will produce correct numbers at roughly")
        print("  1/50th the speed. Reinstall with the CUDA wheel index above, or")
        print("  build from source with CMAKE_ARGS=\"-DGGML_CUDA=on\".")
    return offload


def check_torch() -> bool:
    print("\ntier B - torch (optional, robustness check only)")
    try:
        import torch
    except ImportError:
        print("  not installed - fine unless you are running the AWQ arm")
        return True

    ok = torch.cuda.is_available()
    print(f"  version         : {torch.__version__}")
    print(f"  built for cuda  : {torch.version.cuda}")
    print(f"  cuda available  : {ok}")
    if ok:
        print(f"  device          : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  memory          : {free / 2**30:.1f} / {total / 2**30:.1f} GiB free")
    return ok


def main() -> None:
    a = check_llamacpp()
    check_torch()
    print()
    if a:
        print("Tier A is on the GPU. Every headline result comes from this stack.")
    else:
        print("Tier A cannot reach the GPU - fix this before ./run.sh models.")
        sys.exit(1)


if __name__ == "__main__":
    main()
