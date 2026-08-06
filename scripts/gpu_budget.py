"""What will fit, on which card, given what is already running.

    python scripts/gpu_budget.py

The card is shared with other work, so the question before a long grid is not
"is there a GPU" but "is there enough headroom on it right now". Running into
the ceiling mid-grid does not produce an out-of-memory error - it produces NaN
logits, which look like answers until something checks.

Prints free memory per card and the estimated footprint per arm, then names the
card to pin. Advisory: it cannot see what the other job will do next.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Rough bytes per parameter by precision, plus the fixed costs.
BYTES_PER_PARAM = {"F16": 2.0, "Q8_0": 1.06, "Q4_K_M": 0.60, "Q3_K_M": 0.48}
VOCAB = 152_000
SAFETY = 1.25   # KV cache, activations, allocator slack


def gpu_memory() -> list[tuple[int, int, int]]:
    """(index, used MiB, total MiB) per card."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"cannot query nvidia-smi: {exc}")
    rows = []
    for line in out.strip().splitlines():
        i, used, total = (int(x.strip()) for x in line.split(","))
        rows.append((i, used, total))
    return rows


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    n_ctx = cfg["runtime"]["n_ctx"]
    logits_gib = n_ctx * VOCAB * 4 / 2**30

    print(f"logits buffer at n_ctx={n_ctx}: {logits_gib:.2f} GiB per loaded model\n")

    print("estimated footprint per arm")
    worst = 0.0
    for model in cfg["models"]:
        params = float(model["params"].rstrip("Bb")) * 1e9
        for var in model["variants"]:
            if var.get("tier", "A") != "A":
                continue
            weights = params * BYTES_PER_PARAM.get(var["precision"], 2.0) / 2**30
            need = (weights + logits_gib) * SAFETY
            worst = max(worst, need)
            print(f"  {model['id']:<16} {var['precision']:<8} "
                  f"weights {weights:5.2f} + logits {logits_gib:4.2f} "
                  f"-> ~{need:5.2f} GiB")

    print(f"\nlargest arm needs about {worst:.1f} GiB\n")

    print("cards right now")
    best, best_free = None, -1.0
    for i, used, total in gpu_memory():
        free = (total - used) / 1024
        verdict = "ok" if free > worst * 1.5 else "tight" if free > worst else "TOO FULL"
        print(f"  gpu{i}  {used / 1024:5.1f} used / {total / 1024:5.1f} GiB  "
              f"free {free:5.1f}  {verdict}")
        if free > best_free:
            best, best_free = i, free

    print()
    if best_free < worst:
        print("No card has room for the largest arm. Running now would hit the")
        print("ceiling mid-grid, and llama.cpp signals that with NaN logits rather")
        print("than an allocation error. Wait, or drop the 3B arm for this pass.")
        sys.exit(1)

    print(f"Pin to gpu{best}:")
    print(f"  CUDA_VISIBLE_DEVICES={best} ./run.sh main")
    if best_free < worst * 1.5:
        print()
        print("Headroom is thin. The runner now aborts an arm once degenerate")
        print("cells pass 5%, so a squeeze costs one arm and a rerun rather than")
        print("a corrupted result - but rerun that arm, do not analyse it.")


if __name__ == "__main__":
    main()
