from .base import (Backend, ScoreResult, GenResult, MockBackend, BoundaryError,
                   DegenerateArm, DegenerateOutput)

__all__ = ["Backend", "ScoreResult", "GenResult", "MockBackend", "BoundaryError", "DegenerateOutput", "DegenerateArm", "load_backend"]


def load_backend(spec: dict):
    """Instantiate a backend from a models.yaml variant entry.

    Imports are deferred so that the analysis code runs on a machine with
    neither llama-cpp-python nor torch installed.
    """
    kind = spec["backend"]
    if kind == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(**{k: v for k, v in spec.items() if k != "backend"})
    if kind == "mock":
        return MockBackend(**{k: v for k, v in spec.items() if k != "backend"})
    if kind == "hf_awq":
        try:
            from .hf import HFBackend
        except ModuleNotFoundError as exc:
            raise NotImplementedError(
                "the tier B (AWQ) backend is not written yet. Tier A - the "
                "llama.cpp precision ladder that produces every headline result - "
                "does not depend on it; run with --tier A. Tier B is the "
                "cross-quantizer robustness check (PLAN 1.3)."
            ) from exc
        return HFBackend(**{k: v for k, v in spec.items() if k != "backend"})
    raise ValueError(f"unknown backend {kind!r}")
