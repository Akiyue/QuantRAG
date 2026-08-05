from .base import Backend, ScoreResult, GenResult, MockBackend, BoundaryError

__all__ = ["Backend", "ScoreResult", "GenResult", "MockBackend", "BoundaryError", "load_backend"]


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
        from .hf import HFBackend
        return HFBackend(**{k: v for k, v in spec.items() if k != "backend"})
    raise ValueError(f"unknown backend {kind!r}")
