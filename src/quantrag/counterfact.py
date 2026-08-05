"""Cached access to CounterFact rows.

Both the coverage survey and the dataset build need the same 21,919 rows, and
the HF datasets server serves them 100 at a time. Fetching that twice is 440
requests for data that does not change, which is how the survey earned a 429.

So: one cache on disk, one backoff policy, shared.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "NeelNanda/counterfact-tracing"
UA = "QuantRAG/0.1 (academic research)"
PAGE = 100


def _get(url: str, max_retries: int = 6, delay: float = 0.3) -> dict:
    for _ in range(max_retries):
        time.sleep(delay)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = (float(retry_after) if retry_after and retry_after.isdigit()
                     else max(delay * 2, 2.0))
        except urllib.error.URLError:
            delay = max(delay * 2, 2.0)
    raise RuntimeError(f"datasets server kept failing: {url}")


def load_rows(cache_path: str | Path, limit: int = 25_000,
              progress: bool = True) -> list[dict]:
    """Rows from the on-disk cache, fetching only what is missing.

    Appends as it goes, so an interrupted fetch resumes from where it stopped
    rather than starting the whole download again.
    """
    cache = Path(cache_path)
    rows: list[dict] = []
    if cache.exists():
        rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l]
        if len(rows) >= limit:
            if progress:
                print(f"  cached: {len(rows)} rows")
            return rows[:limit]
        if progress:
            print(f"  cached: {len(rows)} rows, fetching more")

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "a", encoding="utf-8") as fh:
        while len(rows) < limit:
            q = urllib.parse.urlencode({
                "dataset": DATASET, "config": "default", "split": "train",
                "offset": len(rows), "length": min(PAGE, limit - len(rows)),
            })
            payload = _get(f"{ROWS_API}?{q}")
            batch = [r["row"] for r in payload.get("rows", [])]
            if not batch:
                break
            for r in batch:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            rows.extend(batch)
            if progress:
                print(f"\r  fetched {len(rows)}", end="", flush=True)
    if progress:
        print()
    return rows
