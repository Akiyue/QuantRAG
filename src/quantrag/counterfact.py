"""Cached access to CounterFact rows.

The whole dataset is 21,919 rows and 1.1 MB, published as a single parquet
file. Reading it through the datasets server's paginated rows API instead means
220 requests for it, which is enough to get rate-limited half way through - and
then the survey and the dataset build each do it again.

So: fetch the parquet once, cache it as JSONL, and fall back to the paginated
API only if the file cannot be reached.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "NeelNanda/counterfact-tracing"
API = f"https://huggingface.co/api/datasets/{DATASET}"
PARQUET_API = "https://datasets-server.huggingface.co/parquet"
ROWS_API = "https://datasets-server.huggingface.co/rows"
UA = "QuantRAG/0.1 (academic research)"
PAGE = 100


def _get_json(url: str, max_retries: int = 5, delay: float = 0.5) -> dict:
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
                     else max(delay * 2, 3.0))
        except urllib.error.URLError:
            delay = max(delay * 2, 3.0)
    raise RuntimeError(f"gave up on {url}")


def _parquet_urls() -> list[str]:
    """Direct parquet locations, repo copy first.

    The repo's own file has a stable path; the datasets-server conversion is a
    fallback for when it does not.
    """
    urls: list[str] = []
    try:
        meta = _get_json(API)
        for s in meta.get("siblings", []):
            name = s.get("rfilename", "")
            if name.endswith(".parquet"):
                urls.append(f"https://huggingface.co/datasets/{DATASET}"
                            f"/resolve/main/{urllib.parse.quote(name)}")
    except Exception:  # noqa: BLE001 - fall through to the conversion endpoint
        pass
    try:
        conv = _get_json(f"{PARQUET_API}?dataset={DATASET}")
        urls += [f["url"] for f in conv.get("parquet_files", [])
                 if f.get("split") == "train"]
    except Exception:  # noqa: BLE001
        pass
    return urls


def _read_parquet(url: str) -> list[dict]:
    import io

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to read the parquet") from exc

    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=180) as resp:
        blob = resp.read()
    df = pd.read_parquet(io.BytesIO(blob))
    return df.to_dict(orient="records")


def _fetch_paginated(existing: list[dict], limit: int,
                     progress: bool) -> list[dict]:
    rows = list(existing)
    while len(rows) < limit:
        q = urllib.parse.urlencode({
            "dataset": DATASET, "config": "default", "split": "train",
            "offset": len(rows), "length": min(PAGE, limit - len(rows)),
        })
        payload = _get_json(f"{ROWS_API}?{q}")
        batch = [r["row"] for r in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        if progress:
            print(f"\r  fetched {len(rows)}", end="", flush=True)
    if progress:
        print()
    return rows


def load_rows(cache_path: str | Path, limit: int = 25_000,
              progress: bool = True) -> list[dict]:
    """Rows from the on-disk cache, fetching once if it is short."""
    cache = Path(cache_path)
    rows: list[dict] = []
    if cache.exists():
        rows = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l]
    if len(rows) >= limit:
        if progress:
            print(f"  cached: {len(rows)} rows")
        return rows[:limit]

    if rows and progress:
        print(f"  cache holds {len(rows)} rows, short of {limit}")

    fetched: list[dict] = []
    for url in _parquet_urls():
        try:
            if progress:
                print(f"  downloading parquet ({url.rsplit('/', 1)[-1]})")
            fetched = _read_parquet(url)
            break
        except Exception as exc:  # noqa: BLE001 - try the next source
            if progress:
                print(f"    failed: {exc}")

    if not fetched:
        if progress:
            print("  parquet unavailable, falling back to the paginated API")
        fetched = _fetch_paginated(rows, limit, progress)

    # Normalise: parquet may hand back numpy scalars, which json cannot write.
    fetched = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
               for r in fetched]

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        for r in fetched:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if progress:
        print(f"  cached {len(fetched)} rows -> {cache}")
    return fetched[:limit]
