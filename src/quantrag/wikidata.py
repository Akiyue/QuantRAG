"""Wikidata access with an on-disk cache.

Everything here is cached to data/raw/wikidata_cache.json. The dataset build is
re-run many times while templates and filters are tweaked, and hammering the
public API for answers we already have is both slow and rude.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

API = "https://www.wikidata.org/w/api.php"
UA = "QuantRAG/0.1 (academic research; https://github.com/)"


class Wikidata:
    def __init__(self, cache_path: str | Path, delay: float = 0.2,
                 max_retries: int = 6) -> None:
        self.cache_path = Path(cache_path)
        self.delay = delay
        self.max_retries = max_retries
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self._dirty = False

    # -- plumbing --------------------------------------------------------

    def _get(self, params: dict) -> dict:
        """One API call, with backoff.

        Subject verification issues a search plus a claims fetch per candidate,
        so a full build makes thousands of calls and will hit 429 without this.
        `maxlag` is the Wikimedia convention for asking the API to shed load
        politely rather than being throttled for ignoring it.
        """
        params = {**params, "maxlag": 5}
        url = f"{API}?{urllib.parse.urlencode(params)}"
        delay = self.delay

        for attempt in range(self.max_retries):
            time.sleep(delay)
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.load(resp)
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() \
                    else max(delay * 2, 1.0)
                continue
            except urllib.error.URLError:
                delay = max(delay * 2, 1.0)
                continue

            # maxlag rejections come back as HTTP 200 with an error body.
            if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
                delay = max(delay * 2, 2.0)
                continue
            return data

        raise RuntimeError(f"Wikidata API kept failing after {self.max_retries} tries")

    def save(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
        )
        self._dirty = False

    # -- entities --------------------------------------------------------

    def entities(self, qids: Iterable[str], langs: tuple[str, ...] = ("en", "vi"),
                 with_claims: bool = False) -> dict[str, dict]:
        """Labels, aliases and optionally claims, batched 50 at a time."""
        key_suffix = "|claims" if with_claims else ""
        wanted = [q for q in dict.fromkeys(qids) if f"{q}{key_suffix}" not in self._cache]

        props = "labels|aliases|claims" if with_claims else "labels|aliases"
        for i in range(0, len(wanted), 50):
            chunk = wanted[i:i + 50]
            data = self._get({
                "action": "wbgetentities", "ids": "|".join(chunk),
                "props": props, "languages": "|".join(langs), "format": "json",
            })
            for qid, ent in data.get("entities", {}).items():
                labels = ent.get("labels", {})
                aliases = ent.get("aliases", {})
                rec: dict = {
                    "labels": {lg: labels.get(lg, {}).get("value") for lg in langs},
                    "aliases": {
                        lg: [a["value"] for a in aliases.get(lg, [])] for lg in langs
                    },
                }
                if with_claims:
                    rec["claims"] = _claim_targets(ent.get("claims", {}))
                self._cache[f"{qid}{key_suffix}"] = rec
                self._dirty = True
            # Ask for 50, get back only the ones that exist; record the rest as
            # absent so a missing id is not re-requested on every run.
            for qid in chunk:
                self._cache.setdefault(f"{qid}{key_suffix}", {})
                self._dirty = True

        return {q: self._cache.get(f"{q}{key_suffix}", {}) for q in dict.fromkeys(qids)}

    def search(self, name: str, limit: int = 5) -> list[str]:
        key = f"search:{name}"
        if key not in self._cache:
            data = self._get({
                "action": "wbsearchentities", "search": name, "language": "en",
                "uselang": "en", "type": "item", "limit": limit, "format": "json",
            })
            self._cache[key] = {"ids": [r["id"] for r in data.get("search", [])]}
            self._dirty = True
        return self._cache[key].get("ids", [])

    # -- subject resolution ----------------------------------------------

    def prefetch_searches(self, names: list[str], workers: int = 6) -> None:
        """Warm the search cache concurrently.

        Subject resolution is one search plus a claims fetch per candidate, done
        serially at a polite delay - which makes the dataset build network-bound
        for an hour or more while the machine sits idle. The searches are
        independent, so running a handful at once turns most of that wait into
        overlap. Kept modest: the point is to stop wasting the latency, not to
        hammer a public API.
        """
        from concurrent.futures import ThreadPoolExecutor

        todo = [n for n in dict.fromkeys(names) if f"search:{n}" not in self._cache]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(self.search, todo))

    def prefetch_entities(self, qids: list[str], workers: int = 4) -> None:
        """Warm the claims cache for many entities concurrently."""
        from concurrent.futures import ThreadPoolExecutor

        todo = [q for q in dict.fromkeys(qids) if f"{q}|claims" not in self._cache]
        if not todo:
            return
        chunks = [todo[i:i + 50] for i in range(0, len(todo), 50)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda c: self.entities(c, with_claims=True), chunks))

    def resolve_subject(self, name: str, pid: str, expect_qid: str) -> dict | None:
        """Find the QID for `name` and verify it carries pid -> expect_qid.

        Verification is what makes a name search safe: a bare lookup would
        happily return a different person of the same name, but an entity that
        also carries the relation CounterFact asserts is almost certainly the
        intended one. It doubles as a freshness check - facts that no longer
        hold on Wikidata drop out instead of quietly aging into the dataset.

        Returns None when nothing verifies. That is a false negative sometimes
        (Wikidata may record the relation under a neighbouring property), which
        is an acceptable trade: there are ~22k candidate rows and we need 500,
        so precision is worth far more than recall here.
        """
        candidates = self.search(name)
        if not candidates:
            return None
        ents = self.entities(candidates, with_claims=True)
        for qid in candidates:
            ent = ents.get(qid) or {}
            if expect_qid in set(ent.get("claims", {}).get(pid, [])):
                return {"qid": qid, "labels": ent.get("labels", {}),
                        "aliases": ent.get("aliases", {})}
        return None


def _claim_targets(claims: dict) -> dict[str, list[str]]:
    """Reduce a claims blob to {property: [target QIDs]}."""
    out: dict[str, list[str]] = {}
    for pid, statements in claims.items():
        ids: list[str] = []
        for st in statements:
            val = st.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            if isinstance(val, dict) and "id" in val:
                ids.append(val["id"])
        if ids:
            out[pid] = ids
    return out
