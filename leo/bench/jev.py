"""Live runs against TypeSafe's Jev: rate-limited, and every response cached on disk.

A run of thousands of requests survives an interruption (power cut, network drop, Ctrl+C): rerun the same
command and only requests missing from the cache are sent, so nothing is paid for twice. The cache key
includes a hash of the model name and request body, so a changed prompt never reuses an old answer.

Jev's answers are used for scoring only; nothing here produces training data.

    from leo.bench.jev import JevRunner
    responses = JevRunner().run("my-suite", [("row-0", {"state": ..., "questions": {...}}), ...])
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from leo.client import SystemOneClient

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results" / "cache" / "jev"
PRICE_PER_MTOK = 0.042  # USD per million input tokens (docs.typesafe.ai/models); output tokens are free


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def request_key(rid: str, model: str, body: dict[str, Any]) -> str:
    """Cache key. Key order is kept: option order changes Jev's answers, so a permuted request is a new request."""
    blob = json.dumps({"model": model, **body}, ensure_ascii=False).encode("utf-8")
    return f"{rid}:o{hashlib.sha1(blob).hexdigest()[:12]}"


def legacy_key(rid: str, model: str, body: dict[str, Any]) -> str:
    """The first cache format hashed with sorted keys, so it could not tell option orders apart."""
    blob = json.dumps({"model": model, **body}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{rid}:{hashlib.sha1(blob).hexdigest()[:12]}"


class RateLimiter:
    """Spaces request starts evenly; TypeSafe allows 1,200 requests per minute (20/s) per account."""

    def __init__(self, per_second: float) -> None:
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next)
            self.next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


class JevRunner:
    def __init__(self, model: str = "jev-latest", per_second: float = 15.0, workers: int = 12, timeout: float = 120.0) -> None:
        load_env()
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("TYPESAFE_API_KEY is not set; add it to .env")
        base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
        self.model = model
        self.client = SystemOneClient(base, api_key=key, model=model, timeout=timeout, retries=8)
        self.limiter = RateLimiter(per_second)
        self.workers = workers

    @staticmethod
    def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
        cached: dict[str, dict[str, Any]] = {}
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # a line torn by a crash mid-write
                        continue
                    cached[rec["key"]] = rec["response"]
        return cached

    def _call(self, body: dict[str, Any]) -> dict[str, Any]:
        self.limiter.wait()
        return self.client.system_one(body["state"], body["questions"])

    def run(
        self, name: str, requests: list[tuple[str, dict[str, Any]]], rounds: int = 3, legacy_ok: bool = False
    ) -> dict[str, dict[str, Any]]:
        """Send every (request_id, {"state", "questions"}) not yet cached; return request_id -> response.

        ``legacy_ok`` reuses answers stored under the first (order-blind) key format. Only safe for suites
        that never send two option orders of the same request, such as the held-out four and JevBench.
        """
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{name}.jsonl"
        keyed = [(rid, request_key(rid, self.model, body), body) for rid, body in requests]
        cached = self._read_cache(path)
        if legacy_ok:
            adopted = 0
            with open(path, "a", encoding="utf-8") as fh:
                for rid, k, body in keyed:
                    old = legacy_key(rid, self.model, body)
                    if k not in cached and old in cached:
                        cached[k] = cached[old]
                        fh.write(json.dumps({"key": k, "response": cached[k]}, ensure_ascii=False) + "\n")
                        adopted += 1
            if adopted:
                print(f"[jev] {name}: re-keyed {adopted} cached responses to the order-aware format", flush=True)
        todo = [(rid, k, b) for rid, k, b in keyed if k not in cached]
        if len(todo) < len(keyed):
            print(f"[jev] {name}: {len(keyed) - len(todo)} of {len(keyed)} responses already cached", flush=True)
        errors: dict[str, str] = {}
        for attempt in range(rounds):
            if not todo:
                break
            t0 = time.perf_counter()
            failed = []
            with open(path, "a", encoding="utf-8") as fh, ThreadPoolExecutor(self.workers) as pool:
                futures = {pool.submit(self._call, b): (rid, k, b) for rid, k, b in todo}
                for n, fut in enumerate(as_completed(futures), start=1):
                    rid, k, b = futures[fut]
                    try:
                        resp = fut.result()
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in (401, 403):
                            raise SystemExit(f"Jev rejected the API key ({e.response.status_code})") from None
                        errors[rid] = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                        failed.append((rid, k, b))
                        continue
                    except Exception as e:  # network trouble; retried in the next round
                        errors[rid] = f"{type(e).__name__}: {e}"
                        failed.append((rid, k, b))
                        continue
                    fh.write(json.dumps({"key": k, "response": resp}, ensure_ascii=False) + "\n")
                    fh.flush()
                    cached[k] = resp
                    if n % 500 == 0:
                        rate = n / (time.perf_counter() - t0)
                        print(f"[jev] {name}: {n}/{len(todo)} ({rate:.1f} req/s)", flush=True)
            todo = failed
            if todo and attempt < rounds - 1:
                print(f"[jev] {name}: {len(todo)} failed, retrying in 10 s", flush=True)
                time.sleep(10)
        missing = [rid for rid, k, _ in keyed if k not in cached]
        if missing:
            sample = "; ".join(f"{rid}: {errors.get(rid, '?')}" for rid in missing[:3])
            raise RuntimeError(f"{len(missing)} Jev requests failed after {rounds} rounds (rerun to retry). {sample}")
        out = {rid: cached[k] for rid, k, _ in keyed}
        tokens = sum(r.get("usage", {}).get("input_tokens", 0) for r in out.values())
        print(f"[jev] {name}: {len(out)} responses, {tokens:,} input tokens (${tokens * PRICE_PER_MTOK / 1e6:.4f} at list price)", flush=True)
        return out


def summarize_latency(responses: dict[str, dict[str, Any]]) -> dict[str, float]:
    import numpy as np

    client = [r["client_latency_ms"] for r in responses.values() if "client_latency_ms" in r]
    server = [r["server_ms"] for r in responses.values() if "server_ms" in r]
    out: dict[str, float] = {}
    if client:
        out |= {"client_p50_ms": float(np.percentile(client, 50)), "client_p95_ms": float(np.percentile(client, 95))}
    if server:
        out |= {"server_p50_ms": float(np.percentile(server, 50)), "server_p95_ms": float(np.percentile(server, 95))}
    return out
