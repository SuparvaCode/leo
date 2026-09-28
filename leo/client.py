"""Minimal client for any POST /v1/systemone server: Leo's own server or TypeSafe's hosted Jev."""
from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504, 529}


class SystemOneClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str | None = None,
        model: str = "leo-latest",
        timeout: float = 60.0,
        retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = retries
        headers = {"Content-Type": "application/json"}
        key = api_key if api_key is not None else os.environ.get("LEO_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(timeout=timeout, headers=headers)

    def system_one(self, state: Any, questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        body = {"state": state, "model": model or self.model, "questions": questions}
        delay = 0.5
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._http.post(f"{self.base_url}/v1/systemone", json=body)
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
            else:
                if r.status_code not in RETRY_STATUS or attempt == self.retries:
                    r.raise_for_status()
                    out = r.json()
                    out["client_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    upstream = r.headers.get("x-envoy-upstream-service-time")
                    if upstream and upstream.isdigit():
                        out["server_ms"] = int(upstream)
                    return out
                ra = r.headers.get("retry-after")
                if ra and ra.replace(".", "", 1).isdigit():
                    delay = max(delay, float(ra))
            time.sleep(delay * (1 + random.random() * 0.25))
            delay = min(delay * 2, 20.0)
        raise RuntimeError("unreachable")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SystemOneClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
