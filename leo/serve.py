"""HTTP server with TypeSafe's /v1/systemone wire format, so TypeSafe SDKs work by changing the base URL.

    LEO_API_KEY=... python -m leo.serve --model checkpoints/leo-0.6b --port 8000

Security defaults: binds to 127.0.0.1; requires ``Authorization: Bearer <LEO_API_KEY>`` whenever a key
is set; refuses to bind a non-loopback address without a key; caps body size and question count.
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import threading
from datetime import date
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from leo.encode import QuestionTooLong
from leo.infer import Leo
from leo.schema import SystemOneRequest

MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_QUESTIONS = 2000


def _error(status: int, kind: str, message: str, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"type": kind, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status)


def create_app(leo: Leo, api_key: str | None = None, max_body_bytes: int = MAX_BODY_BYTES,
               max_questions: int = MAX_QUESTIONS) -> FastAPI:
    app = FastAPI(title="Leo System One", version="0.1.0", docs_url=None, redoc_url=None)
    lock = threading.Lock()

    def authorized(request: Request) -> bool:
        if not api_key:
            return True
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), api_key.encode())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": leo.name, "device": str(leo.device), "temperatures": leo.temperatures}

    @app.get("/v1/models")
    def models(request: Request) -> Any:
        if not authorized(request):
            return _error(401, "authentication_error", "missing or invalid bearer token")
        today = date.today().isoformat()
        desc = f"Leo System One decision model ({leo.config.get('base_model', '?')} backbone)"
        return {"models": [{"name": leo.name, "description": desc, "release_date": today},
                           {"name": "leo-latest", "description": f"alias of {leo.name}", "release_date": today}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Any:
        if not authorized(request):
            return _error(401, "authentication_error", "missing or invalid bearer token")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_body_bytes:
            return _error(413, "request_too_large", f"body exceeds {max_body_bytes} bytes")
        raw = await request.body()
        if len(raw) > max_body_bytes:
            return _error(413, "request_too_large", f"body exceeds {max_body_bytes} bytes")
        try:
            payload = SystemOneRequest.model_validate_json(raw)
        except ValidationError as e:
            details = json.loads(e.json(include_url=False, include_input=False))
            return _error(422, "invalid_request", "request failed validation", details)
        if len(payload.questions) > max_questions:
            return _error(422, "invalid_request", f"at most {max_questions} questions per request")

        def run() -> dict[str, Any]:
            with lock:
                return leo.predict_many([payload])[0]

        try:
            result = await run_in_threadpool(run)
        except QuestionTooLong as e:
            return _error(422, "invalid_request", str(e))
        result["model"] = leo.name
        return result

    return app


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=os.environ.get("LEO_DTYPE", "auto"), choices=["auto", "bf16", "fp32"],
                    help="fp32 = exact, question-order-independent answers; bf16 = faster (default on GPU)")
    ap.add_argument("--precise", action="store_true",
                    help="4-decimal probabilities plus a latency_ms field, instead of TypeSafe's exact response shape")
    ap.add_argument("--order-views", type=int, default=int(os.environ.get("LEO_ORDER_VIEWS", "1")),
                    help="average each choice over this many option orders (less order-sensitive, slower)")
    args = ap.parse_args()
    key = os.environ.get("LEO_API_KEY")
    if not key and not _is_loopback(args.host):
        raise SystemExit("Refusing to bind a non-loopback address without LEO_API_KEY set.")
    if not key:
        print("LEO_API_KEY is not set: the server accepts unauthenticated requests on loopback only.")
    leo = Leo.load(args.model, device=args.device, dtype=args.dtype, jev_exact=not args.precise,
                   order_views=args.order_views)
    uvicorn.run(create_app(leo, api_key=key), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
