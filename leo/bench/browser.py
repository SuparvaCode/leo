"""Browser-automation benchmark on browser-use/jev-ultrafast, with Jev or Leo as the decision model.

jev-ultrafast (external/jev-ultrafast, MIT) is Jev's own browser agent: every step it builds one
/v1/systemone request (an ``operation`` question plus speculative ``*_target`` questions over the observed
elements), validates the answer and executes it. This runner leaves the agent loop, the request builder,
the response validation, the DOM snapshot and the executor untouched. It changes one thing: who answers
the request.

* ``jev``: the request goes to api.typesafe.ai exactly as jev-ultrafast sends it.
* ``leo``: the identical request body is answered in-process by a local Leo checkpoint.

Both arms share the same dedicated Chrome profile (never the user's own), viewport, goals, step budgets,
text helper for TYPE_TEXT and independent result checks. A run passes only if the agent stops with DONE
*and* the checks on the final page hold; the model's DONE alone is never trusted.

    python -m leo.bench.browser --backends jev,leo --leo checkpoints/leo-0.6b-v1 --name v1 --repeats 3
    python -m leo.bench.browser --backends leo --leo checkpoints/leo-0.6b-v1 --tasks travel-casa-flora --repeats 1
    python -m leo.bench.browser --report --name v1

TYPE_TEXT needs a small text model. If TEXT_MODEL_API_KEY is set, jev-ultrafast's own OpenAI-compatible
helper is used; otherwise a local Qwen3-1.7B (greedy, thinking off) writes the value with jev-ultrafast's
prompt and output checks. Either way both arms use the same helper, so it is a controlled variable.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
ULTRAFAST = ROOT / "external" / "jev-ultrafast"
OUT = ROOT / "results" / "browser"
PROFILE = OUT / "chrome-profile"  # dedicated, throwaway automation profile
CDP_PORT = 9333
FIXTURE_PORT = 8767
DAEMON = "leo-bench"
CHROME_PATHS = [
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
]

# browser-harness reads these at import time: talk to our dedicated Chrome over CDP, under our own daemon
# name, and send no telemetry (its events can include the task text).
os.environ.setdefault("BU_CDP_URL", f"http://127.0.0.1:{CDP_PORT}")
os.environ.setdefault("BU_NAME", DAEMON)
os.environ.setdefault("BH_TELEMETRY", "0")
os.environ.setdefault("BH_TAB_MARKER", "0")


# ------------------------------------------------------------------------------ tasks

@dataclass(frozen=True)
class Task:
    name: str
    url: str  # "{fixture}" is replaced with the local fixture origin
    goal: str
    verify: Callable[[Any, dict[str, Any]], dict[str, bool]]
    live: bool = False  # a real third-party website
    needs_text: bool = True  # has a text field to fill, so the text helper is loaded up front


def _body_text(agent: Any) -> str:
    try:
        return agent.browser.evaluate("document.body.innerText") or ""
    except Exception:
        return ""


def _stay(slug: str, filters: str) -> Callable[[Any, dict[str, Any]], dict[str, bool]]:
    # Case-insensitive: the fixture's own search is (it lower-cases the query), so "LISBON" is a correct search.
    def verify(agent: Any, page: dict[str, Any]) -> dict[str, bool]:
        return {"opened": page["url"].endswith(f"#{slug}"),
                "filters": f"Your filters: {filters}".lower() in _body_text(agent).lower()}

    return verify


def _article(slug: str, title: str) -> Callable[[Any, dict[str, Any]], dict[str, bool]]:
    def verify(agent: Any, page: dict[str, Any]) -> dict[str, bool]:
        return {"opened": page["url"].endswith(f"#{slug}"), "title": page["title"].startswith(title)}

    return verify


def _wikipedia(title: str) -> Callable[[Any, dict[str, Any]], dict[str, bool]]:
    def verify(agent: Any, page: dict[str, Any]) -> dict[str, bool]:
        u = urlparse(page["url"])
        return {"host": u.hostname == "en.wikipedia.org", "article": unquote(u.path) == "/wiki/" + title.replace(" ", "_")}

    return verify


def _flights(origin: str, dest: str, iso: str, short: str, long: str) -> Callable[[Any, dict[str, Any]], dict[str, bool]]:
    """jev-ultrafast's examples/flights.py checks, parameterised by date."""
    import base64

    def verify(agent: Any, page: dict[str, Any]) -> dict[str, bool]:
        parsed = urlparse(page["url"])
        encoded = parse_qs(parsed.query).get("tfs", [""])[0]
        try:
            date_in_url = iso.encode() in base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except ValueError:
            date_in_url = False
        values = {a["label"].strip(): a.get("value") for a in page["actions"]}
        flights = [a["label"] for a in page["actions"] if "Select flight" in a["label"]]
        return {
            "search_page": parsed.hostname == "www.google.com" and parsed.path == "/travel/flights/search",
            "one_way": values.get("Change ticket type. One way") == "One way",
            "origin": values.get("Where from?") == origin,
            "destination": values.get("Where to?") == dest,
            "date": values.get("Departure") == short,
            "year": date_in_url or f"departing {iso}" in page["text"],
            "results": bool(flights) and all(long in f for f in flights),
        }

    return verify


PROVIDER_ERRORS = ("Model connection failed", "Model provider returned HTTP 5", "Model unavailable")
FIXTURE_TRAVEL = "{fixture}/fixture.html?scenario=travel"
FIXTURE_RESEARCH = "{fixture}/fixture.html?scenario=research"
TASKS: dict[str, Task] = {t.name: t for t in [
    # jev-ultrafast's own checks (scripts/smoke.py, static/app.js) on its local fixture
    Task("travel-casa-flora", FIXTURE_TRAVEL,
         "Use the destination search and filters to find Design stays in Lisbon with Free cancellation, then open Casa Flora.",
         _stay("casa-flora", "Design · Free cancellation enabled · Destination Lisbon")),
    Task("research-finite-choices", FIXTURE_RESEARCH,
         "Open the article about using finite choices to control browser agents.",
         _article("choices", "A browser is a choice, not a conversation"), needs_text=False),
    # new goals on the same fixture
    Task("travel-glasshouse", FIXTURE_TRAVEL,
         "Find a Design stay in Copenhagen with Free cancellation and open The Glasshouse.",
         _stay("the-glasshouse", "Design · Free cancellation enabled · Destination Copenhagen")),
    Task("travel-serra-lodge", FIXTURE_TRAVEL,
         "Search for Nature stays in Lisbon and open Serra Lodge. Leave Free cancellation off.",
         _stay("serra-lodge", "Nature · Free cancellation off · Destination Lisbon")),
    Task("research-confidence", FIXTURE_RESEARCH,
         "Open the article explaining what a peaked probability distribution can and cannot tell you.",
         _article("uncertainty", "Confidence is not correctness"), needs_text=False),
    # jev-ultrafast's live checks (README, docs/performance.md); the flight date moved from the past
    Task("wikipedia-godel", "https://en.wikipedia.org/wiki/Main_Page",
         "Find and open the Wikipedia article about Gödel’s incompleteness theorems.",
         _wikipedia("Gödel's incompleteness theorems"), live=True),
    Task("google-flights", "https://www.google.com/travel/flights?hl=en",
         "Find one-way flights from Zurich to London on October 20, 2026, for one adult in economy. "
         "Stop when matching flight options are visible. Do not select or book a flight.",
         _flights("Zürich", "London", "2026-10-20", "Tue, Oct 20", "Tuesday, October 20"), live=True),
]}


# ------------------------------------------------------------------------------ infrastructure

def find_chrome() -> str:
    for p in CHROME_PATHS:
        if p.exists():
            return str(p)
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if not found:
        raise SystemExit("Chrome not found")
    return found


def _cdp_version(port: int, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
                return json.loads(r.read())
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


class Chrome:
    """A dedicated Chrome on a throwaway profile, loopback CDP only."""

    def __init__(self, port: int = CDP_PORT, headless: bool = True) -> None:
        self.port = port
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"port {port} is already in use; stop that process or pick another --cdp-port")
        PROFILE.mkdir(parents=True, exist_ok=True)
        self.args = [find_chrome(), f"--remote-debugging-port={port}", "--remote-debugging-address=127.0.0.1",
                     f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check", "--disable-sync",
                     "--disable-extensions", "--window-size=1120,780", "--lang=en-US", "--accept-lang=en-US,en"]
        if headless:
            self.args.append("--headless=new")
        self.proc = self._start(self.args)
        ua = _cdp_version(port).get("User-Agent", "")
        if "HeadlessChrome" in ua:  # present as ordinary Chrome; some sites serve headless clients differently
            self.close()
            self.proc = self._start(self.args + [f"--user-agent={ua.replace('HeadlessChrome', 'Chrome')}"])
            ua = _cdp_version(port).get("User-Agent", "")
        self.info = {"user_agent": ua, "headless": headless, "version": _cdp_version(port).get("Browser")}

    def _start(self, args: list[str]) -> subprocess.Popen:
        proc = subprocess.Popen(args + ["about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _cdp_version(self.port)
        return proc

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        # wait until the port is free again
        for _ in range(50):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", self.port)) != 0:
                    return
            time.sleep(0.1)


def serve_fixture(port: int = FIXTURE_PORT) -> http.server.ThreadingHTTPServer:
    static = ULTRAFAST / "jev_ultrafast" / "static"

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            pass

    class Server(http.server.ThreadingHTTPServer):
        def handle_error(self, request: Any, client_address: Any) -> None:  # Chrome drops keep-alive sockets on close
            pass

    server = Server(("127.0.0.1", port), functools.partial(Quiet, directory=str(static)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ------------------------------------------------------------------------------ text helper

class LocalTextHelper:
    """Stand-in for jev-ultrafast's OpenAI-compatible text helper: same prompt, same output validation."""

    def __init__(self, model: str = "Qwen/Qwen3-1.7B", device: str = "cuda") -> None:
        self.name = f"local:{model}"
        self.model_id = model
        self.device = device
        self.model = None  # loaded up front by main() when a task needs text

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        free = torch.cuda.mem_get_info()[0] / 2**30 if self.device.startswith("cuda") else 1e9
        if free < 5.0:  # sharing an 8 GB card with Leo: 4-bit weights (about 1.2 GB instead of 3.4 GB)
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=q, device_map=self.device).eval()
            self.name += " (nf4)"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to(self.device).eval()

    def __call__(self, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        import torch
        from jev_ultrafast.questions import TEXT_VALUE

        if self.model is None:
            self._load()
        # jev-ultrafast's system prompt and JSON context, plus one line restating the goal and field after the
        # page text. Without it this small model copies page text ("LISBON · DESIGN") into the field.
        user = (json.dumps(context, ensure_ascii=False) + f'\n\nWrite the value for the field '
                f'"{context["field"].get("label", "")}" that this goal requires: {context["goal"]}')
        messages = [{"role": "system", "content": TEXT_VALUE}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=64, do_sample=False, pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        latency = round((time.perf_counter() - t0) * 1000)
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            output = json.loads(match.group(0) if match else raw)
            value = output["text"]
            if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise ValueError("Text helper returned no valid field value; nothing typed.") from None
        usage = {"prompt_tokens": int(ids["input_ids"].shape[1]), "completion_tokens": int(out.shape[1] - ids["input_ids"].shape[1])}
        return value, {"model": self.name, "latency_ms": latency, "usage": usage}


# ------------------------------------------------------------------------------ decision backends

class Router:
    """Replaces jev_ultrafast.model.post_json: /v1/systemone goes to the selected backend, the rest passes."""

    def __init__(self, leo: Any | None) -> None:
        import jev_ultrafast.model as jm

        self.jm = jm
        self.original = jm.post_json
        self.leo = leo
        self.backend = "jev"
        jm.post_json = self

    def __call__(self, url: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
        if not url.endswith("/v1/systemone") or self.backend == "jev":
            return self.original(url, key, body)
        if self.leo is None:
            raise RuntimeError("no Leo checkpoint loaded")
        try:
            return self.leo.system_one(body["state"], body["questions"])
        except Exception as e:  # surfaced like a provider error: no action executed
            raise RuntimeError(f"Leo failed on this request: {type(e).__name__}: {e}") from None


def fix_snapshot_encoding() -> None:
    """jev-ultrafast reads snapshot.js with the platform default encoding. On Windows that is cp1252, which
    turns the UTF-8 arrow in its dropdown labels ("Category → Design") into "â†’", so the Python side no
    longer splits element names from option names. Re-read it as UTF-8, as on macOS and Linux."""
    import jev_ultrafast.browser as jb

    jb.READ_STATE = Path(jb.__file__).with_name("snapshot.js").read_text(encoding="utf-8")
    jb.MARKER = f"(() => {{ const state={jb.READ_STATE}; return state?.marker ?? null; }})()"


def load_leo(path: str, dtype: str = "fp32") -> Any:
    from leo.infer import Leo

    # Browser states run to several thousand tokens (page text plus up to 250 elements): nothing is cut.
    leo = Leo.load(path, dtype=dtype, max_state_tokens=12288, max_row_tokens=20480, max_batch_tokens=20480)
    leo.encoder.max_question_tokens = 6144
    leo.encoder.max_instruction_tokens = 1024
    return leo


# ------------------------------------------------------------------------------ one run

def run_once(task: Task, fixture_origin: str, time_limit_s: float, record: Path) -> dict[str, Any]:
    from jev_ultrafast import Agent

    url = task.url.replace("{fixture}", fixture_origin)
    t0 = time.perf_counter()
    error = None
    agent = Agent(url, task.goal)
    try:
        for state in agent.run():
            if time.perf_counter() - t0 > time_limit_s:
                error = f"stopped at the {time_limit_s:.0f} s wall-clock limit"
                break
    except Exception as e:  # invalid model response, provider failure, budget stop, ...
        error = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0
    try:
        snap = agent.snapshot()
        page = snap["page"]
        checks = task.verify(agent, page)
    finally:
        agent.close()
    history = snap["history"]
    decisions = snap["decisions"]
    lat = [d["latency_ms"] for d in decisions]
    result = {
        "task": task.name,
        "status": snap["status"],
        "passed": snap["status"] == "done" and all(checks.values()),
        "goal_state_reached": all(checks.values()),
        "checks": checks,
        "error": error,
        "elapsed_ms": snap["elapsed_ms"],
        "wall_s": round(wall, 2),
        "decisions": len(decisions),
        "actions": len(history),
        "text_calls": len(snap["text_calls"]),
        "decision_ms_p50": sorted(lat)[len(lat) // 2] if lat else None,
        "input_tokens": sum(d.get("usage", {}).get("input_tokens", 0) for d in decisions),
        "final_url": page["url"],
        "trace": [{k: h.get(k) for k in ("step", "action", "kind", "operation", "target", "probability", "confidence",
                                         "text", "page_changed", "url")} for h in history],
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    full = {**result, "goal": task.goal, "decisions_full": [
        {k: v for k, v in d.items() if k not in ("request",)} | {"state": d.get("request", {}).get("state")}
        for d in decisions], "text_calls_full": snap["text_calls"]}
    record.write_text(json.dumps(full, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return result


# ------------------------------------------------------------------------------ report

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["task"], r["backend"]), []).append(r)
    return {f"{t}|{b}": {"runs": len(v), "passed": sum(x["passed"] for x in v),
                         "goal_state": sum(x["goal_state_reached"] for x in v),
                         "median_elapsed_ms": sorted(x["elapsed_ms"] for x in v)[len(v) // 2],
                         "median_decisions": sorted(x["decisions"] for x in v)[len(v) // 2],
                         "median_decision_ms": sorted((x["decision_ms_p50"] or 0) for x in v)[len(v) // 2]}
            for (t, b), v in by.items()}


def report(name: str) -> str:
    path = OUT / name / "runs.jsonl"
    latest: dict[tuple[str, str, int], dict[str, Any]] = {}  # an interrupted invocation may have left rows behind
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["task"], r["backend"], r["repeat"])] = r
    rows = list(latest.values())
    backends = sorted({r["backend"] for r in rows}, key=lambda b: (not b.startswith("jev"), b))
    tasks = [t for t in TASKS if any(r["task"] == t for r in rows)]
    s = summarize(rows)
    head = "| task | " + " | ".join(f"{b} passed | {b} median time / decisions" for b in backends) + " |"
    lines = [head, "|---|" + "---|---|" * len(backends)]
    totals = {b: [0, 0] for b in backends}
    for t in tasks:
        cells = []
        for b in backends:
            m = s.get(f"{t}|{b}")
            if not m:
                cells += ["-", "-"]
                continue
            totals[b][0] += m["passed"]
            totals[b][1] += m["runs"]
            cells += [f"{m['passed']}/{m['runs']}", f"{m['median_elapsed_ms'] / 1000:.1f} s / {m['median_decisions']}"]
        live = " (live site)" if TASKS[t].live else ""
        lines.append(f"| {t}{live} | " + " | ".join(cells) + " |")
    lines.append("| **all** | " + " | ".join(f"**{p}/{n}** | " for p, n in totals.values()) + " |")
    fails = [r for r in rows if not r["passed"]]
    if fails:
        lines += ["", "Failures:", ""]
        for r in fails:
            last = r["trace"][-1]["action"] if r["trace"] else "-"
            why = r["error"] or f"status {r['status']}, checks {','.join(k for k, v in r['checks'].items() if not v) or 'ok'}"
            lines.append(f"- {r['backend']} / {r['task']} #{r['repeat']}: {why}; {r['actions']} actions, last: {last}")
    text = "\n".join(lines)
    (OUT / name / "report.md").write_text(text + "\n", encoding="utf-8")
    return text


# ------------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="jev,leo", help="comma list of jev, leo")
    ap.add_argument("--leo", help="Leo checkpoint for the leo backend")
    ap.add_argument("--leo-dtype", default="fp32", choices=["fp32", "bf16", "auto"])
    ap.add_argument("--name", required=True, help="results/browser/<name>/")
    ap.add_argument("--tasks", default="all", help="comma list, 'fixture' (local only) or 'all'")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=120.0, help="seconds per run")
    ap.add_argument("--headed", action="store_true", help="show the automation Chrome window")
    ap.add_argument("--text-model", default="Qwen/Qwen3-1.7B", help="local text helper when TEXT_MODEL_API_KEY is not set")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        print(report(args.name))
        return

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    backends = args.backends.split(",")
    if "jev" in backends and not os.environ.get("TYPESAFE_API_KEY"):
        raise SystemExit("TYPESAFE_API_KEY is not set; add it to .env")
    os.environ.setdefault("TYPESAFE_API_KEY", "unused-by-leo")  # choose() reads it even when Leo answers
    if "leo" in backends and not args.leo:
        raise SystemExit("--leo is required for the leo backend")
    if args.tasks == "all":
        tasks = list(TASKS.values())
    elif args.tasks == "fixture":
        tasks = [t for t in TASKS.values() if not t.live]
    else:
        tasks = [TASKS[t] for t in args.tasks.split(",")]

    import jev_ultrafast.agent as ja
    from browser_harness.admin import restart_daemon

    fix_snapshot_encoding()

    # Leo first, so the local text helper sees how much GPU memory is left and picks bf16 or 4-bit.
    router = Router(load_leo(args.leo, args.leo_dtype) if "leo" in backends else None)
    helper: Any
    if os.environ.get("TEXT_MODEL_API_KEY"):
        helper, helper_name = ja.field_text, os.environ.get("TEXT_MODEL", "deepseek-chat")
    else:
        helper = LocalTextHelper(args.text_model)
        if any(t.needs_text for t in tasks):
            helper._load()  # before any clock starts
        helper_name = helper.name
    ja.field_text = helper  # agent.py imported field_text by name

    out_dir = OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    chrome = Chrome(headless=not args.headed)
    fixture = serve_fixture()
    meta = {"chrome": chrome.info, "text_helper": helper_name, "leo": args.leo, "leo_dtype": args.leo_dtype,
            "jev_model": os.environ.get("TYPESAFE_MODEL", "jev-latest"), "time_limit_s": args.time_limit,
            "jev_ultrafast_commit": subprocess.run(["git", "-C", str(ULTRAFAST), "rev-parse", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip()}
    meta["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_dir / "meta.jsonl", "a", encoding="utf-8") as fh:  # one line per invocation
        fh.write(json.dumps(meta) + "\n")
    print(json.dumps(meta), flush=True)
    origin = f"http://127.0.0.1:{FIXTURE_PORT}"
    labels = {"jev": "jev", "leo": f"leo:{Path(args.leo).name}" if args.leo else "leo"}
    try:
        with open(out_dir / "runs.jsonl", "a", encoding="utf-8") as fh:
            for task in tasks:
                for rep in range(args.repeats):
                    # alternate the arms within each repeat, as jev-ultrafast's matched comparison does
                    order = backends if rep % 2 == 0 else backends[::-1]
                    for backend in order:
                        router.backend = backend
                        rec = out_dir / "traces" / f"{task.name}-{labels[backend].replace(':', '_')}-{rep}.json"
                        r: dict[str, Any] = {}
                        for attempt in range(2):
                            try:
                                r = run_once(task, origin, args.time_limit, rec)
                            except Exception as e:  # could not even start (browser, network)
                                r = {"task": task.name, "status": "error", "passed": False, "goal_state_reached": False,
                                     "checks": {}, "error": f"{type(e).__name__}: {e}", "elapsed_ms": 0, "wall_s": 0,
                                     "decisions": 0, "actions": 0, "decision_ms_p50": None, "trace": []}
                            # jev-ultrafast never retries a failed provider call; rerun the whole task once, in a
                            # fresh tab, so a network drop is not scored as a wrong decision. Same rule for both arms.
                            if not any(s in (r["error"] or "") for s in PROVIDER_ERRORS):
                                break
                            print(f"{task.name} {labels[backend]} #{rep}: provider error, rerunning once", flush=True)
                        r |= {"backend": labels[backend], "repeat": rep, "attempts": attempt + 1}
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                        fh.flush()
                        why = r["error"] or ",".join(k for k, v in r["checks"].items() if not v)
                        print(f"{task.name:<24} {labels[backend]:<16} #{rep} {'PASS' if r['passed'] else 'FAIL'} status={r['status']:<8} "
                              f"{r['elapsed_ms'] / 1000:6.1f} s {r['decisions']:>3} decisions {r['actions']:>3} actions {why}",
                              flush=True)
    finally:
        fixture.shutdown()
        try:
            restart_daemon(DAEMON)
        except Exception:
            pass
        chrome.close()
    print(report(args.name))


if __name__ == "__main__":
    main()
