/*
 * leo.kognare.com: the Leo-1 pitch page plus a free trial API with TypeSafe's /v1/systemone wire format.
 *
 * Callers authenticate with "Authorization: Bearer free". Requests are forwarded to the Leo-1 GPU endpoint on
 * Modal with a server-side key that never reaches the browser. Limits per client IP (IPv6 grouped by /64):
 * 20 requests per minute, 200 per UTC day. When the trial is busy, new IPs are turned away (529) while
 * already-active users keep working. Per-IP token usage is recorded (IPs stored only as salted hashes);
 * request and response bodies are never stored or logged.
 *
 * Environment:
 *   LEO_UPSTREAM_URL       Leo-1 endpoint base URL (required)
 *   LEO_UPSTREAM_KEY       bearer key for it (required, secret)
 *   LEO_FALLBACK_URL/_KEY  optional second endpoint used when the first fails
 *   ADMIN_KEY              enables GET /admin/usage (optional, secret)
 *   IP_HASH_SALT           salt for hashing IPs in the usage file (random per start if unset)
 *   DATA_DIR               where usage.json is written (default ./data)
 *   RPM, RPD, MAX_ACTIVE_USERS, ACTIVE_WINDOW_MIN, MAX_INFLIGHT, GLOBAL_RPD, MAX_BODY_KB, MAX_QUESTIONS,
 *   MAX_STATE_CHARS, TRUST_PROXY_HOPS
 */
"use strict";

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");
const express = require("express");

const env = (k, d) => (process.env[k] === undefined || process.env[k] === "" ? d : process.env[k]);
const num = (k, d) => Number(env(k, d));

const CFG = {
  port: num("PORT", 3000),
  upstream: env("LEO_UPSTREAM_URL", "").replace(/\/+$/, ""),
  upstreamKey: env("LEO_UPSTREAM_KEY", ""),
  fallback: env("LEO_FALLBACK_URL", "").replace(/\/+$/, ""),
  fallbackKey: env("LEO_FALLBACK_KEY", ""),
  adminKey: env("ADMIN_KEY", ""),
  salt: env("IP_HASH_SALT", crypto.randomBytes(16).toString("hex")),
  dataDir: env("DATA_DIR", path.join(__dirname, "data")),
  rpm: num("RPM", 20),
  rpd: num("RPD", 200),
  maxActiveUsers: num("MAX_ACTIVE_USERS", 25),
  activeWindowMs: num("ACTIVE_WINDOW_MIN", 10) * 60_000,
  maxInflight: num("MAX_INFLIGHT", 6),
  globalRpd: num("GLOBAL_RPD", 4000),
  maxBodyKb: num("MAX_BODY_KB", 64),
  maxQuestions: num("MAX_QUESTIONS", 16),
  maxStateChars: num("MAX_STATE_CHARS", 20000),
  trustProxyHops: num("TRUST_PROXY_HOPS", 1),
  upstreamTimeoutMs: num("UPSTREAM_TIMEOUT_MS", 150_000),
  model: "leo-1",
};
const TRIAL_KEY = "free";

// ------------------------------------------------------------------ usage records (no request content)

const USAGE_FILE = path.join(CFG.dataDir, "usage.json");
const day = () => new Date().toISOString().slice(0, 10);
let usage = { days: {} }; // days[YYYY-MM-DD] = { total: {...}, ips: { hash: {...} } }
try {
  usage = JSON.parse(fs.readFileSync(USAGE_FILE, "utf8"));
} catch (_) { /* first start */ }
const blank = () => ({ requests: 0, ok: 0, errors: 0, rejected: 0, input_tokens: 0, output_tokens: 0 });
function bucket(d, ipHash) {
  const D = (usage.days[d] ||= { total: blank(), ips: {} });
  return ipHash ? (D.ips[ipHash] ||= blank()) : D.total;
}
let dirty = false;
function saveUsage() {
  if (!dirty) return;
  dirty = false;
  const keep = Object.keys(usage.days).sort().slice(-30); // 30 days of records
  usage.days = Object.fromEntries(keep.map((k) => [k, usage.days[k]]));
  fs.mkdirSync(CFG.dataDir, { recursive: true });
  const tmp = USAGE_FILE + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(usage));
  fs.renameSync(tmp, USAGE_FILE);
}
setInterval(saveUsage, 15_000).unref();
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => { saveUsage(); process.exit(0); });

function record(ipHash, field, tokens) {
  const d = day();
  for (const b of [bucket(d), bucket(d, ipHash)]) {
    b[field] += 1;
    if (field !== "rejected") b.requests += 1;
    if (tokens) {
      b.input_tokens += tokens.input_tokens || 0;
      b.output_tokens += tokens.output_tokens || 0;
    }
  }
  dirty = true;
}

// ------------------------------------------------------------------ client identity and limits

function clientKey(ip) {
  let a = String(ip || "").replace(/^::ffff:/, "");
  if (a.includes(":")) a = a.split(":").slice(0, 4).join(":") + "::/64"; // one IPv6 /64 = one client
  return a;
}
const hashIp = (k) => crypto.createHash("sha256").update(CFG.salt + k).digest("hex").slice(0, 16);

const minute = new Map(); // key -> [timestamps within the last 60 s]
const daily = new Map(); // key -> { day, n }
const lastSeen = new Map(); // key -> ms of last accepted request
let inflight = 0;

setInterval(() => {
  const now = Date.now();
  for (const [k, t] of lastSeen) if (now - t > CFG.activeWindowMs) lastSeen.delete(k);
  for (const [k, ts] of minute) if (!ts.length || now - ts[ts.length - 1] > 60_000) minute.delete(k);
  const d = day();
  for (const [k, v] of daily) if (v.day !== d) daily.delete(k);
}, 30_000).unref();

function dailyCount(k) {
  const d = day();
  const v = daily.get(k);
  return v && v.day === d ? v.n : 0;
}

function limitHeaders(res, k) {
  const now = Date.now();
  const ts = (minute.get(k) || []).filter((t) => now - t < 60_000);
  res.set("X-RateLimit-Limit-Requests", String(CFG.rpm));
  res.set("X-RateLimit-Remaining-Requests", String(Math.max(0, CFG.rpm - ts.length)));
  res.set("X-RateLimit-Limit-Requests-Day", String(CFG.rpd));
  res.set("X-RateLimit-Remaining-Requests-Day", String(Math.max(0, CFG.rpd - dailyCount(k))));
}

/** null when allowed (and counted), else [status, type, message, retryAfterSeconds]. */
function admit(k) {
  const now = Date.now();
  const ts = (minute.get(k) || []).filter((t) => now - t < 60_000);
  if (ts.length >= CFG.rpm) return [429, "rate_limit_error", `Free trial limit: ${CFG.rpm} requests per minute per IP.`, Math.ceil((ts[0] + 60_000 - now) / 1000)];
  if (dailyCount(k) >= CFG.rpd) {
    const tomorrow = new Date(); tomorrow.setUTCHours(24, 0, 0, 0);
    return [429, "rate_limit_error", `Free trial limit: ${CFG.rpd} requests per day per IP (resets 00:00 UTC).`, Math.ceil((tomorrow - now) / 1000)];
  }
  if (bucket(day()).ok + inflight >= CFG.globalRpd) return [529, "overloaded_error", "The free trial has used today's shared capacity. Please try again tomorrow.", 3600];
  if (!lastSeen.has(k) && lastSeen.size >= CFG.maxActiveUsers) return [529, "overloaded_error", "The free trial is full right now (too many active users). Please try again in a few minutes.", 300];
  if (inflight >= CFG.maxInflight) return [529, "overloaded_error", "Leo-1 is busy. Please retry in a few seconds.", 5];
  ts.push(now);
  minute.set(k, ts);
  daily.set(k, { day: day(), n: dailyCount(k) + 1 });
  lastSeen.set(k, now);
  return null;
}

// ------------------------------------------------------------------ app

const app = express();
app.disable("x-powered-by");
app.set("trust proxy", CFG.trustProxyHops);

const err = (res, status, type, message, extra) => res.status(status).json({ error: { type, message, ...(extra || {}) } });

app.use((req, res, next) => {
  res.set("X-Content-Type-Options", "nosniff");
  res.set("Referrer-Policy", "no-referrer");
  if (req.path.startsWith("/v1/")) {
    res.set("Access-Control-Allow-Origin", "*");
    res.set("Access-Control-Allow-Headers", "Authorization, Content-Type");
    res.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    if (req.method === "OPTIONS") return res.status(204).end();
  }
  next();
});

function authorized(req) {
  const [scheme, token] = String(req.get("authorization") || "").split(/\s+/, 2);
  return /^bearer$/i.test(scheme || "") && token === TRIAL_KEY;
}

app.get("/v1/models", (req, res) => {
  if (!authorized(req)) return err(res, 401, "authentication_error", 'Use "Authorization: Bearer free" for the Leo-1 free trial.');
  res.json({ models: [
    { name: CFG.model, description: "Leo-1: open-weight decision model (Qwen3-4B base, LoRA, calibrated). Free trial.", release_date: "2026-09-30" },
    { name: "leo-latest", description: `alias of ${CFG.model}`, release_date: "2026-09-30" },
  ] });
});

app.get("/v1/usage", (req, res) => {
  if (!authorized(req)) return err(res, 401, "authentication_error", 'Use "Authorization: Bearer free".');
  const k = clientKey(req.ip);
  limitHeaders(res, k);
  res.json({ day: day(), limits: { rpm: CFG.rpm, rpd: CFG.rpd }, used_today: { ...bucket(day(), hashIp(k)) } });
});

const jsonBody = express.json({ limit: `${CFG.maxBodyKb}kb`, strict: true });

async function forward(base, key, body) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), CFG.upstreamTimeoutMs);
  try {
    const r = await fetch(`${base}/v1/systemone`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctl.signal,
    });
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch (_) { data = null; }
    return { status: r.status, data };
  } finally {
    clearTimeout(timer);
  }
}

app.post("/v1/systemone", (req, res, next) => {
  if (!authorized(req)) return err(res, 401, "authentication_error", 'Use "Authorization: Bearer free" for the Leo-1 free trial.');
  next();
}, (req, res, next) => jsonBody(req, res, (e) => {
  if (!e) return next();
  if (e.type === "entity.too.large") return err(res, 413, "request_too_large", `Free trial requests are limited to ${CFG.maxBodyKb} KB.`);
  return err(res, 400, "invalid_request", "Body must be valid JSON.");
}), async (req, res) => {
  const k = clientKey(req.ip);
  const ipHash = hashIp(k);
  const b = req.body;
  if (!b || typeof b !== "object" || Array.isArray(b)) return err(res, 422, "invalid_request", "Body must be a JSON object with state and questions.");
  if (!b.questions || typeof b.questions !== "object" || Array.isArray(b.questions) || !Object.keys(b.questions).length)
    return err(res, 422, "invalid_request", "questions must be a non-empty object mapping ids to questions.");
  if (Object.keys(b.questions).length > CFG.maxQuestions)
    return err(res, 422, "invalid_request", `The free trial allows at most ${CFG.maxQuestions} questions per request.`);
  if (b.state === undefined) return err(res, 422, "invalid_request", "state is required (string, object or array).");
  if (JSON.stringify(b.state).length > CFG.maxStateChars)
    return err(res, 422, "invalid_request", `The free trial allows states up to ${CFG.maxStateChars} characters.`);

  const denied = admit(k);
  limitHeaders(res, k);
  if (denied) {
    const [status, type, message, retry] = denied;
    record(ipHash, "rejected");
    res.set("Retry-After", String(Math.max(1, retry)));
    return err(res, status, type, message);
  }
  const body = { state: b.state, model: CFG.model, questions: b.questions };
  inflight += 1;
  try {
    let out;
    try {
      out = await forward(CFG.upstream, CFG.upstreamKey, body);
      if (CFG.fallback && (out.status >= 500 || out.data === null)) out = await forward(CFG.fallback, CFG.fallbackKey, body);
    } catch (e) {
      if (!CFG.fallback) throw e;
      out = await forward(CFG.fallback, CFG.fallbackKey, body);
    }
    if (out.status === 200 && out.data && out.data.answers) {
      out.data.model = CFG.model;
      record(ipHash, "ok", out.data.usage);
      return res.json({ model: out.data.model, answers: out.data.answers, usage: out.data.usage });
    }
    record(ipHash, "errors");
    if (out.status === 422 || out.status === 413) return res.status(out.status).json(out.data || { error: { type: "invalid_request", message: "Request failed validation." } });
    return err(res, 529, "overloaded_error", "Leo-1 is unavailable right now. Please retry shortly.");
  } catch (e) {
    record(ipHash, "errors");
    const cold = e && e.name === "AbortError";
    return err(res, 529, "overloaded_error", cold ? "Leo-1 took too long to answer (it may be starting). Please retry in a minute." : "Leo-1 is unavailable right now. Please retry shortly.");
  } finally {
    inflight -= 1;
  }
});

app.get("/admin/usage", (req, res) => {
  if (!CFG.adminKey) return err(res, 404, "not_found", "Not found.");
  const got = Buffer.from(String(req.get("x-admin-key") || ""));
  const want = Buffer.from(CFG.adminKey);
  if (got.length !== want.length || !crypto.timingSafeEqual(got, want)) return err(res, 401, "authentication_error", "Bad admin key.");
  const out = {};
  for (const [d, D] of Object.entries(usage.days)) {
    out[d] = { total: D.total, unique_ips: Object.keys(D.ips).length,
      top_ips: Object.entries(D.ips).sort((a, b) => b[1].requests - a[1].requests).slice(0, 20) };
  }
  res.json({ now: new Date().toISOString(), active_users: lastSeen.size, inflight, config: {
    rpm: CFG.rpm, rpd: CFG.rpd, max_active_users: CFG.maxActiveUsers, global_rpd: CFG.globalRpd, fallback: !!CFG.fallback }, days: out });
});

app.get("/health", (req, res) => res.json({ status: "ok", model: CFG.model, active_users: lastSeen.size }));

app.use("/v1", (req, res) => err(res, 404, "not_found", "Unknown endpoint. Use POST /v1/systemone or GET /v1/models."));
app.use(express.static(path.join(__dirname, "public"), { extensions: ["html"], maxAge: "10m" }));
app.use((req, res) => res.status(404).sendFile(path.join(__dirname, "public", "index.html")));

// Hostinger's Node.js runner loads this file through its own wrapper, so listen() must run unconditionally.
// Tests set LEO_SITE_NO_LISTEN=1 and start their own listener.
if (process.env.LEO_SITE_NO_LISTEN !== "1") {
  if (!CFG.upstream || !CFG.upstreamKey) {
    console.error(JSON.stringify({ level: "error", msg: "LEO_UPSTREAM_URL and LEO_UPSTREAM_KEY must be set; /v1 will return 529" }));
  }
  app.listen(CFG.port, () => console.log(JSON.stringify({ level: "info", msg: "leo trial site listening", port: CFG.port })));
}

module.exports = { app, CFG, clientKey };
