// Local tests of the trial site against a fake upstream (no GPU, no network):  node test.js
"use strict";
const http = require("http");
const assert = require("assert");

let upstreamCalls = 0;
const fake = http.createServer((req, res) => {
  let raw = "";
  req.on("data", (c) => (raw += c));
  req.on("end", () => {
    upstreamCalls += 1;
    if (req.headers.authorization !== "Bearer secret-upstream") { res.writeHead(401); return res.end("{}"); }
    const b = JSON.parse(raw);
    const answers = Object.fromEntries(Object.keys(b.questions).map((k) => [k, { type: "noul", noul: 0.9 }]));
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ model: "x", answers, usage: { input_tokens: 10, output_tokens: 5 } }));
  });
});

(async () => {
  await new Promise((r) => fake.listen(0, r));
  Object.assign(process.env, { LEO_UPSTREAM_URL: `http://127.0.0.1:${fake.address().port}`, LEO_UPSTREAM_KEY: "secret-upstream",
    LEO_SITE_NO_LISTEN: "1", RPM: "20", RPD: "25", MAX_ACTIVE_USERS: "3", ADMIN_KEY: "adm", DATA_DIR: require("os").tmpdir() + "/leo-site-test", TRUST_PROXY_HOPS: "1" });
  require("fs").rmSync(process.env.DATA_DIR, { recursive: true, force: true });
  const { app } = require("./server");
  const srv = await new Promise((r) => { const s = app.listen(0, () => r(s)); });
  const base = `http://127.0.0.1:${srv.address().port}`;
  const post = (body, ip = "1.1.1.1", key = "free") => fetch(`${base}/v1/systemone`, { method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "X-Forwarded-For": ip }, body: JSON.stringify(body) });
  const ok = { state: "hi", questions: { q: { type: "noul", instructions: "angry" } } };

  let r = await post(ok, "1.1.1.1", "wrong");
  assert.equal(r.status, 401, "bad key");
  r = await post(ok);
  let j = await r.json();
  assert.equal(r.status, 200); assert.equal(j.model, "leo-1"); assert.deepEqual(Object.keys(j), ["model", "answers", "usage"]);
  assert.equal(r.headers.get("x-ratelimit-remaining-requests"), "19");
  r = await post({ state: "x", questions: {} });
  assert.equal(r.status, 422, "empty questions");
  r = await post({ state: "x".repeat(30000), questions: ok.questions });
  assert.equal(r.status, 422, "state too long");
  r = await fetch(`${base}/v1/systemone`, { method: "POST", headers: { Authorization: "Bearer free", "Content-Type": "application/json" }, body: "{bad" });
  assert.equal(r.status, 400, "bad json");

  // 20 per minute: 19 more succeed, the 21st is 429
  for (let i = 0; i < 19; i++) assert.equal((await post(ok)).status, 200);
  r = await post(ok);
  assert.equal(r.status, 429, "rpm"); assert.ok(Number(r.headers.get("retry-after")) >= 1);

  // IPv6 in the same /64 share one limit
  const v6a = "2001:db8:1:2::10", v6b = "2001:db8:1:2::99";
  for (let i = 0; i < 20; i++) assert.equal((await post(ok, i % 2 ? v6a : v6b)).status, 200);
  assert.equal((await post(ok, v6a)).status, 429, "ipv6 /64 grouped");

  // max 3 active users: 1.1.1.1 and the v6 /64 are active; one more fits, the next new IP is turned away
  assert.equal((await post(ok, "3.3.3.3")).status, 200);
  r = await post(ok, "4.4.4.4");
  assert.equal(r.status, 529, "trial full for new users"); j = await r.json(); assert.equal(j.error.type, "overloaded_error");
  assert.equal((await post(ok, "3.3.3.3")).status, 200, "active user still served");

  // upstream key never leaks to clients; admin usage works and holds no raw IPs or content
  r = await fetch(`${base}/admin/usage`, { headers: { "x-admin-key": "adm" } });
  const text = await r.text();
  assert.equal(r.status, 200); assert.ok(!text.includes("1.1.1.1") && !text.includes("secret-upstream") && !text.includes("angry"));
  const day = Object.values(JSON.parse(text).days)[0];
  assert.equal(day.total.ok, 42, "successful calls recorded");
  assert.equal(day.total.input_tokens, 420, "input tokens recorded");
  assert.equal(day.total.output_tokens, 210, "output tokens recorded");
  assert.ok(day.total.rejected >= 3, "rejections recorded");
  assert.equal((await fetch(`${base}/admin/usage`, { headers: { "x-admin-key": "no" } })).status, 401);
  r = await fetch(`${base}/v1/models`, { headers: { Authorization: "Bearer free" } });
  assert.equal((await r.json()).models[0].name, "leo-1");
  r = await fetch(`${base}/`);
  assert.equal(r.status, 200); assert.ok((await r.text()).includes("Leo-1"));
  console.log(`all site tests passed (${upstreamCalls} upstream calls)`);
  srv.close(); fake.close(); process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
