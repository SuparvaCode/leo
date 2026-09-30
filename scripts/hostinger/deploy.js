// Deploy site/ to a Hostinger Node.js website through the Hostinger API (same flow as Hostinger's own
// MCP "deploy JS application" tool): archive -> upload URL -> TUS upload -> build settings + env -> build.
//
//   node scripts/hostinger/deploy.js leo.kognare.com
//
// Reads HOSTINGER_API_TOKEN and the site's secrets from the repo .env (git-ignored). Secrets are sent as
// Hostinger environment variables; they are never put in the archive.
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const tus = require("tus-js-client");

const ROOT = path.resolve(__dirname, "..", "..");
const SITE = path.join(ROOT, "site");
const API = "https://developers.hostinger.com/api/hosting/v1";
const envFile = Object.fromEntries(fs.readFileSync(path.join(ROOT, ".env"), "utf8").split(/\r?\n/)
  .filter((l) => /^[A-Z_]+=/.test(l)).map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]));
const TOKEN = envFile.HOSTINGER_API_TOKEN;
const domain = process.argv[2];
if (!TOKEN || !domain) { console.error("usage: node deploy.js <domain>  (HOSTINGER_API_TOKEN in .env)"); process.exit(1); }

async function api(method, p, body) {
  const r = await fetch(API + p, { method, headers: { Authorization: `Bearer ${TOKEN}`, Accept: "application/json",
    ...(body ? { "Content-Type": "application/json" } : {}) }, body: body ? JSON.stringify(body) : undefined });
  const text = await r.text();
  let data; try { data = JSON.parse(text); } catch (_) { data = text; }
  if (!r.ok) throw new Error(`${method} ${p} -> ${r.status}: ${typeof data === "string" ? data.slice(0, 400) : JSON.stringify(data).slice(0, 400)}`);
  return data;
}

function archive() {
  const out = path.join(os.tmpdir(), `leo-site-${Date.now()}.zip`);
  const files = ["package.json", "package-lock.json", "server.js", "public"].map((f) => path.join(SITE, f));
  const ps = `Compress-Archive -Path ${files.map((f) => `'${f}'`).join(",")} -DestinationPath '${out}' -Force`;
  execFileSync("powershell", ["-NoProfile", "-Command", ps], { stdio: "inherit" });
  return out;
}

async function upload(file, url, authKey, restKey) {
  const name = path.basename(file);
  const target = `${url.replace(/\/$/, "")}/${name}?override=true`;
  const size = fs.statSync(file).size;
  const headers = { "X-Auth": authKey, "X-Auth-Rest": restKey, "upload-length": String(size), "upload-offset": "0" };
  const pre = await fetch(target, { method: "POST", headers, body: "" });
  if (pre.status !== 201) throw new Error(`pre-upload -> ${pre.status}: ${(await pre.text()).slice(0, 300)}`);
  await new Promise((resolve, reject) => {
    const up = new tus.Upload(fs.createReadStream(file), { uploadUrl: target, retryDelays: [1000, 2000, 4000, 8000],
      uploadDataDuringCreation: false, parallelUploads: 1, chunkSize: 10 * 1024 * 1024, headers,
      removeFingerprintOnSuccess: true, uploadSize: size, metadata: { filename: name },
      onError: reject, onSuccess: resolve });
    up.start();
  });
  return name;
}

(async () => {
  const site = (await api("GET", `/websites?domain=${encodeURIComponent(domain)}`)).data.find((w) => w.domain === domain);
  if (!site) throw new Error(`no website ${domain}`);
  const user = site.username;
  const base = `/accounts/${user}/websites/${domain}/nodejs`;
  const zip = archive();
  console.log("archive", zip, fs.statSync(zip).size, "bytes");
  const cred = await api("POST", "/files/upload-urls", { username: user, domain });
  const name = await upload(zip, cred.url, cred.auth_key, cred.rest_auth_key);
  console.log("uploaded", name);
  const settings = { node_version: 22, app_type: "express", root_directory: ".", entry_file: "server.js",
    build_script: null, output_directory: null, package_manager: "npm" };
  await api("PUT", `${base}/builds/settings`, settings).then(() => console.log("build settings saved"));
  const vars = { LEO_UPSTREAM_URL: envFile.LEO_UPSTREAM_URL, LEO_UPSTREAM_KEY: envFile.LEO_UPSTREAM_KEY,
    ADMIN_KEY: envFile.LEO_SITE_ADMIN_KEY, IP_HASH_SALT: envFile.LEO_SITE_IP_SALT, NODE_ENV: "production" };
  for (const [k, v] of Object.entries(vars)) if (!v) throw new Error(`missing ${k} source in .env`);
  await api("PUT", `${base}/builds/settings/env`, { env_vars: Object.entries(vars).map(([key, value]) => ({ key, value })) })
    .then(() => console.log("environment variables set:", Object.keys(vars).join(", ")));
  const build = await api("POST", `${base}/builds`, { ...settings, source_type: "archive", source_options: { archive_path: name } });
  const uuid = build.uuid || (build.data && build.data.uuid);
  console.log("build started", uuid || JSON.stringify(build).slice(0, 200));
  for (let i = 0; uuid && i < 80; i++) {
    await new Promise((r) => setTimeout(r, 8000));
    const b = await api("GET", `${base}/builds/${uuid}`);
    const state = b.state || (b.data && b.data.state);
    process.stdout.write(`build ${state}\n`);
    if (state === "completed" || state === "failed") {
      if (state === "failed") {
        const logs = await api("GET", `${base}/builds/${uuid}/logs`);
        console.log(JSON.stringify(logs).slice(-3000));
        process.exit(2);
      }
      break;
    }
  }
  fs.unlinkSync(zip);
})().catch((e) => { console.error("DEPLOY FAILED:", e.message); process.exit(1); });
