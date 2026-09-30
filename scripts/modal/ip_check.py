"""One request to leo.kognare.com from a Modal container (a different network than this PC), to check that the
trial site tells clients apart by IP.   modal run scripts/modal/ip_check.py"""
import modal

app = modal.App("leo-ip-check")
image = modal.Image.debian_slim(python_version="3.12").pip_install("httpx==0.28.1")


@app.function(image=image, cpu=0.25, memory=256, timeout=300)
def probe() -> dict:
    import httpx

    ip = httpx.get("https://api.ipify.org", timeout=30).text
    r = httpx.post("https://leo.kognare.com/v1/systemone", headers={"Authorization": "Bearer free"}, timeout=180,
                   json={"state": "hello from a second network", "questions": {"q": {"type": "noul", "instructions": "greeting"}}})
    u = httpx.get("https://leo.kognare.com/v1/usage", headers={"Authorization": "Bearer free"}, timeout=60)
    return {"egress_ip": ip, "status": r.status_code, "usage_seen_by_site": u.json()}


@app.local_entrypoint()
def main() -> None:
    print(probe.remote())
