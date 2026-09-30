"""Live check of https://leo.kognare.com (uses a few of this machine's 200 free daily requests).

    python site/live_check.py
"""
import time

import httpx

U = "https://leo.kognare.com"
H = {"Authorization": "Bearer free"}
c = httpx.Client(timeout=180)


def show(label, r):
    body = r.text if len(r.text) < 400 else r.text[:400] + "..."
    print(f"{label:<34} {r.status_code}  {body}")


r = c.get(U + "/")
print(f"{'front page':<34} {r.status_code}  {len(r.text)} bytes, playground: {'id=\"play\"' in r.text}, notice: {'Notice' in r.text}")
show("GET /v1/models without key", c.get(U + "/v1/models"))
show("GET /v1/models, key free", c.get(U + "/v1/models", headers=H))
show("POST wrong key", c.post(U + "/v1/systemone", headers={"Authorization": "Bearer abc"}, json={"state": "x", "questions": {"q": {"type": "noul"}}}))
show("POST invalid score (1 level)", c.post(U + "/v1/systemone", headers=H,
                                             json={"state": "x", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}}))
body = {"model": "jev-latest",
        "state": {"ticket": {"subject": "Charged twice", "body": "Third time writing: I was billed twice and nobody refunded me. Fix it today."}},
        "questions": {"angry": {"type": "noul", "instructions": "angry"},
                      "refund": {"type": "noul", "instructions": "asking for a refund"},
                      "spam": {"type": "noul", "instructions": "spam"},
                      "team": {"type": "choice", "instructions": "Which team should handle `ticket`?",
                               "criteria": {"billing": "charges, refunds", "technical": "bugs", "other": None}},
                      "anger": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "very angry"]}}}
for i in range(3):
    t = time.perf_counter()
    r = c.post(U + "/v1/systemone", headers=H, json=body)
    ms = (time.perf_counter() - t) * 1000
    show(f"POST /v1/systemone #{i + 1} ({ms:.0f} ms)", r)
print("rate-limit headers:", {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")})
show("GET /v1/usage", c.get(U + "/v1/usage", headers=H))
show("GET /admin/usage without key", c.get(U + "/admin/usage"))
