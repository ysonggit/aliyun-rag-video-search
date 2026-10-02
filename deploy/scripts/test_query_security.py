"""Security smoke test for the hardened query-service (no server, no DashVector calls).

Uses Flask's in-process test client to verify:
  - token gate (401 without / wrong token)
  - input validation (400 on bad input)
  - health check open without token

Run: python deploy/scripts/test_query_security.py
"""
import os
import sys
import importlib.util

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUERY_SVC = os.path.join(PROJECT, "deploy", "fc", "query-service")

# Env must be set BEFORE importing index.py (it reads at import time).
os.environ.setdefault("INTERNAL_TOKEN", "smoke-test-secret-token-12345")
os.environ.setdefault("DASHSCOPE_API_KEY", "dummy")
os.environ.setdefault("DASHVECTOR_API_KEY", "dummy")
os.environ.setdefault("DASHVECTOR_ENDPOINT", "dummy.endpoint")

sys.path.insert(0, PROJECT)
sys.path.insert(0, QUERY_SVC)

# index.py lives in a dir with a hyphen → import as a standalone module.
spec = importlib.util.spec_from_file_location(
    "qs_index", os.path.join(QUERY_SVC, "index.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app
client = app.test_client()

CORRECT = "smoke-test-secret-token-12345"

tests = [
    ("health / (no token)",      client.get("/"),                                                           200),
    ("/search no token",        client.get("/search?q=test"),                                              401),
    ("/search wrong token",     client.get("/search?q=test", headers={"X-Internal-Token": "wrong"}),       401),
    ("/search missing q",       client.get("/search", headers={"X-Internal-Token": CORRECT}),              400),
    ("/search too-long query",  client.get("/search?q=" + "x"*600, headers={"X-Internal-Token": CORRECT}), 400),
    ("/search invalid top_k",   client.get("/search?q=test&top_k=abc", headers={"X-Internal-Token": CORRECT}), 400),
    ("/search too many objects", client.get("/search?q=test&objects=" + ",".join(["obj"]*25),
                                              headers={"X-Internal-Token": CORRECT}),                       400),
]

print("=" * 70)
print("QUERY-SERVICE SECURITY SMOKE TEST")
print("=" * 70)
all_pass = True
for desc, resp, expected in tests:
    got = resp.status_code
    ok = got == expected
    all_pass = all_pass and ok
    body = resp.get_json()
    msg = body.get("error", "") if body else ""
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {desc}: got {got}, expected {expected}" + (f" — {msg}" if msg else ""))

print("=" * 70)
print("RESULT:", "ALL PASS" if all_pass else "SOME FAILED")
sys.exit(0 if all_pass else 1)
