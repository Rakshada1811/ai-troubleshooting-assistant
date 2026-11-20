import time

import requests

URL = "http://127.0.0.1:8000/ask"

# (query, expected KB id or None if it should be refused)
TESTS = [
    ("The payment API is returning a 500 error", "KB001"),
    ("Getting too many connections error from the database", "KB002"),
    ("API responses are very slow and timing out", "KB003"),
    ("No space left on device on the server", "KB004"),
    ("Site shows certificate expired warning", "KB005"),
    ("Pipeline failed during the release", "KB006"),
    ("Server CPU is at 100 percent", "KB007"),
    ("Service keeps getting OOM killed", "KB008"),
    ("Users getting 401 unauthorized on login", "KB009"),
    ("Kafka consumer lag is very high", "KB010"),
    ("Cannot reach the service, connection refused", "KB011"),
    ("Bank gateway is not responding", "KB012"),
    ("Settlement file is missing and counts do not match", "KB013"),
    ("Pod keeps restarting in kubernetes", "KB014"),
    ("Backup did not run last night", "KB015"),
    ("My laptop screen is flickering", None),
    ("What is the capital of France", None),
    ("The printer on floor 3 is jammed", None),
]

passed = 0
for query, expected in TESTS:
    start = time.time()
    try:
        d = requests.post(URL, json={"message": query}, timeout=200).json()
    except Exception as ex:
        print(f"ERROR   | {query} | {ex}")
        continue
    secs = round(time.time() - start, 1)
    got = d["sources"][0]["id"] if d["sources"] else None
    ok = got == expected
    passed += ok
    print(f"{'PASS' if ok else 'FAIL'}    | {query}")
    print(f"         expected={expected} got={got} mode={d['mode']} time={secs}s")
    for err in d.get("errors", []):
        print(f"         error: {err[:200]}")

print(f"\n{passed}/{len(TESTS)} passed")