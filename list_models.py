import requests

r = requests.get("https://openrouter.ai/api/v1/models")
for m in r.json()["data"]:
    if m["id"].endswith(":free"):
        print(m["id"])