import json
from pathlib import Path

import requests

BASE_URL = "http://127.0.0.1:8000/chat"
here = Path(__file__).parent

for f in sorted(here.glob("example_*.json")):
    body = json.loads(f.read_text())["request"]["body"]
    payload = {"user_id": "test", "prompt": body["user_input"]}
    r = requests.post(BASE_URL, json=payload, timeout=120)
    print(f"\n{f.name} -> {r.status_code}")
    print(r.json())
