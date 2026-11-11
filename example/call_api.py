import json
import os
from pathlib import Path

import requests

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
API_KEY = os.environ["API_KEY"]  # create with: python -m core.auth create <user>
here = Path(__file__).parent

for f in sorted(here.glob("example_*.json")):
    body = json.loads(f.read_text())["request"]["body"]
    r = requests.post(BASE_URL, json={"prompt": body["user_input"]}, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120)
    print(f"\n{f.name} -> {r.status_code}")
    print(r.json())
