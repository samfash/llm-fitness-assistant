import requests, json

BASE_URL = "http://127.0.0.1:8000/query"

examples = [
    "example_1_basic_query.json",
    "example_2_diet_tracking.json",
    "example_3_workout_planner.json"
]

for file in examples:
    data = json.load(open(f"examples/{file}"))
    r = requests.post(BASE_URL, json=data["request"]["body"])
    print(f"\n🔹 {file}")
    print("Response:", r.json())
