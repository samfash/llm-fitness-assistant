---
title: LLM Fitness Assistant
emoji: 🏋️‍♂️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
python_version: "3.11"
pinned: false
---
<p align="center">
  <img src="https://raw.githubusercontent.com/samfash/llm-fitness-assistant/main/assets/banner.png" alt="LLM Fitness Assistant Banner" width="250">
</p>

<h1 align="center">🏋️‍♂️ LLM Fitness Assistant</h1>
<p align="center">Your personalized AI coach for workouts, diet, and motivation — powered by LLMs.</p>



This assistant helps users track workouts, manage diet, and stay motivated — combining LangChain, LoRA fine-tuning, and Chroma vector search for intelligence and memory.



---
## 🚀 Live Demo

👉 Try it here: [Hugging Face Space Link](https://huggingface.co/spaces/sam351/llm-fitness-assistant)

---


## 🛡️ Guardrails

`core/guardrails.py` is a deterministic safety layer (no model calls) around the LLM:

- **Input, hard block (model never called):** chest pain/fainting/breathing trouble, purging or extreme restriction, pregnancy, calorie or weight-loss targets for under-18s, daily intake below 1200 kcal, injury diagnosis requests, and attempts to extract the system prompt. Each returns a fixed, reviewed response that points to a professional.
- **Input, notice only:** prompt-injection text and cross-user requests. The model is warned to treat them as data, and the legitimate part of the request (e.g. logging the meal) still goes through.
- **Output rewrite:** replies that leak the system prompt, assert a diagnosis, or recommend a dangerously low daily intake.
- **Narrow on purpose:** negations ("no chest pain"), units ("12 weeks into my program", "15 minutes late"), deficits ("a 500 calorie deficit") and comma numbers ("2,000 calories") are handled, and 10 benign cases in `evals/cases/precision.json` must never be blocked.

Disable for A/B runs with `GUARDRAILS=off` or `python -m evals.run --no-guardrails`. Limits: regex rules miss paraphrases and non-English input, so treat this as a floor, not a complete safety system.

---

## 🧪 Evaluation

`evals/` holds a golden set of 33 cases (14 tool-calling, 9 safety, 10 guardrail-precision) and a harness that scores real model behaviour:

- **Tool calling:** did it call the right tool with the right arguments? Did it avoid logging plans or vague statements?
- **Numeric grounding:** every number >= 100 in a reply must come from a tool result or the user's own message, so the model cannot invent calorie or macro figures.
- **Safety:** referral on red-flag symptoms, no diagnosis, no extreme-diet or eating-disorder coaching, no system-prompt leak, no `user_id` smuggling, no obeying instructions embedded in logged text.
- **Gate:** `evals/thresholds.json` sets minimum pass rates; `--baseline` fails the build on regressions.

```bash
python -m evals.run --mode live --record --report evals/results/report.md   # needs HF_TOKEN
cp evals/results/latest.json evals/baseline.json                           # commit as the baseline
python -m evals.run --mode replay                                          # CI: deterministic, no network
```

`--record` saves each model transcript to `evals/recordings/`. Committing them lets CI replay the exact model outputs on every push, which catches regressions in the agent loop, tools, validation and scoring without paying for or depending on a live model. The weekly `evals.yml` workflow re-measures the live model.

Results (fill in after your first live run): _not yet measured_.

---

## ✨ What it does (v2)

Say "Bench 80kg 3x8, then chicken salad ~450 kcal" and the assistant **records it**: the LLM calls validated tools that write to a database, and all numbers (BMR/TDEE, macros, 1RM, progress) come from tested Python, not from the model.

| Layer | File | Notes |
|---|---|---|
| Schemas (Pydantic) | `core/schemas.py` | Validation + the JSON schemas the LLM sees |
| Calculators | `core/calculators.py` | Mifflin-St Jeor, Epley 1RM, calorie floors |
| Persistence | `core/db.py` | SQLite (set `FITNESS_DB_PATH`); ephemeral on free Spaces |
| Tools | `core/tools.py` | `user_id` is injected by the app, never by the model |
| Agent loop | `core/agent.py` | Tool-calling loop, step limit, error feedback |
| Tests | `tests/` | `pytest`, uses a scripted fake LLM (no network) |

Requires an `HF_TOKEN` Space secret for tool calling. Without it the app falls back to basic chat. Set `HF_MODEL` to a chat model that supports tool calling on Inference Providers.

Run the tests: `pip install pydantic pytest && pytest -q`

---

🧠 Overview



The LLM Fitness Assistant blends modern AI techniques to simulate a real fitness coach:



💬 Conversational interaction — natural language input for workouts and meals  

🔍 RAG (Retrieval-Augmented Generation) — fetches up-to-date fitness and nutrition data  

🧩 LoRA fine-tuning — adapts the base model’s tone and behavior to act like a coach  

🧠 Memory (via Chroma) — remembers user goals, progress, and preferences  

⚙ FastAPI backend — lightweight, scalable REST API for serving responses  



---



🏗️ System Architecture



```text

User Input

&nbsp;  ↓

Prompt Template (Fitness-specific)

&nbsp;  ↓

RAG Retriever (Chroma Vector DB)

&nbsp;  ↓

LoRA Fine-Tuned Model (Hugging Face Transformers)

&nbsp;  ↓

LangChain Agent (decides actions: track, suggest, motivate)

&nbsp;  ↓

Memory Store (Chroma)

&nbsp;  ↓

FastAPI Endpoint → JSON Response
```


🚀 Quick Start



1️⃣ Clone the Repository
```bash
git clone https://github.com/samfash/llm-fitness-assistant.git
cd llm-fitness-assistant
 ```

2️⃣ Install Dependencies
 ```
pip install -r requirements.txt
 ```

3️⃣ Set Environment Variables

Create a .env file in the project root:
 ```
HUGGINGFACE_API_KEY=your_huggingface_token
CHROMA_PATH=./chroma_db
 ```

4️⃣ Run the FastAPI Server
 ```
uvicorn api.api_server:app --reload
 ```


Then visit 👉 http://127.0.0.1:8000/docs
 for the Swagger UI.


🧩 Example API Usage

Request
 ```
POST /chat
{
  "user_id": "demo",
  "prompt": "Plan my workout for the next 3 days focused on strength training."
}
 ```

Response
 ```
{
  "assistant_reply": "Here's a 3-day strength training plan: Day 1 - Push (Chest/Triceps)..."
}
 ```
⚡ Run the UI

Once your FastAPI backend is running (uvicorn api.api_server:app --reload):

Open another terminal:
```
streamlit run ui/ui_streamlit.py
```

Then go to:
👉 http://localhost:8501
---
🧰 Tech Stack
Component	Tool
LLM Orchestration	LangChain

Model	Hugging Face LoRA fine-tuned model
Vector Database	ChromaDB

API Framework	FastAPI

Memory & RAG	LangChain + Chroma retriever
Environment	Python 3.11+
---
🧪 Development Notes

LoRA fine-tuning is done on a base instruction model (e.g., mistralai/Mistral-7B-Instruct or falcon-7b).

Chroma stores embeddings of user logs, nutrition data, and workout templates.

FastAPI serves as the interface for both a frontend app and other systems (e.g., mobile or chatbot UI).

---
🧩 Future Improvements

Integrate progress visualization dashboard (Streamlit)

Add reminder agent for workout schedules

Support multimodal inputs (e.g., food photo → calorie estimation)

🧑‍💻 Example Prompt Template
 ```
You are a friendly and motivating fitness assistant.
You help users track workouts, plan meals, and stay consistent.
Use the retrieved context to make informed recommendations.

Context:
{retrieved_data}

User Query:
{user_input}
 ```
---
📄 License

MIT License © 2025 [Samuel Fasanya](https://fashrockweb.vercel.app/)

📫 Connect: [LinkedIn](https://www.linkedin.com/in/samuel-fasanya-b9ba41112/)



