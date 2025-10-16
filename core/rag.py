import random

fitness_knowledge = {
    "protein": "Protein helps build and repair muscle tissue.",
    "cardio": "Cardio exercises improve heart health and burn calories.",
    "hydration": "Drink enough water to aid metabolism and recovery."
}

def retrieve_context(prompt):
    # Simple keyword match simulation
    for key, val in fitness_knowledge.items():
        if key in prompt.lower():
            return val
    # fallback
    return random.choice(list(fitness_knowledge.values()))
