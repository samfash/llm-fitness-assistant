"""LLM backend.

Order of preference:
1. Hugging Face Inference API (needs an HF_TOKEN secret) - works on free CPU Spaces.
2. A small local model on CPU (fallback, slow but self-contained).
3. Optional LoRA adapter on the local model if LORA_REPO is set (a Hub repo id,
   NOT a local folder that was never committed).

Nothing heavy is loaded at import time, so the Space boots immediately.
"""
import logging
import os
from functools import lru_cache

log = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
REMOTE_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LORA_REPO = os.getenv("LORA_REPO")  # e.g. "sam351/fitness-lora" (must match LOCAL_MODEL's base)

SYSTEM_PROMPT = (
    "You are a friendly, motivating personal fitness assistant. "
    "Use the user's past conversation and the retrieved knowledge to give "
    "practical, safe advice about workouts, diet and progress. Be concise."
)


def _build_messages(user_input, context="", knowledge=""):
    system = SYSTEM_PROMPT
    if knowledge:
        system += f"\n\nRetrieved knowledge:\n{knowledge}"
    if context:
        system += f"\n\nPast conversation:\n{context}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]


def _remote(messages):
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=REMOTE_MODEL, token=HF_TOKEN, timeout=60)
    out = client.chat_completion(messages, max_tokens=400, temperature=0.7)
    return out.choices[0].message.content.strip()


@lru_cache(maxsize=1)
def _local_pipe():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, torch_dtype=torch.float32)
    if LORA_REPO:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, LORA_REPO)
    model.eval()
    return tok, model


def _local(messages):
    import torch

    tok, model = _local_pipe()
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)
    # decode only the newly generated tokens (no echo of the prompt)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()


def generate_response(user_input, context="", knowledge=""):
    messages = _build_messages(user_input, context, knowledge)
    if HF_TOKEN:
        try:
            return _remote(messages)
        except Exception as e:  # network, quota, model not served...
            log.warning("Remote inference failed (%s); falling back to local model", e)
    return _local(messages)
