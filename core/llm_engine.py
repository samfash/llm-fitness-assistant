from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

model_name = "mistralai/Mistral-7B-v0.1"  # or "tiiuae/falcon-7b-instruct" or "meta-llama/Llama-2-7b-chat-hf"

# Load fine-tuned weights
base = AutoModelForCausalLM.from_pretrained( model_name)
model = PeftModel.from_pretrained(base, "./lora-fitness")
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt):
    out = pipe(prompt, max_length=200, do_sample=True)
    return out[0]["generated_text"]
