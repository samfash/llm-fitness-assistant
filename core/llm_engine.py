from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Load fine-tuned weights
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(base, "./lora-fitness")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt):
    out = pipe(prompt, max_length=200, do_sample=True)
    return out[0]["generated_text"]
