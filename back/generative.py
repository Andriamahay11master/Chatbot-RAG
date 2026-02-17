import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#Load model function
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        #load_in_4bit=True,
        device_map="auto"
    )
    return tokenizer, model

def generate_answer(tokenizer, model, query, retrieved_chunks):
    context = "\n\n".join([
        f"Regulation {c['regulation']}({c['clause']}): {c['text']}"
        for c in retrieved_chunks
    ])

    prompt = f"""
You are a regulatory assistant for the UTM Resource Centre.
Answer strictly using the provided regulations.
Cite the regulation number.

Regulations:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
