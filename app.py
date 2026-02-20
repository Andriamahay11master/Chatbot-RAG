# Load your regulation text
from back.embeddings import create_vector_index
from back.preprocessing import clause_aware_chunking, clean_text, extract_pdf_text
from back.generative import evaluate_model


pdf_path = "/data/RCentre.pdf"  # Your PDF file

raw_text = extract_pdf_text(pdf_path)
cleaned_text = clean_text(raw_text)

chunks = clause_aware_chunking(cleaned_text)

index, embeddings = create_vector_index(chunks)

#Pipeline execution for three models
results_qwen = evaluate_model("Qwen/Qwen2.5-7B-Instruct")
print(results_qwen)

results_mistral = evaluate_model("mistralai/Mistral-7B-Instruct-v0.2")
print(results_mistral)

results_phi = evaluate_model("microsoft/Phi-3.5-mini-instruct")
print(results_phi)