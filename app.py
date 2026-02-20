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

# start the Flask app
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    
    # Here you would implement the logic to use the vector index and generative model to get a response
    # For demonstration, we'll just return the user query
    response = {
        'response': f"Received query: {user_query}"
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
    