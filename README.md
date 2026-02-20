# Chatbot-RAG: Regulatory Question-Answering System

A powerful **Retrieval-Augmented Generation (RAG)** chatbot designed to provide accurate answers about regulations and compliance requirements. This system combines semantic search with large language models to deliver precise, cited responses based on official regulatory documents.

## Overview

The Chatbot-RAG system leverages advanced NLP techniques to:

- **Extract and process** regulatory documents from PDFs
- **Retrieve** relevant regulatory clauses using semantic similarity search
- **Generate** accurate answers with proper citations using state-of-the-art language models
- **Provide** a user-friendly interface for regulatory queries

## Key Features

✨ **Semantic Search** - Uses FAISS and sentence-transformers for fast, accurate document retrieval  
🤖 **LLM-Powered** - Leverages Hugging Face Transformers and Mistral AI for natural language generation  
📄 **PDF Support** - Automatically extracts and processes regulatory documents  
🔗 **Citation Tracking** - Provides regulation references for full traceability  
⚡ **Flask API** - RESTful API with CORS support for easy integration  
🎯 **RAG Architecture** - Combines retrieval and generation for factually grounded responses

## Architecture

```
Input Query
    ↓
Embedding & Retrieval (FAISS + Sentence-Transformers)
    ↓
Retrieved Regulatory Clauses
    ↓
Generative Model (Mistral AI via Hugging Face)
    ↓
Cited Response with Regulation References
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)
- 8GB+ RAM

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Chatbot-RAG
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support, ensure CUDA and cuDNN are properly installed. The requirements include both `faiss-cpu` and `faiss-gpu` options.

## Project Structure

```
Chatbot-RAG/
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── replacements.txt            # Text replacement rules for PDF processing
│
├── back/                       # Backend modules
│   ├── embeddings.py          # Embedding model initialization
│   ├── retrieval.py           # Vector search and document retrieval
│   ├── preprocessing.py       # PDF extraction and text cleaning
│   └── generative.py          # LLM-based answer generation
│
├── data/                       # Data storage (PDFs, embeddings, indices)
│   └── (regulatory documents)
│
└── experience/                 # Experiments and notebooks
    └── experiment.ipynb       # Experiment and testing notebooks
```

## Usage

### Quick Start

1. **Prepare your data:**
   - Place regulatory PDF documents in the `data/` directory
   - Run preprocessing to extract and clean text
   - Build FAISS index for semantic search

2. **Run the application:**

```bash
python app.py
```

3. **Query the chatbot:**
   - Send HTTP requests to the Flask API
   - Provide your regulatory question
   - Receive answers with regulation citations

### Example Query

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the requirements for UTM resource allocation?"}'
```

## Configuration

### Model Selection

Edit the model paths in `back/generative.py` and `back/embeddings.py`:

- **Generative Model:** Currently uses Mistral AI
- **Embedding Model:** Uses sentence-transformers (all-MiniLM-L6-v2 recommended)

### Retrieval Settings

In `back/retrieval.py`, adjust `top_k` parameter to control the number of retrieved clauses (default: 3).

## Dependencies

### Core Libraries

- **torch, transformers** - Deep learning and language models
- **sentence-transformers** - Semantic text embeddings
- **faiss-cpu/gpu** - Vector similarity search
- **pdfplumber** - PDF text extraction
- **flask** - Web framework

See [requirements.txt](requirements.txt) for complete dependency list and versions.

## Development

### Running Experiments

Use the Jupyter notebook for experimentation:

```bash
jupyter notebook experience/experiment.ipynb
```

### PDF Processing Pipeline

1. **Extraction** (`preprocessing.py`):
   - Extract text from PDFs page by page
   - Remove repeated headers and page numbers
2. **Chunking:**
   - Split documents into regulation clauses
   - Maintain section hierarchy and numbering
3. **Cleaning:**
   - Apply text replacements (see `replacements.txt`)
   - Normalize formatting

## API Reference

### POST /query

Query the chatbot with a regulatory question.

**Request:**

```json
{
  "question": "your regulatory question here"
}
```

**Response:**

```json
{
  "answer": "answer text",
  "citations": [
    {
      "regulation": "Regulation ID",
      "clause": "Clause number",
      "text": "relevant excerpt"
    }
  ]
}
```

## Performance Optimization

- **GPU Acceleration:** Install CUDA-compatible PyTorch and faiss-gpu for faster inference
- **Model Quantization:** Consider INT4 or INT8 quantization for memory efficiency
- **Batch Processing:** Process multiple queries in parallel when possible

## Troubleshooting

### Common Issues

**Out of Memory (OOM):**

- Reduce batch size
- Enable model quantization in `generative.py` (uncomment `load_in_4bit=True`)
- Use CPU inference as fallback

**Slow Retrieval:**

- Ensure FAISS index is properly built
- Check that GPU acceleration is enabled if available

**Poor Answer Quality:**

- Verify PDF documents are properly processed
- Increase `top_k` in retrieval settings
- Review retrieved clauses for relevance

## Contributing

Contributions are welcome! Please:

1. Create a feature branch
2. Make your changes
3. Test thoroughly with the experiment notebook
4. Submit a pull request

## License

[Add your license information here]

## Contact

For questions or support, please contact the development team.

---

**Last Updated:** February 2026
