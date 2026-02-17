import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# Export the variable
__all__ = ["embedding_model"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_index(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings
