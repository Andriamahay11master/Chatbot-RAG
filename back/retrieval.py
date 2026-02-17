from .embeddings import embedding_model

def retrieve(query, index, chunks, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []
    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks
