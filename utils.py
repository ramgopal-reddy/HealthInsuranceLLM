import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

persist_dir = "chroma_store"
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_chunks_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    raw_chunks = content.split("--- Chunk ")
    chunks = []
    for chunk in raw_chunks[1:]:
        _, chunk_text = chunk.split("---\n", 1)
        chunks.append(chunk_text.strip())
    return chunks

def setup_chroma_collection(chunks):
    chroma_client = chromadb.Client(Settings(
        persist_directory=persist_dir,
        anonymized_telemetry=False
    ))

    collection = chroma_client.get_or_create_collection("insurance_clauses")

    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": f"chunk_{i}"}],
            embeddings=[embedding.tolist()],
            ids=[f"chunk_{i}"]
        )
    return collection

def query_clauses(collection, query):
    query_embedding = embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    return "\n\n".join(results["documents"][0])
