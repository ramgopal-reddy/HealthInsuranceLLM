from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

genai.configure(api_key=GOOGLE_API_KEY)

# -------------------------
# Utility Functions
# -------------------------
def load_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    raw_chunks = content.split("--- Chunk ")
    chunks = []
    for chunk in raw_chunks[1:]:
        _, chunk_text = chunk.split("---\n", 1)
        chunks.append(chunk_text.strip())
    return chunks

def embed_text(text, task_type):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type=task_type
    )
    return response['embedding']

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embeddings.append(embed_text(chunk, task_type="retrieval_document"))
    return np.array(embeddings)

def get_top_k_chunks(query, chunks, chunk_embeddings, k=5):
    query_embedding = np.array(embed_text(query, task_type="retrieval_query")).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:k]
    return [chunks[i] for i in top_indices]

def generate_decision(user_query, retrieved_clauses):
    prompt = f"""
You are a health insurance assistant. Based on the user query and the retrieved policy clauses, make a decision.

## User Query:
{user_query}

## Retrieved Clauses:
{retrieved_clauses}

## Task:
1. Decide if the case should be APPROVED or REJECTED.
2. If approved, specify payout amount if mentioned.
3. Give a short justification using exact clause references.
4. Output result in the following JSON format:

{{
  "decision": "approved/rejected",
  "amount": "if mentioned",
  "justification": "clear reason based on clause"
}}
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# -------------------------
# FastAPI Setup
# -------------------------
app = FastAPI(title="Insurance Assistant API (Google Embeddings)")

class Query(BaseModel):
    user_query: str

# -------------------------
# Load Chunks + Embed Once
# -------------------------
CHUNKS = load_chunks("insurance_chunks.txt")
CHUNK_EMBEDDINGS = embed_chunks(CHUNKS)

# -------------------------
# API Endpoint
# -------------------------
@app.post("/analyze")
def analyze_insurance(query: Query):
    try:
        top_chunks = get_top_k_chunks(query.user_query, CHUNKS, CHUNK_EMBEDDINGS, k=5)
        retrieved_clauses = "\n\n".join(top_chunks)
        decision = generate_decision(query.user_query, retrieved_clauses)
        return {
            "query": query.user_query,
            "retrieved_clauses": retrieved_clauses,
            "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
