from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables")
if not JINA_API_KEY:
    raise ValueError("Missing JINA_API_KEY in environment variables")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Configure Jina
JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_HEADERS = {"Authorization": f"Bearer {JINA_API_KEY}"}


# --- Load chunks from text file ---
def load_chunks(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    raw_chunks = content.split("--- Chunk ")
    chunks = []
    for chunk in raw_chunks[1:]:
        _, chunk_text = chunk.split("---\n", 1)
        chunks.append(chunk_text.strip())
    return chunks


# --- Embedding Functions ---
def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts with Jina API"""
    try:
        resp = requests.post(
            JINA_URL,
            headers=JINA_HEADERS,
            json={"input": texts, "model": "jina-embeddings-v2-base-en"}
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array([d["embedding"] for d in data["data"]])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Jina API error: {str(e)}")


def get_top_k_chunks(query: str, chunks: list[str], k=5):
    """Return top-k most relevant chunks for a query"""
    query_emb = embed_texts([query])
    chunk_embs = embed_texts(chunks)
    sims = cosine_similarity(query_emb, chunk_embs)[0]
    top_indices = sims.argsort()[::-1][:k]
    return [chunks[i] for i in top_indices]


# --- Gemini Decision Function ---
def generate_decision(user_query: str, retrieved_clauses: str):
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
    response = model.generate_content(prompt)
    return response.text


# --- FastAPI App ---
app = FastAPI(title="Insurance Assistant API")

class Query(BaseModel):
    user_query: str


# Load chunks once
CHUNKS = load_chunks("insurance_chunks.txt")


@app.post("/analyze")
def analyze_insurance(query: Query):
    try:
        top_chunks = get_top_k_chunks(query.user_query, CHUNKS)
        retrieved = "\n\n".join(top_chunks)
        decision = generate_decision(query.user_query, retrieved)
        return {
            "query": query.user_query,
            "retrieved_clauses": retrieved,
            "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
