import requests
import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# Load env vars
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Load precomputed embeddings
with open("embeddings.pkl", "rb") as f:
    CHUNKS, CHUNK_EMBEDDINGS = pickle.load(f)

# Hugging Face embedding API for queries
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def embed_query(text: str):
    response = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text}
    )
    if response.status_code != 200:
        raise Exception(f"HuggingFace API error: {response.text}")
    return np.array(response.json()).mean(axis=0)  # average pooling

# Get top K chunks
def get_top_k_chunks(query, chunks, chunk_embeddings, k=5):
    query_embedding = embed_query(query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:k]
    return [chunks[i] for i in top_indices]

# Generate decision using Gemini
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
    response = model.generate_content(prompt)
    return response.text

# FastAPI app
app = FastAPI(title="Insurance Assistant API")

class Query(BaseModel):
    user_query: str

@app.post("/analyze")
def analyze_insurance(query: Query):
    try:
        top_chunks = get_top_k_chunks(query.user_query, CHUNKS, CHUNK_EMBEDDINGS)
        retrieved = "\n\n".join(top_chunks)
        decision = generate_decision(query.user_query, retrieved)
        return {
            "query": query.user_query,
            "retrieved_clauses": retrieved,
            "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
