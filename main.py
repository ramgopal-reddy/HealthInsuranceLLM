from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import google.generativeai as genai
from utils import load_chunks_from_file, setup_chroma_collection, query_clauses

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# FastAPI app
app = FastAPI(title="Health Insurance Assistant API")

# Load and embed chunks on startup
chunks = load_chunks_from_file("insurance_chunks.txt")
collection = setup_chroma_collection(chunks)

# Request schema
class InsuranceQuery(BaseModel):
    query: str

# Gemini decision logic
def generate_insurance_decision(user_query, retrieved_clauses):
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

# API endpoint
@app.post("/analyze")
def analyze_insurance(query: InsuranceQuery):
    try:
        retrieved_clauses = query_clauses(collection, query.query)
        decision = generate_insurance_decision(query.query, retrieved_clauses)
        return {
            "query": query.query,
            "retrieved_clauses": retrieved_clauses,
            "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
