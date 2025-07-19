# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import google.generativeai as genai
# from dotenv import load_dotenv
# import os

# # Load .env file (for GOOGLE_API_KEY)
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Load the Google embedding model
# embedding_model = genai.embed_content

# def load_chunks(file_path):
#     with open(file_path, "r", encoding="utf-8") as f:
#         content = f.read()

#     raw_chunks = content.split("--- Chunk ")
#     chunks = []
#     for chunk in raw_chunks[1:]:
#         _, chunk_text = chunk.split("---\n", 1)
#         chunks.append(chunk_text.strip())
#     return chunks

# def embed_chunks(chunks):
#     embeddings = []
#     for chunk in chunks:
#         res = embedding_model(
#             model="models/embedding-001",
#             content=chunk,
#             task_type="retrieval_document"
#         )
#         embeddings.append(res['embedding'])
#     return np.array(embeddings)

# def get_top_k_similar_chunks(query, chunks, chunk_embeddings, k=5):
#     query_embedding = embedding_model(
#         model="models/embedding-001",
#         content=query,
#         task_type="retrieval_query"
#     )['embedding']

#     query_embedding = np.array(query_embedding).reshape(1, -1)
#     similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
#     top_indices = similarities.argsort()[::-1][:k]
#     return [chunks[i] for i in top_indices]
