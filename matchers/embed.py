import numpy as np

# ---------- OpenAI ----------
def embed_openai(texts, api_key, model="text-embedding-3-large"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

# ---------- Gemini ----------
def embed_gemini(texts, api_key, model="text-embedding-004"):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    # The embeddings API expects a single string; we batch:
    embs = []
    for t in texts:
        r = genai.embed_content(model=model, content=t, task_type="semantic_similarity")
        embs.append(r["embedding"])
    return np.array(embs, dtype=np.float32)

def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float((a @ b.T))
