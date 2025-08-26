import re, numpy as np
from collections import Counter
from .embed import cosine

SKILL_WORDS_HINT = [
    # lightweight default vocabulary; you can extend in-app
    "python","java","go","c++","docker","kubernetes","terraform","aws","gcp","azure",
    "linux","bash","ci/cd","jenkins","github actions","prometheus","grafana",
    "redis","postgres","mysql","microservices","rest","grpc","spark","hadoop",
    "pandas","numpy","scikit-learn","mlops","airflow","dbt","snowflake","databricks"
]

def normalize(txt):
    return re.sub(r"\s+", " ", (txt or "")).strip()

def token_counts(text):
    toks = re.findall(r"[a-zA-Z][a-zA-Z+\-/\.#0-9]*", text.lower())
    return Counter(toks)

def keyword_coverage(resume_text, jd_text, extra_vocab=None):
    vocab = set((extra_vocab or []) + SKILL_WORDS_HINT)
    rc, jc = token_counts(resume_text), token_counts(jd_text)
    present = sorted([w for w in vocab if rc[w] > 0 and jc[w] > 0])
    missing = sorted([w for w in vocab if rc[w] == 0 and jc[w] > 0])
    return present, missing

def score_similarity(resume_emb, jd_emb):
    return round(100 * cosine(resume_emb, jd_emb), 2)

def chunk(text, max_chars=4000):
    text = normalize(text)
    parts = []
    while text:
        parts.append(text[:max_chars])
        text = text[max_chars:]
    return parts or [""]
