import re
from collections import Counter

def _tokens(s: str):
    return [w.lower() for w in re.findall(r"[A-Za-z0-9#+.]+", s)]

def match_score(resume: str, jd: str):
    r, j = Counter(_tokens(resume)), Counter(_tokens(jd))
    overlap = sum(min(r[t], j[t]) for t in j)
    total = sum(j.values()) or 1
    score = round(100 * overlap / total, 1)
    return {"match_score": score}

def keyword_gaps(resume: str, jd: str, top_k: int = 30):
    rset = set(_tokens(resume))
    jset = _tokens(jd)
    missing = []
    for w in jset:
        if w not in rset and w not in missing:
            missing.append(w)
        if len(missing) >= top_k:
            break
    return {"missing": [(w, 1.0) for w in missing]}
