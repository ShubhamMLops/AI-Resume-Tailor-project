# ai/jd_keywords.py
from typing import List, Dict
import re
import logging

logger = logging.getLogger(__name__)

# Optional packages
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    import yake
    YAKE_AVAILABLE = True
except Exception:
    YAKE_AVAILABLE = False

# strong JD stopwords / blacklist
STOPWORDS = set("""
the and a an of for to in on with by as that is are be or from at you your we us our will should ability
role roles team teams experience skills responsibilities responsibility candidate candidates requirement
requirements preferred including work working years year strong good knowledge knowledgeable ability able perform
provide ensure etc such may position positions project projects software system systems application applications
business task tasks company organization
""".split())

# token regex
token_re = re.compile(r"[A-Za-z0-9\+\#\.\-/]+")

# --- spaCy lazy loader ---
_spacy_nlp = None
def _get_spacy(nlp_name: str = "en_core_web_sm"):
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    if not SPACY_AVAILABLE:
        return None
    try:
        _spacy_nlp = spacy.load(nlp_name)
    except Exception:
        try:
            _spacy_nlp = spacy.blank("en")
        except Exception:
            _spacy_nlp = None
    return _spacy_nlp

def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', (s or "").strip()).lower()

def simple_tokenize(s: str) -> List[str]:
    return [t for t in token_re.findall(s or "") if t and t.lower() not in STOPWORDS]

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    nlp = _get_spacy()
    if not nlp:
        return [t.lower() for t in tokens]
    doc = nlp(" ".join(tokens))
    return [t.lemma_.lower() for t in doc if t.text.strip()]

# Candidate generation
def _candidate_phrases_basic(text: str, max_ngram: int = 3) -> List[str]:
    cleaned = re.sub(r'[^A-Za-z0-9\+\#\.\-/ ]+', ' ', text or "")
    tokens = [t for t in cleaned.split() if t and t.lower() not in STOPWORDS and len(t) > 1]
    candidates = set()
    n = len(tokens)
    for k in range(1, max_ngram + 1):
        for i in range(0, n - k + 1):
            phrase = " ".join(tokens[i:i + k])
            if not re.match(r'^\d+$', phrase) and len(phrase) >= 2:
                candidates.add(phrase)
    return list(candidates)

def _candidate_phrases_spacy(jd_text: str, max_phrases: int = 300) -> List[str]:
    nlp = _get_spacy()
    if not nlp:
        return []
    doc = nlp(jd_text)
    candidates = []
    for nc in doc.noun_chunks:
        txt = nc.text.strip()
        if txt and txt.lower() not in STOPWORDS:
            candidates.append(txt)
    for ent in doc.ents:
        e = ent.text.strip()
        if e and e.lower() not in STOPWORDS:
            candidates.append(e)
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and token.is_alpha and len(token.text) > 1:
            candidates.append(token.text)
    seen = set()
    out = []
    for c in candidates:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            out.append(c)
    return out[:max_phrases]

def _merge_yake(jd_text: str, candidates: List[str], max_keywords: int = 80) -> List[str]:
    if not YAKE_AVAILABLE:
        return candidates
    try:
        kw_ex = yake.KeywordExtractor(lan="en", top=max_keywords)
        yake_kws = [kw for kw, score in kw_ex.extract_keywords(jd_text)]
        merged = []
        seen = set()
        for kw in (yake_kws + candidates):
            lc = kw.lower()
            if lc not in seen:
                seen.add(lc)
                merged.append(kw)
        return merged
    except Exception as e:
        logger.debug("YAKE merge failed: %s", e)
        return candidates

def _score_candidates_tfidf(jd_text: str, candidates: List[str]) -> List[tuple]:
    if SKLEARN_AVAILABLE and candidates:
        try:
            vectorizer = TfidfVectorizer(vocabulary=candidates, ngram_range=(1,3), use_idf=True, smooth_idf=True)
            tfidf = vectorizer.fit_transform([jd_text])
            scores = tfidf.toarray()[0]
            mapping = vectorizer.get_feature_names_out()
            scored = [(mapping[i], float(scores[i])) for i in range(len(mapping))]
            maxv = max((s for _, s in scored), default=1.0)
            if maxv > 0:
                scored = [(k, float(v/maxv)) for k, v in scored]
            return sorted(scored, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.debug("TF-IDF scoring failed: %s", e)

    txt = (jd_text or "").lower()
    scored = []
    for cand in candidates:
        freq = txt.count(cand.lower())
        score = freq * (1 + 0.15 * len(cand.split()))
        scored.append((cand, float(score)))
    if scored:
        max_raw = max(s for _, s in scored)
        if max_raw > 0:
            scored = [(k, float(v/max_raw)) for k, v in scored]
    return sorted(scored, key=lambda x: x[1], reverse=True)

def _filter_candidate_noise(cand: str) -> bool:
    if not cand or not cand.strip():
        return False
    cand = cand.strip()
    digits = sum(ch.isdigit() for ch in cand)
    if digits > 0 and digits / max(1, len(cand)) > 0.5:
        return False
    cleaned = re.sub(r'[^A-Za-z0-9]', '', cand)
    if len(cleaned) < 2:
        return False
    if cand.lower() in STOPWORDS:
        return False
    if re.match(r'^\d+\.*$', cand):
        return False
    return True

def _classify_keyword(kw: str) -> str:
    l = kw.lower()
    if re.search(r'\b(python|java|c\+\+|c#|sql|javascript|typescript|go|rust|scala|r)\b', l):
        return "skill"
    if re.search(r'\b(aws|azure|gcp|kubernetes|docker|terraform|ansible|jenkins|github)\b', l):
        return "tool"
    if re.search(r'\b(manage|design|develop|implement|deploy|automate|build|test|monitor|scale|lead|maintain)\b', l):
        return "action"
    if re.search(r'\b(bachelor|master|phd|degree|certification|certified|[0-9]+ years)\b', l):
        return "qualification"
    return "other"

def _make_explanation(keyword: str, category: str) -> str:
    base = f"'{keyword}'"
    if category == "skill":
        return f"{base} is a core technical skill used for development, automation, or data work; the JD expects hands-on use of this skill."
    if category == "tool":
        return f"{base} is a platform/tool used for building, operating, or deploying systems; familiarity indicates practical experience with the stack."
    if category == "action":
        return f"{base} signals a responsibility or activity the hire will perform."
    if category == "qualification":
        return f"{base} indicates formal qualifications or experience level the employer is seeking."
    return f"{base} is mentioned in the JD and relates to the role's domain or requirements."

def extract_keywords_from_jd(jd_text: str, top_n: int = 30, min_score: float = 0.20, use_spacy: bool = True) -> List[Dict]:
    if not jd_text or not jd_text.strip():
        return []

    text = re.sub(r'\s+', ' ', jd_text.strip())
    candidates = []
    if use_spacy and SPACY_AVAILABLE:
        try:
            candidates.extend(_candidate_phrases_spacy(text))
        except Exception as e:
            logger.debug("spaCy extraction failed: %s", e)
    candidates.extend(_candidate_phrases_basic(text, max_ngram=3))
    if YAKE_AVAILABLE:
        try:
            candidates = _merge_yake(text, candidates, max_keywords=200)
        except Exception:
            pass

    seen = set()
    unique_cands = []
    for c in candidates:
        lc = c.lower().strip()
        if lc not in seen:
            seen.add(lc)
            unique_cands.append(c.strip())

    filtered = [c for c in unique_cands if _filter_candidate_noise(c)]
    if not filtered:
        return []

    scored = _score_candidates_tfidf(text, filtered)
    final = []
    for kw, score in scored:
        if score < min_score:
            continue
        final.append((kw, float(score)))
        if len(final) >= top_n:
            break

    out = []
    for kw, score in final:
        cat = _classify_keyword(kw)
        out.append({"keyword": kw, "score": float(score), "category": cat, "explanation": _make_explanation(kw, cat)})
    return out
