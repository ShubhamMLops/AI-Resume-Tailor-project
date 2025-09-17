from __future__ import annotations
from typing import Dict, Any, Optional, List
import re, json
from ai.prompts import (
    SYSTEM_TAILOR, USER_TAILOR,
    SYSTEM_KEYWORDS, USER_KEYWORDS,
    SYSTEM_SAMPLE_RESUME, USER_SAMPLE_RESUME,
    SYSTEM_TAILOR_JSON, USER_TAILOR_JSON,
    SYSTEM_CONTACTS, USER_CONTACTS,
    SYSTEM_SUMMARY_BULLETS, USER_SUMMARY_BULLETS,
    SYSTEM_ATS, USER_ATS,
    SYSTEM_KEYWORD_SENTENCES, USER_KEYWORD_SENTENCES,
)
from ai.selector import get_provider
from ai.matcher import match_score, keyword_gaps
# Add this line among the other imports in pipeline.py
from ai.jd_keywords import extract_keywords_from_jd

from pprint import pprint
import textwrap

# -----------------------
# Helper: remove resume placeholder headings
# -----------------------
import re

# common headings to treat as placeholders when they are empty or standalone
_PLACEHOLDER_HEADING_RX = re.compile(
    r'^\s*(technical\s*skills|skills|core\s*competenc(?:y|ies)|core\s*skills|expertise|toolbox|technical\s*expertise|profile\s*summary|summary)\s*[:\-\—\–]?\s*$',
    flags=re.I
)

def remove_resume_placeholders(resume_text: str) -> str:
    """
    Remove lines that are heading-only placeholders (e.g. "Technical Skills:", "Skills")
    while preserving real skill lists (e.g. "Python, Docker") and other resume content.
    Returns cleaned resume string.
    """
    if not resume_text:
        return resume_text or ""

    lines = resume_text.splitlines()
    cleaned_lines = []
    for i, ln in enumerate(lines):
        s = (ln or "").strip()
        if not s:
            # preserve blank lines (you may also drop them)
            cleaned_lines.append("")
            continue

        # If line looks exactly like a placeholder heading, skip it
        if _PLACEHOLDER_HEADING_RX.match(s):
            # Skip the line, but do not skip if the next non-empty line looks like a skill list
            # e.g. "Technical Skills:" followed by "Python, Docker" -> in that case we want to keep heading or keep list.
            # So peek next non-empty line:
            j = i + 1
            next_line = ""
            while j < len(lines):
                nxt = (lines[j] or "").strip()
                if nxt:
                    next_line = nxt
                    break
                j += 1
            # if next_line contains comma or slash (likely an actual skill list), keep current heading line removed but keep next_line
            if next_line and ("," in next_line or "/" in next_line or re.search(r'\b[A-Za-z0-9\+\#\.\-]{2,}\b', next_line)):
                # do NOT append heading line, but allow next_line to be processed (it will be preserved)
                continue
            # otherwise skip the placeholder heading completely
            continue

        # Otherwise include the line unchanged
        cleaned_lines.append(ln)

    # Rebuild text and return
    return "\n".join(cleaned_lines)


# -----------------------------
# Keyword object sanitizer (add right after imports)
# -----------------------------
from typing import Iterable


def _first_json_block(text: str) -> Optional[str]:
    if not text: return None
    start = text.find('{')
    if start == -1: return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

# -----------------------------
# Keyword normalization helper
# -----------------------------
def _normalize_keyword_list(raw_kw_list):
    """
    Ensure keywords list is a list of dicts with stable keys:
    { "term": str, "variants": [str], "evidence": [str], "rank": int|None, "category": str }
    If input is malformed, convert safely. Returns list (possibly empty).
    """
    out = []
    if not raw_kw_list:
        return out
    # if single dict provided
    if isinstance(raw_kw_list, dict):
        raw_kw_list = [raw_kw_list]
    # if it's a string, return empty
    if isinstance(raw_kw_list, str):
        return out
    for item in raw_kw_list:
        try:
            if not isinstance(item, dict):
                # if item is a simple string, convert to minimal dict
                if isinstance(item, str) and item.strip():
                    out.append({"term": item.strip(), "variants": [], "evidence": [], "rank": None, "category": ""})
                continue
            term = (item.get("term") or "") if item.get("term") is not None else ""
            # normalize types
            if not isinstance(term, str):
                term = str(term)
            variants = item.get("variants") or []
            if not isinstance(variants, list):
                variants = [variants] if variants else []
            variants = [str(v).strip() for v in variants if v is not None and str(v).strip()]
            evidence = item.get("evidence") or []
            if not isinstance(evidence, list):
                evidence = [evidence] if evidence else []
            evidence = [str(e).strip() for e in evidence if e is not None and str(e).strip()]
            rank = item.get("rank")
            try:
                rank = int(rank) if rank is not None else None
            except Exception:
                rank = None
            category = (item.get("category") or "") if item.get("category") is not None else ""
            if not isinstance(category, str):
                category = str(category)
            out.append({
                "term": term.strip(),
                "variants": variants,
                "evidence": evidence,
                "rank": rank,
                "category": category.strip()
            })
        except Exception:
            # skip malformed entries silently (but could log)
            continue
    return out


# -----------------------------
# Regex helpers
# -----------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?\b(?:\d[\s-]?){8,14}\b")
LINKEDIN_RE = re.compile(r"(https?://)?(www\.)?linkedin\.com/[^\s\)\]]+", re.I)
GITHUB_RE = re.compile(r"(https?://)?(www\.)?github\.com/[^\s\)\]]+", re.I)

# -----------------------------
# Contacts
# -----------------------------
def extract_contacts_regex(text: str) -> Dict[str, str]:
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    linkedin = LINKEDIN_RE.search(text)
    github = GITHUB_RE.search(text)

    name = ""
    for line in text.splitlines()[:15]:
        s = line.strip()
        if not s or len(s) > 60:
            continue
        if any(ch.isdigit() for ch in s) or any(x in s for x in ("@", "http", "|", "/", "\\", "•", " - ", ",", "(", ")", ":")):
            continue
        words = s.split()
        if 2 <= len(words) <= 6:
            name = s
            break

    return {
        "name": name or "",
        "email": email.group(0) if email else "",
        "phone": phone.group(0) if phone else "",
        "linkedin": linkedin.group(0) if linkedin else "",
        "github": github.group(0) if github else "",
    }

def extract_contacts_llm(resume_text: str, provider_pref: Optional[str], model_name: Optional[str], keys: Dict[str, str]) -> Dict[str, str]:
    provider = _provider_from_keys(provider_pref, keys)
    raw = provider.chat(model=model_name, system=SYSTEM_CONTACTS, user=USER_CONTACTS.format(resume=resume_text), temperature=0, max_tokens=256)
    raw = (raw or "").strip().strip("`").strip()
    try:
        start = raw.find('{'); end = raw.rfind('}') + 1
        obj = json.loads(raw[start:end])
        return {k: (obj.get(k) or "") for k in ["name", "email", "phone", "linkedin", "github"]}
    except Exception:
        return extract_contacts_regex(resume_text)

# -----------------------------
# Sanitizer
# -----------------------------
def sanitize_markdown(md: str) -> str:
    """
    Clean up LLM markdown outputs into resume-friendly text
    """
    KNOWN_HEADINGS = {
        "profile summary", "professional summary", "summary",
        "core skills", "core competencies",
        "technical skills", "work experience", "experience",
        "education", "certifications", "projects", "other"
    }

    md = (md or "").replace("**", "").replace("\t", " ")
    md = re.sub(r"[ ]{3,}", "  ", md)
    md = re.sub(r"\r\n?", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)

    lines = [ln.rstrip() for ln in md.split("\n")]
    out, prev_blank = [], True

    def is_heading(s: str) -> bool:
        return s.strip().lower() in KNOWN_HEADINGS

    for raw in lines:
        s = raw.strip()
        if re.match(r"^[-•▪‣·*]\s+", s):
            s = "• " + re.sub(r"^[-•▪‣·*]\s+", "", s)
        out.append(s)

        if is_heading(s):
            out.append("")
            prev_blank = True
            continue

        if s == "":
            if prev_blank:
                continue
            prev_blank = True
        else:
            prev_blank = False

    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()

# -----------------------------
# Provider selector
# -----------------------------
def _provider_from_keys(provider_preference: Optional[str], keys: Dict[str, str]):
    provider = get_provider(provider_preference, keys)
    if not provider:
        raise RuntimeError("No API key provided in the app. Enter a key in the sidebar.")
    return provider

def jd_to_json_via_llm(jd_text: str, provider, model_name: Optional[str],
                       temperature: float = 0.0, max_tokens: int = 800) -> Dict[str, Any]:
    """
    Ask the LLM to read the JD line-by-line, normalize/trim whitespace and return a strict JSON object:
    {
      "lines": [ {"index": 0, "text": "<verbatim JD line trimmed>"}, ... ],
      "flat": "<concat of trimmed lines with single space separators>"
    }
    The LLM must return JSON only. If the LLM fails to return valid JSON, this function falls back to a
    deterministic local normalization.
    """
    if not jd_text or not jd_text.strip():
        return {"lines": [], "flat": ""}

    system_prompt = (
        "You are a strict JSON-only transformer. Read the JOB DESCRIPTION verbatim, line-by-line. "
        "For each non-empty line produce a JSON array entry with {\"index\": <line_index>, \"text\": \"<trimmed_line>\"}. "
        "Trim leading/trailing whitespace and collapse internal repeated spaces to a single space in each line. "
        "Do NOT change wording, do NOT invent or remove words. Return a single JSON object only with keys: "
        "\"lines\" (array of objects as described) and \"flat\" (string: all trimmed lines joined by single spaces). "
        "If no valid lines, return {\"lines\": [], \"flat\": \"\"}."
    )

    user_prompt = f"JOB DESCRIPTION (verbatim):\n{jd_text}\n\nReturn only the JSON object as specified."

    try:
        raw = provider.chat(model=model_name, system=system_prompt, user=user_prompt, temperature=temperature, max_tokens=max_tokens or 800)
        raw = (raw or "").strip().strip("`").strip()
        # extract first {...} block
        start = raw.find("{"); end = raw.rfind("}") + 1
        if start != -1 and end > start:
            block = raw[start:end]
            obj = json.loads(block)
            # basic sanity check
            if isinstance(obj, dict) and "lines" in obj and "flat" in obj:
                return obj
    except Exception:
        pass

    # Fallback deterministic normalization if LLM fails or returned invalid JSON
    lines = []
    for i, ln in enumerate([l for l in (jd_text or "").splitlines()]):
        s = ln.strip()
        s = re.sub(r"\s+", " ", s)
        if s:
            lines.append({"index": i, "text": s})
    flat = " ".join([l["text"] for l in lines])
    return {"lines": lines, "flat": flat}

# -----------------------------
# Analysis
# -----------------------------
def readability(text: str) -> Dict[str, float]:
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z0-9']+", text)
    words_count = max(1, len(words))
    fre = 206.835 - 1.015*(words_count/sentences) - 84.6*(1/words_count)
    return {"flesch_reading_ease": round(fre, 1), "sentences": sentences, "words": words_count}

def ats_checks(text: str) -> Dict[str, Any]:
    flags = []
    if len(text) < 400:
        flags.append("Resume is very short (<400 chars). Add more detail.")
    if re.search(r"\bI\b|\bme\b|\bmy\b", text, re.I):
        flags.append("Avoid first-person pronouns in bullets.")
    if re.search(r"\b(References available on request)\b", text, re.I):
        flags.append("Remove 'References available on request'.")
    return {"warnings": flags}

def analyze(resume_text: str, jd_text: str) -> Dict[str, Any]:
    return {
        "ats": ats_checks(resume_text),
        "readability": readability(resume_text),
        "match": match_score(resume_text, jd_text),
        "keywords_bow": keyword_gaps(resume_text, jd_text, top_k=30),
        "contacts": extract_contacts_regex(resume_text),
    }


def extract_core_competencies_from_jd(jd_text: str) -> List[str]:
    """
    Deterministic extraction of 'Core Competencies' from a JD.
    - If JD contains an explicit heading like 'Core Competencies' (case-insensitive),
      collect the non-empty lines under that heading until a blank line or next heading.
    - Otherwise, look for short comma-separated lists or single-line lists containing many short tokens
      near the top of the JD that are likely competency lists (heuristic).
    - Always return items verbatim as found in the JD (no invention), deduped preserving order.
    """
    if not jd_text:
        return []

    lines = [ln.rstrip() for ln in jd_text.splitlines()]
    n = len(lines)
    out = []
    seen = set()

    # 1) Find explicit "Core Competencies"/"Core Skills" heading
    heading_re = re.compile(r"(?i)^\s*(core\s*competenc(?:ies|y)|core\s*skills|core\s*expertise)\s*[:\-–—]?\s*$")
    section_heading_re = re.compile(r"^[A-Za-z0-9 \-]{1,80}\s*:$")  # generic heading
    for i, ln in enumerate(lines):
        if heading_re.match(ln.strip()):
            # collect subsequent non-empty lines until blank or next section heading
            for j in range(i+1, n):
                nxt = lines[j].strip()
                if not nxt:
                    break
                if section_heading_re.match(nxt):
                    break
                # split lists like "A, B, C" into items else take line as-is
                parts = [p.strip() for p in re.split(r",|\u2022|;|\t", nxt) if p.strip()]
                if parts:
                    for p in parts:
                        key = p.lower()
                        if key not in seen:
                            seen.add(key); out.append(p)
                else:
                    key = nxt.lower()
                    if key not in seen:
                        seen.add(key); out.append(nxt)
            if out:
                return out

    # 2) Heuristic: locate lines with many short tokens (comma separated) near top
    # e.g. "AWS, Terraform, Docker, Kubernetes"
    for ln in lines[:20]:  # only look at first 20 lines to avoid scanning entire JD
        if not ln.strip():
            continue
        parts = [p.strip() for p in re.split(r",|\u2022|;|\band\b|\bor\b", ln) if p.strip()]
        if len(parts) >= 3:  # likely a competency list
            for p in parts:
                key = p.lower()
                if key not in seen:
                    seen.add(key); out.append(p)
            if out:
                return out

    # 3) Nothing found
    return []

# -----------------------------
# Keywords (LLM)
# -----------------------------
# pipeline.py — wrapper replacement (paste inside existing file)
from typing import Optional, List, Dict, Any
from ai.jd_keywords import extract_keywords_from_jd, normalize_text, simple_tokenize, lemmatize_tokens  # normalize_text helper exists above? if not use ai.jd_keywords.normalize_text
import re
import logging

logger = logging.getLogger(__name__)

# Optional fuzzy lib
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

def _fuzzy_match(a: str, b: str) -> int:
    """
    Returns a fuzzy similarity 0..100 between a and b.
    Uses rapidfuzz if available, otherwise simple heuristics.
    """
    if not a or not b:
        return 0
    if RAPIDFUZZ_AVAILABLE:
        try:
            return int(fuzz.token_set_ratio(a, b))
        except Exception:
            pass
    # fallback: token overlap ratio
    toks_a = set(re.findall(r'\w+', a.lower()))
    toks_b = set(re.findall(r'\w+', b.lower()))
    if not toks_a or not toks_b:
        return 0
    inter = toks_a.intersection(toks_b)
    score = int(100 * (2 * len(inter) / (len(toks_a) + len(toks_b))))
    return score

def _extract_evidence_sentences(resume_text: str, term: str, max_sentences: int = 2) -> List[str]:
    """
    Return up to `max_sentences` sentences from resume_text that contain the keyword term.
    Simple regex-based sentence splitting.
    """
    sents = re.split(r'(?<=[.!?])\s+', (resume_text or "").strip())
    out = []
    tnorm = (term or "").lower()
    for s in sents:
        if tnorm in s.lower():
            out.append(s.strip())
            if len(out) >= max_sentences:
                break
    return out

# ---------------------------
# STEP 0: Domain extraction (LLM)
# ---------------------------
# ---------------------------
# STEP 0: Domain extraction (LLM + IT domain mapping)
# ---------------------------

STEP0_SYSTEM = (
    "You are an assistant that identifies the primary IT domain/discipline of a job description. "
    "Return exactly one IT domain such as 'backend', 'frontend', 'fullstack', 'data science', "
    "'machine learning', 'ai', 'cloud', 'devops', 'security', 'mobile', 'testing', 'qa', "
    "'blockchain', 'database', 'infrastructure', 'networking', 'product management', 'ui/ux'. "
    "If multiple apply, pick the one that dominates most of the responsibilities."
)
STEP0_USER_TMPL = "Job Description:\n\n{jd}\n\nReturn the single most relevant IT domain."

# Canonical IT domains
IT_DOMAINS = {
    "backend": ["backend", "server-side", "api", "microservices"],
    "frontend": ["frontend", "ui", "react", "angular", "vue", "javascript"],
    "fullstack": ["fullstack", "end-to-end"],
    "data science": ["data science", "analytics", "machine learning", "ml engineer"],
    "machine learning": ["ml", "machine learning", "ai engineer", "deep learning"],
    "ai": ["ai", "artificial intelligence", "nlp", "computer vision"],
    "cloud": ["cloud", "aws", "azure", "gcp", "cloud engineer"],
    "devops": ["devops", "sre", "infrastructure as code", "terraform", "ansible", "cicd"],
    "security": ["security", "infosec", "cybersecurity"],
    "mobile": ["mobile", "android", "ios", "react native", "flutter"],
    "testing": ["testing", "qa", "automation testing", "selenium"],
    "blockchain": ["blockchain", "web3", "solidity", "crypto"],
    "database": ["database", "db admin", "sql", "nosql"],
    "infrastructure": ["infrastructure", "systems engineer", "platform"],
    "networking": ["network", "tcp/ip", "router", "firewall"],
    "product management": ["product manager", "pm", "product owner"],
    "ui/ux": ["ui", "ux", "design", "figma", "user experience"],
}

def llm_domain_extract(jd_text: str, model: Optional[str] = None, temperature: float = 0.0,
                       provider: Optional[str] = None, provider_keys: Optional[Dict[str, str]] = None) -> str:
    """
    Step 0: Extract the IT domain of the JD.
    1) Ask provider for domain guess.
    2) Normalize and map against IT_DOMAINS.
    3) If no match, fallback to 'general-it'.
    """
    if not jd_text or not jd_text.strip():
        return "general-it"

    try:
        out = _call_llm_system_user(
            STEP0_SYSTEM,
            STEP0_USER_TMPL.format(jd=jd_text),
            model=model,
            temperature=temperature,
            max_tokens=64,
            provider=provider,
            provider_keys=provider_keys,
        )
    except Exception as e:
        logger.debug("llm_domain_extract provider call failed: %s", e)
        out = ""

    if not out or not isinstance(out, str):
        return "general-it"

    guess = out.strip().lower()
    guess = re.sub(r'[^a-z0-9\s\-/]', '', guess)  # clean up punctuation

    # Match guess to IT domains
    for domain, patterns in IT_DOMAINS.items():
        for pat in patterns:
            if pat in guess:
                return domain

    # fallback: try keyword search in JD text itself
    jd_low = jd_text.lower()
    for domain, patterns in IT_DOMAINS.items():
        for pat in patterns:
            if pat in jd_low:
                return domain

    # default fallback
    return "general-it"


# Keyword extraction 3 steps pipeline:
# --- 3-layer JD keyword extraction (drop-in) ---
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# LLM client: OpenAI wrapper (adjust to your provider if needed)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # change as you like
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# NLP & utilities (optional libs with fallbacks)
try:
    import spacy
    SPACY_AVAILABLE = True
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    SPACY_AVAILABLE = False
    _nlp = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SENTEVAL_AVAILABLE = True
    SENTEVAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    SENTEVAL_AVAILABLE = False
    SENTEVAL_MODEL = None

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

# --- Helper: safe LLM call with JSON extraction ---
def _call_llm_system_user(system_prompt: str, user_prompt: str,
                          model: Optional[str] = None,
                          temperature: float = 0.0,
                          max_tokens: int = 800,
                          provider: Optional[str] = None,
                          provider_keys: Optional[Dict[str, str]] = None) -> str:
    """
    Provider-aware LLM call used by the 3-layer extraction pipeline.
    REQUIREMENTS:
      - provider: string that matches the provider selectbox in the UI (e.g. "openai", "gemini", "anthropic")
      - provider_keys: dict of keys (e.g. {"openai": "sk-...", "gemini": "...", ...})
    Behavior:
      1. Try ai.providers.call_llm(...) if available.
      2. Else, look for ProviderClass in ai.providers (e.g. OpenAIProvider) and instantiate it with provider_keys.
      3. If neither is present or a call fails, raise RuntimeError with a clear message.
    Note: This function intentionally does NOT fallback to environment OPENAI_API_KEY — the GUI must supply provider + key.
    """
    # Require provider selected by the UI
    if not provider:
        raise RuntimeError("No LLM provider selected. Please select a provider in the UI and provide its API key.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Import providers module
    try:
        import ai.providers as providers_module
    except Exception as e:
        raise RuntimeError(f"Failed to import ai.providers module: {e}")

    # 1) If providers_module exposes call_llm, use it (support a couple of common signatures)
    if hasattr(providers_module, "call_llm") and callable(getattr(providers_module, "call_llm")):
        try:
            # Preferred signature:
            return providers_module.call_llm(provider_name=provider,
                                             messages=messages,
                                             model=model,
                                             temperature=temperature,
                                             max_tokens=max_tokens,
                                             keys=provider_keys or {})
        except TypeError:
            # Fallback signature: call_llm(messages, provider, **opts)
            try:
                return providers_module.call_llm(messages, provider, model=model,
                                                 temperature=temperature, max_tokens=max_tokens, keys=provider_keys or {})
            except Exception as e:
                raise RuntimeError(f"ai.providers.call_llm raised an error (fallback signature): {e}")
        except Exception as e:
            raise RuntimeError(f"ai.providers.call_llm raised an error: {e}")

    # 2) If call_llm not present, try Provider class lookup and instantiation
    # Map common provider names to class names (you can extend this map)
    provider_map = {
        "openai": "OpenAIProvider",
        "gemini": "GeminiProvider",
        "anthropic": "AnthropicProvider"
    }
    cls_name = provider_map.get(provider.lower(), provider.capitalize() + "Provider")

    # Ensure class exists in ai.providers
    if not hasattr(providers_module, cls_name):
        # list available provider class candidates for helpful error message
        candidates = [n for n in dir(providers_module) if n.endswith("Provider")]
        raise RuntimeError(f"Provider class '{cls_name}' not found in ai.providers. Available provider classes: {candidates}")

    ProviderClass = getattr(providers_module, cls_name)

    # instantiate provider class with provider_keys (constructor signature may differ)
    try:
        # If provider constructor expects dict, pass provider_keys; else try no-arg init -> set keys attribute
        try:
            provider_instance = ProviderClass(provider_keys or {})
        except TypeError:
            # try no-arg constructor then set keys if attribute present
            provider_instance = ProviderClass()
            if hasattr(provider_instance, "set_keys") and callable(provider_instance.set_keys):
                provider_instance.set_keys(provider_keys or {})
            elif hasattr(provider_instance, "keys"):
                try:
                    setattr(provider_instance, "keys", provider_keys or {})
                except Exception:
                    pass
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate provider class '{cls_name}': {e}")

    # Ensure provider instance has a chat/complete method
    chat_fn = None
    if hasattr(provider_instance, "chat") and callable(getattr(provider_instance, "chat")):
        chat_fn = provider_instance.chat
    elif hasattr(provider_instance, "complete") and callable(getattr(provider_instance, "complete")):
        chat_fn = provider_instance.complete
    elif hasattr(provider_instance, "call") and callable(getattr(provider_instance, "call")):
        chat_fn = provider_instance.call

    if not chat_fn:
        raise RuntimeError(f"Provider class '{cls_name}' does not implement a callable 'chat' / 'complete' / 'call' method.")

    # Call provider's chat method. Many provider chat methods accept system+user separately, or a single messages list.
    try:
        # Try calling with (model, system, user, temperature, max_tokens)
        try:
            return chat_fn(model=model, system=system_prompt, user=user_prompt, temperature=temperature, max_tokens=max_tokens)
        except TypeError:
            # Try calling with single messages parameter
            try:
                return chat_fn(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
            except TypeError:
                # Try positional messages
                return chat_fn(messages, model, temperature, max_tokens)
    except Exception as e:
        raise RuntimeError(f"Provider '{cls_name}' chat call failed: {e}")


def _extract_json_from_text(text: str) -> Any:
    """
    Robust JSON extraction from an LLM response string.
    Returns a parsed JSON object (dict/list) or None on failure.
    Attempts:
      1. Direct json.loads(text)
      2. Regex extract first {...} or [...] block and json.loads
      3. Progressive trimming of trailing characters to salvage near-JSON
    """
    import json
    import re
    if not text or not isinstance(text, str):
        return None

    # 1) direct parse fast-path
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) find first {...} or [...] block
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if not m:
        return None

    candidate = m.group(1).strip()

    # 3) try to parse candidate. If it fails, progressively trim trailing characters and retry.
    # This helps when the LLM appends extra commentary after JSON.
    for trim in range(0, min(200, len(candidate))):
        try:
            # attempt to parse progressively smaller suffixes removed
            maybe = candidate[:len(candidate) - trim]
            return json.loads(maybe)
        except Exception:
            continue

    # 4) final attempt: try replacing single quotes with double quotes (some LLMs use JS-style single quotes)
    try:
        alt = candidate.replace("'", "\"")
        return json.loads(alt)
    except Exception:
        pass

    # If we can't parse, return None and let caller fall back to deterministic logic
    return None


# ---------------------------
# STEP 1: Broad LLM extraction
# ---------------------------
STEP1_SYSTEM = (
    "You are a meticulous reader. Extract every possible keyword (technical skills, tools, "
    "frameworks, methodologies, soft skills, and responsibilities) from the job description. "
    "Output as a comma-separated list only—do NOT output explanation or JSON—just items separated by commas."
)
STEP1_USER_TMPL = "Job Description:\n\n{jd}\n\nExtract keywords as comma-separated list."

def llm_broad_extract(jd_text: str, model: Optional[str] = None, temperature: float = 0.0,
                      provider: Optional[str] = None, provider_keys: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Step 1: broad generative extraction via the selected provider.
    """
    try:
        out = _call_llm_system_user(STEP1_SYSTEM,
                                   STEP1_USER_TMPL.format(jd=jd_text),
                                   model=model,
                                   temperature=temperature,
                                   max_tokens=600,
                                   provider=provider,
                                   provider_keys=provider_keys)
    except Exception as e:
        logger.exception("LLM broad extract failed: %s", e)
        return []

    # split by common delimiters (commas, semicolons, newlines, bullets, dashes)
    parts = re.split(r'[,;\n•\-\r]+', out)
    tokens = [p.strip() for p in parts if p and p.strip()]
    return tokens



# ---------------------------
# STEP 2: LLM refinement & classification
# ---------------------------
STEP2_SYSTEM = (
    "You are a strict data cleaner. Receive a list of items and return a JSON object with three arrays: "
    "'Technical Skills', 'Soft Skills', 'Responsibilities'. Remove duplicates and items that are not true keywords. "
    "Normalize items (trim whitespace, remove excessive punctuation). Output valid JSON ONLY."
)
STEP2_USER_TMPL = "Raw items:\n\n{items}\n\nClassify and clean them into JSON."

def llm_refine_and_classify(items: List[str], model: Optional[str] = None, temperature: float = 0.0,
                            provider: Optional[str] = None, provider_keys: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """
    Step 2: use LLM to refine Step1 items into categories and perform tech/noise reconciliation.
    Returns dict with keys: Technical Skills, Noise, Soft Skills, Responsibilities
    """
    # prepare default return
    empty_struct = {"Technical Skills": [], "Noise": [], "Soft Skills": [], "Responsibilities": []}
    if not items:
        return empty_struct

    joined = ", ".join(items)
    try:
        out = _call_llm_system_user(
            STEP2_SYSTEM,
            STEP2_USER_TMPL.format(items=joined),
            model=model,
            temperature=temperature,
            max_tokens=800,
            provider=provider,
            provider_keys=provider_keys
        )
    except Exception as e:
        logger.exception("LLM refine failed (provider call): %s", e)
        # fallback deterministic classification to at least fill Tech/Noise/Soft/Resp
        tech, soft, resp, noise = [], [], [], []
        for it in items:
            low = (it or "").lower()
            if re.search(r'\b(python|java|c#|c\+\+|sql|docker|kubernetes|aws|azure|gcp|terraform|ansible|jenkins|git|scala|rust|go|typescript|react|node)\b', low):
                tech.append(it.strip())
            elif re.search(r'\b(manage|lead|collaborate|communicat|organize|present|team|design|develop)\b', low):
                resp.append(it.strip())
            elif len(low.split()) <= 3 and re.search(r'^[a-zA-Z]+$', low):
                # short single words are probably tech / keep heuristic
                tech.append(it.strip())
            else:
                noise.append(it.strip())
        return {"Technical Skills": list(dict.fromkeys(tech)),
                "Noise": list(dict.fromkeys(noise)),
                "Soft Skills": list(dict.fromkeys(soft)),
                "Responsibilities": list(dict.fromkeys(resp))}

    parsed = _extract_json_from_text(out)
    if not parsed or not isinstance(parsed, dict):
        # log raw output for debugging
        logger.debug("LLM refine output (non-JSON or parse failed): %s", out)
        # Use deterministic fallback as above
        tech, soft, resp, noise = [], [], [], []
        for it in items:
            low = (it or "").lower()
            if re.search(r'\b(python|java|c#|c\+\+|sql|docker|kubernetes|aws|azure|gcp|terraform|ansible|jenkins|git|scala|rust|go|typescript|react|node)\b', low):
                tech.append(it.strip())
            elif re.search(r'\b(manage|lead|collaborate|communicat|organize|present|team|design|develop)\b', low):
                resp.append(it.strip())
            elif len(low.split()) <= 3 and re.search(r'^[a-zA-Z]+$', low):
                tech.append(it.strip())
            else:
                noise.append(it.strip())
        return {"Technical Skills": list(dict.fromkeys(tech)),
                "Noise": list(dict.fromkeys(noise)),
                "Soft Skills": list(dict.fromkeys(soft)),
                "Responsibilities": list(dict.fromkeys(resp))}

    # Normalize parsed outputs to lists for expected keys; accept extra keys as Noise
    tech = parsed.get("Technical Skills", []) if isinstance(parsed.get("Technical Skills", []), list) else []
    soft = parsed.get("Soft Skills", []) if isinstance(parsed.get("Soft Skills", []), list) else []
    resp = parsed.get("Responsibilities", []) if isinstance(parsed.get("Responsibilities", []), list) else []
    # Any other items from parsed put into noise; also include parsed.get("Noise")
    noise = parsed.get("Noise", []) if isinstance(parsed.get("Noise", []), list) else []

    # also catch stray values under unknown keys
    for k, v in parsed.items():
        if k not in ("Technical Skills", "Soft Skills", "Responsibilities", "Noise") and isinstance(v, list):
            for it in v:
                noise.append(it)

    # deterministic verification: move suspicious items between tech <-> noise
    verified_tech = []
    verified_noise = []

    tech_patterns = re.compile(r'\b(python|java|c#|c\+\+|sql|docker|kubernetes|aws|azure|gcp|terraform|ansible|jenkins|git|scala|rust|go|typescript|react|node|django|flask|spring)\b', re.I)
    # If an item in tech looks like noise (too long generic phrase or contains stopwords), move to noise
    for t in tech:
        tstr = (t or "").strip()
        low = tstr.lower()
        if len(low.split()) > 6 or low in STRONG_STOPWORDS or re.search(r'\b(experience|years|responsible|work|team|role|company|apply)\b', low):
            verified_noise.append(tstr)
        else:
            verified_tech.append(tstr)

    # If item in noise looks like tech, move to tech
    for n in noise:
        nstr = (n or "").strip()
        if tech_patterns.search(nstr):
            verified_tech.append(nstr)
        else:
            verified_noise.append(nstr)

    # dedupe preserving order
    def dedupe_keep_order(seq):
        seen = set(); out = []
        for s in seq:
            if not s: continue
            k = s.lower().strip()
            if k not in seen:
                seen.add(k); out.append(s)
        return out

    verified_tech = dedupe_keep_order(verified_tech)
    verified_noise = dedupe_keep_order(verified_noise)
    soft = dedupe_keep_order(soft)
    resp = dedupe_keep_order(resp)

    return {"Technical Skills": verified_tech, "Noise": verified_noise, "Soft Skills": soft, "Responsibilities": resp}


# ---------------------------
# STEP 3: Deterministic filtering & normalization
# ---------------------------
# strong stoplist
STRONG_STOPWORDS = set([
    "experience","years","year","candidate","responsibilities","responsibility","work","works","working",
    "requirements","preferred","should","will","including","including:","knowledge","knowledgeable","ability","able"
])

def _normalize_token(tok: str) -> str:
    tok = tok.strip()
    # remove surrounding punctuation
    tok = re.sub(r'^[^\w]+|[^\w]+$', '', tok)
    tok = tok.replace('_',' ').strip()
    return tok

def deterministic_filter_and_normalize(structured: Optional[Dict[str, List[str]]],
                                       jd_text: str,
                                       top_n: int = 40,
                                       min_token_len: int = 2,
                                       use_spacy: bool = True,
                                       use_semantic_grouping: bool = False,
                                       semantic_threshold: float = 0.88) -> Dict[str, List[str]]:
    """
    Final deterministic cleaning:
      - Accepts structured (may be None) and returns cleaned dict with keys:
        'Technical Skills', 'Soft Skills', 'Responsibilities'
      - Defensive: if structured is None or malformed, returns empty lists for keys.
    """
    # Defensive default if structured is None or not a mapping
    if not structured or not isinstance(structured, dict):
        logger.debug("deterministic_filter_and_normalize: received invalid structured input; substituting empty structure.")
        structured = {"Technical Skills": [], "Soft Skills": [], "Responsibilities": []}

    # flatten all items into a list of (category, raw)
    flat: List[Tuple[str, str]] = []
    for cat in ("Technical Skills", "Soft Skills", "Responsibilities"):
        items = structured.get(cat) if isinstance(structured.get(cat, []), list) else []
        for it in items:
            flat.append((cat, it))

    cleaned_by_cat: Dict[str, List[str]] = {"Technical Skills": [], "Soft Skills": [], "Responsibilities": []}
    seen_norm: Set[str] = set()

    # spaCy doc of JD for optional context
    jd_doc = _nlp(jd_text) if use_spacy and SPACY_AVAILABLE and _nlp else None

    for cat, raw in flat:
        if not raw or not str(raw).strip():
            continue
        norm = _normalize_token(str(raw))
        if not norm or len(re.sub(r'[^A-Za-z]', '', norm)) < min_token_len:
            continue
        if re.search(r'\d', norm) and not re.search(r'[A-Za-z]', norm):
            # drop pure numbers
            continue
        if norm.lower() in STRONG_STOPWORDS:
            continue

        # POS filtering if spaCy is available
        if use_spacy and SPACY_AVAILABLE and jd_doc is not None:
            try:
                tok_doc = _nlp(norm)
                keep = False
                for t in tok_doc:
                    if t.pos_ in ("NOUN", "PROPN"):
                        keep = True
                    if not t.is_alpha and any(ch.isdigit() or ch in "+#./-" for ch in t.text):
                        keep = True
                if not keep:
                    continue
            except Exception:
                # If spaCy parsing fails for this token, continue with best-effort keep
                logger.debug("spaCy parsing failed for token '%s' — keeping by fallback", norm)

        final_norm = norm.strip()
        if final_norm.lower() in seen_norm:
            continue
        seen_norm.add(final_norm.lower())
        cleaned_by_cat.setdefault(cat, []).append(final_norm)

    # Optional semantic grouping handled elsewhere; keep ordering deterministic
    for k in cleaned_by_cat:
        cleaned_by_cat[k] = sorted(list(dict.fromkeys(cleaned_by_cat[k])), key=lambda x: x.lower())

    return cleaned_by_cat

# ---------------------------
# Orchestrator that runs all 3 steps
# ---------------------------
def three_layer_extract_and_normalize(jd_text: str,
                                      model_step1: Optional[str] = None,
                                      model_step2: Optional[str] = None,
                                      provider: Optional[str] = None,
                                      provider_keys: Optional[Dict[str, str]] = None,
                                      use_spacy: bool = True,
                                      use_semantic_grouping: bool = False,
                                      semantic_threshold: float = 0.88,
                                      top_n: int = 40,
                                      min_score: float = 0.0) -> Dict[str, List[str]]:
    """
    Orchestrator enhanced:
      Step0: domain extraction
      Step1: broad LLM extraction
      Step2: refine & classify (with tech<->noise reconciliation)
      Step2.1: ensure JD technical keywords are included
      Step2.2: re-run reconciliation
      Step3: deterministic filter & normalize
    """
    # Step 0: domain
    domain = None
    try:
        domain = llm_domain_extract(jd_text, model=model_step1, temperature=0.0, provider=provider, provider_keys=provider_keys)
        logger.debug("Detected JD domain: %s", domain)
    except Exception as e:
        logger.debug("Domain extraction failed: %s", e)
        domain = None

    # Step 1: broad extraction (LLM)
    try:
        step1 = llm_broad_extract(jd_text, model=model_step1, temperature=0.0, provider=provider, provider_keys=provider_keys)
    except Exception as e:
        logger.exception("Step1 (broad extract) failed: %s", e)
        step1 = []

    # Step 2: refine & classify via LLM into Technical Skills / Noise / Soft / Responsibilities
    try:
        step2_struct = llm_refine_and_classify(step1, model=model_step2, temperature=0.0, provider=provider, provider_keys=provider_keys)
    except Exception as e:
        logger.exception("Step2 (refine & classify) raised an exception: %s", e)
        step2_struct = {"Technical Skills": [], "Noise": [], "Soft Skills": [], "Responsibilities": []}

    # Defensive coercion
    if not step2_struct or not isinstance(step2_struct, dict):
        step2_struct = {"Technical Skills": [], "Noise": [], "Soft Skills": [], "Responsibilities": []}
    for k in ("Technical Skills", "Noise", "Soft Skills", "Responsibilities"):
        if not isinstance(step2_struct.get(k, []), list):
            step2_struct[k] = [str(step2_struct.get(k))] if step2_struct.get(k) else []

    # Step 2.1: ensure JD deterministic technical keywords are present
    try:
        from ai.jd_keywords import extract_keywords_from_jd as deterministic_jd_kw
        # get deterministic high-precision tech keywords from JD
        det_cands = deterministic_jd_kw(jd_text or "", top_n=80, min_score=0.18, use_spacy=use_spacy) or []
        # extract only skill-category keywords returned by jd_keywords
        det_techs = []
        for c in det_cands:
            kw = c.get("keyword") if isinstance(c, dict) else c
            cat = c.get("category") if isinstance(c, dict) else None
            if kw and (not cat or cat.lower() == "skill" or cat.lower() == "tool"):
                det_techs.append(str(kw).strip())
        # Add any deterministic tech keywords missing in step2_struct['Technical Skills']
        existing_lower = set([t.lower() for t in step2_struct.get("Technical Skills", [])])
        for dt in det_techs:
            if dt and dt.lower() not in existing_lower:
                logger.debug("Step2.1: adding deterministic tech '%s' to Technical Skills", dt)
                step2_struct["Technical Skills"].append(dt)
                existing_lower.add(dt.lower())
    except Exception as e:
        logger.debug("Step2.1 deterministic augment failed: %s", e)

    # Step 2.2: re-run tech<->noise reconciliation using same verification logic
    try:
        # reuse llm_refine_and_classify verification logic by calling a small in-memory reconciliation
        # Here implement same deterministic verification as in llm_refine_and_classify (without LLM calls)
        tech_list = list(step2_struct.get("Technical Skills", []))
        noise_list = list(step2_struct.get("Noise", []))

        tech_patterns = re.compile(r'\b(python|java|c#|c\+\+|sql|docker|kubernetes|aws|azure|gcp|terraform|ansible|jenkins|git|scala|rust|go|typescript|react|node|django|flask|spring)\b', re.I)
        verified_tech = []
        verified_noise = []

        for t in tech_list:
            tstr = (t or "").strip()
            low = tstr.lower()
            if len(low.split()) > 6 or low in STRONG_STOPWORDS or re.search(r'\b(experience|years|responsible|work|team|role|company|apply)\b', low):
                verified_noise.append(tstr)
            else:
                verified_tech.append(tstr)

        for n in noise_list:
            nstr = (n or "").strip()
            if tech_patterns.search(nstr):
                verified_tech.append(nstr)
            else:
                verified_noise.append(nstr)

        # dedupe
        def dedupe_keep_order(seq):
            seen = set(); out = []
            for s in seq:
                if not s: continue
                k = s.lower().strip()
                if k not in seen:
                    seen.add(k); out.append(s)
            return out

        step2_struct["Technical Skills"] = dedupe_keep_order(verified_tech)
        step2_struct["Noise"] = dedupe_keep_order(verified_noise)
    except Exception as e:
        logger.debug("Step2.2 reconciliation failed: %s", e)

    # Step 3: deterministic filter & normalization using existing function
    # Build a structure accepted by deterministic_filter_and_normalize:
    # We will pass 'Technical Skills' as Technical Skills, merge Noise into a 'Noise' bucket
    structured_for_step3 = {
        "Technical Skills": step2_struct.get("Technical Skills", []),
        "Soft Skills": step2_struct.get("Soft Skills", []),
        "Responsibilities": step2_struct.get("Responsibilities", [])
    }
    # We purposely ignore Noise in final deterministic normalization (it will be filtered out by rules).
    cleaned = deterministic_filter_and_normalize(structured_for_step3, jd_text,
                                                 top_n=top_n, use_spacy=use_spacy,
                                                 use_semantic_grouping=use_semantic_grouping,
                                                 semantic_threshold=semantic_threshold)
    # Optionally attach the noise list for debugging downstream
    cleaned["_noise_raw"] = step2_struct.get("Noise", [])
    cleaned["_domain"] = domain
    return cleaned




# ---------------------------
# Backwards-compatible wrapper for your app
# ---------------------------
def extract_keywords_llm(resume_text: str, jd_text: str,
                         provider_pref: Optional[str] = None, model_name: Optional[str] = None,
                         temperature: float = 0.0, max_tokens: int = 1024, keys: Dict[str, str] = {}) -> Dict[str, Any]:
    """
    Wrapper preserving old signature. Runs three-layer extraction and then does resume matching
    (you can keep your existing strict resume-match logic after we get final keywords).
    Returns a dictionary with 'keywords' (list matched), 'missing', 'summary', '_raw_extraction'.
    """
    # run three-layer extraction (models for steps can be customized via model_name or env)
    try:
        cleaned = three_layer_extract_and_normalize(jd_text,
                                                    model_step1=None,
                                                    model_step2=None,
                                                    provider=provider_pref,
                                                    provider_keys=keys,
                                                    use_spacy=True,
                                                    use_semantic_grouping=False)


    except Exception as e:
        logger.exception("Three-layer extraction failed: %s", e)
        cleaned = {"Technical Skills": [], "Soft Skills": [], "Responsibilities": []}

    # flattened "final JD keywords"
    final_keywords = []
    for cat in ["Technical Skills", "Soft Skills", "Responsibilities"]:
        for kw in cleaned.get(cat, []):
            final_keywords.append({"keyword": kw, "category": cat, "explanation": ""})

    # now perform resume filtering (strict) — minimal version: exact substring / token subset / lemma subset / fuzzy
    resume_norm = (resume_text or "").lower()
    resume_norm = re.sub(r'\s+', ' ', resume_norm).strip()
    resume_tokens = re.findall(r'\w+', resume_norm)

    matched = []
    missing = []
    rank = 1
    for item in final_keywords:
        term = item["keyword"]
        term_norm = term.lower()
        found = False
        # exact
        if term_norm in resume_norm:
            found = True
        # token subset
        if not found:
            tks = re.findall(r'\w+', term_norm)
            if tks and set(tks).issubset(set(resume_tokens)):
                found = True
        # fuzzy
        if not found and RAPIDFUZZ_AVAILABLE:
            score = fuzz.token_set_ratio(term_norm, resume_norm)
            if score >= 88:
                found = True
        if found:
            matched.append({"rank": rank, "term": term, "category": item["category"], "explanation": item.get("explanation",""), "evidence": _extract_evidence_sentences(resume_text, term)})
            rank += 1
        else:
            missing.append({"term": term, "category": item["category"], "explanation": item.get("explanation","")})

    result = {
        "keywords": matched,
        "missing": missing,
        "weak": [],
        "summary": f"Three-layer extraction: {len(matched)} matched, {len(missing)} missing.",
        "_raw_extraction": cleaned
    }
    return result
# ---------------------------

def enforce_jd_keywords(obj, jd_text: str, resume_text: str):
    """
    Compute presence/coverage info for extracted keywords vs resume.

    Mutates/returns `obj` with added keys:
      - obj["present"]  : list of keyword terms (strings) found in resume
      - obj["missing"]  : list of keyword terms not found in resume
      - obj["coverage"] : float percentage (0-100) of keywords present
      - obj["present_map"]: optional mapping term -> match_rule for debugging (exact|compact|tokens|variant|evidence)
    Matching rules mirror tokenization used across pipeline:
      - sequential token match
      - compact match (concatenated tokens)
      - all tokens present (for multi-word)
      - singular/plural heuristic for single-token words
    """
    # normalize incoming keywords list
    if isinstance(obj, dict):
        obj["keywords"] = _normalize_keyword_list(obj.get("keywords", []))
    else:
        obj = {"keywords": _normalize_keyword_list(obj)}

    if not isinstance(obj, dict):
        return obj

    keywords = obj.get("keywords") or []

    # tokenization regex should match the one used elsewhere in the file
    token_re = re.compile(r"[A-Za-z0-9#+.]+")
    def _tok_seq(s: str):
        return [t.lower() for t in token_re.findall(s or "")]

    # prepare resume tokens & compact representation
    resume_tokens = _tok_seq(resume_text or "")
    resume_compact = "".join(resume_tokens)
    resume_set = set(resume_tokens)

    def _sequential_present(seq_tokens: List[str]) -> bool:
        L = len(seq_tokens)
        if L == 0:
            return False
        if L <= len(resume_tokens):
            for i in range(0, len(resume_tokens) - L + 1):
                if resume_tokens[i:i+L] == seq_tokens:
                    return True
        return False

    def _compact_present(seq_tokens: List[str]) -> bool:
        if not seq_tokens:
            return False
        return "".join(seq_tokens) in resume_compact

    def _all_tokens_present(seq_tokens: List[str]) -> bool:
        if not seq_tokens:
            return False
        return all(tok in resume_set for tok in seq_tokens)

    def _single_token_plural_heuristic(tok: str) -> bool:
        if not tok or len(tok) <= 3:
            return False
        base = tok
        alt = base[:-1] if base.endswith("s") else base + "s"
        return base in resume_set or alt in resume_set

    def _present_by_rules(seq_tokens: List[str]) -> (bool, str):
        """Return (found, rule_name)"""
        if not seq_tokens:
            return False, ""
        if _sequential_present(seq_tokens):
            return True, "sequential"
        if _compact_present(seq_tokens):
            return True, "compact"
        if len(seq_tokens) > 1 and _all_tokens_present(seq_tokens):
            return True, "all_tokens"
        if len(seq_tokens) == 1 and _single_token_plural_heuristic(seq_tokens[0]):
            return True, "singular_plural"
        return False, ""

    present = []
    missing = []
    present_map = {}

    # For each keyword, attempt matches against term, variants, and evidence lines
    for kw in keywords:
        term = (kw.get("term") or "").strip()
        variants = [v for v in (kw.get("variants") or []) if v and v.strip()]
        evidence = kw.get("evidence") or []

        found = False
        found_rule = ""

        # 1) check main term
        if term:
            seq = _tok_seq(term)
            ok, rule = _present_by_rules(seq)
            if ok:
                found = True
                found_rule = f"term:{rule}"

        # 2) check variants if not found
        if not found:
            for v in variants:
                seqv = _tok_seq(v)
                ok, rule = _present_by_rules(seqv)
                if ok:
                    found = True
                    found_rule = f"variant:{rule}"
                    break

        # 3) check evidence lines (if LLM supplied lines from JD) - see if tokens from evidence are present in resume
        if not found and isinstance(evidence, list) and evidence:
            for e in evidence:
                seqe = _tok_seq(e)
                ok, rule = _present_by_rules(seqe)
                if ok:
                    found = True
                    found_rule = f"evidence:{rule}"
                    break

        # 4) fallback: if term empty but variants exist, mark missing/skip
        display_term = term if term else (variants[0] if variants else "")

        if found:
            present.append(display_term)
            present_map[display_term] = found_rule
        else:
            missing.append(display_term)
            present_map[display_term] = "not_found"

    total = max(1, len(keywords))
    coverage = round((len(present) / total) * 100.0, 1)

    # Attach computed fields to object (do not clobber existing missing if present? we overwrite with computed result)
    obj["present"] = [p for p in present if p]
    obj["missing"] = [m for m in missing if m]
    obj["coverage"] = coverage
    obj["present_map"] = present_map

    return obj


# -----------------------------
# Gaps (compare Top Keywords vs Resume)
# -----------------------------
def extract_gaps(resume_text: str, kw_obj: Dict[str, Any]) -> List[str]:
    """
    Improved gaps extraction:
    - Build candidate list from kw_obj['keywords'] (term + variants), preserve rank order.
    - Normalize and dedupe candidates (collapse punctuation/spacing).
    - Return only those candidates that are NOT present in resume (token-level/compact checks).
    - Ignore tiny tokens and a small stoplist.
    """
    # normalize kw_obj keywords to stable dict list
    if isinstance(kw_obj, dict):
        kw_obj["keywords"] = _normalize_keyword_list(kw_obj.get("keywords", []))
    else:
        kw_obj = {"keywords": _normalize_keyword_list(kw_obj)}

    if not kw_obj or not isinstance(kw_obj.get("keywords"), list) or len(kw_obj.get("keywords")) == 0:
        return []

    token_re = re.compile(r"[A-Za-z0-9#+.]+")
    resume_tokens = [t.lower() for t in token_re.findall(resume_text or "")]
    resume_compact = "".join(resume_tokens)

    def _tok_seq(s: str):
        return [t.lower() for t in token_re.findall(s or "")]

    def _present(seq_tokens: List[str]) -> bool:
        if not seq_tokens:
            return False
        L = len(seq_tokens)
        # sequential token match
        for i in range(0, len(resume_tokens) - L + 1):
            if resume_tokens[i:i+L] == seq_tokens:
                return True
        # compact match (ci/cd -> cicd)
        if "".join(seq_tokens) in resume_compact:
            return True
        # singular/plural heuristic for single-word tokens
        if L == 1 and len(seq_tokens[0]) > 3:
            base = seq_tokens[0]
            alt = base[:-1] if base.endswith("s") else base + "s"
            if base in resume_tokens or alt in resume_tokens:
                return True
        return False

    # helper: normalize candidate for dedupe key (collapse punctuation/whitespace)
    def _norm_key(s: str) -> str:
        if not s:
            return ""
        k = re.sub(r"[^A-Za-z0-9]+", " ", s).strip().lower()
        k = re.sub(r"\s+", " ", k)
        return k

    stoplist = {"open-source"}  # add any noisy tokens you want ignored (normalized form)

    # Build ordered candidate list (preserve keyword rank order, then variant order)
    ordered_candidates = []
    seen_keys = set()
    for kw in (kw_obj.get("keywords") or []):
        term = (kw.get("term") or "").strip()
        variants = kw.get("variants") or []
        # primary term first
        items = [term] + [v for v in variants if v and v.strip()]
        for it in items:
            key = _norm_key(it)
            if not key:
                continue
            if key in seen_keys:
                continue
            # ignore very short tokens
            if len(key) <= 2:
                continue
            if key in stoplist:
                continue
            seen_keys.add(key)
            ordered_candidates.append({"orig": it.strip(), "key": key})

    # Now check each candidate for presence in resume; if absent, include original display form
    gaps = []
    for cand in ordered_candidates:
        seq = _tok_seq(cand["orig"])
        # if normalization changed spacing/punctuation, but token seq empty, try from key:
        if not seq:
            seq = cand["key"].split()
        if not _present(seq):
            gaps.append(cand["orig"])

    # final dedupe (just in case) preserving order
    final = []
    seen_final = set()
    for g in gaps:
        k = _norm_key(g)
        if k not in seen_final:
            seen_final.add(k)
            final.append(g)

    return final

def _first_json_block(text: str) -> Optional[str]:
    """Return first balanced {...} JSON-like block found in text, else None."""
    if not text:
        return None
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items or []:
        if not isinstance(it, str):
            continue
        v = it.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# -----------------------------
# Generate Technical Skills (LLM) -> now returns structured Technical Skills (heading -> list)
# -----------------------------
def generate_keyword_sentences(resume_text: str, jd_text: str, target_keywords: List[str],
                               provider_pref: Optional[str], model_name: Optional[str],
                               temperature: float, max_tokens: int, keys: Dict[str, str]) -> str:
    """
    Append-only merge of target_keywords into the resume's Technical Skills block.
    - Preserve original Technical Skills lines verbatim.
    - Append new keywords only (do not remove or alter existing items).
    - If no heading matches a gap keyword, create a meaningful heading and add the keyword.
    - Use provider only once (optional) to map remaining gaps -> headings; if provider fails, fallback to "Other Technical Skills".
    """
    try:
        # --- helpers ---
        def _log(msg, *args):
            try:
                # concise console logging (replace with logger if available)
                print("[generate_keyword_sentences]", msg % args if args else msg)
            except Exception:
                pass

        def _safe_strip(s):
            try:
                return (s or "").strip()
            except Exception:
                return str(s or "")

        def _is_short_skill(s: str) -> bool:
            """Relaxed definition of a skill token."""
            if not s or not s.strip():
                return False
            s = s.strip()
            if re.fullmatch(r"[-•▪‣·\s]+", s):
                return False
            words = s.split()
            if len(words) > 14:
                return False
            if re.search(r"\b(role|responsib|project|experience|since|from|to|with|present|manager|joined|company)\b", s, flags=re.I):
                return False
            return True

        def _split_line_to_parts(line: str):
            return [p.strip() for p in re.split(r"[,\u2022\u2023\u2024\u2025/|;]+", (line or "")) if p and p.strip()]

        # --- sanitize inputs ---
        resume_text = resume_text or ""
        jd_text = jd_text or ""
        input_candidates = []
        if isinstance(target_keywords, list):
            for t in target_keywords:
                if isinstance(t, str):
                    if t.strip():
                        input_candidates.append(t.strip())
                elif isinstance(t, dict):
                    # support dict items e.g. {"term":"Kubernetes"} or {"keyword":"Kubernetes"}
                    term = t.get("term") or t.get("keyword") or t.get("name") or t.get("kw")
                    if term and isinstance(term, str) and term.strip():
                        input_candidates.append(term.strip())
                    else:
                        # if dict has single string-like value, try to find it
                        for v in t.values():
                            if isinstance(v, str) and v.strip():
                                input_candidates.append(v.strip()); break
                else:
                    try:
                        s = str(t).strip()
                        if s:
                            input_candidates.append(s)
                    except Exception:
                        pass
        elif isinstance(target_keywords, dict):
            # allow passing dict -> take keys or values
            for k, v in target_keywords.items():
                if isinstance(v, str) and v.strip():
                    input_candidates.append(v.strip())
                elif isinstance(k, str) and k.strip():
                    input_candidates.append(k.strip())
        elif isinstance(target_keywords, str):
            if target_keywords.strip():
                input_candidates = [target_keywords.strip()]

        # dedupe preserving order (case-insensitive)
        seen_ck = set(); merged_candidates = []
        for c in input_candidates:
            key = c.strip().lower()
            if key and key not in seen_ck:
                seen_ck.add(key); merged_candidates.append(c.strip())

        _log("Incoming target keywords count: %d", len(merged_candidates))

        # --- extract original Technical Skills block verbatim ---
        def _find_tech_block_lines(text: str):
            lines = text.splitlines()
            start = None; end = None
            for i, ln in enumerate(lines):
                if re.match(r"(?i)^(technical\s*skills|skills|core\s*competenc(?:y|ies)|core\s*skills|expertise|toolbox|technical\s*expertise)\s*[:\-–—]?\s*$", ln.strip()):
                    start = i + 1
                    # find block end
                    for j in range(start, len(lines)):
                        nxt = lines[j].strip()
                        if not nxt:
                            end = j; break
                        if re.match(r"(?i)^(work\s*experience|experience|education|projects|certifications|awards|publications|professional\s*summary|profile\s*summary)\s*[:\-–—]?\s*$", nxt):
                            end = j; break
                    if end is None:
                        end = len(lines)
                    return [lines[k].rstrip() for k in range(start, end)]
            return []

        orig_block_lines = _find_tech_block_lines(resume_text)
        _log("Found %d lines in original tech block.", len(orig_block_lines))

        # parse original block into headings -> exact original items (no normalization)
        headings = []  # order
        items_by_heading = {}  # heading -> [exact items]

        # default top-level heading
        DEFAULT_HEADING = "Technical Skills"
        headings.append(DEFAULT_HEADING)
        items_by_heading[DEFAULT_HEADING] = []

        # parse lines: if "Heading: a, b" present, create subheading; else add to current heading
        current_heading = DEFAULT_HEADING
        for ln in orig_block_lines:
            ln = ln.rstrip()
            if ":" in ln:
                left, right = ln.split(":", 1)
                h = _safe_strip(left)
                parts = _split_line_to_parts(right)
                if parts:
                    if h not in items_by_heading:
                        headings.append(h); items_by_heading[h] = []
                    for p in parts:
                        if p not in items_by_heading[h]:
                            items_by_heading[h].append(p)
                    current_heading = h
                    continue
            # otherwise parse tokens and add to current heading
            parts = _split_line_to_parts(ln)
            if parts:
                for p in parts:
                    if p not in items_by_heading[current_heading]:
                        items_by_heading[current_heading].append(p)

        # If resume had no technical block, ensure DEFAULT exists but empty (we will copy nothing)
        if DEFAULT_HEADING not in items_by_heading:
            headings.insert(0, DEFAULT_HEADING)
            items_by_heading.setdefault(DEFAULT_HEADING, [])

        # --- placement: append-only ---
        global_seen = set(x.lower().strip() for arr in items_by_heading.values() for x in (arr or []))
        remaining_gaps = []

        def _find_best_heading_for_kw(kw: str) -> Optional[str]:
            kl = kw.lower()
            # exact match to an existing item's heading
            for h, items in items_by_heading.items():
                for it in (items or []):
                    if it and it.strip().lower() == kl:
                        return h
            # heading token overlap
            k_toks = set(re.findall(r'\w+', kl))
            if k_toks:
                best_h = None; best_score = 0.0
                for h in headings:
                    h_toks = set(re.findall(r'\w+', h.lower()))
                    if not h_toks: continue
                    inter = k_toks & h_toks
                    smaller = min(len(k_toks), len(h_toks))
                    if smaller > 0:
                        score = len(inter) / smaller
                        if score > best_score:
                            best_score = score; best_h = h
                if best_score >= 0.5:
                    return best_h
            # substring with heading
            for h in headings:
                if kl in h.lower() or h.lower() in kl:
                    return h
            # no match
            return None

        for kw in merged_candidates:
            if not kw or not kw.strip(): continue
            k = kw.strip(); kl = k.lower()
            if kl in global_seen:
                # already present verbatim somewhere; skip
                continue
            placed = False
            # 1) deterministic best-heading
            h = None
            try:
                h = _find_best_heading_for_kw(k)
            except Exception:
                h = None
            if h:
                # append only if not present
                if k not in items_by_heading.get(h, []):
                    items_by_heading.setdefault(h, []).append(k)
                    global_seen.add(kl)
                placed = True

            # 2) fuzzy heading match fallback (if rapidfuzz available)
            if not placed and 'RAPIDFUZZ_AVAILABLE' in globals() and RAPIDFUZZ_AVAILABLE:
                try:
                    best_score = 0; best_h = None
                    for hname in headings:
                        score = fuzz.token_set_ratio(kl, hname.lower())
                        if score > best_score:
                            best_score = score; best_h = hname
                    if best_score >= 84 and best_h:
                        if k not in items_by_heading.get(best_h, []):
                            items_by_heading.setdefault(best_h, []).append(k); global_seen.add(kl)
                        placed = True
                except Exception:
                    pass

            if not placed:
                remaining_gaps.append(k)

        _log("Placement done: placed=%d remaining_gaps=%d", len(merged_candidates) - len(remaining_gaps), len(remaining_gaps))

        # --- if there are remaining gaps, try provider once to map them
        if remaining_gaps:
            _log("Attempting provider mapping for %d remaining gaps", len(remaining_gaps))
            provider = None
            try:
                provider = _provider_from_keys(provider_pref, keys or {})
            except Exception as e:
                _log("Provider resolution failed: %s", e)
                provider = None

            mapped = None
            if provider:
                sys_prompt = (
                    "You are a concise resume assistant. Given remaining technical keywords and the job description, "
                    "return a VALID JSON object mapping heading -> [keywords]. Use existing headings when appropriate. "
                    "Example: {\"Cloud / DevOps\": [\"terraform\",\"kubernetes\"]}. Output JSON ONLY."
                )
                user_prompt = (
                    "Job Description:\n" + jd_text + "\n\n"
                    "Existing resume headings:\n" + (", ".join([h for h in headings if items_by_heading.get(h)]) or "<none>") + "\n\n"
                    "Remaining keywords:\n" + (", ".join(remaining_gaps)) + "\n\n"
                    "Return a JSON mapping heading to keywords (arrays). Use existing headings when appropriate, "
                    "but do NOT overwrite or remove any existing resume skill items. If you propose a new heading, "
                    "return its name exactly as you want it to appear."
                )
                try:
                    out = provider.chat(model=model_name, system=sys_prompt, user=user_prompt,
                                        temperature=max(0.0, min(0.6, temperature)), max_tokens=max_tokens or 400) or ""
                    mapped = None
                    # try to extract json
                    try:
                        start = out.find("{"); end = out.rfind("}") + 1
                        if start != -1 and end > start:
                            mapped = json.loads(out[start:end])
                    except Exception:
                        # best-effort: try to parse whole string
                        try:
                            mapped = json.loads(out)
                        except Exception:
                            mapped = None
                except Exception as e:
                    _log("Provider chat failed or quota: %s", e)
                    mapped = None

            # merge mapped dict append-only if available
            if mapped and isinstance(mapped, dict):
                for new_h, kwlist in mapped.items():
                    if not kwlist:
                        continue
                    # normalize incoming
                    incoming = []
                    if isinstance(kwlist, list):
                        for v in kwlist:
                            if isinstance(v, str) and v.strip():
                                incoming.append(v.strip())
                            else:
                                incoming.append(str(v).strip())
                    elif isinstance(kwlist, str):
                        incoming = _split_line_to_parts(kwlist)
                    else:
                        incoming = [str(kwlist).strip()]
                    # choose canonical heading: prefer existing non-empty headings that match; else use new_h
                    canonical = None
                    nh_lower = new_h.strip().lower()
                    for h in headings:
                        if items_by_heading.get(h) and len(items_by_heading.get(h, [])) > 0:
                            if nh_lower == h.lower() or nh_lower in h.lower() or h.lower() in nh_lower:
                                canonical = h; break
                    if not canonical:
                        for h in headings:
                            if nh_lower == h.lower() or nh_lower in h.lower() or h.lower() in nh_lower:
                                canonical = h; break
                    if not canonical:
                        canonical = new_h.strip()
                        if canonical.lower() in (hh.lower() for hh in headings):
                            base = canonical; i = 1
                            while f"{base} ({i})".lower() in (hh.lower() for hh in headings):
                                i += 1
                            canonical = f"{base} ({i})"
                        if canonical not in headings:
                            headings.append(canonical); items_by_heading.setdefault(canonical, [])
                    for kw in incoming:
                        if not kw: continue
                        if kw.strip().lower() in global_seen: continue
                        if kw not in items_by_heading.get(canonical, []):
                            items_by_heading.setdefault(canonical, []).append(kw); global_seen.add(kw.strip().lower())
            else:
                # fallback: put remaining into "Other Technical Skills"
                bucket = "Other Technical Skills"
                if bucket not in items_by_heading:
                    headings.append(bucket); items_by_heading[bucket] = []
                for k in remaining_gaps:
                    if k.strip().lower() not in global_seen:
                        items_by_heading[bucket].append(k); global_seen.add(k.strip().lower())

        # --- final rendering: copy original items verbatim; appended items will follow
        out_lines = []
        # keep headings order stable and dedupe case-insensitively
        seen_h = set()
        ordered_headings = []
        for h in headings:
            key = h.strip().lower()
            if key and key not in seen_h:
                seen_h.add(key); ordered_headings.append(h)

        for h in ordered_headings:
            arr = items_by_heading.get(h, []) or []
            if not arr:
                continue
            out_lines.append(f"{h}: {', '.join(arr)}")

        if not out_lines:
            return "Technical Skills:"
        return "\n".join(["Technical Skills"] + out_lines).strip()
    except Exception as exc:
        # extreme fallback: do not touch resume at all; simply return original technical block plus the raw merged_candidates appended under Other Technical Skills
        try:
            print("generate_keyword_sentences fatal error:", exc)
        except Exception:
            pass
        try:
            orig_block = "\n".join(_find_tech_block_lines(resume_text)) if 'resume_text' in locals() else ""
            if merged_candidates:
                appended = ", ".join([c for c in merged_candidates if c])
                return (("Technical Skills\n" + (orig_block + ("\nOther Technical Skills: " + appended if appended else ""))) if orig_block else ("Technical Skills:\nOther Technical Skills: " + appended))
            else:
                return "Technical Skills:\n" + (orig_block or "")
        except Exception:
            return "Technical Skills:"


def polish_keyword_sentences(resume_text: str, bullets_text: str, jd_text: str,
                             provider_pref: Optional[str], model_name: Optional[str],
                             temperature: float, max_tokens: int, keys: Dict[str, str]) -> str:
    from ai.prompts import SYSTEM_KEYWORD_SENTENCES_POLISH, USER_KEYWORD_SENTENCES_POLISH
    provider = _provider_from_keys(provider_pref, keys or {})
    user = USER_KEYWORD_SENTENCES_POLISH.format(resume=resume_text or "", bullets=bullets_text or "", jd=jd_text or "")
    out = provider.chat(model=model_name, system=SYSTEM_KEYWORD_SENTENCES_POLISH, user=user, temperature=temperature, max_tokens=min(max_tokens, 800))
    return sanitize_markdown(out or "")

def polish_core_competencies(original_bullets: str, new_bullets: str,
                             provider_pref: Optional[str], model_name: Optional[str],
                             temperature: float, max_tokens: int, keys: Dict[str, str]) -> str:
    from ai.prompts import SYSTEM_CORE_COMPETENCIES_POLISH, USER_CORE_COMPETENCIES_POLISH
    provider = _provider_from_keys(provider_pref, keys or {})
    user = USER_CORE_COMPETENCIES_POLISH.format(original=original_bullets or "", new=new_bullets or "")
    raw = provider.chat(model=model_name, system=SYSTEM_CORE_COMPETENCIES_POLISH, user=user, temperature=temperature, max_tokens=min(max_tokens, 900))
    return sanitize_markdown(raw or "").strip()



# -----------------------------
# Keyword Sentences Polishing (left as-is)
# -----------------------------

def generate_summary_bullets(resume_text: str, jd_text: str, focus: str,
                             provider_pref: Optional[str], model_name: Optional[str],
                             temperature: float, max_tokens: int, keys: Dict[str, str]) -> str:
    provider = _provider_from_keys(provider_pref, keys or {})
    raw = provider.chat(model=model_name, system=SYSTEM_SUMMARY_BULLETS, user=USER_SUMMARY_BULLETS.format(jd=jd_text, resume=resume_text, focus=(focus or "")), temperature=temperature, max_tokens=max_tokens)
    return sanitize_markdown(raw or "").strip()

def bulletize_summary_preserve_meaning(summary_text: str,
                                       provider_pref: Optional[str], model_name: Optional[str],
                                       temperature: float, max_tokens: int, keys: Dict[str, str]) -> str:
    summary_text = (summary_text or "").strip()
    if not summary_text: return ""
    provider = _provider_from_keys(provider_pref, keys or {})
    raw = provider.chat(model=model_name, system=SYSTEM_SUMMARY_BULLETS, user=USER_SUMMARY_BULLETS.format(jd="", resume=summary_text, focus=""), temperature=temperature, max_tokens=max_tokens)
    return sanitize_markdown(raw or "").strip()

# -----------------------------
# Render JSON -> text (unchanged)
# -----------------------------
def _limit_words(s: str, max_words: int = 22) -> str:
    parts = s.split()
    return s if len(parts)<=max_words else " ".join(parts[:max_words])

def _clean_bullets(arr: List[str]) -> List[str]:
    cleaned = []
    for b in arr or []:
        b = b.strip().lstrip("- ").lstrip("• ").strip()
        b = re.sub(r"[\.;:]+$", "", b)
        cleaned.append(_limit_words(b))
    return cleaned

def render_text_from_json(obj: Dict[str, Any]) -> str:
    h = obj.get("header", {}) or {}
    name = (h.get("name") or "").strip()
    contact_line = " • ".join([x for x in [h.get("email","").strip(), h.get("phone","").strip(), h.get("linkedin","").strip(), h.get("github","").strip()] if x])

    lines = []
    if name: lines += [name]
    if contact_line: lines += [contact_line, ""]

    def section(title: str): lines.append(title)

    if obj.get("summary"):
        section("Profile Summary"); lines += [obj["summary"].strip(), ""]

    # place technical_skills here if present (tailor JSON)
    ts = obj.get("technical_skills")
    if ts:
        section("Technical Skills")
        if isinstance(ts, dict):
            for k, arr in ts.items():
                arr = [a for a in arr if a]
                if arr: lines.append(k + ": " + ", ".join(arr))
        elif isinstance(ts, list):
            for s in ts: lines.append("• " + s)
        lines.append("")

    exp = obj.get("experience") or []
    if exp:
        section("Work Experience")
        for role in exp:
            company = (role.get("company") or "").strip()
            title = (role.get("title") or "").strip()
            dates = (role.get("dates") or "").strip()
            location = (role.get("location") or "").strip()
            header = " | ".join([x for x in [title, company, location, dates] if x])
            if header: lines.append(header)
            for b in _clean_bullets(role.get("bullets") or []):
                if b: lines.append("• " + b)
            lines.append("")
        lines.append("")

    edu = obj.get("education") or []
    if edu:
        section("Education")
        for e in edu:
            header = " | ".join([x for x in [(e.get("degree") or "").strip(), (e.get("school") or "").strip(), (e.get("location") or "").strip(), (e.get("dates") or "").strip()] if x])
            if header: lines.append(header)
            for d in _clean_bullets(e.get("details") or []):
                if d: lines.append("• " + d)
        lines.append("")

    certs = obj.get("certifications") or []
    if certs:
        section("Certifications")
        for c in certs: lines.append("• " + c)
        lines.append("")

    projs = obj.get("projects") or []
    if projs:
        section("Projects")
        for p in projs:
            pname = (p.get("name") or "").strip()
            tech = ", ".join(p.get("tech") or [])
            header = " • ".join([x for x in [pname, tech] if x])
            if header: lines.append(header)
            for b in _clean_bullets(p.get("bullets") or []):
                if b: lines.append("• " + b)
            lines.append("")
        lines.append("")

    return sanitize_markdown("\n".join(lines))

def _remove_technical_skills_section(text: str) -> str:
    """
    Remove an existing 'Technical Skills' section entirely (heading + body).
    """
    if not text:
        return text
    pattern = re.compile(r"(?im)^\s*technical[\s\-_:]*skills\s*[:\-–—]?\s*$")
    m = pattern.search(text)
    if not m:
        return text
    start = m.start()
    end = m.end()
    after = text[end:]
    nxt = re.search(
        r"(?im)^\s*(work\s*experience|experience|education|projects|certifications|awards|publications|profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$",
        after
    )
    block_end = end + (nxt.start() if nxt else len(after))
    return (text[:start] + text[block_end:]).strip()

# -----------------------------
# Remove Core Competencies + insert technical block
# -----------------------------
def _remove_core_competencies_section(text: str) -> str:
    """
    Remove a 'Core Competencies' section entirely (heading + body).
    """
    if not text:
        return text
    pattern = re.compile(r"(?im)^\s*core[\s\-_:]*competencies\s*[:\-–—]?\s*$")
    m = pattern.search(text)
    if not m:
        return text
    start = m.start()
    end = m.end()
    after = text[end:]
    # find next section heading to mark end of block
    nxt = re.search(
        r"(?im)^\s*(technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$",
        after
    )
    block_end = end + (nxt.start() if nxt else len(after))
    return (text[:start] + text[block_end:]).strip()


def _parse_block_to_headings(block: str) -> Dict[str, List[str]]:
    """
    Parse a skills_block (lines like "Heading: a, b" or raw 'Technical Skills' block)
    into { heading: [item, ...] } preserving item text.
    """
    out = {}
    if not block:
        return out
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    # if first line is "Technical Skills", drop it for parsing headings
    if lines and lines[0].lower().startswith("technical"):
        lines = lines[1:]
    for ln in lines:
        if ":" in ln:
            left, right = ln.split(":", 1)
            heading = left.strip()
            items = [p.strip() for p in re.split(r"[,\u2022\u2023\u2024\u2025/|;]+", right) if p.strip()]
            if items:
                out.setdefault(heading, []).extend(items)
        else:
            # treat as inline items under fallback heading
            items = [p.strip() for p in re.split(r",|\u2022|\t", ln) if p.strip()]
            if items:
                out.setdefault("Technical Skills", []).extend(items)
    # normalize lists: strip whitespace, remove empty
    for h in list(out.keys()):
        out[h] = [i for i in (x.strip() for x in out[h]) if i]
        if not out[h]:
            del out[h]
    return out


def _merge_preserve_resume(resume_text: str, parsed_new: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Read existing resume Technical Skills sections, preserve them exactly,
    and merge new parsed_new headings/items by:
    - if heading exists in resume, append only items not already present (case-insensitive)
    - if heading missing, add it at the end in the order parsed_new provides
    - ensure no duplicates (case-insensitive), do not change order of existing items
    Returns merged heading->items map and insertion order (list of headings).
    """
    # reuse resume parsing logic (small inline helper similar to _find_resume_skill_sections)
    def _find_resume_skill_sections_local(text: str):
        lines = (text or "").splitlines()
        sections = []
        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            if re.match(r"(?i)^(technical\s*skills|skills|core\s*competencies|core\s*skills|expertise|toolbox|technical\s*expertise)\s*[:\-–—]?\s*$", s):
                start = i + 1
                end = len(lines)
                for j in range(start, len(lines)):
                    nxt = lines[j].strip()
                    if not nxt:
                        end = j
                        break
                    if re.match(r"(?i)^(work\s*experience|experience|education|projects|certifications|awards|publications|professional\s*summary|profile\s*summary)\s*[:\-–—]?\s*$", nxt):
                        end = j
                        break
                block = [lines[k].rstrip() for k in range(start, end) if lines[k].strip()]
                sections.append((s.rstrip(":"), block))
        return sections

    resume_sections = _find_resume_skill_sections_local(resume_text)
    merged_headings = []
    merged_skills = {}

    # preserve resume headings and items exactly (order)
    for heading, block_lines in resume_sections:
        items = []
        for ln in block_lines:
            ln = ln.strip()
            # a line may be "Heading: a, b" or a bullet; attempt parsing
            if ":" in ln:
                _, vals = ln.split(":", 1)
                parts = [p.strip() for p in re.split(r",|\u2022", vals) if p.strip()]
                items.extend(parts)
            else:
                # bullet or comma separated
                parts = [p.strip() for p in re.split(r",|\u2022|\t", ln) if p.strip()]
                items.extend(parts)
        # dedupe preserving order (case-insensitive)
        seen = set(); final_items = []
        for it in items:
            key = it.lower()
            if key in seen:
                continue
            seen.add(key); final_items.append(it)
        if final_items:
            merged_headings.append(heading)
            merged_skills[heading] = final_items

    # now merge parsed_new: append new items into existing headings (without modifying existing order)
    for new_h in parsed_new:
        new_items = parsed_new.get(new_h, []) or []
        if not new_items:
            continue
        if new_h in merged_skills:
            exist_keys = {x.lower() for x in merged_skills[new_h]}
            for it in new_items:
                if it and it.strip() and it.lower() not in exist_keys:
                    merged_skills[new_h].append(it.strip()); exist_keys.add(it.lower())
        else:
            # add heading at the end in the order parsed_new provides
            merged_headings.append(new_h)
            # dedupe new items preserving their order
            seen_n = set(); final_n = []
            for it in new_items:
                k = it.lower()
                if k not in seen_n:
                    seen_n.add(k); final_n.append(it)
            merged_skills[new_h] = final_n

    return {"headings": merged_headings, "skills": merged_skills}

def _render_skills_block(merged: Dict[str, Any]) -> str:
    """
    Produce a skills_block (lines like "Heading: a, b") from merged structure.
    """
    out_lines = []
    for h in merged.get("headings", []):
        items = merged.get("skills", {}).get(h, []) or []
        if not items:
            continue
        out_lines.append(f"{h}: {', '.join(items)}")
    if not out_lines:
        return "Technical Skills"
    return "Technical Skills\n" + "\n".join(out_lines)

# ---------- NEW helpers for robust merge+insert ----------
def _collect_existing_skill_blocks(text: str):
    """
    Find all existing skill-like blocks (Technical Skills, Skills, Core Competencies),
    parse their contents into a dict {heading: [items...]}, preserve order, and return:
      cleaned_text (with those blocks removed), existing_parsed (dict), headings_order (list)
    """
    if not text:
        return text, {}, []

    heading_pat = re.compile(r"(?im)^\s*(technical\s*skills|skills|core\s*competenc(?:ies|y))\s*[:\-–—]?\s*$", re.M)
    next_section_pat = re.compile(
        r"(?im)^\s*(work\s*experience|experience|education|projects|certifications|awards|publications|profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$",
        re.M
    )

    matches = list(heading_pat.finditer(text))
    if not matches:
        return text, {}, []

    ranges = []
    accumulated = {}  # heading -> [items]
    headings_order = []

    for m in matches:
        hname = m.group(1).strip()
        start = m.start()
        end = m.end()
        after = text[end:]
        nxt = next_section_pat.search(after)
        block_end = end + (nxt.start() if nxt else len(after))
        ranges.append((start, block_end))

        # body lines (excluding the heading line)
        body = text[end:block_end]
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

        # parse each line: "Heading: a, b" OR bullets/comma lists -> attach under hname
        for ln in lines:
            # If line contains a ":" treat as sub-heading line
            if ":" in ln:
                left, right = ln.split(":", 1)
                subh = left.strip()
                items = [p.strip() for p in re.split(r",|\u2022|;|\t", right) if p.strip()]
                if items:
                    if subh not in accumulated:
                        accumulated[subh] = []
                        headings_order.append(subh)
                    for it in items:
                        if it and it not in accumulated[subh]:
                            accumulated[subh].append(it)
            else:
                # plain bullets or comma separated items -> attach under main heading hname
                parts = [p.strip() for p in re.split(r",|\u2022|;|\t", ln) if p.strip()]
                if parts:
                    if hname not in accumulated:
                        accumulated[hname] = []
                        headings_order.append(hname)
                    for it in parts:
                        if it and it not in accumulated[hname]:
                            accumulated[hname].append(it)

    # remove ranges from text (build new text skipping those ranges)
    if not ranges:
        return text, accumulated, headings_order

    ranges = sorted(ranges, key=lambda x: x[0])
    out_parts = []
    cursor = 0
    for (s, e) in ranges:
        if cursor < s:
            out_parts.append(text[cursor:s])
        cursor = max(cursor, e)
    if cursor < len(text):
        out_parts.append(text[cursor:])
    cleaned = "".join(out_parts).strip()
    return cleaned, accumulated, headings_order


def _merge_parsed_skill_dicts(existing: Dict[str, List[str]], new: Dict[str, List[str]]):
    """
    Merge two heading->items dicts:
    - keep existing headings & their items (order preserved)
    - append items from `new` into existing heading if missing (case-insensitive)
    - append new headings (in their order) at the end if not present
    Returns merged dict and headings order list.
    """
    merged = {}
    headings = []

    # helper lower mapping for existing headings to match case-insensitively
    existing_heading_map = {h.lower(): h for h in (existing.keys() or [])}

    # start with existing headings & items
    for h in (existing.keys() or []):
        items = existing.get(h, []) or []
        seen = set()
        kept = []
        for it in items:
            if not isinstance(it, str):
                it = str(it)
            key = it.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            kept.append(it.strip())
        if kept:
            merged[h] = kept
            headings.append(h)

    # now merge new items into existing headings where the heading matches (case-insensitive)
    for nh in (new.keys() or []):
        nitems = new.get(nh, []) or []
        # find matching existing heading (case-insensitive)
        match_h = existing_heading_map.get(nh.lower())
        if match_h:
            exist_items = merged.setdefault(match_h, [])
            exist_seen = {it.lower() for it in exist_items}
            for it in nitems:
                if not it: continue
                if it.strip().lower() not in exist_seen:
                    exist_items.append(it.strip()); exist_seen.add(it.strip().lower())
        else:
            # new heading not present in existing: add at end
            cleaned = []
            seen_local = set()
            for it in nitems:
                if not it: continue
                k = it.strip().lower()
                if k in seen_local: continue
                seen_local.add(k); cleaned.append(it.strip())
            if cleaned:
                merged[nh] = cleaned
                headings.append(nh)

    return merged, headings


def _global_dedupe_preserve_order(headings: List[str], skills_map: Dict[str, List[str]]):
    """
    Remove duplicate skill tokens across headings (case-insensitive).
    Keep first-seen occurrence (heading order then item order).
    Returns deduped headings list and skills_map.
    """
    global_seen = set()
    deduped_headings = []
    deduped_skills = {}

    for h in headings:
        items = skills_map.get(h, []) or []
        new_items = []
        for it in items:
            if not it: continue
            key = it.strip().lower()
            if key in global_seen:
                continue
            global_seen.add(key)
            new_items.append(it.strip())
        if new_items:
            deduped_headings.append(h)
            deduped_skills[h] = new_items

    return deduped_headings, deduped_skills


def _merge_and_insert_skills(full_text: str, skills_block: str,
                             jd_core_competencies: Optional[List[str]] = None,
                             prefer_after_summary: bool = True) -> str:
    """
    Robust routine:
    1) Collect all existing skill blocks (without losing content).
    2) Parse incoming skills_block.
    3) Merge existing + incoming (preserve order, append new items).
    4) Merge JD core competencies if provided under "Core Competencies".
    5) Dedupe globally (case-insensitive).
    6) Render a single block and insert (prefer after summary if requested).
    """
    if not full_text:
        return full_text

    text = full_text

    # 1) collect & strip existing blocks
    cleaned_text, existing_parsed, existing_order = _collect_existing_skill_blocks(text)

    # 2) parse incoming block
    parsed_new = _parse_block_to_headings(skills_block or "") or {}

    # if incoming empty but JD cores provided, make parsed_new hold them
    if (not parsed_new) and jd_core_competencies:
        parsed_new = {"Core Competencies": list(jd_core_competencies)}

    # 3) merge existing & new
    merged_map, merged_headings = _merge_parsed_skill_dicts(existing_parsed, parsed_new)

    # 4) merge JD core competencies into merged_map under canonical heading
    if jd_core_competencies:
        core_key = None
        # try to find existing core heading key case-insensitively
        for h in merged_headings:
            if h.lower().startswith("core"):
                core_key = h; break
        if not core_key:
            core_key = "Core Competencies"
            # insert core_key near the front (prefer before Technical Skills)
            if core_key not in merged_headings:
                merged_headings.insert(0, core_key)
        cur = merged_map.get(core_key, [])
        cur_lower = {c.lower() for c in cur}
        for jd_item in jd_core_competencies or []:
            if jd_item and jd_item.strip() and jd_item.lower() not in cur_lower:
                cur.append(jd_item.strip()); cur_lower.add(jd_item.lower())
        if cur:
            merged_map[core_key] = cur

    # 5) global dedupe
    final_headings, final_map = _global_dedupe_preserve_order(merged_headings, merged_map)

    # Ensure Technical Skills is present for fallback ordering if nothing else exists
    if not final_headings:
        # try to populate with tokens from parsed_new targets (if any)
        fallback_items = []
        for arr in (parsed_new.values() or []):
            for it in arr:
                if it and _is_short_skill(it):
                    fallback_items.append(it.strip())
        if fallback_items:
            final_headings = ["Technical Skills"]
            final_map = {"Technical Skills": _dedupe_preserve_order(fallback_items)}
        else:
            # nothing meaningful to insert
            return cleaned_text

    # 6) ensure "Technical Skills" heading exists (if not present, appended at end)
    if not any(h.lower().startswith("technical") for h in final_headings):
        final_headings.append("Technical Skills")
        # if there are no items under Technical Skills, nothing to add there (ok)

    # Build merged structure compatible with _render_skills_block
    merged_struct = {"headings": final_headings, "skills": final_map}
    block = _render_skills_block(merged_struct)

    # 7) Insert block into cleaned_text
    # Prefer inserting after Profile Summary heading if requested
    if prefer_after_summary:
        summary_heading_re = re.compile(r"(?im)^\s*(profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$")
        m = summary_heading_re.search(cleaned_text)
        if m:
            head_end = m.end()
            after = cleaned_text[head_end:]
            nxt = re.search(r"(?im)^\s*(technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$", after)
            insert_pos = head_end + (nxt.start() if nxt else len(after))
            new_text = cleaned_text[:insert_pos].rstrip() + "\n\n" + block + "\n\n" + cleaned_text[insert_pos:].lstrip()
            return new_text

    # fallback: insert after first non-empty line (name/contacts)
    parts = cleaned_text.splitlines()
    idx = 0
    while idx < len(parts) and not parts[idx].strip():
        idx += 1
    insert_at = min(len(parts), idx + 1)
    new_lines = parts[:insert_at] + ["", block, ""] + parts[insert_at:]
    return "\n".join(new_lines).strip()


# ---------- Replacements / wrappers (keeps your original API) ----------
def insert_technical_skills(full_text: str, skills_block: str) -> str:
    """
    Backwards-compatible: merge skill block into resume but do not create duplicates.
    Default behavior: insert near Summary or at top (same as previous).
    """
    return _merge_and_insert_skills(full_text, skills_block, jd_core_competencies=None, prefer_after_summary=False)


def insert_technical_skills_after_summary(full_text: str, skills_block: str,
                                         jd_core_competencies: Optional[List[str]] = None) -> str:
    """
    Backwards-compatible wrapper that prefers inserting after the Profile Summary.
    """
    return _merge_and_insert_skills(full_text, skills_block, jd_core_competencies=jd_core_competencies, prefer_after_summary=True)


# -----------------------------
# Tailor (JSON-first, fallback)
# -----------------------------
def tailor(resume_text: str, jd_text: str,
           provider_preference: str = None, model_name: str = None,
           temperature: float = 0.2, max_tokens: int = 1500,
           keys: Dict[str,str] = None, target_keywords: Optional[List[str]] = None,
           override_contacts: Optional[Dict[str,str]] = None) -> str:
    provider = _provider_from_keys(provider_preference, keys or {})
    kw_blob = "\n".join(f"- {k}" for k in (target_keywords or []))
    contacts = override_contacts if override_contacts is not None else extract_contacts_regex(resume_text)
    contact_block = f"""name={contacts.get('name','')}
email={contacts.get('email','')}
phone={contacts.get('phone','')}
linkedin={contacts.get('linkedin','')}
github={contacts.get('github','')}"""

    raw = provider.chat(model=model_name, system=SYSTEM_TAILOR_JSON, user=USER_TAILOR_JSON.format(jd=jd_text, resume=resume_text, contact=contact_block, keywords=kw_blob), temperature=temperature, max_tokens=max_tokens).strip()

    try:
        start = raw.find("{"); end = raw.rfind("}") + 1
        obj = json.loads(raw[start:end])
        txt = render_text_from_json(obj)
        if txt: return txt
    except Exception:
        pass

    out = provider.chat(model=model_name, system=SYSTEM_TAILOR, user=USER_TAILOR.format(jd=jd_text, resume=resume_text, keywords=kw_blob), temperature=temperature, max_tokens=max_tokens)
    return sanitize_markdown(out)

def replace_core_competencies(full_text: str, new_bullets: str) -> str:
    """
    Replace 'Core Competencies' section with new bullets
    (Kept for backward compatibility but NOT used in the new flow.)
    """
    new_bullets = (new_bullets or "").strip()
    if not new_bullets: return full_text

    lines = []
    for ln in new_bullets.splitlines():
        s = ln.strip()
        if not s: continue
        if not s.startswith("• "):
            s = "• " + s.lstrip("-• ").strip()
        lines.append(s)
    block = "\n".join(lines)

    pattern = re.compile(r"(?im)^(core\s*competencies)\s*[:\-–—]?\s*$")
    m = pattern.search(full_text or "")
    if m:
        head = full_text[:m.end()]
        after = full_text[m.end():]
        nxt = re.search(r"(?im)^\s*(skills|technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$", after)
        section_end = m.end() + (nxt.start() if nxt else len(after))
        tail = full_text[section_end:]
        return (head + "\n" + block + "\n\n" + tail).strip()
    else:
        insertion_point = re.search(r"(?im)^\s*(technical\s*skills)\s*[:\-–—]?\s*$", full_text or "")
        if insertion_point:
            idx = insertion_point.start()
            return full_text[:idx] + "\nCore Competencies\n" + block + "\n\n" + full_text[idx:]
        return full_text.rstrip() + "\n\nCore Competencies\n" + block

# -----------------------------
# AI ATS
# -----------------------------
def extract_ats_llm_from_optimizer(resume_text: str, optimizer_obj: Dict[str, Any],
                                   provider_pref: Optional[str], model_name: Optional[str],
                                   temperature: float, max_tokens: int, keys: Dict[str, str],
                                   jd_text: Optional[str] = "") -> Dict[str, Any]:
    provider = _provider_from_keys(provider_pref, keys)
    # 🔹 normalize optimizer_obj keywords to be safe
    optimizer_obj = optimizer_obj or {}
    optimizer_obj["keywords"] = _normalize_keyword_list(optimizer_obj.get("keywords", []))

    kws = []
    for item in (optimizer_obj.get("keywords") or []):
        term = (item.get("term") or "").strip()
        if term:
            kws.append({"rank": item.get("rank", None), "term": term, "variants": [v for v in (item.get("variants") or []) if v]})
    payload = {"keywords": kws, "missing": optimizer_obj.get("missing") or [], "weak": optimizer_obj.get("weak") or [], "summary": optimizer_obj.get("summary") or ""}
    optimizer_json = json.dumps(payload, ensure_ascii=False)

    raw = provider.chat(model=model_name, system=SYSTEM_ATS, user=USER_ATS.format(resume=resume_text, optimizer_json=optimizer_json, jd=(jd_text or "")), temperature=0, max_tokens=min(max_tokens, 1200))
    raw = (raw or "").strip().strip('`').strip()

    try:
        start = raw.find('{'); end = raw.rfind('}') + 1
        obj = json.loads(raw[start:end])
    except Exception:
        obj = {}

    try:
        obj["score"] = int(max(0, min(100, int(obj.get("score", 0)))))
    except Exception:
        obj["score"] = 0
    for k in ["present", "missing", "suggestions", "coverage"]:
        if not isinstance(obj.get(k), list):
            obj[k] = []
    fixed_sugg = []
    for s in obj.get("suggestions", []):
        if isinstance(s, dict):
            fixed_sugg.append({"term": s.get("term", ""), "section": s.get("section", "Core Competencies"), "how": s.get("how", "")})
    obj["suggestions"] = fixed_sugg
    return obj

# ----------------------------
# Non-invasive tracing / instrumentation
# Add this block AFTER the original function definitions.
# It wraps the existing functions and prints human-readable step info,
# but DOES NOT change their behavior or return values.
# ----------------------------

from functools import wraps
from pprint import pprint
import time

def _short_repr(x, maxlen=400):
    """Short safe string representation for logging."""
    try:
        if x is None:
            return "(none)"
        if isinstance(x, str):
            s = x.strip()
            return (s[:maxlen] + "...") if len(s) > maxlen else s
        if isinstance(x, (list, tuple, set)):
            return f"{type(x).__name__}({len(x)}) -> {str(list(x)[:20])[:maxlen]}"
        if isinstance(x, dict):
            # show keys only
            ks = list(x.keys())[:50]
            return f"dict(keys={ks})"
        return str(x)[:maxlen]
    except Exception:
        return "<unprintable>"

def trace_step(step_name: str, show_inputs: bool = True, show_output: bool = True):
    """
    Decorator factory to trace function calls in a human-readable terminal format.
    Use like: @trace_step('Step 1: Broad Extract')
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            print("\n" + "=" * 80)
            print(f"[PIPELINE TRACE] START: {step_name}")
            if show_inputs:
                # print a short summary of main inputs
                try:
                    # attempt to find jd_text or items/resume_text in args/kwargs
                    if kwargs.get("jd_text") is not None:
                        print(" JD (preview):", _short_repr(kwargs.get("jd_text")))
                    elif len(args) >= 1 and isinstance(args[0], str) and len(args[0]) > 0:
                        # assume first arg is jd_text for many functions
                        print(" Preview input[0]:", _short_repr(args[0]))
                    elif kwargs.get("items") is not None:
                        print(" Items count:", len(kwargs.get("items")) if kwargs.get("items") else 0)
                    elif len(args) >= 1 and isinstance(args[0], (list,tuple)):
                        print(" Items count:", len(args[0]))
                except Exception:
                    pass
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                duration = time.time() - start
                print(f"[PIPELINE TRACE] {step_name} RAISED EXCEPTION after {duration:.2f}s: {e}")
                print("=" * 80 + "\n")
                raise
            duration = time.time() - start

            if show_output:
                # print a concise but human-readable summary of the output
                try:
                    if isinstance(result, dict):
                        # show main keys and counts
                        keys = list(result.keys())
                        print(" Output keys:", keys)
                        # If token lists present, show counts and preview
                        for k in ("tokens", "Technical Skills", "Noise", "Soft Skills", "Responsibilities"):
                            if k in result:
                                v = result[k]
                                try:
                                    if isinstance(v, list):
                                        print(f"  - {k}: count={len(v)}  preview={_short_repr(v[:10])}")
                                    else:
                                        print(f"  - {k}: {_short_repr(v)}")
                                except Exception:
                                    pass
                    elif isinstance(result, list):
                        print(f" Output: list of length {len(result)}; preview: {_short_repr(result[:20])}")
                    elif isinstance(result, str):
                        print(" Output (string preview):", _short_repr(result, maxlen=800))
                    else:
                        print(" Output preview:", _short_repr(result))
                except Exception:
                    print(" Output: (unprintable summary)")
            print(f"[PIPELINE TRACE] END: {step_name}  (elapsed {duration:.2f}s)")
            print("=" * 80 + "\n")
            return result
        return wrapper
    return decorator

# Wrap existing functions non-destructively.
# Save originals in case you need to call them directly.
try:
    # wrap llm_broad_extract
    if 'llm_broad_extract' in globals():
        _orig_llm_broad_extract = llm_broad_extract
        llm_broad_extract = trace_step("Step 1 — Broad LLM extraction (llm_broad_extract)")(llm_broad_extract)

    # wrap llm_refine_and_classify
    if 'llm_refine_and_classify' in globals():
        _orig_llm_refine_and_classify = llm_refine_and_classify
        llm_refine_and_classify = trace_step("Step 2 — Refine & classify (llm_refine_and_classify)")(llm_refine_and_classify)

    # wrap three_layer_extract_and_normalize
    if 'three_layer_extract_and_normalize' in globals():
        _orig_three_layer = three_layer_extract_and_normalize
        three_layer_extract_and_normalize = trace_step("Orchestrator — three_layer_extract_and_normalize", show_inputs=True, show_output=True)(three_layer_extract_and_normalize)

    # Optionally wrap extract_keywords_llm to show resume matching summary
    if 'extract_keywords_llm' in globals():
        _orig_extract_keywords_llm = extract_keywords_llm
        def _extract_wrapper(*args, **kwargs):
            # Print header with provider/resume/JD preview
            provider = kwargs.get("provider_pref", None) if "provider_pref" in kwargs else (args[2] if len(args) > 2 else None)
            jd_preview = None
            resume_preview = None
            try:
                jd_preview = kwargs.get("jd_text", None) or (args[1] if len(args) > 1 else None)
                resume_preview = kwargs.get("resume_text", None) or (args[0] if len(args) > 0 else None)
            except Exception:
                pass
            print("\n" + "#" * 80)
            print("PIPELINE TRACE — extract_keywords_llm called")
            print(" Provider selected:", provider)
            print(" JD preview:", _short_repr(jd_preview, maxlen=300))
            print(" Resume preview:", _short_repr(resume_preview, maxlen=200))
            print("#" * 80 + "\n")
            start = time.time()
            out = _orig_extract_keywords_llm(*args, **kwargs)
            dur = time.time() - start
            # print summary
            try:
                matched = out.get("keywords", [])
                missing = out.get("missing", [])
                print("\n" + "-" * 60)
                print(f"extract_keywords_llm SUMMARY: matched={len(matched)}, missing={len(missing)}, elapsed={dur:.2f}s")
                if len(matched) > 0:
                    print(" First matched items:")
                    for m in matched[:10]:
                        print("  -", _short_repr(m.get("term") or m.get("keyword") or m))
                if len(missing) > 0:
                    print(" First missing items:")
                    for m in missing[:10]:
                        print("  -", _short_repr(m.get("term") or m))
                print("-" * 60 + "\n")
            except Exception:
                pass
            return out
        extract_keywords_llm = _extract_wrapper

except Exception as e:
    # If wrapping fails, log to logger but do not raise so pipeline still works.
    try:
        logger.exception("Failed to install pipeline tracers: %s", e)
    except Exception:
        print("Failed to install pipeline tracers:", e)
#--------------------
