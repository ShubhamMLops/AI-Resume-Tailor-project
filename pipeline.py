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
def extract_keywords_llm(resume_text: str, jd_text: str,
                         provider_pref: Optional[str], model_name: Optional[str],
                         temperature: float, max_tokens: int, keys: Dict[str, str]) -> Dict[str, Any]:
    """
    Run LLM keyword extractor, but post-filter results so that only terms/variants
    that actually appear in the provided JD are kept.

    Domain-agnostic behavior:
    - No stoplist or domain-specific filtering.
    - Strict presence rules:
      * single-word: sequential token match OR compact match
      * multi-word: exact sequential match OR all tokens present somewhere OR compact match
    - Returns obj with 'keywords' filtered and '_filtered_out' listing removed entries.
    """
    provider = _provider_from_keys(provider_pref, keys)
    # Convert raw JD to normalized JSON via LLM (for more stable downstream extraction).
    _jd_json = jd_to_json_via_llm(jd_text, provider, model_name, temperature=0.0, max_tokens=800)
    # Print the JD JSON to terminal (not GUI)
    print("\n=== NORMALIZED JD JSON (used for extraction) ===")
    print(json.dumps(_jd_json, indent=2, ensure_ascii=False))
    print("==============================================\n")
    # Replace jd_text with the normalized flat representation for downstream use
    jd_text = _jd_json.get("flat", jd_text)


    # If JD is empty, return a clean empty structure immediately
    if not (jd_text and jd_text.strip()):
        return {"keywords": [], "missing": [], "weak": [], "summary": "", "_raw_json": "", "_filtered_out": []}

    raw = provider.chat(
        model=model_name,
        system=SYSTEM_KEYWORDS,
        user=USER_KEYWORDS.format(jd=jd_text, resume=resume_text),
        temperature=temperature,
        max_tokens=max_tokens
    )

    # normalize raw output and extract JSON safely
    raw = (raw or "").strip()

    # remove common prefixes like "json" or ```json
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.strip("`").strip()

    # find first "{" and last "}"
    start, end = raw.find("{"), raw.rfind("}") + 1
    candidate = raw[start:end] if start != -1 and end > start else raw

    # clean up common issues (trailing commas, smart quotes)
    candidate = re.sub(r",\s*(\]|\})", r"\1", candidate)
    candidate = candidate.replace("“", '"').replace("”", '"').replace("’", "'")

    try:
        obj = json.loads(candidate)
        raw = candidate
    except Exception:
        obj = {"keywords": [], "missing": [], "weak": [], "summary": ""}
        raw = json.dumps(obj)

    # pretty-print parsed JSON to terminal/logs
    try:
        pretty = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        pretty = raw  # fallback to raw string if obj isn't a dict

    print("\n=== RAW LLM OUTPUT (cleaned & pretty) ===")
    print(pretty)
    print("======================\n")


    # Normalize schema keys
    if not isinstance(obj.get("keywords"), list):
        obj["keywords"] = []
    if not isinstance(obj.get("missing"), list):
        obj["missing"] = []
    if not isinstance(obj.get("weak"), list):
        obj["weak"] = []
    obj["summary"] = obj.get("summary", "")
    obj["_raw_json"] = raw

    # --- prepare JD tokens for strict matching (domain-agnostic) ---
    token_re = re.compile(r"[A-Za-z0-9#+.]+")
    jd_tokens = [t.lower() for t in token_re.findall(jd_text or "")]
    jd_compact = "".join(jd_tokens)
    jd_token_set = set(jd_tokens)

    def _tok_seq(s: str):
        return [t.lower() for t in token_re.findall(s or "")]

    def _sequential_match(seq_tokens: List[str]) -> bool:
        if not seq_tokens:
            return False
        L = len(seq_tokens)
        for i in range(0, len(jd_tokens) - L + 1):
            if jd_tokens[i:i+L] == seq_tokens:
                return True
        return False

    def _all_tokens_present(seq_tokens: List[str]) -> bool:
        if not seq_tokens:
            return False
        return all(tok in jd_token_set for tok in seq_tokens)

    def _compact_match(seq_tokens: List[str]) -> bool:
        if not seq_tokens:
            return False
        return "".join(seq_tokens) in jd_compact

    def _present_in_jd_strict(seq_tokens: List[str]) -> bool:
        """
        Strict presence rules (domain-agnostic):
        - single token: sequential OR compact match
        - multi-token: sequential OR all tokens present OR compact match
        """
        if not seq_tokens:
            return False
        if len(seq_tokens) == 1:
            return _sequential_match(seq_tokens) or _compact_match(seq_tokens)
        # multi-word
        if _sequential_match(seq_tokens):
            return True
        if _all_tokens_present(seq_tokens):
            return True
        if _compact_match(seq_tokens):
            return True
        return False

    # 2) Post-filter LLM keywords to ensure they actually appear in the JD (strict)
    filtered_keywords = []
    filtered_out = []

    for kw in (obj.get("keywords") or []):
        term = (kw.get("term") or "").strip()
        variants = [v for v in (kw.get("variants") or []) if v and v.strip()]

        kept = False

        # Check term itself (strict)
        if term:
            seq = _tok_seq(term)
            if _present_in_jd_strict(seq):
                kept = True

        # Check variants (strict)
        if not kept:
            for v in variants:
                seqv = _tok_seq(v)
                if _present_in_jd_strict(seqv):
                    kept = True
                    break

        if kept:
            safe_kw = {
                "rank": kw.get("rank"),
                "term": term,
                "category": kw.get("category", ""),
                "variants": variants
            }
            # preserve evidence if LLM provided it
            if isinstance(kw.get("evidence"), list):
                safe_kw["evidence"] = kw.get("evidence")
            filtered_keywords.append(safe_kw)
        else:
            filtered_out.append({"term": term, "variants": variants, "reason": "not_in_jd"})

    obj["keywords"] = filtered_keywords
    obj["_filtered_out"] = filtered_out

    # 3) FALLBACK: if no keywords after filtering, run a deterministic JD line-by-line extractor
    if not obj["keywords"]:
        jd_lines = [ln.rstrip() for ln in (jd_text or "").splitlines() if ln.strip()]
        fallback_keywords = []
        seen = set()
        rank = 1

        # Helper: tech-like candidate checks and normalization
        token_re_local = re.compile(r"[A-Za-z0-9\+#\-/\.]+")
        stop_verbs = re.compile(r"\b(design|implement|maintain|develop|manage|ensure|work|automate|analyze|write|partner|integrate|provide|perform|configure)\b", re.I)
        noise_words = {"responsibilities", "requirements", "experience", "responsibilit", "development", "management"}

        def is_tech_candidate(s: str) -> bool:
            s = s.strip().strip(",:;.-()[]")
            if not s or len(s) <= 1:
                return False
            # reject obviously long prose (>7 words)
            if len(s.split()) > 7:
                return False
            # reject pure verbs/prose lines
            if stop_verbs.search(s):
                # allow if the string contains clear tech tokens like '/' (RDS/DynamoDB) or dots or '+' or capitalized acronym
                if not re.search(r"[/#\.+]|[A-Z]{2,}", s):
                    return False
            # must contain at least one alpha/numeric token of length >=2
            toks = token_re_local.findall(s)
            if not toks:
                return False
            if all(len(t) <= 1 for t in toks):
                return False
            # filter out generic headings/noise
            low = s.lower()
            if any(w in low for w in noise_words):
                return False
            return True

        # Preferred extraction strategy:
        # 1) split on common separators (commas, 'such as', 'e.g.', 'or', ';')
        # 2) also capture slash tokens (RDS/DynamoDB), parentheses content, and short phrases (<=4 tokens)
        sep_re = re.compile(r",|\band\b|\bor\b|\bsuch as\b|\be\.g\.\b|;|\u2022", re.I)
        slash_or_paren_re = re.compile(r"[A-Za-z0-9\+#\-/\.]{2,}(?:/[A-Za-z0-9\+#\-/\.]{2,})?")

        for line in jd_lines:
            # 1. extract parenthetical content first (e.g., "(Jenkins, Bitbucket, Git)")
            for par in re.findall(r"\(([^)]+)\)", line):
                for part in sep_re.split(par):
                    cand = part.strip()
                    if not cand:
                        continue
                    if not is_tech_candidate(cand):
                        continue
                    key = cand.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    fallback_keywords.append({"rank": rank, "term": cand, "category": "", "variants": [], "evidence": [line]})
                    rank += 1
                    if rank > 50:
                        break
                if rank > 50:
                    break
            if rank > 50:
                break

            # 2. split line on separators to get short candidate fragments
            parts = [p.strip() for p in sep_re.split(line) if p.strip()]
            for part in parts:
                # further split long fragments by "such as" or "e.g." already handled; check slash tokens
                # capture explicit slash tokens
                for m in slash_or_paren_re.finditer(part):
                    cand = m.group(0).strip().strip(",:;.-")
                    if not cand or len(cand) <= 1:
                        continue
                    if not is_tech_candidate(cand):
                        continue
                    key = cand.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    fallback_keywords.append({"rank": rank, "term": cand, "category": "", "variants": [], "evidence": [line]})
                    rank += 1
                    if rank > 50:
                        break
                if rank > 50:
                    break

                # If no slash-like tokens, consider short phrase fragments (<=4 tokens)
                words = [w for w in token_re_local.findall(part)]
                if 0 < len(words) <= 4:
                    cand = " ".join(words)
                    if not cand:
                        continue
                    if not is_tech_candidate(cand):
                        continue
                    key = cand.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    fallback_keywords.append({"rank": rank, "term": cand, "category": "", "variants": [], "evidence": [line]})
                    rank += 1
                    if rank > 50:
                        break
            if rank > 50:
                break

        # Final cleanup: prefer shorter, clearly-technical tokens and limit results
        cleaned = []
        seen_c = set()
        for ent in fallback_keywords:
            t = ent["term"].strip()
            # prefer tokens with letters/digits and reasonable length
            if len(t) < 2 or len(t) > 70:
                continue
            # avoid pure stop-verb fragments
            if stop_verbs.fullmatch(t):
                continue
            k = t.lower()
            if k in seen_c:
                continue
            seen_c.add(k)
            cleaned.append({"rank": len(cleaned) + 1, "term": t, "category": "", "variants": [], "evidence": ent.get("evidence", [])})
            if len(cleaned) >= 25:
                break

        if cleaned:
            obj["keywords"] = cleaned
            obj["missing"] = []
            obj["weak"] = []
            obj["summary"] = "Fallback deterministic JD line-by-line extraction used to derive explicit JD tokens."
            obj["_raw_json"] = raw
            obj["_filtered_out"] = obj.get("_filtered_out", [])

    # Keep compatibility enforcement (original behavior)
    obj = enforce_jd_keywords(obj, jd_text, resume_text)

    # Ensure lists are present
    if not isinstance(obj.get("keywords"), list):
        obj["keywords"] = []
    if not isinstance(obj.get("missing"), list):
        obj["missing"] = []
    if not isinstance(obj.get("weak"), list):
        obj["weak"] = []

    return obj




def enforce_jd_keywords(obj, jd_text: str, resume_text: str):
    """
    Compute true Gaps = JD/Top Keywords not present in resume
    """
    jd_terms = list(set(re.findall(r"\b[A-Z][A-Za-z0-9\+\-_/]{2,}\b", jd_text)))

    all_kw = []
    for item in obj.get("keywords", []):
        if item.get("term"):
            all_kw.append(item["term"])
        all_kw.extend(item.get("variants", []))

    def _tok_seq(s: str): return [t.lower() for t in re.findall(r"[A-Za-z0-9#+.]+", s or "")]
    resume_tokens = _tok_seq(resume_text or "")
    resume_compact = "".join(resume_tokens)

    def _present(term: str) -> bool:
        kt = _tok_seq(term)
        if not kt: return False
        L = len(kt)
        for i in range(0, len(resume_tokens) - L + 1):
            if resume_tokens[i:i+L] == kt:
                return True
        if "".join(kt) in resume_compact:
            return True
        if L == 1 and len(kt[0]) > 3:
            base = kt[0]; alt = base[:-1] if base.endswith("s") else base + "s"
            if base in resume_tokens or alt in resume_tokens:
                return True
        return False


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
# Keyword Sentences -> now returns structured Technical Skills (heading -> list)
# -----------------------------
def generate_keyword_sentences(resume_text: str, jd_text: str, target_keywords: List[str],
                               provider_pref: Optional[str], model_name: Optional[str],
                               temperature: float, max_tokens: int, keys: Dict[str, str]) -> str:
    """
    Generate a grouped Technical Skills block for insertion into the resume.

    Behavior (non-domain-specific):
    - Preserve resume headings & order; resume content is source-of-truth.
    - Merge LLM-provided headings/items but do not invent domain-specific headings.
    - Place new keywords in an existing heading if a simple heading-token overlap exists,
      otherwise append to fallback "Technical Skills".
    - Remove duplicates and avoid injecting long prose as skill items.
    - Return plain text block starting with "Technical Skills" then "Heading: item1, item2" lines.
    """
    provider = _provider_from_keys(provider_pref, keys or {})

    # 1) Call LLM (leave this behavior unchanged)
    kws_blob = "\n".join(f"- {k}" for k in (target_keywords or []))
    user_prompt = USER_KEYWORD_SENTENCES.format(jd=(jd_text or ""), resume=(resume_text or ""), keywords=kws_blob)
    raw = provider.chat(model=model_name, system=SYSTEM_KEYWORD_SENTENCES, user=user_prompt, temperature=temperature, max_tokens=max_tokens or 600)
    raw = (raw or "").strip().strip("`").strip()

    # 2) Try to parse JSON output; otherwise parse heuristically
    skills_obj = {"skills": {}}
    def _try_parse_json(s: str):
        try:
            start = s.find("{"); end = s.rfind("}") + 1
            if start >= 0 and end > start:
                obj = json.loads(s[start:end])
                if isinstance(obj, dict) and "skills" in obj and isinstance(obj["skills"], dict):
                    return obj
        except Exception:
            pass
        return None

    parsed = _try_parse_json(raw)
    if parsed:
        skills_obj = {"skills": {}}
        for h, arr in (parsed.get("skills") or {}).items():
            items = []
            if isinstance(arr, list):
                for v in arr:
                    if isinstance(v, str) and v.strip():
                        items.append(v.strip())
                    else:
                        items.append(str(v).strip())
            elif isinstance(arr, str):
                items = [p.strip() for p in re.split(r",|\u2022", arr) if p.strip()]
            else:
                items = [str(arr).strip()]
            if items:
                skills_obj["skills"][h.strip()] = items
    else:
        # fallback: parse "Heading: a, b" lines or bullets
        skills_obj = {"skills": {}}
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ":" in ln:
                left, right = ln.split(":", 1)
                heading = left.strip()
                items = [p.strip() for p in re.split(r",|\u2022", right) if p.strip()]
                if items:
                    skills_obj["skills"].setdefault(heading, []).extend(items)
            else:
                toks = [t.strip() for t in re.split(r",|\u2022|\t", ln) if t.strip()]
                if toks:
                    skills_obj["skills"].setdefault("Technical Skills", []).extend(toks)

    # Helpers: detect short, skill-like fragments and extract tokens from resume lines
    def _is_short_skill(s: str) -> bool:
        if not s or not s.strip():
            return False
        s = s.strip()
        # drop markers only
        if re.fullmatch(r"[-•▪‣·\s]+", s):
            return False
        words = s.split()
        # drop long prose (> 8 words)
        if len(words) > 8:
            return False
        # drop obvious role/prose lines
        if re.search(r"\b(role|responsib|project|experience|since|from|to|with|present|manager|engineer|joined|company)\b", s, flags=re.I):
            return False
        return True

    def _extract_skill_tokens_from_line(line: str) -> List[str]:
        if not line or not line.strip():
            return []
        s = line.strip()
        s = re.sub(r"^[-•▪‣·*]\s*", "", s).strip()
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return [p for p in parts if _is_short_skill(p)]
        if _is_short_skill(s):
            return [s]
        return []

    # 3) Parse resume to find existing skill headings and items (preserve order)
    def _find_resume_skill_sections(text: str):
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

    resume_sections = _find_resume_skill_sections(resume_text)

    merged_headings = []
    merged_skills = {}

    for heading, block_lines in resume_sections:
        items = []
        for ln in block_lines:
            toks = _extract_skill_tokens_from_line(ln)
            for t in toks:
                if t and t.strip():
                    items.append(t.strip())
        seen_local = set(); final_items = []
        for it in items:
            kl = it.lower()
            if kl in seen_local:
                continue
            seen_local.add(kl); final_items.append(it)
        if final_items:
            merged_headings.append(heading)
            merged_skills[heading] = final_items

    # 4) Merge LLM-provided headings/items without domain inference
    for h, arr in (skills_obj.get("skills") or {}).items():
        if not arr:
            continue
        items_short = [i.strip() for i in arr if isinstance(i, str) and _is_short_skill(i.strip())]
        if not items_short:
            continue
        if h in merged_skills:
            exist = {x.lower() for x in merged_skills[h]}
            for it in items_short:
                if it.lower() not in exist:
                    merged_skills[h].append(it); exist.add(it.lower())
        else:
            # avoid adding headings that obviously look like prose sections
            if re.match(r"(?i)^(work\s*experience|experience|education|projects|requirements|role|responsibilit)s?", h):
                continue
            merged_headings.append(h)
            merged_skills[h] = []
            seen_h = set()
            for it in items_short:
                if it.lower() not in seen_h:
                    merged_skills[h].append(it); seen_h.add(it.lower())

    # 5) Build a simple JD heading map (short fragments only) - used only to help placement, not to infer domains
    jd_lines = [ln.strip() for ln in (jd_text or "").splitlines() if ln.strip()]
    jd_heading_map = {}
    for i, ln in enumerate(jd_lines):
        s = ln.strip()
        if re.match(r"^[A-Za-z0-9 \-]{1,80}\s*:$", s):
            h = s.rstrip(":").strip()
            start = i + 1
            end = len(jd_lines)
            for j in range(start, len(jd_lines)):
                nxt = jd_lines[j].strip()
                if not nxt or re.match(r"^[A-Za-z0-9 \-]{1,80}\s*:$", nxt):
                    end = j
                    break
            block = [ln2.strip() for ln2 in jd_lines[start:end] if ln2.strip()]
            tokens = []
            for bl in block:
                parts = [p.strip() for p in re.split(r",|\u2022", bl) if p.strip()]
                for p in parts:
                    if _is_short_skill(p):
                        tokens.append(p)
            if tokens:
                jd_heading_map[h] = tokens

    # 6) Insert target_keywords preserving resume order; append to matching heading or fallback
    fallback = "Technical Skills"
    global_seen = set()
    for h in merged_headings:
        for it in merged_skills.get(h, []):
            global_seen.add(it.lower())

    # process each target keyword in order
    for kw in (target_keywords or []):
        if not kw or not kw.strip():
            continue
        kws = kw.strip()
        kl = kws.lower()
        if kl in global_seen:
            continue
        placed = False
        # 6a) place under an existing resume heading if simple token overlap
        for heading in merged_headings:
            h_low = heading.lower()
            h_tokens = re.findall(r"[A-Za-z0-9]+", h_low)
            k_tokens = re.findall(r"[A-Za-z0-9]+", kl)
            if any(ht in kt or kt in ht for ht in h_tokens for kt in k_tokens):
                merged_skills.setdefault(heading, []).append(kws)
                global_seen.add(kl)
                placed = True
                break
        if placed:
            continue
        # 6b) place under JD heading if exact short fragment match
        for jh, tokens in jd_heading_map.items():
            if any(kws.lower() == t.lower() for t in tokens):
                if jh in merged_skills:
                    merged_skills[jh].append(kws)
                else:
                    merged_headings.append(jh)
                    merged_skills[jh] = [kws]
                global_seen.add(kl)
                placed = True
                break
        if placed:
            continue
        # 6c) fallback - append to Technical Skills at the end (ensure fallback exists last)
        if fallback not in merged_skills:
            merged_headings.append(fallback)
            merged_skills[fallback] = []
        merged_skills[fallback].append(kws)
        global_seen.add(kl)

    # 7) Final cleanup: dedupe each heading preserving order and remove non-short items
    out_lines = []
    for heading in merged_headings:
        items = merged_skills.get(heading, []) or []
        cleaned = []
        seen_local = set()
        for it in items:
            if not isinstance(it, str):
                it = str(it)
            it_s = it.strip()
            if not it_s:
                continue
            if not _is_short_skill(it_s):
                # allow if exact match present in JD heading tokens (rare)
                if not any(it_s.lower() == t.lower() for vs in jd_heading_map.values() for t in vs):
                    continue
            key = it_s.lower()
            if key in seen_local:
                continue
            seen_local.add(key)
            cleaned.append(it_s)
        if cleaned:
            out_lines.append(f"{heading}: {', '.join(cleaned)}")

    if not out_lines:
        # fallback output if nothing found
        filtered_targets = [k.strip() for k in (target_keywords or []) if _is_short_skill(k)]
        if filtered_targets:
            out_lines = [f"Technical Skills: {', '.join(filtered_targets)}"]
        else:
            out_lines = ["Technical Skills:"]

    # Prepend the UI title line "Technical Skills" as before
    return "\n".join(["Technical Skills"] + out_lines).strip()


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
# Technical Skills insertion utility
# -----------------------------
def _remove_core_competencies_section(text: str) -> str:
    """
    Remove the first Core Competencies section (heading + body) if present.
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
    nxt = re.search(r"(?im)^\s*(skills|technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$", after)
    block_end = end + (nxt.start() if nxt else len(after))
    return (text[:start] + text[block_end:]).strip()


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
            items = [p.strip() for p in re.split(r",|\u2022", right) if p.strip()]
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

def insert_technical_skills(full_text: str, skills_block: str) -> str:
    """
    Replace or insert Technical Skills while preserving existing technical skills exactly.
    - Remove Core Competencies (kept behavior)
    - If existing Technical Skills present, replace its body with merged block that preserves
      original items and appends new ones (no reordering of original items).
    - If no Technical Skills heading, insert near Summary or top preserving resume structure.
    """
    if not full_text:
        return full_text
    text = full_text

    # remove core competencies only (preserve existing skills)
    text = _remove_core_competencies_section(text)

    # parse incoming skills block into headings->items
    parsed_new = _parse_block_to_headings(skills_block or "")

    # if incoming block is empty, just return text (no destructive change)
    if not parsed_new:
        return text

    # merge with resume existing skill headings preserving original items & order
    merged = _merge_preserve_resume(text, parsed_new)
    block = _render_skills_block(merged)

    # replace existing Technical Skills if present
    pattern = re.compile(r"(?im)^\s*technical\s*skills\s*[:\-–—]?\s*$")
    m = pattern.search(text or "")
    if m:
        start = m.start()
        end = m.end()
        after = text[end:]
        nxt = re.search(r"(?im)^\s*(work\s*experience|experience|education|projects|certifications|awards|publications|profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$", after)
        section_end = end + (nxt.start() if nxt else len(after))
        head = text[:end].rstrip()
        tail = text[section_end:].lstrip("\n")
        new_text = (head + "\n" + block + "\n\n" + tail).strip()
        return new_text
    else:
        # If no Technical Skills heading, try to insert after Summary or after Contact block (first non-empty line)
        summary_heading_re = re.compile(r"(?im)^\s*(profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$")
        m2 = summary_heading_re.search(text)
        if m2:
            head_end = m2.end()
            after = text[head_end:]
            nxt = re.search(r"(?im)^\s*(work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$", after)
            insert_pos = head_end + (nxt.start() if nxt else len(after))
            new_text = text[:insert_pos].rstrip() + "\n\n" + block + "\n\n" + text[insert_pos:].lstrip()
            return new_text
        else:
            # fallback: insert near the top after first non-empty line
            parts = text.splitlines()
            idx = 0
            while idx < len(parts) and not parts[idx].strip():
                idx += 1
            insert_at = min(len(parts), idx + 1)
            new_lines = parts[:insert_at] + ["", block, ""] + parts[insert_at:]
            return "\n".join(new_lines).strip()

def insert_technical_skills_after_summary(full_text: str, skills_block: str,
                                         jd_core_competencies: Optional[List[str]] = None) -> str:
    """
    Ensure exactly one Technical Skills block: remove Core Competencies and any existing
    Technical Skills sections, then insert the supplied skills_block immediately after
    the Profile Summary (or after name/contacts if no summary).

    Behavior tweak:
    - If jd_core_competencies is provided and non-empty:
        * Preserve any existing resume Core Competencies and MERGE JD core competencies
          (JD items are inserted above Technical Skills under a 'Core Competencies' heading).
    - If jd_core_competencies is None or empty:
        * Remove any existing Core Competencies sections from the resume (do not preserve).
    """
    if not full_text:
        return full_text

    text = full_text

    # If JD did NOT provide core competencies, remove existing Core Competencies
    if not jd_core_competencies:
        text = _remove_core_competencies_section(text)
    # If JD provided core competencies, preserve existing resume Core Competencies (don't remove),
    # but we'll later ensure merging/insertion happens.

    # Always remove existing Technical Skills to avoid duplicates (we will re-insert)
    text = _remove_technical_skills_section(text)

    # Normalize skills_block into lines and skip if empty
    lines = [ln.rstrip() for ln in (skills_block or "").splitlines() if ln.strip()]
    if not lines:
        # Nothing to insert; if JD provided core competencies, still insert them
        if jd_core_competencies:
            # prepare core competencies block from JD (merge with any existing resume ones)
            existing_cores = []
            # extract existing resume core lines (if any)
            # simple parse: find existing heading earlier (if present)
            # reuse _find_resume_core_lines logic-like approach inline
            res_lines = (text or "").splitlines()
            for i, ln in enumerate(res_lines):
                if re.match(r"(?i)^\s*(core\s*competenc(?:ies|y)|core\s*skills)\s*[:\-–—]?\s*$", ln.strip()):
                    # collect subsequent non-empty lines
                    for j in range(i+1, len(res_lines)):
                        nxt = res_lines[j].strip()
                        if not nxt:
                            break
                        if re.match(r"(?im)^\s*(technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$", nxt):
                            break
                        parts = [p.strip() for p in re.split(r",|\u2022|;|\t", nxt) if p.strip()]
                        if parts:
                            for p in parts:
                                if p not in existing_cores:
                                    existing_cores.append(p)
                        else:
                            if nxt not in existing_cores:
                                existing_cores.append(nxt)
                    break
            # merge JD cores (only add JD items not already present)
            merged = list(existing_cores)
            for jd_item in jd_core_competencies:
                if jd_item not in merged:
                    merged.append(jd_item)
            # build block and insert after summary (reuse insertion logic below)
            skills_block_to_insert = "Core Competencies\n" + "\n".join(merged)
        else:
            return text
    else:
        # If lines are present, produce block that includes Core Competencies if provided
        if lines[0].strip().lower().startswith("technical"):
            block_body_lines = lines[1:] if lines[0].strip().lower().startswith("technical") else lines
        else:
            block_body_lines = lines

        # Build core competencies block if JD has them
        if jd_core_competencies:
            # Find existing resume core entries to merge
            existing_cores = []
            res_lines = (text or "").splitlines()
            for i, ln in enumerate(res_lines):
                if re.match(r"(?i)^\s*(core\s*competenc(?:ies|y)|core\s*skills)\s*[:\-–—]?\s*$", ln.strip()):
                    for j in range(i+1, len(res_lines)):
                        nxt = res_lines[j].strip()
                        if not nxt:
                            break
                        if re.match(r"(?im)^\s*(technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$", nxt):
                            break
                        parts = [p.strip() for p in re.split(r",|\u2022|;|\t", nxt) if p.strip()]
                        if parts:
                            for p in parts:
                                if p not in existing_cores:
                                    existing_cores.append(p)
                        else:
                            if nxt not in existing_cores:
                                existing_cores.append(nxt)
                    break
            merged_cores = list(existing_cores)
            for jd_item in jd_core_competencies:
                if jd_item not in merged_cores:
                    merged_cores.append(jd_item)
            # assemble final block: Core Competencies (if any), then Technical Skills lines
            block_lines = []
            if merged_cores:
                block_lines.append("Core Competencies: " + ", ".join(merged_cores))
            # Append the rest (normalize headings -> Technical Skills below will be added)
            # If the LLM supplied headings in skills_block, keep them as-is
            block_lines.extend(block_body_lines)
            skills_block_to_insert = "\n".join(block_lines)
        else:
            # No JD cores: straightforward block is just skills_block (wrapped under Technical Skills later)
            skills_block_to_insert = "\n".join(lines)

    # Ensure the inserted block is prefixed with "Technical Skills" if not already
    if not skills_block_to_insert.strip().lower().startswith("technical") and not skills_block_to_insert.strip().lower().startswith("core"):
        skills_block_to_insert = "Technical Skills\n" + skills_block_to_insert

    # Insert after Profile Summary if present (same as prior behavior)
    summary_heading_re = re.compile(r"(?im)^\s*(profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$")
    m = summary_heading_re.search(text)
    if m:
        head_end = m.end()
        after = text[head_end:]
        nxt = re.search(r"(?im)^\s*(technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$", after)
        insert_pos = head_end + (nxt.start() if nxt else len(after))
        new_text = text[:insert_pos].rstrip() + "\n\n" + skills_block_to_insert + "\n\n" + text[insert_pos:].lstrip()
        return new_text

    # fallback: insert after first non-empty line (name/contacts)
    parts = text.splitlines()
    idx = 0
    while idx < len(parts) and not parts[idx].strip():
        idx += 1
    insert_at = min(len(parts), idx + 1)
    new_lines = parts[:insert_at] + ["", skills_block_to_insert, ""] + parts[insert_at:]
    return "\n".join(new_lines).strip()


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