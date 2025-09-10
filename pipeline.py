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

# -----------------------------
# Keywords (LLM)
# -----------------------------
def extract_keywords_llm(resume_text: str, jd_text: str,
                         provider_pref: Optional[str], model_name: Optional[str],
                         temperature: float, max_tokens: int, keys: Dict[str, str]) -> Dict[str, Any]:
    """
    Run LLM keyword extractor using SYSTEM_KEYWORDS, then strictly filter so ONLY
    terms/variants that actually appear in the JD remain.
    Domain-agnostic; no stoplist or domain-specific heuristics.
    Returns obj with 'keywords' filtered and '_filtered_out' listing removed entries.
    """
    provider = _provider_from_keys(provider_pref, keys)

    # If JD is empty, return a clean empty structure immediately
    if not (jd_text and jd_text.strip()):
        return {"keywords": [], "missing": [], "weak": [], "summary": "", "_raw_json": "", "_filtered_out": []}

    # call LLM
    raw = provider.chat(
        model=model_name,
        system=SYSTEM_KEYWORDS,
        user=USER_KEYWORDS.format(jd=jd_text, resume=resume_text),
        temperature=temperature,
        max_tokens=max_tokens
    )

    raw = (raw or "").strip().strip("`").strip()
    print("\n=== RAW LLM OUTPUT ===")
    print(raw)
    print("======================\n")

    try:
        start = raw.find("{"); end = raw.rfind("}") + 1
        obj = json.loads(raw[start:end])
    except Exception as e:
        obj = {"keywords": [], "missing": [], "weak": [], "summary": ""}
        obj["_parse_error"] = str(e)
        obj["_raw_json"] = raw

    # normalize schema
    if not isinstance(obj.get("keywords"), list):
        obj["keywords"] = []
    if not isinstance(obj.get("missing"), list):
        obj["missing"] = []
    if not isinstance(obj.get("weak"), list):
        obj["weak"] = []
    obj["summary"] = obj.get("summary", "")
    obj["_raw_json"] = raw

    # prepare JD tokens for strict matching (domain-agnostic)
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
            filtered_keywords.append(safe_kw)
        else:
            filtered_out.append({"term": term, "variants": variants, "reason": "not_in_jd"})

    # Replace keywords with filtered list; clear 'missing' so deterministic gaps function is used later.
    obj["keywords"] = filtered_keywords
    obj["missing"] = []
    obj["weak"] = obj.get("weak", []) if isinstance(obj.get("weak", []), list) else []
    obj["_filtered_out"] = filtered_out

    # enforce_jd_keywords kept for compatibility if you use it
    obj = enforce_jd_keywords(obj, jd_text, resume_text)

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


# -----------------------------
# Keyword Sentences -> now returns structured Technical Skills (heading -> list)
# -----------------------------
def generate_keyword_sentences(resume_text: str, jd_text: str, target_keywords: List[str],
                               provider_pref: Optional[str], model_name: Optional[str],
                               temperature: float, max_tokens: int, keys: Dict[str, str]) -> str:
    """
    Generate a grouped Technical Skills block for insertion into the resume.
    This function asks the LLM to return a JSON mapping: { "skills": { "Cloud Computing": ["AWS","EC2"], ... } }
    Then it converts that JSON into a human-readable block:

    Technical Skills
    Cloud Computing: AWS, EC2, S3
    Databases: MySQL, PostgreSQL
    ...

    We intentionally REMOVE any 'Core Competencies' output and instead structure everything under Technical Skills.
    """
    provider = _provider_from_keys(provider_pref, keys or {})

    # If no explicit target keywords passed, try to use gaps (caller may compute them), but we accept empty list.
    kws_blob = "\n".join(f"- {k}" for k in (target_keywords or []))

    # Ask the LLM for a JSON mapping of grouped skills
    user_prompt = USER_KEYWORD_SENTENCES.format(jd=(jd_text or ""), resume=(resume_text or ""), keywords=kws_blob)
    raw = provider.chat(model=model_name, system=SYSTEM_KEYWORD_SENTENCES, user=user_prompt, temperature=temperature, max_tokens=max_tokens or 600)
    raw = (raw or "").strip().strip("`").strip()
    # Try to extract JSON block
    skills_obj = {}
    try:
        start = raw.find('{'); end = raw.rfind('}') + 1
        if start >= 0 and end > start:
            skills_obj = json.loads(raw[start:end])
        else:
            # If LLM didn't return strict JSON, try to parse line-by-line headings: "Heading: a, b, c"
            skills_obj = {"skills": {}}
            for ln in raw.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                if ":" in ln:
                    h, vals = ln.split(":", 1)
                    items = [v.strip() for v in re.split(r",|\u2022", vals) if v.strip()]
                    skills_obj["skills"][h.strip()] = items
    except Exception:
        # fallback: try to parse free text into one heading
        try:
            # take everything as a single Technical Skills line
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if lines:
                skills_obj = {"skills": {"Technical Skills": []}}
                for ln in lines:
                    if ":" in ln:
                        h, vals = ln.split(":", 1)
                        items = [v.strip() for v in re.split(r",|\u2022", vals) if v.strip()]
                        skills_obj["skills"][h.strip()] = items
                    else:
                        # add individual tokens
                        tokens = [t.strip() for t in re.split(r",|\u2022|\s{2,}", ln) if t.strip()]
                        skills_obj["skills"]["Technical Skills"].extend(tokens)
        except Exception:
            skills_obj = {"skills": {}}

    # Normalize the skills object and build text block
    skills = skills_obj.get("skills") or {}
    # If empty, fallback to target_keywords grouped under "Technical Skills"
    if not skills:
        if target_keywords:
            skills = {"Technical Skills": list(dict.fromkeys(target_keywords))}
        else:
            skills = {}

    # Build a readable block
    out_lines = []
    out_lines.append("Technical Skills")
    for heading, items in skills.items():
        # skip any accidental 'Core Competencies' headings (we must remove core competencies)
        if heading.strip().lower().startswith("core"):
            continue
        # normalize items
        items_arr = []
        for it in items or []:
            if isinstance(it, str) and it.strip():
                items_arr.append(it.strip())
        if items_arr:
            out_lines.append(f"{heading}: {', '.join(items_arr)}")

    return "\n".join(out_lines).strip()

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

def insert_technical_skills(full_text: str, skills_block: str) -> str:
    """
    Ensure Core Competencies is removed and Technical Skills section is inserted or replaced.
    skills_block is a plain text block starting with "Technical Skills" followed by lines "Heading: item, item"
    """
    if not full_text:
        return full_text
    text = full_text

    # 1) Remove Core Competencies entirely
    text = _remove_core_competencies_section(text)

    # 2) Normalize skills_block
    lines = [ln.rstrip() for ln in (skills_block or "").splitlines() if ln.strip()]
    if not lines:
        # nothing to insert; simply remove core competencies and return
        return text

    # If the skills_block already starts with "Technical Skills", keep as-is, else try to wrap
    if lines[0].strip().lower().startswith("technical"):
        block = "\n".join(lines).strip()
    else:
        # wrap the whole block under Technical Skills heading
        block = "Technical Skills\n" + "\n".join(lines).strip()

    # 3) Replace existing Technical Skills if present
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
        # Insert after Summary if exists
        summary_heading_re = re.compile(r"(?im)^\s*(profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$")
        m2 = summary_heading_re.search(text)
        if m2:
            # find end of summary section
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


def insert_technical_skills_after_summary(full_text: str, skills_block: str) -> str:
    """
    Ensure exactly one Technical Skills block: remove Core Competencies and any existing
    Technical Skills sections, then insert the supplied skills_block immediately after
    the Profile Summary (or after name/contacts if no summary).
    """
    if not full_text:
        return full_text

    text = full_text

    # Remove Core Competencies and existing Technical Skills to avoid duplicates
    text = _remove_core_competencies_section(text)
    text = _remove_technical_skills_section(text)

    # Normalize skills_block into lines and skip if empty
    lines = [ln.rstrip() for ln in (skills_block or "").splitlines() if ln.strip()]
    if not lines:
        return text

    # If the user supplied a heading "Technical Skills" already, keep as-is; else add heading
    if lines[0].strip().lower().startswith("technical"):
        block = "\n".join(lines).strip()
    else:
        block = "Technical Skills\n" + "\n".join(lines).strip()

    # Find the Profile Summary heading location
    summary_heading_re = re.compile(r"(?im)^\s*(profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$")
    m = summary_heading_re.search(text)
    if m:
        # Insert after the existing summary block (end of its body)
        head_end = m.end()
        after = text[head_end:]
        nxt = re.search(r"(?im)^\s*(technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$", after)
        insert_pos = head_end + (nxt.start() if nxt else len(after))
        new_text = text[:insert_pos].rstrip() + "\n\n" + block + "\n\n" + text[insert_pos:].lstrip()
        return new_text

    # If no Profile Summary heading, insert after first non-empty line (name/contacts)
    parts = text.splitlines()
    idx = 0
    while idx < len(parts) and not parts[idx].strip():
        idx += 1
    insert_at = min(len(parts), idx + 1)
    new_lines = parts[:insert_at] + ["", block, ""] + parts[insert_at:]
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
