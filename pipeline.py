from __future__ import annotations
from typing import Dict, Any, Optional, List
import re, json
from ai.prompts import SYSTEM_SUMMARY_BULLETS, USER_SUMMARY_BULLETS
from ai.selector import get_provider
from ai.matcher import match_score, keyword_gaps
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
def extract_contacts_regex(text: str) -> Dict[str,str]:
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    linkedin = LINKEDIN_RE.search(text)
    github = GITHUB_RE.search(text)
    name = ""
    for line in text.splitlines()[:15]:
        s = line.strip()
        if not s or len(s) > 60:
            continue
        if any(ch.isdigit() for ch in s) or any(x in s for x in ("@", "http", "|", "/", "\\", "•", " - ", ",", "(", ")", ":" )):
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

def extract_contacts_llm(resume_text: str, provider_pref: Optional[str], model_name: Optional[str], keys: Dict[str,str]) -> Dict[str,str]:
    provider = _provider_from_keys(provider_pref, keys)
    raw = provider.chat(model=model_name, system=SYSTEM_CONTACTS, user=USER_CONTACTS.format(resume=resume_text), temperature=0, max_tokens=256)
    raw = (raw or "").strip().strip('`').strip()
    try:
        start = raw.find('{'); end = raw.rfind('}') + 1
        obj = json.loads(raw[start:end])
        out = {k: (obj.get(k) or "") for k in ["name","email","phone","linkedin","github"]}
        return out
    except Exception:
        return extract_contacts_regex(resume_text)

# -----------------------------
# Sanitizer
# -----------------------------
def sanitize_markdown(md: str) -> str:
    """
    Post-format the model output so it looks clean & professional:
    - collapse >2 blank lines to a single blank line
    - ensure a blank line after known section headings
    - normalize bullet markers to '• ' where a bullet already exists
    - trim trailing/leading spaces; remove stray markdown markers
    """
    import re

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
    out = []
    prev_blank = True

    def is_heading(s: str) -> bool:
        t = s.strip().lower()
        return t in KNOWN_HEADINGS

    for raw in lines:
        s = raw.strip()

        # normalize bullets
        if re.match(r"^[-•▪‣·*]\s+", s):
            s = "• " + re.sub(r"^[-•▪‣·*]\s+", "", s)

        out.append(s)

        if is_heading(s):
            out.append("")  # force single blank line after heading
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

    txt = "\n".join(out)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

# -----------------------------
# Provider selector
# -----------------------------
def _provider_from_keys(provider_preference: Optional[str], keys: Dict[str,str]):
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
    return {"flesch_reading_ease": round(fre,1), "sentences": sentences, "words": words_count}

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
    report = {}
    report["ats"] = ats_checks(resume_text)
    report["readability"] = readability(resume_text)
    report["match"] = match_score(resume_text, jd_text)
    report["keywords_bow"] = keyword_gaps(resume_text, jd_text, top_k=30)
    report["contacts"] = extract_contacts_regex(resume_text)
    return report

# -----------------------------
# Keywords (LLM)
# -----------------------------
def extract_keywords_llm(resume_text: str, jd_text: str,
                         provider_pref: Optional[str], model_name: Optional[str],
                         temperature: float, max_tokens: int, keys: Dict[str,str]) -> Dict[str, Any]:
    provider = _provider_from_keys(provider_pref, keys)
    raw = provider.chat(
        model=model_name,
        system=SYSTEM_KEYWORDS,
        user=USER_KEYWORDS.format(jd=jd_text, resume=resume_text),
        temperature=temperature,
        max_tokens=max_tokens
    )
    raw = (raw or "").strip().strip('`').strip()
    try:
        start = raw.find('{'); end = raw.rfind('}') + 1
        obj = json.loads(raw[start:end])
    except Exception:
        obj = {"keywords": [], "missing": [], "weak": [], "summary": ""}
    return obj

# -----------------------------
# Keyword sentences (Core Competencies only)
# -----------------------------
def generate_keyword_sentences(resume_text: str, jd_text: str, target_keywords: List[str],
                               provider_pref: Optional[str], model_name: Optional[str],
                               temperature: float, max_tokens: int, keys: Dict[str,str]) -> str:
    """
    Ask the LLM to produce ATS-friendly Core Competencies bullets for the provided keywords.
    Returns plain text (one '• ' bullet per line).
    """
    provider = _provider_from_keys(provider_pref, keys or {})
    kw_blob = "\n".join(f"- {k}" for k in (target_keywords or []))
    resp = provider.chat(
        model=model_name,
        system=SYSTEM_KEYWORD_SENTENCES,
        user=USER_KEYWORD_SENTENCES.format(jd=jd_text, resume=resume_text, keywords=kw_blob),
        temperature=temperature,
        max_tokens=max_tokens
    )
    return sanitize_markdown(resp or "").strip()

# -----------------------------
# Polish keyword sentences
# -----------------------------
def polish_keyword_sentences(resume_text: str, bullets_text: str, jd_text: str,
                             provider_pref: Optional[str], model_name: Optional[str],
                             temperature: float, max_tokens: int, keys: Dict[str,str]) -> str:
    """
    Rewrites generated keyword bullets to be sharper, resume-native, and ATS-friendly.
    Keeps colon style and one-per-line format.
    """
    from ai.prompts import SYSTEM_KEYWORD_SENTENCES_POLISH, USER_KEYWORD_SENTENCES_POLISH
    provider = _provider_from_keys(provider_pref, keys or {})
    user = USER_KEYWORD_SENTENCES_POLISH.format(
        resume=resume_text or "",
        bullets=bullets_text or "",
        jd=jd_text or ""
    )
    out = provider.chat(
        model=model_name,
        system=SYSTEM_KEYWORD_SENTENCES_POLISH,
        user=user,
        temperature=temperature,
        max_tokens=min(max_tokens, 800)
    )
    return sanitize_markdown(out or "")

# -----------------------------
# Summary bullets (Profile/Professional)
# -----------------------------
def generate_summary_bullets(
    resume_text: str,
    jd_text: str,
    focus: str,
    provider_pref: Optional[str],
    model_name: Optional[str],
    temperature: float,
    max_tokens: int,
    keys: Dict[str, str],
) -> str:
    """
    Generate a professional summary as bullet points (• …) using only resume facts,
    aligned to the JD. No hard cap here; model should avoid redundancy.
    """
    provider = _provider_from_keys(provider_pref, keys or {})
    raw = provider.chat(
        model=model_name,
        system=SYSTEM_SUMMARY_BULLETS,
        user=USER_SUMMARY_BULLETS.format(jd=jd_text, resume=resume_text, focus=(focus or "")),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return sanitize_markdown(raw or "").strip()

# --- Summary bulletizer (preserve meaning & order) ---
def bulletize_summary_preserve_meaning(
    summary_text: str,
    provider_pref: Optional[str],
    model_name: Optional[str],
    temperature: float,
    max_tokens: int,
    keys: Dict[str, str],
) -> str:
    """
    Re-emit the ORIGINAL Summary as bullets:
    - preserves content & order (no new facts)
    - each line starts with '• '
    - plain text only
    """
    summary_text = (summary_text or "").strip()
    if not summary_text:
        return ""

    provider = _provider_from_keys(provider_pref, keys or {})
    raw = provider.chat(
        model=model_name,
        system=SYSTEM_SUMMARY_BULLETS,                       # <-- use existing prompts
        user=USER_SUMMARY_BULLETS.format(
            jd="",                                          # we don't need JD here; we are preserving content
            resume=summary_text,                            # pass ONLY the extracted summary body
            focus=""                                        # keep empty; template tolerates it
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return sanitize_markdown(raw or "").strip()



# -----------------------------
# Render JSON -> text (used by tailoring JSON path)
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

    if obj.get("core_skills"):
        section("Core Skills")
        for s in obj["core_skills"]: lines.append("• " + s)
        lines.append("")

    if obj.get("core_competencies"):
        section("Core Competencies")
        for s in obj["core_competencies"]: lines.append("• " + s)
        lines.append("")

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

# -----------------------------
# Tailor (JSON-first, fallback to text)
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

    raw = provider.chat(
        model=model_name,
        system=SYSTEM_TAILOR_JSON,
        user=USER_TAILOR_JSON.format(jd=jd_text, resume=resume_text, contact=contact_block, keywords=kw_blob),
        temperature=temperature,
        max_tokens=max_tokens
    ).strip()

    try:
        start = raw.find("{"); end = raw.rfind("}") + 1
        obj = json.loads(raw[start:end])
        txt = render_text_from_json(obj)
        if txt: return txt
    except Exception:
        pass

    out = provider.chat(
        model=model_name,
        system=SYSTEM_TAILOR,
        user=USER_TAILOR.format(jd=jd_text, resume=resume_text, keywords=kw_blob),
        temperature=temperature,
        max_tokens=max_tokens
    )
    return sanitize_markdown(out)

def replace_core_competencies(full_text: str, new_bullets: str) -> str:
    """
    Replace the 'Core Competencies' section body with provided bullets.
    If section doesn't exist, append it before Technical Skills or at the end.
    """
    new_bullets = (new_bullets or "").strip()
    if not new_bullets:
        return full_text

    # Normalize bullets
    lines = []
    for ln in new_bullets.splitlines():
        s = ln.strip()
        if not s:
            continue
        if not s.startswith("• "):
            s = "• " + s.lstrip("-• ").strip()
        lines.append(s)
    block = "\n".join(lines)

    # Find Core Competencies section
    pattern = re.compile(r"(?im)^(core\s*competencies)\s*[:\-–—]?\s*$")
    m = pattern.search(full_text or "")
    if m:
        head = full_text[:m.end()]
        after = full_text[m.end():]
        nxt = re.search(
            r"(?im)^\s*(skills|technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$",
            after
        )
        section_end = m.end() + (nxt.start() if nxt else len(after))
        tail = full_text[section_end:]
        return (head + "\n" + block + "\n\n" + tail).strip()
    else:
        # If not found, add before Technical Skills or at end
        insertion_point = re.search(r"(?im)^\s*(technical\s*skills)\s*[:\-–—]?\s*$", full_text or "")
        if insertion_point:
            idx = insertion_point.start()
            return full_text[:idx] + "\nCore Competencies\n" + block + "\n\n" + full_text[idx:]
        return full_text.rstrip() + "\n\nCore Competencies\n" + block

# -----------------------------
# AI ATS: read final resume + optimizer JSON
# -----------------------------
def extract_ats_llm_from_optimizer(
    resume_text: str,
    optimizer_obj: Dict[str, Any],
    provider_pref: Optional[str],
    model_name: Optional[str],
    temperature: float,
    max_tokens: int,
    keys: Dict[str, str],
    jd_text: Optional[str] = ""
) -> Dict[str, Any]:
    provider = _provider_from_keys(provider_pref, keys)

    # compact optimizer JSON to essentials (term + variants + rank + missing/weak/summary)
    kws = []
    for item in (optimizer_obj.get("keywords") or []):
        term = (item.get("term") or "").strip()
        if term:
            kws.append({
                "rank": item.get("rank", None),
                "term": term,
                "variants": [v for v in (item.get("variants") or []) if v]
            })
    payload = {
        "keywords": kws,
        "missing": optimizer_obj.get("missing") or [],
        "weak": optimizer_obj.get("weak") or [],
        "summary": optimizer_obj.get("summary") or ""
    }
    optimizer_json = json.dumps(payload, ensure_ascii=False)

    raw = provider.chat(
        model=model_name,
        system=SYSTEM_ATS,
        user=USER_ATS.format(resume=resume_text, optimizer_json=optimizer_json, jd=(jd_text or "")),
        temperature=0,
        max_tokens=min(max_tokens, 1200)
    )
    raw = (raw or "").strip().strip('`').strip()
    try:
        start = raw.find('{'); end = raw.rfind('}') + 1
        obj = json.loads(raw[start:end])
    except Exception:
        obj = {}

    # normalize
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
            fixed_sugg.append({
                "term": s.get("term", ""),
                "section": s.get("section", "Core Competencies"),
                "how": s.get("how", "")
            })
    obj["suggestions"] = fixed_sugg
    return obj

