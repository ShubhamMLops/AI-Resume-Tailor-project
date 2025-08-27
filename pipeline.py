from __future__ import annotations
from typing import Dict, Any, Optional, List
import re, json
from ai.selector import get_provider
from ai.matcher import match_score, keyword_gaps
from ai.prompts import (
    SYSTEM_TAILOR, USER_TAILOR, 
    SYSTEM_KEYWORDS, USER_KEYWORDS, 
    SYSTEM_SAMPLE_RESUME, USER_SAMPLE_RESUME,
    SYSTEM_TAILOR_JSON, USER_TAILOR_JSON,
    SYSTEM_CONTACTS, USER_CONTACTS
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?\b(?:\d[\s-]?){8,14}\b")
LINKEDIN_RE = re.compile(r"(https?://)?(www\.)?linkedin\.com/[^\s\)\]]+", re.I)
GITHUB_RE = re.compile(r"(https?://)?(www\.)?github\.com/[^\s\)\]]+", re.I)

def extract_contacts_regex(text: str) -> Dict[str,str]:
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    linkedin = LINKEDIN_RE.search(text)
    github = GITHUB_RE.search(text)
    # naive name: first simple text line
    name = ""
    for i, line in enumerate(text.splitlines()[:15]):
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

def sanitize_markdown(md: str) -> str:
    # We output plain text, but still clean stray markers & spacing
    md = md.replace("**", "")
    md = md.replace("\t", " ")
    md = re.sub(r"[ ]{3,}", "  ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()

def readability(text: str) -> Dict[str, float]:
    sentences = max(1, len(re.findall(r"[.!?]+", text)) )
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

def _provider_from_keys(provider_preference: Optional[str], keys: Dict[str,str]):
    provider = get_provider(provider_preference, keys)
    if not provider:
        raise RuntimeError("No API key provided in the app. Enter a key in the sidebar.")
    return provider

def extract_contacts_llm(resume_text: str, provider_pref: Optional[str], model_name: Optional[str], keys: Dict[str,str]) -> Dict[str,str]:
    provider = _provider_from_keys(provider_pref, keys)
    raw = provider.chat(model=model_name, system=SYSTEM_CONTACTS, user=USER_CONTACTS.format(resume=resume_text), temperature=0, max_tokens=256)
    raw = raw.strip().strip('`').strip()
    try:
        start = raw.find('{'); end = raw.rfind('}') + 1
        obj = json.loads(raw[start:end])
        out = {k: (obj.get(k) or "") for k in ["name","email","phone","linkedin","github"]}
        return out
    except Exception:
        return extract_contacts_regex(resume_text)

def extract_keywords_llm(resume_text: str, jd_text: str, provider_pref: Optional[str], model_name: Optional[str], temperature: float, max_tokens: int, keys: Dict[str,str]) -> Dict[str, Any]:
    provider = _provider_from_keys(provider_pref, keys)
    raw = provider.chat(model=model_name, system=SYSTEM_KEYWORDS, user=USER_KEYWORDS.format(jd=jd_text, resume=resume_text), temperature=temperature, max_tokens=max_tokens)
    raw = raw.strip().strip('`').strip()
    try:
        start = raw.find('{'); end = raw.rfind('}') + 1
        obj = json.loads(raw[start:end])
    except Exception:
        obj = {"keywords": [], "missing": [], "weak": [], "summary": ""}
    return obj

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

def tailor(resume_text: str, jd_text: str, provider_preference: str = None, model_name: str = None, temperature: float = 0.2, max_tokens: int = 1500, keys: Dict[str,str] = None, target_keywords: Optional[List[str]] = None, override_contacts: Optional[Dict[str,str]] = None) -> str:
    provider = _provider_from_keys(provider_preference, keys or {})

    # SAME tokenizer as extractor
    token_re = re.compile(r"[A-Za-z0-9#+.]+")
    def _canon(s: str) -> str:
        return " ".join(t.lower() for t in token_re.findall(s or ""))

    def _has_term(text: str, term: str) -> bool:
        tt = token_re.findall((text or "").lower())
        kt = token_re.findall((term or "").lower())
        if not kt:
            return False
        L = len(kt)
        for i in range(0, len(tt) - L + 1):
            if tt[i:i+L] == kt:
                return True
        return False

    # 1) Deduplicate incoming keywords (ranked + gaps) using same tokenizer
    seen = set()
    deduped = []
    for k in (target_keywords or []):
        c = _canon(k)
        if c and c not in seen:
            seen.add(c)
            deduped.append(k)

    # 2) FILTER OUT keywords already present in the ORIGINAL resume (A ∩ B)
    allowed = [k for k in deduped if not _has_term(resume_text, k)]

    kw_blob = "\n".join(f"- {k}" for k in allowed)

    contacts = override_contacts if override_contacts is not None else extract_contacts_regex(resume_text)
    contact_block = f"""name={contacts.get('name','')}
email={contacts.get('email','')}
phone={contacts.get('phone','')}
linkedin={contacts.get('linkedin','')}
github={contacts.get('github','')}"""

    # ---- Try structured JSON path first
    raw = provider.chat(
        model=model_name,
        system=SYSTEM_TAILOR_JSON,
        user=USER_TAILOR_JSON.format(jd=jd_text, resume=resume_text, contact=contact_block, keywords=kw_blob),
        temperature=temperature,
        max_tokens=max_tokens
    ).strip()

    txt = ""
    try:
        start = raw.find("{"); end = raw.rfind("}") + 1
        obj = json.loads(raw[start:end])
        txt = render_text_from_json(obj)
    except Exception:
        pass

    if not txt:
        out = provider.chat(
            model=model_name,
            system=SYSTEM_TAILOR,
            user=USER_TAILOR.format(jd=jd_text, resume=resume_text, keywords=kw_blob),
            temperature=temperature,
            max_tokens=max_tokens
        )
        txt = sanitize_markdown(out)

    # 3) GUARANTEE COVERAGE: ensure every 'allowed' keyword appears at least once
    missing_terms = [k for k in allowed if not _has_term(txt, k)]
    if missing_terms:
        lines = txt.splitlines()

        # Find "Core Competencies" section; else we'll create it before Technical Skills / Work Experience / end
        comp_idx = None
        for idx, ln in enumerate(lines):
            if ln.strip() == "Core Competencies":
                comp_idx = idx
                break

        # Build one-liner bullets for missing terms
        bullets = [f"• {k}: role-aligned capability as required by the job description." for k in missing_terms]

        if comp_idx is not None:
            # Insert bullets right after the "Core Competencies" heading, keeping spacing tidy
            insert_at = comp_idx + 1
            # Ensure one blank line after heading if needed
            if insert_at >= len(lines) or (lines[insert_at].strip() and not lines[insert_at].startswith("• ")):
                lines.insert(insert_at, "")
                insert_at += 1
            for b in bullets:
                lines.insert(insert_at, b)
                insert_at += 1
            # Add a trailing blank line if next line is non-blank and not a bullet
            if insert_at < len(lines) and lines[insert_at].strip() and not lines[insert_at].startswith("• "):
                lines.insert(insert_at, "")
        else:
            # Create section near the top (before common anchors) or at end
            anchors = {"Technical Skills","Work Experience","Education","Certifications","Projects"}
            anchor_idx = None
            for idx, ln in enumerate(lines):
                if ln.strip() in anchors:
                    anchor_idx = idx
                    break
            block = ["Core Competencies", *bullets, ""]
            if anchor_idx is not None:
                lines[anchor_idx:anchor_idx] = block
            else:
                if lines and lines[-1].strip():
                    lines.append("")  # ensure spacing
                lines.extend(block)

        txt = sanitize_markdown("\n".join(lines))

    return txt
    
    