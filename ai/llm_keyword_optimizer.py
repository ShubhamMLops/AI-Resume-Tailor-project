# ai/llm_keyword_optimizer.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import json, re, math, os

from ai.selector import get_provider

# -------------------------
# Helpers
# -------------------------
def _first_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def _normalize_line(s: str) -> str:
    if not s:
        return ""
    s2 = s.strip().strip("`\"'")
    s2 = re.sub(r"\s+", " ", s2)
    return s2.strip()

def _norm_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", " ", (s or "").lower()).strip()

def _provider_from_pref(provider_pref: Optional[str], keys: Dict[str,str]):
    provider = get_provider(provider_pref, keys or {})
    if not provider:
        raise RuntimeError("No provider found. Add API key.")
    return provider

# -------------------------
# Deterministic candidate extractor from JD lines
# -------------------------
GENERIC_STOP = set([
    "responsibilities","requirements","experience","qualifications","about","role",
    "responsible","responsible for","preferred","must","required","skills","skills:",
    "job summary","summary"
])

token_re = re.compile(r"[A-Za-z0-9\+#\-.]+")

def extract_candidates_from_jd_lines(jd_lines: List[str]) -> Tuple[List[Dict[str, Any]], Dict[int,List[str]]]:
    """
    For each JD line produce candidate technical phrases with evidence mapping.
    Returns:
      - candidates: list of {term, evidence_lines:[indexes], sample_line}
      - map_line_to_candidates: {line_index: [candidate_terms]}
    Conservative rules:
      - split on commas, bullets, semicolons, slashes, "or", "and"
      - extract parenthetical tokens separately
      - extract tokens with special chars or curated tokens (AWS, Terraform, Docker, Prometheus, SQL, Python, etc.)
      - only accept candidate phrases up to 5 tokens (default)
    """
    curated = {"aws","azure","gcp","terraform","cloudformation","ansible","docker","kubernetes","helm",
               "prometheus","grafana","elasticsearch","elk","lambda","rds","s3","mysql","postgresql",
               "postgres","mongodb","redis","dynamodb","ci/cd","ci","cd","jenkins","github","gitlab",
               "python","java","go","typescript","javascript","scala","rust","sql","spark","hadoop",
               "terraform","ansible","kubernetes","docker","terraform"}
    candidates = {}
    line_map = {}

    def add_candidate(term: str, li: int, line_text: str):
        t = term.strip().strip(":").strip()
        if not t:
            return
        # discard obviously generic single-word tokens
        if t.lower() in GENERIC_STOP:
            return
        # discard tokens that are only punctuation or single-letter
        if len(re.sub(r"[^A-Za-z0-9]+","",t)) < 2:
            return
        key = _norm_key(t)
        if not key:
            return
        ent = candidates.get(key)
        if not ent:
            candidates[key] = {"term": t, "evidence_lines": [li], "sample_line": line_text}
        else:
            if li not in ent["evidence_lines"]:
                ent["evidence_lines"].append(li)
        line_map.setdefault(li, []).append(t)

    for i, ln in enumerate(jd_lines):
        line = (ln or "").strip()
        if not line:
            continue
        # Skip lines that are pure headings / short non-technical words
        low = line.lower().strip()
        if low.rstrip(":") in GENERIC_STOP:
            continue

        # 1) extract parenthetical phrase contents
        for par in re.findall(r"\(([^)]+)\)", line):
            # split paren content
            for p in re.split(r",|/|;|\band\b|\bor\b", par):
                p = p.strip()
                if p:
                    add_candidate(p, i, line)

        # 2) split by common delimiters - comma / semicolon / bullet / " - "
        for part in re.split(r",|;|\u2022|\||\t|\band\b|\bor\b|/|- ", line):
            part = part.strip()
            if not part:
                continue
            # if part contains many tokens, extract shorter token-like items within it
            toks = token_re.findall(part)
            if not toks:
                continue
            # Heuristics: if part length in words <=5, consider it as candidate phrase
            if len(part.split()) <= 5:
                add_candidate(part, i, line)
            else:
                # otherwise, extract contiguous short token sequences <=4 tokens
                for L in range(1,4+1):
                    for start in range(0, max(1, len(toks) - L + 1)):
                        seq = " ".join(toks[start:start+L])
                        if len(seq.split()) <= 4:
                            add_candidate(seq, i, line)

        # 3) explicit token extraction: single tokens that are curated or contain special chars
        for tk in token_re.findall(line):
            lowtk = tk.lower()
            if lowtk in curated or any(ch in tk for ch in "+#/.") or re.search(r"\d", tk):
                add_candidate(tk, i, line)

    # flatten candidates into list
    cand_list = []
    for k, v in candidates.items():
        cand_list.append({"term": v["term"], "evidence_lines": v["evidence_lines"], "sample_line": v["sample_line"]})
    return cand_list, line_map

# -------------------------
# jd_resume_to_json
# -------------------------
def jd_resume_to_json(jd_text: str, resume_text: str,
                      provider_pref: Optional[str], model_name: Optional[str],
                      keys: Dict[str, str]) -> Dict[str, Any]:
    def local_norm(text):
        lines = []
        for i, ln in enumerate([l for l in (text or "").splitlines()]):
            s = _normalize_line(ln)
            if s:
                lines.append({"index": i, "text": s})
        return {"lines": lines, "flat": " ".join([l["text"] for l in lines])}
    jd_json = local_norm(jd_text or "")
    res_json = local_norm(resume_text or "")
    return {"jd": jd_json, "resume": res_json}

# -------------------------
# Main extractor (validator-classifier approach)
# -------------------------
def extract_ranked_and_gap_keywords(jd_json: Dict[str, Any], resume_json: Dict[str, Any],
                                    provider_pref: Optional[str], model_name: Optional[str],
                                    keys: Dict[str, str], top_k: int = 40) -> Dict[str, Any]:
    """
    Conservative extractor:
      1) deterministically extract candidate phrases from JD lines
      2) ask LLM to classify each candidate as accept/reject + category (NO new terms)
      3) post-validate accepted items (must have evidence lines)
      4) rank and return keywords with ranks, evidence, category
    """
    provider = None
    try:
        provider = _provider_from_pref(provider_pref, keys or {})
    except Exception:
        provider = None

    jd_lines_objs = (jd_json or {}).get("lines") or []
    jd_lines = [ln.get("text","") if isinstance(ln, dict) else str(ln) for ln in jd_lines_objs]
    jd_lines_trim = [_normalize_line(l) for l in jd_lines]
    resume_flat = (resume_json or {}).get("flat","") or ""

    # 1) deterministic candidate extraction
    candidates, line_map = extract_candidates_from_jd_lines(jd_lines_trim)

    # if no candidates found, return empty clean result
    if not candidates:
        res_empty = {"keywords": [], "missing": [], "weak": [], "summary": "No candidates extracted from JD.", "_raw_json": "", "_filtered_out": [], "domain": "unknown"}
        print("\n=== LLM KEYWORD OPTIMIZER DIAGNOSTICS ===")
        print("No candidates found in JD lines - returning empty result.")
        print("=== DIAGNOSTICS END ===\n")
        return res_empty

    # Build concise payload for LLM: only candidate terms + 1-2 evidence lines each (indices)
    cand_payload = []
    for c in candidates:
        termsamp = c["term"]
        evs = c["evidence_lines"][:3]
        lines = [jd_lines_trim[i] for i in evs]
        cand_payload.append({"term": termsamp, "evidence": lines})

    # 2) LLM classification prompt (validator). We instruct LLM: DO NOT invent new terms.
    system = (
        "You are a strict JSON-only classifier for job-description keywords. "
        "You will be given a JSON array 'candidates', each with 'term' and 'evidence' (exact JD line strings). "
        "For each candidate, return whether it should be ACCEPTED as a technical keyword (true/false) and a category "
        "from the set: Tool, Language, Platform, Database, Concept, Other. "
        "DO NOT invent new terms or new keywords. Return EXACTLY one JSON object with keys: 'domain' (one-word) and 'candidates' (array). "
        "Each candidate object must be {\"term\":..., \"accept\": true|false, \"category\":\"...\"}. "
        "If unsure, mark accept=false. JSON only, no commentary."
    )

    user = json.dumps({"candidates": cand_payload}, ensure_ascii=False, indent=2)
    raw = ""
    parsed = None
    if provider is not None:
        try:
            raw = provider.chat(model=model_name, system=system, user=user, temperature=0.0, max_tokens=1200) or ""
            raw = (raw or "").strip().strip("`").strip()
            block = _first_json_block(raw) or raw
            parsed = json.loads(block)
        except Exception as e:
            # log provider error and fall back to conservative deterministic accept rules
            parsed = None
            print("Warning: provider.chat failed or returned non-JSON in validator step:", repr(e))
    else:
        parsed = None

    accepted = []
    filtered_out = []

    # Conservative deterministic fallback if parsed is None: Accept only curated tokens or tokens with special chars/digits
    def deterministic_accept(term: str) -> Tuple[bool,str]:
        t = term.strip()
        if not t:
            return False,"empty"
        # curated checks
        low = t.lower()
        curated_keywords = {"aws","azure","gcp","terraform","cloudformation","ansible","docker","kubernetes","prometheus","grafana","jenkins","python","java","sql","postgres","mysql","mongodb","redis","s3","rds"}
        if any(ch in t for ch in "+#/.") or re.search(r"\d", t):
            return True,"specialchar_or_digit"
        toks = token_re.findall(t)
        if not toks:
            return False,"no_tokens"
        for tk in toks:
            if tk.lower() in curated_keywords:
                return True,"curated_match"
        # allow short phrases <=3 tokens that contain at least one alpha token (non-stop)
        stop = {"the","and","or","with","for","to","in","on","of","a","an"}
        if len(toks) <= 3 and not all(x.lower() in stop for x in toks):
            return True,"short_phrase"
        return False,"heuristic_reject"

    if parsed and isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
        # Use LLM judgments but still enforce evidence presence
        domain_guess = parsed.get("domain") or "unknown"
        for c in parsed.get("candidates", []):
            term = c.get("term","") if isinstance(c.get("term",""), str) else str(c.get("term",""))
            accept = bool(c.get("accept") is True)
            category = c.get("category") if isinstance(c.get("category",""), str) else "Other"
            # verify evidence exists (we earlier created candidates with evidence)
            matched = []
            for k in candidates:
                if _norm_key(k["term"]) == _norm_key(term):
                    matched = k["evidence_lines"]
                    break
            if not matched:
                filtered_out.append({"term": term, "reason": "no_evidence_found_after_classify"})
                continue
            if accept:
                accepted.append({"term": term, "variants": [], "evidence_lines": matched, "category": category})
            else:
                filtered_out.append({"term": term, "reason": "llm_rejected"})
    else:
        # fallback deterministic acceptance
        domain_guess = "unknown"
        for c in candidates:
            term = c["term"]
            ok, reason = deterministic_accept(term)
            if ok:
                accepted.append({"term": term, "variants": [], "evidence_lines": c["evidence_lines"], "category": "Other"})
            else:
                filtered_out.append({"term": term, "reason": reason})

    # Post-validate: ensure accepted items indeed match JD lines (safety)
    final_accepted = []
    for a in accepted:
        term = a["term"]
        ev_lines = a.get("evidence_lines", [])
        valid_evidence = [jd_lines_trim[i] for i in ev_lines if 0 <= i < len(jd_lines_trim)]
        if not valid_evidence:
            filtered_out.append({"term": term, "reason": "evidence_missing_after_post_check"})
            continue
        final_accepted.append({"term": term, "variants": [], "evidence": valid_evidence, "category": a.get("category","Other")})

    # Deduplicate by normalized key, merge evidence, prefer longest display term
    merged = {}
    for ent in final_accepted:
        k = _norm_key(ent["term"])
        if not k:
            continue
        cur = merged.get(k)
        if not cur:
            merged[k] = {"term": ent["term"], "variants": set(ent.get("variants",[])), "evidence": list(ent.get("evidence",[])), "category": ent.get("category","Other")}
        else:
            # prefer longer term display (more descriptive)
            if len(ent["term"]) > len(cur["term"]):
                cur["term"] = ent["term"]
            for ev in ent.get("evidence",[]):
                if ev not in cur["evidence"]:
                    cur["evidence"].append(ev)
            for v in ent.get("variants",[]):
                cur["variants"].add(v)
            # prefer more specific category
            if cur["category"] in ("Other",""):
                cur["category"] = ent.get("category","Other")

    merged_list = []
    for k,v in merged.items():
        merged_list.append({"term": v["term"], "variants": list(v["variants"]), "evidence": v["evidence"], "category": v.get("category","Other")})

    # Rank by evidence count then earliest occurrence
    def earliest_index(evidence_lines: List[str]) -> int:
        # find earliest index by scanning jd_lines_trim for the evidence strings
        idxs = []
        for ev in evidence_lines:
            try:
                idxs.append(jd_lines_trim.index(ev))
            except Exception:
                pass
        return min(idxs) if idxs else 9999

    ranked = []
    for item in merged_list:
        ev_count = len(item.get("evidence",[]))
        eidx = earliest_index(item.get("evidence",[]))
        score = ev_count * 10 - (0.01 * eidx)
        ranked.append({"term": item["term"], "variants": item.get("variants",[]), "evidence": item.get("evidence",[]), "category": item.get("category","Other"), "score": score, "earliest": eidx})

    ranked.sort(key=lambda x: (-x["score"], x["earliest"]))

    # Build final list with serial ranks (1..N)
    final_keywords = []
    for i, ent in enumerate(ranked[:top_k], start=1):
        final_keywords.append({"rank": i, "term": ent["term"], "variants": ent.get("variants",[]), "evidence": ent.get("evidence",[]), "category": ent.get("category","Other")})

    # Compute missing / weak relative to resume deterministically
    resume_tokens = [t.lower() for t in token_re.findall(resume_flat or "")]
    resume_compact = "".join(resume_tokens)

    def present_in_resume(term: str) -> bool:
        toks = [t.lower() for t in token_re.findall(term)]
        if not toks:
            return False
        L = len(toks)
        if L <= len(resume_tokens):
            for j in range(0, len(resume_tokens) - L + 1):
                if resume_tokens[j:j+L] == toks:
                    return True
        if "".join(toks) in resume_compact:
            return True
        if L == 1 and len(toks[0]) > 3:
            base = toks[0]; alt = base[:-1] if base.endswith("s") else base + "s"
            if base in resume_tokens or alt in resume_tokens:
                return True
        return False

    missing = []
    weak = []
    for ent in final_keywords:
        if not present_in_resume(ent["term"]):
            toks = [t.lower() for t in token_re.findall(ent["term"])]
            if "".join(toks) in resume_compact:
                weak.append(ent["term"])
            else:
                missing.append(ent["term"])

    # sanity dedupe lists
    def dedupe_list(arr, limit=None):
        out=[]; s=set()
        for x in arr:
            if not isinstance(x, str): continue
            k = x.strip().lower()
            if not k or k in s: continue
            s.add(k); out.append(x)
            if limit and len(out) >= limit: break
        return out

    missing = dedupe_list(missing, limit=10)
    weak = dedupe_list(weak, limit=10)

    result = {
        "keywords": final_keywords,
        "missing": missing,
        "weak": weak,
        "summary": f"Deterministic-candidate + LLM-validated extraction; accepted={len(final_keywords)}",
        "_raw_json": raw if raw else "",
        "_filtered_out": filtered_out,
        "domain": domain_guess if domain_guess else "unknown"
    }

    # Diagnostics printed cleanly to terminal
    try:
        print("\n=== LLM KEYWORD OPTIMIZER DIAGNOSTICS ===")
        print("Domain detected (validator):", result["domain"])
        print("\n=== ACCEPTED KEYWORDS (final) ===")
        if not result["keywords"]:
            print("  ⚠️  No keywords accepted after validation.")
        for k in result["keywords"]:
            print(f"  #{k['rank']:02d} {k['term']} [{k.get('category')}] evidence_count={len(k.get('evidence',[]))}")
            for ev in k.get('evidence', [])[:3]:
                print("     >", ev)
        print("\n=== LINES WITH NO ACCEPTED KEYWORDS (skipped) ===")
        # report line numbers (1-based) that had no accepted keywords
        accepted_lines = set()
        for k in result["keywords"]:
            for ev in k.get("evidence",[]):
                try:
                    accepted_lines.add(jd_lines_trim.index(ev))
                except Exception:
                    pass
        for i, l in enumerate(jd_lines_trim):
            if i not in accepted_lines:
                # only print first 40 lines to avoid noise
                if i < 40:
                    print(f"  line {i+1:02d}: (no accepted keyword) {l}")
        if result.get("_filtered_out"):
            print("\n=== FILTERED OUT SAMPLE ===")
            for f in result["_filtered_out"][:80]:
                print("  -", f.get("term"), ":", f.get("reason"))
        # save raw LLM JSON for inspection
        raw_path = "last_llm_optimizer_raw.json"
        try:
            with open(raw_path, "w", encoding="utf-8") as fh:
                fh.write(result.get("_raw_json","") or "")
            print(f"\nRaw LLM JSON (validator output) saved to: {os.path.abspath(raw_path)}")
        except Exception:
            pass
        print("=== DIAGNOSTICS END ===\n")
    except Exception:
        pass

    return result

# -------------------------
# compatibility alias
# -------------------------
def jd_to_json(jd_text: str, provider_pref: Optional[str], model_name: Optional[str],
               temperature: float = 0.0, max_tokens: int = 800, keys: Optional[Dict[str,str]] = None) -> Dict[str, Any]:
    keys = keys or {}
    try:
        combined = jd_resume_to_json(jd_text, "", provider_pref, model_name, keys)
        return combined.get("jd", {"lines": [], "flat": ""})
    except Exception:
        lines = []
        for i, ln in enumerate([l for l in (jd_text or "").splitlines()]):
            s = (ln or "").strip()
            s = re.sub(r"\s+", " ", s)
            if s:
                lines.append({"index": i, "text": s})
        flat = " ".join([l["text"] for l in lines])
        return {"lines": lines, "flat": flat}
