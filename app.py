import io, os, json, re
import streamlit as st
from utils import export_docx, export_pdf
from pipeline import (
    analyze,
    tailor,
    extract_keywords_llm,
    extract_contacts_llm,
    polish_keyword_sentences,
    sanitize_markdown,
    extract_ats_llm_from_optimizer,  # AI ATS helper
    generate_keyword_sentences
)

# -------------------------
# Fix 1A: Stable editor/download state (ADD ONCE NEAR TOP)
# -------------------------
if "tailored_text" not in st.session_state:
    st.session_state["tailored_text"] = ""
if "tailored_edit" not in st.session_state:
    st.session_state["tailored_edit"] = ""
if "tailored_saved" not in st.session_state:
    st.session_state["tailored_saved"] = False

# Reusable token regex used elsewhere as well
_TOKEN_RE = re.compile(r"[A-Za-z0-9#+.]+")
def _tok_seq(s: str):
    return [t.lower() for t in _TOKEN_RE.findall(s or "")]

def _canon(s: str) -> str:
    return " ".join(_tok_seq(s))

def _present_line(base_text: str, line: str) -> bool:
    base_tokens = _tok_seq(base_text)
    base_compact = "".join(base_tokens)
    kt = _tok_seq(line)
    if not kt:
        return True
    L = len(kt)
    for i in range(0, len(base_tokens) - L + 1):
        if base_tokens[i:i+L] == kt:
            return True
    if "".join(kt) in base_compact:
        return True
    if L == 1 and len(kt[0]) > 3:
        w = kt[0]
        alt = w[:-1] if w.endswith("s") else w + "s"
        if w in base_tokens or alt in base_tokens:
            return True
    return False

def insert_block_after_profile_summary(resume_text: str, block_text: str) -> str:
    """
    Insert block_text immediately after a 'Profile Summary' heading (case-insensitive),
    else prepend it at the top. Keeps plain-text formatting.
    """
    m = re.search(r"(?im)^(profile\s*summary)\s*$", resume_text)
    if not m:
        return (block_text.strip() + "\n\n" + resume_text.strip()).strip()

    insert_pos = m.end()
    before = resume_text[:insert_pos]
    after  = resume_text[insert_pos:]
    glue = "\n" if not before.endswith("\n") else ""
    return (before + glue + "\n" + block_text.strip() + "\n\n" + after.lstrip("\n")).strip()

st.set_page_config(page_title="API-only Resume Tailor (v8 final)", page_icon="🧰", layout="wide")
st.title("🧰 API-only Resume Tailor (v8 final)")
st.caption("v8 layout • Full content read • Colored headings in DOCX/PDF • Plain-text output (no ##/**)")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("🔑 Provider & Model")
    provider = st.selectbox("Provider", ["openai","gemini","anthropic"], index=1)
    model = st.text_input("Model name (optional)", value="")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max tokens (output cap)", 256, 8192, 3000, 64)

    st.header("🔐 API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    keys = {"openai": openai_key.strip(), "gemini": gemini_key.strip(), "anthropic": anthropic_key.strip()}

# -------------------------
# Upload or paste helper
# -------------------------
def read_textarea_or_file(label: str, key_text: str, key_file: str) -> str:
    txt = st.session_state.get(key_text, "")
    up = st.file_uploader(label, type=["txt","md","pdf","docx"], key=key_file)
    if up is not None:
        ext = up.name.lower().split(".")[-1]
        if ext in ("txt","md"):
            txt = up.read().decode("utf-8", errors="ignore")
        elif ext == "pdf":
            try:
                from pdfminer_high_level import extract_text
            except Exception:
                from pdfminer.high_level import extract_text
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up.read()); tmp.flush()
                    txt = extract_text(tmp.name) or ""
            except Exception:
                from PyPDF2 import PdfReader
                up.seek(0); reader = PdfReader(up)
                txt = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext == "docx":
            from docx import Document
            d = Document(up)
            paras = [p.text for p in d.paragraphs]
            for tbl in d.tables:
                for row in tbl.rows:
                    paras.append(" | ".join(cell.text for cell in row.cells))
            txt = "\n".join(paras)
    txt = st.text_area(f"Or paste {label.lower()} text here", value=txt, height=240, key=key_text)
    return txt

# -------------------------
# Inputs
# -------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Resume")
    resume_text = read_textarea_or_file("Resume", "resume_text", "resume_file")
with col2:
    st.subheader("Job Description")
    jd_text = read_textarea_or_file("Job Description", "jd_text", "jd_file")

# Always use text boxes as source of truth
resume_text = st.session_state.get("resume_text", "") or resume_text
jd_text = st.session_state.get("jd_text", "") or jd_text

# -------------------------
# Main
# -------------------------
if resume_text.strip() and jd_text.strip():
    st.divider()
    st.subheader("Analysis")

    report = analyze(resume_text, jd_text)

    if st.button("Enhance contacts with AI", key="btn_contacts_ai"):
        try:
            st.session_state["ai_contacts"] = extract_contacts_llm(
                resume_text, provider_pref=provider, model_name=(model or None), keys=keys
            )
            st.success("Contacts enhanced.")
        except Exception as e:
            st.error(str(e))

    ai_contacts = st.session_state.get("ai_contacts", {})
    merged = {**report["contacts"], **{k: v for k, v in ai_contacts.items() if v}}

    st.markdown("**Contacts**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: merged["name"] = st.text_input("Name", merged.get("name",""))
    with c2: merged["email"] = st.text_input("Email", merged.get("email",""))
    with c3: merged["phone"] = st.text_input("Phone", merged.get("phone",""))
    with c4: merged["linkedin"] = st.text_input("LinkedIn", merged.get("linkedin",""))
    with c5: merged["github"] = st.text_input("GitHub", merged.get("github",""))
    st.session_state["override_contacts"] = merged

    cA, cB, cC = st.columns([1,1,2])
    with cA: st.metric("Match Score", f"{report['match']['match_score']}%")
    with cB: st.metric("Readability (FRE)", report["readability"]["flesch_reading_ease"])
    with cC:
        st.write("ATS warnings:")
        for w in report["ats"]["warnings"]:
            st.write(f"• {w}")

    # -----------------
    # LLM Keyword Optimizer
    # -----------------
    st.subheader("LLM Keyword Optimizer")
    if st.button("Extract ranked keywords with AI", key="btn_kw_extract"):
        try:
            kw = extract_keywords_llm(
                resume_text, jd_text,
                provider_pref=provider, model_name=(model or None),
                temperature=temperature, max_tokens=min(max_tokens, 1200), keys=keys
            )
            st.session_state["kw_llm"] = kw
        except Exception as e:
            st.error(str(e))

    kw_obj = st.session_state.get("kw_llm")

    def _fallback_missing_and_weak(kw_obj, resume_text, jd_text):
        missing_terms = []
        try:
            bow_missing = [w for (w, _) in (report["keywords_bow"]["missing"] or [])]
            seen = set()
            for w in bow_missing:
                wl = w.lower().strip()
                if len(wl) > 2 and wl not in seen:
                    seen.add(wl); missing_terms.append(w)
            missing_terms = missing_terms[:20]
        except Exception:
            missing_terms = []

        weak_terms = []
        try:
            text_low = " " + " ".join(resume_text.lower().split()) + " "
            terms = [(item.get("term") or "").strip() for item in (kw_obj.get("keywords") or [])]
            seen2 = set()
            for t in terms:
                tl = t.lower()
                if not tl or tl in seen2: continue
                seen2.add(tl)
                if text_low.count(" " + tl + " ") == 1:
                    weak_terms.append(t)
            weak_terms = weak_terms[:20]
        except Exception:
            weak_terms = []
        return missing_terms, weak_terms

    # === helper to build target keywords from Top Keywords (ranked) + Gaps, deduped, and excluding ones already in resume ===
    def build_target_keywords_from_optimizer(kw_obj, resume_text: str, jd_text: str):
        if not kw_obj:
            return []

        token_re = re.compile(r"[A-Za-z0-9#+.]+")

        def _canon_local(s: str) -> str:
            return " ".join(t.lower() for t in token_re.findall(s or ""))

        def _tok_seq_local(s: str):
            return [t.lower() for t in token_re.findall(s or "")]

        resume_tokens = _tok_seq_local(resume_text)
        resume_compact = "".join(resume_tokens)

        def _present(term: str) -> bool:
            kt = _tok_seq_local(term)
            if not kt:
                return False
            L = len(kt)
            for i in range(0, len(resume_tokens) - L + 1):
                if resume_tokens[i:i+L] == kt:
                    return True
            k_comp = "".join(kt)
            if k_comp and k_comp in resume_compact:
                return True
            if L == 1 and len(kt[0]) > 3:
                base = kt[0]
                alt = base[:-1] if base.endswith("s") else base + "s"
                if base in resume_tokens or alt in resume_tokens:
                    return True
            return False

        outlets_ordered = []
        for item in (kw_obj.get("keywords") or []):
            base = (item.get("term") or "").strip()
            if not base:
                continue
            variants = [v.strip() for v in (item.get("variants") or []) if v and v.strip()]
            outlets_ordered.append(base)
            outlets_ordered.extend(variants)

        gaps_terms = list(kw_obj.get("missing") or [])
        if not gaps_terms:
            try:
                fallback_missing, _fw = _fallback_missing_and_weak(kw_obj, resume_text, jd_text)
            except Exception:
                fallback_missing = []
            gaps_terms = list(fallback_missing or [])
        for g in gaps_terms:
            g = (g or "").strip()
            if g:
                outlets_ordered.append(g)

        seen, deduped = set(), []
        for t in outlets_ordered:
            c = _canon_local(t)
            if c and c not in seen:
                seen.add(c)
                deduped.append(t)

        target = [t for t in deduped if not _present(t)]
        return target

    if kw_obj:
        st.write(kw_obj.get("summary",""))
        colk1, colk2 = st.columns(2)
        with colk1:
            st.markdown("**Top Keywords (ranked)**")
            if kw_obj.get("keywords"):
                for item in kw_obj["keywords"]:
                    term = item.get("term")
                    cat = item.get("category","general")
                    variants = item.get("variants", [])
                    rank = item.get("rank")
                    suffix = f" · variants: {', '.join(variants)}" if variants else ""
                    st.write(f"{rank}. **{term}** · _{cat}_{suffix}")
            else:
                st.write("No keywords returned.")
        with colk2:
            st.markdown("**Gaps**")
            missing = kw_obj.get("missing")
            weak = kw_obj.get("weak")
            if not missing or not isinstance(missing, list) or not isinstance(weak, list):
                fallback_missing, fallback_weak = _fallback_missing_and_weak(kw_obj, resume_text, jd_text)
                if not isinstance(missing, list) or not missing: missing = fallback_missing
                if not isinstance(weak, list) or not weak: weak = fallback_weak
            st.write("Missing:", ", ".join(missing) if missing else "—")
            st.write("Weak:", ", ".join(weak) if weak else "—")

    # -----------------
    # Keyword Sentence Generator (ATS-friendly) — SEPARATE EDITOR
    # -----------------
    st.subheader("Keyword Sentence Generator (ATS-friendly)")
    st.caption("Generates concise bullets using Top Keywords (ranked) + Gaps. Edit here and Save; Tailor will blend them in.")

    if "kw_sentences_edit" not in st.session_state:
        st.session_state["kw_sentences_edit"] = ""
    if "kw_sentences_saved_text" not in st.session_state:
        st.session_state["kw_sentences_saved_text"] = ""

    col_gen, col_clear = st.columns([1,1])
    with col_gen:
        if st.button("Generate ATS-friendly keyword sentences", key="btn_kw_sentences_generate"):
            if not kw_obj:
                st.warning("Please run the LLM Keyword Optimizer first.")
            else:
                # Build target list: Top Keywords (ranked) + Gaps, deduped
                target_for_sentences = build_target_keywords_from_optimizer(kw_obj, resume_text, jd_text)

                # =========================================
                # NEW: Full target set and accurate present vs new (variants-aware)
                # =========================================
                TOKEN_RE = re.compile(r"[A-Za-z0-9#+.]+")
                def _tok_seq_local2(s: str):
                    return [t.lower() for t in TOKEN_RE.findall(s or "")]
                def _canon_local2(s: str) -> str:
                    return " ".join(_tok_seq_local2(s))

                ranked_terms = []
                for it in (kw_obj.get("keywords") or []):
                    term = (it.get("term") or "").strip()
                    if term:
                        ranked_terms.append(term)

                gaps_terms = list(kw_obj.get("missing") or [])
                if not gaps_terms:
                    try:
                        fallback_missing, _fw = _fallback_missing_and_weak(kw_obj, resume_text, jd_text)
                    except Exception:
                        fallback_missing = []
                    gaps_terms = list(fallback_missing or [])

                seen, target_all = set(), []
                for t in ranked_terms + gaps_terms:
                    c = _canon_local2(t)
                    if c and c not in seen:
                        seen.add(c)
                        target_all.append(t)

                variants_map = {}
                for it in (kw_obj.get("keywords") or []):
                    term = (it.get("term") or "").strip()
                    if not term:
                        continue
                    forms = [term] + [v for v in (it.get("variants") or []) if v and v.strip()]
                    expanded = set()
                    for f in forms:
                        f = f.strip()
                        if not f:
                            continue
                        expanded.add(f)
                        expanded.add(f.replace("-", " "))
                        expanded.add(f.replace("/", " "))
                    variants_map[_canon_local2(term)] = list(expanded)

                presence_text = (st.session_state.get("tailored_edit") or resume_text or "")
                text_tokens = _tok_seq_local2(presence_text)

                def _has_seq(tokens, cand: str) -> bool:
                    seq = _tok_seq_local2(cand)
                    L = len(seq)
                    if L == 0:
                        return False
                    for i in range(0, len(tokens) - L + 1):
                        if tokens[i:i+L] == seq:
                            return True
                    if L == 1 and len(seq[0]) > 3:
                        w = seq[0]
                        alt = w[:-1] if w.endswith("s") else w + "s"
                        return (w in tokens) or (alt in tokens)
                    return False

                present_keywords, new_keywords = [], []
                for t in target_all:
                    key = _canon_local2(t)
                    cand_forms = variants_map.get(key, [t, t.replace("-", " "), t.replace("/", " ")])
                    seen_forms, cand_forms_dedup = set(), []
                    for f in cand_forms:
                        f = (f or "").strip()
                        if not f:
                            continue
                        kf = f.lower()
                        if kf in seen_forms:
                            continue
                        seen_forms.add(kf)
                        cand_forms_dedup.append(f)
                    found = any(_has_seq(text_tokens, f) for f in cand_forms_dedup)
                    (present_keywords if found else new_keywords).append(t)

                with st.expander("🔍 New keywords for your resume (not currently found)", expanded=True):
                    st.write(", ".join(new_keywords) if new_keywords else "— none —")
                with st.expander("✅ Already present (will NOT be generated again)", expanded=False):
                    st.write(", ".join(present_keywords) if present_keywords else "— none —")

                target_for_sentences = new_keywords

                if not target_for_sentences:
                    st.info("All optimizer keywords already appear in the resume. Nothing to add.")
                else:
                    bullets_raw = generate_keyword_sentences(
                        resume_text=presence_text,
                        jd_text=jd_text,
                        target_keywords=target_for_sentences,
                        provider_pref=provider,
                        model_name=(model or None),
                        temperature=temperature,
                        max_tokens=min(max_tokens, 900),
                        keys=keys
                    )

                    try:
                        polished = polish_keyword_sentences(
                            resume_text=presence_text,
                            bullets_text=(bullets_raw or "").strip(),
                            jd_text=jd_text,
                            provider_pref=provider,
                            model_name=(model or None),
                            temperature=temperature,
                            max_tokens=min(max_tokens, 800),
                            keys=keys
                        )
                        out_text = polished
                    except Exception:
                        out_text = (bullets_raw or "")

                    TOKEN_RE2 = re.compile(r"[A-Za-z0-9#+.]+")
                    def _canon_line(s: str) -> str:
                        return " ".join(t.lower() for t in TOKEN_RE2.findall(s or ""))

                    seen_lines, lines_out = set(), []
                    for ln in (out_text.splitlines() if out_text else []):
                        s = ln.strip()
                        if not s:
                            continue
                        c = _canon_line(s)
                        if c in seen_lines:
                            continue
                        seen_lines.add(c)
                        if not s.startswith("• "):
                            s = "• " + s.lstrip("-").lstrip("•").strip()
                        lines_out.append(s)

                    st.session_state["kw_sentences_edit"] = "\n".join(lines_out).strip()
                    st.success("Generated ATS-friendly keyword sentences. Review below, edit, then click Save.")

    with col_clear:
        if st.button("Clear keyword sentences", key="btn_kw_sentences_clear"):
            st.session_state["kw_sentences_edit"] = ""
            st.session_state["kw_sentences_saved_text"] = ""
            st.info("Keyword sentences cleared.")

    kw_edit = st.text_area("Keyword Sentences (editable, plain text)", key="kw_sentences_edit", height=220)

    if st.button("💾 Save keyword sentences", key="btn_kw_sentences_save"):
        st.session_state["kw_sentences_saved_text"] = (kw_edit or "").strip()
        st.success("Saved. Tailor with LLM will integrate these into the resume.")

    # -----------------
    # Tailor with LLM (API-only) — integrates ONLY Resume + SAVED Keyword Sentences,
    # de-duplicates, and refines existing mentions to look native and professional.
    # -----------------
    st.divider()
    st.subheader("Tailor with LLM (API-only)")
    st.caption("Integrates your Resume + SAVED Keyword Sentence Generator text so it reads like original experience (no copy-paste feel), avoids duplicates, and refines existing mentions.")

    if st.button("Generate tailored resume", type="primary", key="btn_tailor_generate"):
        if not resume_text.strip():
            st.warning("Please paste or upload your resume text first.")
        else:
            try:
                # 1) Read SAVED keyword sentences from the separate editor
                saved_kw_sentences = (st.session_state.get("kw_sentences_saved_text", "") or "").strip()

                # 2) Split & clean
                raw_lines = [ln.strip() for ln in (saved_kw_sentences.splitlines() if saved_kw_sentences else [])]
                raw_lines = [ln for ln in raw_lines if ln]

                # 3) Decide: add vs refine (NO placement logic here)
                lines_to_add, refine_hints, seen_lines = [], [], set()
                for ln in raw_lines:
                    ln_clean = ln.lstrip("•").lstrip("-").strip()
                    if not ln_clean:
                        continue
                    c = _canon(ln_clean)
                    if c in seen_lines:
                        continue
                    seen_lines.add(c)

                    if _present_line(resume_text, ln_clean):
                        refine_hints.append(ln_clean)
                    else:
                        if not ln_clean.startswith("• "):
                            ln_clean = "• " + ln_clean
                        lines_to_add.append(ln_clean)

                # 4) Build input for LLM: base resume + integration notes (let AI choose placement)
                blocks = [resume_text.strip()]

                if refine_hints or lines_to_add:
                    guidance = [
                        "",
                        "Integration Notes (for model):",
                        "- Integrate the following without duplicating existing content.",
                        "- If a theme already exists, refine in-place; do not add a new line.",
                        "- Choose the most appropriate existing section (e.g., Core Competencies/Skills, Technical Skills, or a relevant role).",
                        "- NEVER place added lines at the very top of the document or the very end.",
                        # >>> ADD THIS LINE <<<
                        "- Treat ALL added lines below as 'Core Competencies' content; do NOT place them under Work Experience or Projects.",
                    ]
                    if refine_hints:
                        guidance += ["", "Refine these existing themes:"]
                        guidance += [f"- {h}" for h in refine_hints]
                    if lines_to_add:
                        guidance += ["", "Add these lines naturally (responsibility-style):"]
                        guidance += lines_to_add

                    blocks.append("\n".join(guidance))

                base_resume_for_llm = "\n\n".join(blocks).strip()

                override_contacts = st.session_state.get("override_contacts")

                tailored = tailor(
                    base_resume_for_llm,
                    jd_text,
                    provider_preference=provider,
                    model_name=(model or None),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    keys=keys,
                    target_keywords=[],  # do not pull from optimizer in this flow
                    override_contacts=override_contacts
                )

                # -------------------------
                # Fix 2: Normalize spacing after tailoring (REPLACE this part)
                # -------------------------
                def _normalize(s: str) -> str:
                    s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)  # trim trailing spaces per line
                    s = re.sub(r"\n{3,}", "\n\n", s)                   # collapse >2 blank lines
                    return s.strip()

                final_txt = _normalize(sanitize_markdown(tailored))
                st.session_state["tailored_text"] = final_txt
                st.session_state["tailored_edit"]  = final_txt   # keep editor populated
                st.session_state["tailored_saved"] = False

                if lines_to_add and refine_hints:
                    st.success("Tailored resume generated. New keyword sentences were integrated naturally and existing mentions were refined. Review and click Save before exporting.")
                elif lines_to_add:
                    st.success("Tailored resume generated. Your keyword sentences were integrated without duplicates. Review and click Save before exporting.")
                elif refine_hints:
                    st.success("Tailored resume generated. Existing keyword mentions were refined for clarity and impact (no duplicates added). Review and click Save before exporting.")
                else:
                    st.info("No new keyword sentences found and nothing specific to refine. The resume was still tailored for structure and clarity.")
            except Exception as e:
                st.error(str(e))

    # -------------------------
    # Fix 1B: Editor always visible; Save doesn't clear it; downloads show immediately when saved
    # (REPLACE your existing editor + export sections with THIS block)
    # -------------------------
    edited_text = st.text_area("Tailored resume (plain text)", key="tailored_edit", height=420)

    if st.button("💾 Save", key="btn_tailor_save"):
        def _normalize_save(s: str) -> str:
            s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)
            s = re.sub(r"\n{3,}", "\n\n", s)
            return s.strip()
        st.session_state["tailored_text"] = _normalize_save(edited_text or "")
        st.session_state["tailored_saved"] = True
        st.success("Saved. Exports will use your edited text.")

    saved_text = (st.session_state.get("tailored_text", "") or "").strip()
    colx1, colx2 = st.columns(2)
    with colx1:
        if st.session_state.get("tailored_saved") and saved_text:
            out_path = export_docx(saved_text, "tailored_resume.docx")
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download DOCX", f, file_name="tailored_resume.docx", key="dl_docx")
        else:
            st.info("Click Save to enable DOCX download.")
    with colx2:
        if st.session_state.get("tailored_saved") and saved_text:
            out_path = export_pdf(saved_text, "tailored_resume.pdf")
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download PDF", f, file_name="tailored_resume.pdf", key="dl_pdf")
        else:
            st.info("Click Save to enable PDF download.")

    # -----------------
    # ATS Scan (AI) + local reconciliation
    # -----------------
    st.divider()
    st.subheader("ATS Scan (Keyword Coverage vs Final Resume)")

    if st.button("Run ATS analysis on edited resume", key="btn_ats_run"):
        final_text = (st.session_state.get("tailored_edit", "") or "").strip()
        kw_obj_now = st.session_state.get("kw_llm") or {}

        if not final_text:
            st.warning("Please add or generate resume content first.")
        elif not kw_obj_now:
            st.warning("Please run the LLM Keyword Optimizer first.")
        else:
            ats_llm = extract_ats_llm_from_optimizer(
                resume_text=final_text,
                optimizer_obj=kw_obj_now,
                provider_pref=provider,
                model_name=(model or None),
                temperature=temperature,
                max_tokens=max_tokens,
                keys=keys,
                jd_text=jd_text
            )
            st.session_state["final_ats_llm"] = ats_llm

            token_re = re.compile(r"[A-Za-z0-9#+.]+")
            def _canon_local3(s: str) -> str:
                return " ".join(t.lower() for t in token_re.findall(s or ""))

            ranked = [(it.get("term") or "").strip() for it in (kw_obj_now.get("keywords") or []) if (it.get("term") or "").strip()]
            gaps = list(kw_obj_now.get("missing") or [])

            seen_terms, ordered_terms = set(), []
            for t in (ranked + gaps):
                c = _canon_local3(t)
                if c and c not in seen_terms:
                    seen_terms.add(c); ordered_terms.append(t)

            variants_map = {}
            for it in (kw_obj_now.get("keywords") or []):
                term = (it.get("term") or "").strip()
                if term:
                    variants_map[_canon_local3(term)] = [v.strip() for v in (it.get("variants") or []) if v and v.strip()]

            final_tokens = [t.lower() for t in token_re.findall(final_text)]
            final_compact = "".join(final_tokens)

            def _tok_seq3(s: str):
                return [t.lower() for t in token_re.findall(s or "")]

            present, missing, coverage = [], [], []
            for term in ordered_terms:
                cand_list = [term] + (variants_map.get(_canon_local3(term), []))
                found = False
                found_pos = -1
                found_len = 0

                for cand in cand_list:
                    tt = _tok_seq3(cand)
                    if not tt:
                        continue

                    L = len(tt)
                    for i in range(0, len(final_tokens) - L + 1):
                        if final_tokens[i:i+L] == tt:
                            found, found_pos, found_len = True, i, L
                            break
                    if found:
                        break

                    t_comp = "".join(tt)
                    if t_comp and t_comp in final_compact:
                        found, found_pos, found_len = True, -1, L
                        break

                    if L == 1 and len(tt[0]) > 3:
                        base = tt[0]
                        alt = base[:-1] if base.endswith("s") else base + "s"
                        if base in final_tokens or alt in final_tokens:
                            found, found_pos, found_len = True, -1, 1
                            break

                if found:
                    if found_pos >= 0:
                        spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9#+.]+", final_text)]
                        s = spans[found_pos][1] if 0 <= found_pos < len(spans) else 0
                        e_idx = min(found_pos + max(1, found_len) - 1, len(spans) - 1)
                        e = spans[e_idx][2] if spans else s
                        s = max(0, s - 20); e = min(len(final_text), e + 20)
                        ev = final_text[s:e].replace("\n", " ").strip()
                    else:
                        ev = ""
                    present.append(term)
                    coverage.append({"term": term, "present": True, "evidence": ev})
                else:
                    missing.append(term)
                    coverage.append({"term": term, "present": False, "evidence": ""})

            total = max(1, len(ordered_terms))
            score = round(100 * len(present) / total)
            st.session_state["final_ats_llm"] = {
                **(st.session_state.get("final_ats_llm") or {}),
                "score": score,
                "present": present,
                "missing": missing,
                "coverage": coverage,
            }

    # Display ATS results
    ats_llm = st.session_state.get("final_ats_llm")
    if ats_llm:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("AI ATS Keyword Score", f"{ats_llm.get("score",0)}%")
        with c2:
            st.write("Suggestions (where & how to add missing keywords):")
            sugg = ats_llm.get("suggestions") or []
            miss = set(ats_llm.get("missing") or [])
            shown = False
            for s in sugg:
                term = s.get("term","")
                if term in miss:
                    shown = True
                    st.write(f"• {term} → {s.get('section','Core Competencies')}: {s.get('how','')}")
            if not shown:
                st.write("—")

        colp, colm = st.columns(2)
        with colp:
            st.markdown("**Present keywords**")
            pres = ats_llm.get("present") or []
            st.write(", ".join(pres) if pres else "—")
        with colm:
            st.markdown("**Missing keywords**")
            miss = ats_llm.get("missing") or []
            st.write(", ".join(miss) if miss else "—")

        cov = ats_llm.get("coverage") or []
        if cov:
            st.markdown("**Coverage details (sample)**")
            for row in cov[:12]:
                t = row.get("term",""); p = "✅" if row.get("present") else "❌"
                ev = row.get("evidence","")
                st.write(f"{p} {t}: {ev}")

else:
    st.info("Upload or paste both the Resume and the Job Description to begin.")
