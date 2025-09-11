import io
import os
import json
import re
import streamlit as st
from utils import export_docx, export_pdf
from pipeline import (
    analyze,
    tailor,
    extract_keywords_llm,
    extract_contacts_llm,
    polish_keyword_sentences,
    sanitize_markdown,
    extract_ats_llm_from_optimizer,
    generate_keyword_sentences,
    generate_summary_bullets,
    bulletize_summary_preserve_meaning,
    insert_technical_skills_after_summary,
    extract_gaps,
)

# -------------------------
# Stable editor/download state
# -------------------------
if "tailored_text" not in st.session_state:
    st.session_state["tailored_text"] = ""
if "tailored_edit" not in st.session_state:
    st.session_state["tailored_edit"] = ""
if "tailored_saved" not in st.session_state:
    st.session_state["tailored_saved"] = False
if "kw_sentences_edit" not in st.session_state:
    st.session_state["kw_sentences_edit"] = ""
if "kw_sentences_saved_text" not in st.session_state:
    st.session_state["kw_sentences_saved_text"] = ""

# -------------------------
# Token helpers
# -------------------------
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

# -------------------------
# Summary helpers
# -------------------------
_HEADING_RE = r"(?im)^\s*(profile\s*summary|professional\s*summary|summary)\s*[:\-–—]?\s*$"
_NEXT_HEADING_RE = r"(?im)^\s*(profile\s*summary|professional\s*summary|summary|technical\s*skills|work\s*experience|experience|education|projects|certifications|awards|publications)\s*[:\-–—]?\s*$"

_SUMMARY_START_PH = "\n<<<KEEP_SUMMARY_POSITION_START>>>\n"
_SUMMARY_END_PH   = "\n<<<KEEP_SUMMARY_POSITION_END>>>\n"

def _find_summary_bounds(text: str):
    if not text:
        return -1, -1, -1, ""
    m = re.search(_HEADING_RE, text)
    if not m:
        return -1, -1, -1, ""
    head_start, head_end = m.start(), m.end()
    after = text[head_end:]
    nxt = re.search(_NEXT_HEADING_RE, after)
    block_end = head_end + (nxt.start() if nxt else len(after))
    heading_text = text[m.start():m.end()].strip()
    return head_start, head_end, block_end, heading_text

def _insert_summary_placeholders(full_text: str):
    hs, he, be, _ = _find_summary_bounds(full_text or "")
    if hs < 0:
        return full_text, False
    head = full_text[:he]
    body = full_text[he:be]
    tail = full_text[be:]
    return (head.rstrip() + _SUMMARY_START_PH + _SUMMARY_END_PH + tail.lstrip("\n")), True

def _replace_placeholders_with_bullets(full_text: str, bullets_block: str) -> str:
    if not full_text:
        return full_text
    lines = []
    for ln in (bullets_block or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if not s.startswith("• "):
            s = "• " + s.lstrip("-").lstrip("•").strip()
        lines.append(s)
    block = "\n".join(lines).strip()
    if _SUMMARY_START_PH in full_text and _SUMMARY_END_PH in full_text:
        return full_text.replace(_SUMMARY_START_PH, "\n").replace(_SUMMARY_END_PH, "\n" + block + "\n", 1).replace(_SUMMARY_END_PH, "")
    return full_text

def _extract_existing_summary_block(resume_text: str) -> str:
    txt = resume_text or ""
    m = re.search(_HEADING_RE, txt)
    if not m:
        return ""
    start = m.end()
    after = txt[start:]
    n = re.search(_NEXT_HEADING_RE, after)
    if n:
        return after[:n.start()].strip()
    return after.strip()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="API-only Resume Tailor (v8 final)", page_icon="🧰", layout="wide")
st.title("🧰 API-only Resume Tailor (v8 final)")
st.caption("Technical Skills will be inserted after Profile Summary")

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
                from pdfminer.high_level import extract_text
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
# --- Step 3: Clear optimizer cache if JD changed interactively ---
_prev_jd = st.session_state.get("_prev_jd_for_kw", "")
if (jd_text or "").strip() != (_prev_jd or "").strip():
    st.session_state.pop("kw_llm", None)
    st.session_state.pop("final_ats_llm", None)
    st.session_state.pop("ai_contacts", None)
    st.session_state["_prev_jd_for_kw"] = jd_text or ""


resume_text = st.session_state.get("resume_text", "") or resume_text
jd_text = st.session_state.get("jd_text", "") or jd_text
# --- Reset keyword optimizer caches if JD changed ---
_prev_jd = st.session_state.get("_prev_jd_for_kw", "")
if (jd_text or "").strip() != (_prev_jd or "").strip():
    # JD changed — clear previous keyword optimizer outputs so nothing is reused
    st.session_state.pop("kw_llm", None)
    st.session_state.pop("final_ats_llm", None)
    st.session_state.pop("ai_contacts", None)
    # store latest JD for future change detection
    st.session_state["_prev_jd_for_kw"] = jd_text or ""


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
            # Run extraction (always uses current jd_text variable)
            kw = extract_keywords_llm(
                resume_text, jd_text,
                provider_pref=provider, model_name=(model or None),
                temperature=temperature, max_tokens=min(max_tokens, 1200), keys=keys
            )

            # Overwrite session state with fresh results
            st.session_state["kw_llm"] = kw

            # Record the JD used so future changes will clear the cache
            st.session_state["_prev_jd_for_kw"] = jd_text or ""

            # Clear downstream cached ATS results so they will be recomputed
            st.session_state.pop("final_ats_llm", None)

            st.success("Keywords extracted from the current JD.")
        except Exception as e:
            st.error(str(e))


    kw_obj = st.session_state.get("kw_llm")

    if kw_obj and "_raw_extraction" in kw_obj:
        with st.expander("🔍 Raw Extraction from LLM"):
            st.text(kw_obj["_raw_extraction"])

    if kw_obj:
        st.write(kw_obj.get("summary",""))
        colk1, colk2 = st.columns(2)
        with colk1:
            st.markdown("**Top Keywords (ranked)**")
            keywords = kw_obj.get("keywords", []) if isinstance(kw_obj, dict) else []
            if keywords and isinstance(keywords, list):
                for item in keywords:
                    term = item.get("term", "").strip()
                    if not term:
                        continue
                    cat = item.get("category", "general")
                    variants = item.get("variants", [])
                    rank = item.get("rank", "?")
                    suffix = f" · variants: {', '.join(variants)}" if variants else ""
                    st.write(f"{rank}. **{term}** · _{cat}_{suffix}")
            else:
                st.error("⚠️ No parsed keywords available from LLM.")
                st.text("=== RAW JSON FROM LLM ===\n" + str(kw_obj.get("_raw_json", "")))

        with colk2:
            st.markdown("**Gaps**")
            try:
                gaps = extract_gaps(resume_text, kw_obj)
            except Exception:
                gaps = []
            st.write("Gaps:", ", ".join(gaps) if gaps else "—")
            st.caption("🔍 These are Top Keywords missing from your resume.")

    # -----------------
    # Generate Technical Skills (grouped)
    # -----------------
    st.subheader("Generate Technical Skills (LLM)")
    st.caption("Produces grouped Technical Skills headings + keywords (will be inserted after Profile Summary).")

    col_gen, col_clear = st.columns([1,1])
    with col_gen:
        if st.button("Generate Technical Skills (LLM)", key="btn_kw_sentences_generate"):
            if not kw_obj:
                st.warning("Please run the LLM Keyword Optimizer first.")
            else:
                try:
                    new_keywords = extract_gaps(resume_text, kw_obj)
                except Exception:
                    new_keywords = []
                if not new_keywords:
                    new_keywords = list(kw_obj.get("missing") or [])
                if not new_keywords:
                    new_keywords = [it.get("term","") for it in (kw_obj.get("keywords") or [])][:12]
                try:
                    skills_text = generate_keyword_sentences(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        target_keywords=new_keywords,
                        provider_pref=provider,
                        model_name=(model or None),
                        temperature=temperature,
                        max_tokens=min(max_tokens, 900),
                        keys=keys
                    )
                    st.session_state["kw_sentences_edit"] = skills_text or ""
                    st.success("Generated Technical Skills block. Edit below and Save.")
                except Exception as e:
                    st.error(str(e))

    with col_clear:
        if st.button("Clear technical skills", key="btn_kw_sentences_clear"):
            st.session_state["kw_sentences_edit"] = ""
            st.session_state["kw_sentences_saved_text"] = ""
            st.info("Technical skills cleared.")

    kw_edit = st.text_area("Technical Skills (editable, plain text; will be inserted after Profile Summary)", key="kw_sentences_edit", height=220)

    if st.button("💾 Save technical skills", key="btn_kw_sentences_save"):
        st.session_state["kw_sentences_saved_text"] = (kw_edit or "").strip()
        st.success("Saved. Tailor will insert these under 'Technical Skills' immediately after Profile Summary.")

    # -----------------
    # Tailor with LLM
    # -----------------
    st.divider()
    st.subheader("Tailor with LLM (API-only)")
    st.caption("Integrates your Resume + SAVED Technical Skills; Core Competencies will be removed and Technical Skills inserted after Profile Summary.")

    if st.button("Generate tailored resume", type="primary", key="btn_tailor_generate"):
        if not resume_text.strip():
            st.warning("Please paste or upload your resume text first.")
        else:
            try:
                saved_skills_block = (st.session_state.get("kw_sentences_saved_text", "") or "").strip()
                raw_lines = [ln.rstrip() for ln in (saved_skills_block.splitlines() if saved_skills_block else [])]
                raw_lines = [ln for ln in raw_lines if ln]
                lines_to_add, refine_hints, seen_lines = [], [], set()
                for ln in raw_lines:
                    ln_clean = ln.strip()
                    if not ln_clean:
                        continue
                    c = _canon(ln_clean)
                    if c in seen_lines:
                        continue
                    seen_lines.add(c)
                    if _present_line(resume_text, ln_clean):
                        refine_hints.append(ln_clean)
                    else:
                        lines_to_add.append(ln_clean)

                blocks = [resume_text.strip()]
                if refine_hints or lines_to_add:
                    guidance = [
                        "",
                        "Integration Notes (for model):",
                        "- Remove any existing 'Core Competencies' section entirely before insertion.",
                        "- Replace existing 'Technical Skills' if present; otherwise insert new 'Technical Skills' immediately after 'Profile Summary'.",
                        "- NEVER place added lines at the very top of the document or the very end.",
                    ]
                    if refine_hints:
                        guidance += ["", "Refine these existing themes (do not duplicate):"]
                        guidance += [f"- {h}" for h in refine_hints]
                    if lines_to_add:
                        guidance += ["", "Add/replace with these Technical Skills lines (preserve grouping/heading):"]
                        guidance += lines_to_add
                    blocks.append("\n".join(guidance))

                base_resume_for_llm = "\n\n".join(blocks).strip()
                orig_summary_block = _extract_existing_summary_block(resume_text)
                resume_frozen, has_summary = _insert_summary_placeholders(base_resume_for_llm)
                if "Integration Notes (for model):" in resume_frozen:
                    resume_frozen += (
                        "\n- DO NOT move or delete the markers '<<<KEEP_SUMMARY_POSITION_START>>>' "
                        "and '<<<KEEP_SUMMARY_POSITION_END>>>'."
                    )

                override_contacts = st.session_state.get("override_contacts")
                tailored = tailor(
                    resume_frozen,
                    jd_text,
                    provider_preference=provider,
                    model_name=(model or None),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    keys=keys,
                    target_keywords=[],
                    override_contacts=override_contacts
                )

                final_txt = sanitize_markdown(tailored)

                if has_summary and orig_summary_block.strip():
                    bullets_text = generate_summary_bullets(
                        resume_text=orig_summary_block,
                        jd_text=jd_text,
                        focus="summary",
                        provider_pref=provider,
                        model_name=(model or None),
                        temperature=temperature,
                        max_tokens=min(max_tokens, 1000),
                        keys=keys,
                    )
                    if bullets_text:
                        final_txt = _replace_placeholders_with_bullets(final_txt, bullets_text)

                # after you have final_txt and have replaced summary placeholders:
                final_txt = final_txt.replace(_SUMMARY_START_PH, "\n").replace(_SUMMARY_END_PH, "\n")

                # Insert saved technical skills block (if any)
                saved_skills_block = (st.session_state.get("kw_sentences_saved_text", "") or "").strip()
                if saved_skills_block:
                    # insert_technical_skills_after_summary comes from pipeline.py
                    final_txt = insert_technical_skills_after_summary(final_txt, saved_skills_block)


                st.session_state["tailored_text"] = final_txt
                st.session_state["tailored_edit"]  = final_txt
                st.session_state["tailored_saved"] = False

                if lines_to_add and refine_hints:
                    st.success("Tailored resume generated. Technical Skills inserted after Profile Summary and existing mentions refined.")
                elif lines_to_add:
                    st.success("Tailored resume generated. Technical Skills inserted after Profile Summary.")
                elif refine_hints:
                    st.success("Tailored resume generated. Existing mentions refined (no duplicates added).")
                else:
                    st.info("No new Technical Skills detected; resume tailored for structure and clarity.")
            except Exception as e:
                st.error(str(e))

    # -------------------------
    # Editor (always visible) + Save + Downloads
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

    # from utils import export_docx, export_pdf

    saved_text = (st.session_state.get("tailored_text", "") or "").strip()
    colx1, colx2 = st.columns(2)

    with colx1:
        if not saved_text:
            st.info("Click Save to enable DOCX download.")
        else:
            if st.button("Prepare DOCX for download", key="btn_make_docx"):
                docx_bytes = export_docx(saved_text, out_path=None)  # returns bytes
                st.session_state["_last_docx_bytes"] = docx_bytes
                st.success("DOCX ready.")
            docx_b = st.session_state.get("_last_docx_bytes")
            if docx_b:
                st.download_button(
                    "⬇️ Download DOCX",
                    data=docx_b,
                    file_name="tailored_resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_docx"
                )

    with colx2:
        if not saved_text:
            st.info("Click Save to enable PDF download.")
        else:
            if st.button("Prepare PDF for download", key="btn_make_pdf"):
                pdf_bytes = export_pdf(saved_text, out_path=None)  # returns bytes
                st.session_state["_last_pdf_bytes"] = pdf_bytes
                st.success("PDF ready.")
            pdf_b = st.session_state.get("_last_pdf_bytes")
            if pdf_b:
                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_b,
                    file_name="tailored_resume.pdf",
                    mime="application/pdf",
                    key="dl_pdf"
                )


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

    ats_llm = st.session_state.get("final_ats_llm")
    if ats_llm:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("AI ATS Keyword Score", f"{ats_llm.get('score',0)}%")
        with c2:
            st.write("Suggestions (where & how to add missing keywords):")
            sugg = ats_llm.get("suggestions") or []
            miss = set(ats_llm.get("missing") or [])
            shown = False
            for s in sugg:
                term = s.get("term","")
                if term in miss:
                    shown = True
                    st.write(f"• {term} → {s.get('section','Technical Skills')}: {s.get('how','')}")
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
