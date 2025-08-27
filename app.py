import io, os, json, re
import streamlit as st
from pipeline import analyze, tailor, extract_keywords_llm, extract_contacts_llm, sanitize_markdown
from utils import export_docx, export_pdf
from pipeline import analyze, tailor, extract_keywords_llm, extract_contacts_llm, sanitize_markdown, extract_ats_llm


st.set_page_config(page_title="API-only Resume Tailor (v8 final)", page_icon="🧰", layout="wide")
st.title("🧰 API-only Resume Tailor (v8 final)")
st.caption("v8 layout • Full content read • Colored headings in DOCX/PDF • Plain-text output (no ##/**)")

with st.sidebar:
    st.header("🔑 Provider & Model")
    provider = st.selectbox("Provider", ["openai","gemini","anthropic"], index=1)
    model = st.text_input("Model name (optional)", value="")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max tokens", 256, 4096, 1500, 64)

    st.header("🔐 API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password")
    gemini_key = st.text_input("Gemini API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    keys = {"openai": openai_key.strip(), "gemini": gemini_key.strip(), "anthropic": anthropic_key.strip()}

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
                import tempfile
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
            # include docx table text
            for tbl in d.tables:
                for row in tbl.rows:
                    paras.append(" | ".join(cell.text for cell in row.cells))
            txt = "\n".join(paras)
    txt = st.text_area(f"Or paste {label.lower()} text here", value=txt, height=240, key=key_text)
    return txt

col1, col2 = st.columns(2)
with col1:
    st.subheader("Resume")
    resume_text = read_textarea_or_file("Resume", "resume_text", "resume_file")
with col2:
    st.subheader("Job Description")
    jd_text = read_textarea_or_file("Job Description", "jd_text", "jd_file")

# (1) Always use the visible paste boxes as the source of truth for AI steps
resume_text = st.session_state.get("resume_text", "") or resume_text
jd_text = st.session_state.get("jd_text", "") or jd_text

if resume_text.strip() and jd_text.strip():
    st.divider()
    st.subheader("Analysis")

    report = analyze(resume_text, jd_text)

    # Contacts (regex + optional AI + manual)
    if st.button("Enhance contacts with AI"):
        try:
            st.session_state["ai_contacts"] = extract_contacts_llm(resume_text, provider_pref=provider, model_name=(model or None), keys=keys)
            st.success("Contacts enhanced.")
        except Exception as e:
            st.error(str(e))

    ai_contacts = st.session_state.get("ai_contacts", {})
    merged = {**report["contacts"], **{k: v for k,v in ai_contacts.items() if v}}

    st.markdown("**Contacts**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: merged["name"] = st.text_input("Name", merged.get("name",""))
    with c2: merged["email"] = st.text_input("Email", merged.get("email",""))
    with c3: merged["phone"] = st.text_input("Phone", merged.get("phone",""))
    with c4: merged["linkedin"] = st.text_input("LinkedIn", merged.get("linkedin",""))
    with c5: merged["github"] = st.text_input("GitHub", merged.get("github",""))
    st.session_state["override_contacts"] = merged

    cA, cB, cC = st.columns([1,1,2])
    with cA:
        st.metric("Match Score", f"{report['match']['match_score']}%")
    with cB:
        st.metric("Readability (FRE)", report["readability"]["flesch_reading_ease"])
    with cC:
        st.write("ATS warnings:")
        for w in report["ats"]["warnings"]:
            st.write(f"• {w}")

    # LLM-powered keywords (JSON)
    st.subheader("LLM Keyword Optimizer")
    if st.button("Extract ranked keywords with AI"):
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
        """
        If the model didn't return 'missing'/'weak', compute a sensible fallback:
        - missing: token gaps from JD vs resume (bag-of-words)
        - weak: LLM keyword terms that appear only once (or barely) in resume
        """
        try:
            from ai.matcher import re as _re  # not used, but ensures import path correct
        except Exception:
            pass

        # 1) Missing fallback via analyze()’s BOW gaps:
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

        # 2) Weak fallback using the LLM’s keyword list (terms that appear once in resume)
        weak_terms = []
        try:
            text_low = " " + resume_text.lower() + " "
            text_low = " ".join(text_low.split())
            terms = [ (item.get("term") or "").strip() for item in (kw_obj.get("keywords") or []) ]
            seen2 = set()
            for t in terms:
                tl = t.lower()
                if not tl or tl in seen2:
                    continue
                seen2.add(tl)
                count = text_low.count(" " + tl + " ")
                if count == 1:
                    weak_terms.append(t)
            weak_terms = weak_terms[:20]
        except Exception:
            weak_terms = []

        return missing_terms, weak_terms

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
            if not missing or not isinstance(missing, list) or not weak or not isinstance(weak, list):
                fallback_missing, fallback_weak = _fallback_missing_and_weak(kw_obj, resume_text, jd_text)
                if not missing or not isinstance(missing, list):
                    missing = fallback_missing
                if not weak or not isinstance(weak, list):
                    weak = fallback_weak

            st.write("Missing:", ", ".join(missing) if missing else "—")
            st.write("Weak:", ", ".join(weak) if weak else "—")

    st.divider()
    st.subheader("Tailor with LLM (API-only)")
    st.caption("Uses ONLY optimizer keywords; removes those already present in the resume.")

    # Disable generation until optimizer has run
    can_generate = bool(st.session_state.get("kw_llm", {}).get("keywords") or st.session_state.get("kw_llm", {}).get("missing"))
    if st.button("Generate tailored resume", type="primary", disabled=not can_generate):
        try:
            target = []

            # --- Build from Top Keywords (ranked) + Gaps, dedupe with same tokenizer, then filter out ones already in resume ---
            if kw_obj:
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

                ranked_terms = []
                if kw_obj.get("keywords"):
                    for item in kw_obj["keywords"]:
                        term = (item.get("term") or "").strip()
                        if term:
                            ranked_terms.append(term)

                gaps_terms = list(kw_obj.get("missing") or [])
                if not gaps_terms:
                    fallback_missing, _fallback_weak = _fallback_missing_and_weak(kw_obj, resume_text, jd_text)
                    gaps_terms = list(fallback_missing or [])

                # Dedupe first (ranked then gaps)
                seen = set()
                deduped = []
                for t in ranked_terms + gaps_terms:
                    c = _canon(t)
                    if c and c not in seen:
                        seen.add(c)
                        deduped.append(t)

                # Filter out keywords that already appear in the ORIGINAL resume (A ∩ B)
                target = [t for t in deduped if not _has_term(resume_text, t)]

            override_contacts = st.session_state.get("override_contacts")
            tailored = tailor(resume_text, jd_text, provider_preference=provider, model_name=(model or None),
                              temperature=temperature, max_tokens=max_tokens, keys=keys,
                              target_keywords=target, override_contacts=override_contacts)

            # Update stored + editor text; require Save for export
            final_txt = sanitize_markdown(tailored)
            st.session_state["tailored_text"] = final_txt
            if "tailored_edit" not in st.session_state or not st.session_state["tailored_edit"]:
                st.session_state["tailored_edit"] = final_txt
            else:
                # refresh editor with new generation (so users can edit immediately)
                st.session_state["tailored_edit"] = final_txt
            st.session_state["tailored_saved"] = False

            st.success("Tailored resume generated. Review and click Save before exporting.")
        except Exception as e:
            st.error(str(e))

    # (2) Editable tailored text area + Save button; exports use saved text
    if "tailored_edit" not in st.session_state:
        st.session_state["tailored_edit"] = st.session_state.get("tailored_text", "")

    edited_text = st.text_area("Tailored resume (plain text)", key="tailored_edit", height=420)

    if st.button("💾 Save"):
        st.session_state["tailored_text"] = (edited_text or "").strip()
        st.session_state["tailored_saved"] = True
        st.success("Saved. Exports will use your edited text.")
    # # --- ATS Scan on the edited (final) resume text ---
    # st.divider()
    # st.subheader("ATS Scan (Keyword Coverage vs Final Resume)")

    # use_ai_ats = st.checkbox("Use AI-powered ATS (LLM)", value=True, help="Uncheck to run fast local rule-based scan.")

    # if st.button("Run ATS analysis on edited resume"):
    #     final_text = (st.session_state.get("tailored_edit", "") or "").strip()
    #     if not final_text:
    #         st.warning("Please add or generate resume content first.")
    #     else:
    #         if use_ai_ats:
    #             ats_llm = extract_ats_llm(final_text, jd_text, provider_pref=provider, model_name=(model or None),
    #                                     temperature=temperature, max_tokens=max_tokens, keys=keys)
    #             st.session_state["final_ats_llm"] = ats_llm
    #             st.session_state["final_ats_report"] = None  # clear rule-based
    #         else:
    #             final_ats = analyze(final_text, jd_text)  # local, rule-based
    #             st.session_state["final_ats_report"] = final_ats
    #             st.session_state["final_ats_llm"] = None    # clear AI

    # # Display results
    # ats_llm = st.session_state.get("final_ats_llm")
    # final_ats = st.session_state.get("final_ats_report")

    # if use_ai_ats and ats_llm:
    #     c1, c2 = st.columns([1,2])
    #     with c1:
    #         st.metric("AI ATS Score", f"{ats_llm.get('score',0)}%")
    #     with c2:
    #         st.write("Suggestions:")
    #         for s in (ats_llm.get("suggestions") or []):
    #             st.write(f"• {s}")
    #     st.write("Strengths:")
    #     for s in (ats_llm.get("strengths") or []):
    #         st.write(f"• {s}")
    #     st.write("Gaps:")
    #     for g in (ats_llm.get("gaps") or []):
    #         st.write(f"• {g}")
    #     st.write("Missing keywords:", ", ".join(ats_llm.get("missing_keywords") or []) or "—")

    # elif (not use_ai_ats) and final_ats:
    #     c1, c2, c3 = st.columns([1, 1, 2])
    #     with c1:
    #         st.metric("Match Score (final)", f"{final_ats['match']['match_score']}%")
    #     with c2:
    #         st.metric("Readability (FRE)", final_ats["readability"]["flesch_reading_ease"])
    #     with c3:
    #         st.write("ATS warnings:")
    #         for w in final_ats["ats"]["warnings"]:
    #             st.write(f"• {w}")
    #     try:
    #         missing = [w for (w, _) in (final_ats["keywords_bow"]["missing"] or [])]
    #     except Exception:
    #         missing = []
    #     st.write("Top missing keywords:", ", ".join(missing[:20]) if missing else "—")
    # --- ATS Scan on the edited (final) resume text ---
    st.divider()
    st.subheader("ATS Scan (Keyword Coverage vs Final Resume)")

    use_ai_ats = st.checkbox("Use AI-powered ATS (LLM)", value=True,
                            help="Uncheck to run fast local rule-based scan against the JD.")

    if st.button("Run ATS analysis on edited resume"):
        final_text = (st.session_state.get("tailored_edit", "") or "").strip()
        kw_obj = st.session_state.get("kw_llm") or {}

        if not final_text:
            st.warning("Please add or generate resume content first.")
        else:
            if use_ai_ats:
                # Build keyword list from Optimizer output: ranked terms + gaps (missing), then dedupe
                token_re = re.compile(r"[A-Za-z0-9#+.]+")
                def _canon(s: str) -> str:
                    return " ".join(t.lower() for t in token_re.findall(s or ""))

                ranked = []
                for item in (kw_obj.get("keywords") or []):
                    term = (item.get("term") or "").strip()
                    if term:
                        ranked.append(term)

                gaps = list(kw_obj.get("missing") or [])

                seen = set()
                target_keywords = []
                for t in (ranked + gaps):
                    c = _canon(t)
                    if c and c not in seen:
                        seen.add(c)
                        target_keywords.append(t)

                if not target_keywords:
                    st.warning("No optimizer keywords available. Please run the LLM Keyword Optimizer first.")
                else:
                    # IMPORTANT: pass the keyword list (NOT jd_text) to the LLM ATS
                    ats_llm = extract_ats_llm(
                        final_text,
                        target_keywords,
                        provider_pref=provider,
                        model_name=(model or None),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        keys=keys
                    )
                    st.session_state["final_ats_llm"] = ats_llm
                    st.session_state["final_ats_report"] = None  # clear rule-based

            else:
                # Local rule-based ATS vs JD (kept as optional fallback)
                final_ats = analyze(final_text, jd_text)
                st.session_state["final_ats_report"] = final_ats
                st.session_state["final_ats_llm"] = None

    # Display results
    ats_llm = st.session_state.get("final_ats_llm")
    final_ats = st.session_state.get("final_ats_report")

    if use_ai_ats and ats_llm:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("AI ATS Keyword Score", f"{ats_llm.get('score', 0)}%")
        with c2:
            st.write("Suggestions:")
            for s in (ats_llm.get("suggestions") or []):
                st.write(f"• {s}")

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
            for row in cov[:10]:
                t = row.get("term", "")
                p = "✅" if row.get("present") else "❌"
                ev = row.get("evidence", "")
                st.write(f"{p} {t}: {ev}")

    elif (not use_ai_ats) and final_ats:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.metric("Match Score (final)", f"{final_ats['match']['match_score']}%")
        with c2:
            st.metric("Readability (FRE)", final_ats["readability"]["flesch_reading_ease"])
        with c3:
            st.write("ATS warnings:")
            for w in final_ats["ats"]["warnings"]:
                st.write(f"• {w}")
        try:
            missing = [w for (w, _) in (final_ats["keywords_bow"]["missing"] or [])]
        except Exception:
            missing = []
        st.write("Top missing keywords:", ", ".join(missing[:20]) if missing else "—")



    colx1, colx2 = st.columns(2)
    with colx1:
        saved_text = (st.session_state.get("tailored_text", "") or "").strip()
        if st.session_state.get("tailored_saved") and saved_text and st.button("⬇️ Export DOCX"):
            out_path = export_docx(saved_text, "tailored_resume.docx")
            with open(out_path, "rb") as f:
                st.download_button("Download DOCX", f, file_name="tailored_resume.docx")
        elif not st.session_state.get("tailored_saved"):
            st.info("Edit the text and click Save before exporting DOCX.")

    with colx2:
        saved_text = (st.session_state.get("tailored_text", "") or "").strip()
        if st.session_state.get("tailored_saved") and saved_text and st.button("⬇️ Export PDF (professional)"):
            out_path = export_pdf(saved_text, "tailored_resume.pdf")
            with open(out_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="tailored_resume.pdf")
        elif not st.session_state.get("tailored_saved"):
            st.info("Edit the text and click Save before exporting PDF.")
else:
    st.info("Upload or paste both the Resume and the Job Description to begin.")
