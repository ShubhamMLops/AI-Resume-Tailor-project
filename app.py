import io, os, json
import streamlit as st
from pipeline import analyze, tailor, extract_keywords_llm, extract_contacts_llm, sanitize_markdown
from utils import export_docx, export_pdf

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
        # Try to reuse analysis BOW gaps (computed earlier)
        # If you don't have 'report' in scope here, move this function above where 'report' is defined, or pass the gaps in.
        try:
            from ai.matcher import re as _re  # not used, but ensures import path correct
        except Exception:
            pass

        # 1) Missing fallback via analyze()’s BOW gaps:
        # 'report' is defined earlier in this script when we ran analyze(resume_text, jd_text)
        missing_terms = []
        try:
            bow_missing = [w for (w, _) in (report["keywords_bow"]["missing"] or [])]
            # De-duplicate + keep only meaningful tokens (length > 2)
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
            # normalize spaces
            text_low = " ".join(text_low.split())
            terms = [ (item.get("term") or "").strip() for item in (kw_obj.get("keywords") or []) ]
            seen2 = set()
            for t in terms:
                tl = t.lower()
                if not tl or tl in seen2:
                    continue
                seen2.add(tl)
                count = text_low.count(" " + tl + " ")  # crude but effective for single/multi-word terms
                if count == 1:
                    weak_terms.append(t)
            # cap
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
            # Fallback compute if fields are absent or empty
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
    auto_weave = st.checkbox("Use AI keywords automatically", value=True)
    custom_keywords = st.text_input("Optional: comma-separated custom keywords")

    if st.button("Generate tailored resume", type="primary"):
        try:
            target = []
            if auto_weave and kw_obj and kw_obj.get("keywords"):
                for item in kw_obj["keywords"][:15]:
                    term = (item.get("term") or "").strip()
                    if term and term.lower() not in [t.lower() for t in target]:
                        target.append(term)
            if custom_keywords.strip():
                for t in custom_keywords.split(","):
                    term = t.strip()
                    if term and term.lower() not in [x.lower() for x in target]:
                        target.append(term)

            override_contacts = st.session_state.get("override_contacts")
            tailored = tailor(resume_text, jd_text, provider_preference=provider, model_name=(model or None),
                              temperature=temperature, max_tokens=max_tokens, keys=keys,
                              target_keywords=target, override_contacts=override_contacts)
            st.session_state["tailored_text"] = sanitize_markdown(tailored)
            st.success("Tailored resume generated.")
        except Exception as e:
            st.error(str(e))

    tailored_text = st.session_state.get("tailored_text", "")
    st.text_area("Tailored resume (plain text)", value=tailored_text, height=420)

    colx1, colx2 = st.columns(2)
    with colx1:
        if tailored_text and st.button("⬇️ Export DOCX"):
            out_path = export_docx(tailored_text, "tailored_resume.docx")
            with open(out_path, "rb") as f:
                st.download_button("Download DOCX", f, file_name="tailored_resume.docx")
    with colx2:
        if tailored_text and st.button("⬇️ Export PDF (professional)"):
            out_path = export_pdf(tailored_text, "tailored_resume.pdf")
            with open(out_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="tailored_resume.pdf")
else:
    st.info("Upload or paste both the Resume and the Job Description to begin.")
