import streamlit as st
from matchers.generate import generate_tailored_json
from matchers.formatters import build_docx, build_pdf
import numpy as np
from matchers.io_utils import read_file
from matchers.embed import embed_openai, embed_gemini
from matchers.analyze import chunk, score_similarity, keyword_coverage

st.set_page_config(page_title="Resume Matcher (Streamlit)", layout="wide")

st.title("Resume Matcher — Streamlit Edition")
st.caption("Compare your resume with a Job Description using OpenAI or Gemini embeddings. "
           "Keys are kept in memory only (Session State).")

# ---------------- Sidebar: Provider, keys, models ----------------
with st.sidebar:
    st.header("Model Provider")
    provider = st.selectbox("Choose a provider", ["OpenAI", "Google Gemini"])

    if provider == "OpenAI":
        openai_key = st.text_input("OpenAI API Key", type="password", help="Will not be stored.")
        emb_model = st.text_input("Embedding model", value="text-embedding-3-large")
        chat_model = st.text_input("(Optional) Chat model for advice", value="gpt-4o-mini")
    else:
        gemini_key = st.text_input("Google Gemini API Key", type="password", help="Will not be stored.")
        emb_model = st.text_input("Embedding model", value="text-embedding-004")
        chat_model = st.text_input("(Optional) Chat model for advice", value="gemini-1.5-flash")

    extra_vocab = st.text_area(
        "Extra keywords to check (comma separated)",
        value="",
        help="Add domain skills to check presence/absence."
    )
    extra_vocab = [w.strip() for w in extra_vocab.split(",") if w.strip()]

# ---------------- Inputs ----------------
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload your Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
    resume_text = ""
    if resume_file:
        try:
            resume_text = read_file(resume_file)
            st.success(f"Loaded resume: {len(resume_text)} characters")
        except Exception as e:
            st.error(f"Could not read file: {e}")

with col2:
    jd_text = st.text_area("Paste Job Description", height=300, placeholder="Paste the JD here...")

run = st.button("Analyze")

st.write("---")
st.header("AI Tailored Resume (to this JD)")

st.caption("Rewrites your resume to fit the JD, keeping only truthful info from your original resume. "
           "Outputs professional DOCX and PDF. No data is stored.")

gen_model = chat_model  # reuse sidebar chat model field
if provider == "OpenAI":
    st.caption("Uses your OpenAI chat model for rewriting (e.g., gpt-4o-mini).")
else:
    st.caption("Uses your Gemini chat model for rewriting (e.g., gemini-1.5-flash).")

colg1, colg2 = st.columns(2)
with colg1:
    add_skills_hint = st.text_input("(Optional) force-include these truthful keywords if present in your resume (comma-separated)", value="")
with colg2:
    bullets_per_role = st.slider("Max bullets per role", min_value=3, max_value=12, value=8)

go_generate = st.button("Generate Tailored Resume")

if go_generate:
    key_missing = (provider=="OpenAI" and not openai_key) or (provider!="OpenAI" and not gemini_key)
    if key_missing:
        st.error(f"{provider} API key required.")
        st.stop()
    if not resume_text or not jd_text:
        st.error("Please provide both a resume file and a job description.")
        st.stop()
    with st.spinner("Creating tailored resume..."):
        try:
            key = openai_key if provider=="OpenAI" else gemini_key
            data = generate_tailored_json(provider, gen_model, key, jd_text, resume_text)

            # Optional: lightly post-process skills / bullets size
            if add_skills_hint.strip():
                hints = [w.strip() for w in add_skills_hint.split(",") if w.strip()]
                skills = set(data.get("skills") or [])
                for h in hints:
                    if h.lower() not in [s.lower() for s in skills]:
                        # include only if keyword actually appears in original resume
                        if h.lower() in resume_text.lower():
                            skills.add(h)
                data["skills"] = sorted(skills, key=str.lower)

            # trim bullets per role
            for r in (data.get("experience") or []):
                r["bullets"] = (r.get("bullets") or [])[:bullets_per_role]

            # Build files
            docx_bytes = build_docx(data)
            pdf_bytes = build_pdf(data)

            st.success("Tailored resume generated!")
            st.download_button("Download DOCX", data=docx_bytes, file_name="Tailored_Resume.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            st.download_button("Download PDF", data=pdf_bytes, file_name="Tailored_Resume.pdf",
                               mime="application/pdf")

            # Preview (textual)
            with st.expander("Preview structured data (JSON)"):
                import json
                st.code(json.dumps(data, indent=2)[:5000])

        except Exception as e:
            st.error(f"Could not generate: {e}")


# ---------------- Compute ----------------
if run:
    if not resume_text or not jd_text:
        st.warning("Please provide both a resume file and a job description.")
        st.stop()

    # Build embeddings
    st.subheader("Results")
    with st.spinner("Embedding and scoring..."):
        r_chunks = chunk(resume_text, max_chars=5000)
        j_chunks = chunk(jd_text, max_chars=5000)

        if provider == "OpenAI":
            if not openai_key:
                st.error("OpenAI API key required.")
                st.stop()
            r_emb = np.mean(embed_openai(r_chunks, openai_key, model=emb_model), axis=0)
            j_emb = np.mean(embed_openai(j_chunks, openai_key, model=emb_model), axis=0)
        else:
            if not gemini_key:
                st.error("Gemini API key required.")
                st.stop()
            r_emb = np.mean(embed_gemini(r_chunks, gemini_key, model=emb_model), axis=0)
            j_emb = np.mean(embed_gemini(j_chunks, gemini_key, model=emb_model), axis=0)

        sim = score_similarity(r_emb, j_emb)

    st.metric("Overall Similarity", f"{sim} / 100")

    present, missing = keyword_coverage(resume_text, jd_text, extra_vocab=extra_vocab)
    st.write("### Keyword Coverage")
    colp, colm = st.columns(2)
    with colp:
        st.write("**Present in both**")
        if present:
            st.write(", ".join(present))
        else:
            st.write("_No overlaps from the hint list._")
    with colm:
        st.write("**Missing from resume (but in JD)**")
        if missing:
            st.write(", ".join(missing))
        else:
            st.write("_None from the hint list._")

    # Optional: lightweight LLM suggestion (if user provided a chat model)
    if chat_model:
        st.write("### Suggestions (LLM-generated)")
        try:
            if provider == "OpenAI":
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                prompt = (
                    "You are an expert resume coach. Given the Job Description and the "
                    "current resume text, list the top 5 specific, truthful edits the user "
                    "could make to better reflect the JD (no fabrications). "
                    f"\n\nJD:\n{jd_text}\n\nResume:\n{resume_text[:12000]}"
                )
                msg = client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role":"user", "content": prompt}],
                    temperature=0.2
                )
                st.write(msg.choices[0].message.content)
            else:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                model = genai.GenerativeModel(chat_model)
                prompt = (
                    "You are an expert resume coach. Given the Job Description and the "
                    "current resume text, list the top 5 specific, truthful edits the user "
                    "could make to better reflect the JD (no fabrications). "
                    f"\n\nJD:\n{jd_text}\n\nResume:\n{resume_text[:12000]}"
                )
                resp = model.generate_content(prompt)
                st.write(resp.text)
        except Exception as e:
            st.info(f"(Skipping suggestions) {e}")

    # Downloads
    st.write("### Export")
    import json
    export = {
        "similarity": sim,
        "present_keywords": present,
        "missing_keywords": missing,
    }
    st.download_button("Download JSON report", data=json.dumps(export, indent=2),
                       file_name="match_report.json", mime="application/json")
