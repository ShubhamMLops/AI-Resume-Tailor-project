import io, os, json, re
import streamlit as st
from utils import export_docx, export_pdf
from pipeline import (
    analyze,
    tailor,
    extract_keywords_llm,
    extract_contacts_llm,
    sanitize_markdown,
    extract_ats_llm_from_optimizer,  # AI ATS helper
    generate_keyword_sentences
)

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
    max_tokens = st.slider("Max tokens", 256, 4096, 1500, 64)

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

    if st.button("Enhance contacts with AI"):
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

    # === ADDED: helper to build target keywords from Top Keywords (ranked) + Gaps, deduped, and excluding ones already in resume ===
    def build_target_keywords_from_optimizer(kw_obj, resume_text: str, jd_text: str):
        """
        Build final target keywords from:
        - Top Keywords (ranked) + ALL their variants (outlets)
        - Gaps (missing) or fallback
        Then:
        - Dedupe canonically (punctuation/case insensitive)
        - EXCLUDE anything already present in the ORIGINAL resume
            (match against term OR any variant; handles 'CI/CD' ~ 'cicd' and simple plural/singular)
        """
        if not kw_obj:
            return []

        token_re = re.compile(r"[A-Za-z0-9#+.]+")

        def _canon(s: str) -> str:
            return " ".join(t.lower() for t in token_re.findall(s or ""))

        def _tok_seq(s: str):
            return [t.lower() for t in token_re.findall(s or "")]

        # Resume tokens + compacted view (for CI/CD ~ cicd)
        resume_tokens = _tok_seq(resume_text)
        resume_compact = "".join(resume_tokens)

        def _present(term: str) -> bool:
            kt = _tok_seq(term)
            if not kt:
                return False
            L = len(kt)

            # A) exact token-sequence
            for i in range(0, len(resume_tokens) - L + 1):
                if resume_tokens[i:i+L] == kt:
                    return True

            # B) compacted (handles CI/CD -> cicd)
            k_comp = "".join(kt)
            if k_comp and k_comp in resume_compact:
                return True

            # C) simple plural/singular toggle (avoid breaking short acronyms)
            if L == 1 and len(kt[0]) > 3:
                base = kt[0]
                alt = base[:-1] if base.endswith("s") else base + "s"
                if base in resume_tokens or alt in resume_tokens:
                    return True

            return False

        # (1) Collect ALL outlets: each Top Keyword (ranked) followed by its variants
        outlets_ordered = []
        for item in (kw_obj.get("keywords") or []):
            base = (item.get("term") or "").strip()
            if not base:
                continue
            variants = [v.strip() for v in (item.get("variants") or []) if v and v.strip()]
            outlets_ordered.append(base)           # base first
            outlets_ordered.extend(variants)       # then all variants

        # (2) Append Gaps (or fallback if empty)
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

        # (3) Dedupe by canonical form while preserving order
        seen, deduped = set(), []
        for t in outlets_ordered:
            c = _canon(t)
            if c and c not in seen:
                seen.add(c)
                deduped.append(t)

        # (4) Remove anything already present in the original resume
        target = [t for t in deduped if not _present(t)]
        return target
    # === END REPLACE ===

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
                if not missing or not isinstance(missing, list): missing = fallback_missing
                if not weak or not isinstance(weak, list): weak = fallback_weak
            st.write("Missing:", ", ".join(missing) if missing else "—")
            st.write("Weak:", ", ".join(weak) if weak else "—")
    # -----------------
    # Generate ATS-friendly keyword sentences (from Top Keywords (ranked) + Gaps)
    # -----------------
    st.subheader("Keyword Sentence Generator (ATS-friendly)")
    st.caption("Reads your resume context and creates concise bullets that naturally integrate Top Keywords (ranked) + Gaps.")

    if st.button("Generate ATS-friendly keyword sentences"):
        kw_obj_now = st.session_state.get("kw_llm")
        if not kw_obj_now:
            st.warning("Please run the LLM Keyword Optimizer first.")
        else:
            # Build final target list (variants-aware, deduped, skip already-present)
            target_for_sentences = build_target_keywords_from_optimizer(kw_obj_now, resume_text, jd_text)

            if not target_for_sentences:
                st.info("All optimizer keywords already appear in the resume. Nothing to add.")
            else:
                bullets = generate_keyword_sentences(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    target_keywords=target_for_sentences,
                    provider_pref=provider,
                    model_name=(model or None),
                    temperature=temperature,
                    max_tokens=min(max_tokens, 900),
                    keys=keys
                )
                if bullets:
                    # Append a small section into the editor for user review
                    block = "\n".join([
                        "",
                        "Core Competencies",  # safe default section
                        bullets.strip(),
                        ""
                    ])
                    existing = st.session_state.get("tailored_edit", "") or st.session_state.get("tailored_text", "")
                    st.session_state["tailored_edit"] = (existing + "\n" + block).strip()
                    st.session_state["tailored_saved"] = False
                    st.success("Generated ATS-friendly keyword sentences and added them to the editor.")
                else:
                    st.info("No sentences were generated.")

    # -----------------
    # Tailor with LLM (Integrated with Keyword Sentence Generator) — no duplicates
    # -----------------
    st.divider()
    st.subheader("Tailor with LLM (API-only)")
    st.caption("Uses ONLY Top Keywords (ranked) + Gaps from the LLM Keyword Optimizer; tailors first, then fills ONLY still-missing terms with ATS-friendly bullets (no duplicates).")

    # Enable only if the optimizer has Top Keywords (ranked) or Gaps
    can_generate = bool(st.session_state.get("kw_llm", {}).get("keywords") or st.session_state.get("kw_llm", {}).get("missing"))
    if st.button("Generate tailored resume", type="primary", disabled=not can_generate):
        try:
            target = []
            kw_obj_now = st.session_state.get("kw_llm")  # latest optimizer output

            if kw_obj_now:
                # Step 1: Build target keywords from Top Keywords (ranked) + Gaps,
                #         deduped & excluding anything already present in the ORIGINAL resume
                target = build_target_keywords_from_optimizer(kw_obj_now, resume_text, jd_text)

            override_contacts = st.session_state.get("override_contacts")

            # Step 2: Tailor with ONLY the filtered target keywords
            tailored = tailor(
                resume_text,
                jd_text,
                provider_preference=provider,
                model_name=(model or None),
                temperature=temperature,
                max_tokens=max_tokens,
                keys=keys,
                target_keywords=target,
                override_contacts=override_contacts
            )
            tailored_txt = sanitize_markdown(tailored)

            # Step 3: Determine which optimizer keywords are STILL missing in the tailored text
            token_re = re.compile(r"[A-Za-z0-9#+.]+")
            def _canon(s: str) -> str:
                return " ".join(t.lower() for t in token_re.findall(s or ""))

            def _tok_seq(s: str):
                return [t.lower() for t in token_re.findall(s or "")]

            # Build ordered list: Top Keywords (ranked) + Gaps (deduped)
            ranked = [(it.get("term") or "").strip() for it in (kw_obj_now.get("keywords") or []) if (it.get("term") or "").strip()]
            gaps   = list(kw_obj_now.get("missing") or [])
            seen_terms, ordered_terms = set(), []
            for t in (ranked + gaps):
                c = _canon(t)
                if c and c not in seen_terms:
                    seen_terms.add(c); ordered_terms.append(t)

            # Variants map from optimizer
            variants_map = {}
            for it in (kw_obj_now.get("keywords") or []):
                base = (it.get("term") or "").strip()
                if base:
                    variants_map[_canon(base)] = [v.strip() for v in (it.get("variants") or []) if v and v.strip()]

            # Tokenize tailored text & compact form for CI/CD-like cases
            t_tokens   = [t.lower() for t in token_re.findall(tailored_txt)]
            t_compact  = "".join(t_tokens)

            def _present_in_tailored(term: str) -> bool:
                cand_list = [term] + (variants_map.get(_canon(term), []))
                for cand in cand_list:
                    kt = _tok_seq(cand)
                    if not kt:
                        continue
                    L = len(kt)

                    # A) exact token sequence
                    for i in range(0, len(t_tokens) - L + 1):
                        if t_tokens[i:i+L] == kt:
                            return True

                    # B) compacted (CI/CD -> cicd)
                    k_comp = "".join(kt)
                    if k_comp and k_comp in t_compact:
                        return True

                    # C) simple plural/singular toggle for single-token words (avoid short acronyms)
                    if L == 1 and len(kt[0]) > 3:
                        base = kt[0]
                        alt = base[:-1] if base.endswith("s") else base + "s"
                        if base in t_tokens or alt in t_tokens:
                            return True
                return False

            missing_after_tailor = [term for term in ordered_terms if not _present_in_tailored(term)]

            # Step 4: If anything is still missing, generate ATS-friendly bullets ONLY for those missing terms
            appended_block = ""
            if missing_after_tailor:
                bullets = generate_keyword_sentences(
                    resume_text=resume_text,       # read full resume for context; do NOT invent
                    jd_text=jd_text,
                    target_keywords=missing_after_tailor,
                    provider_pref=provider,
                    model_name=(model or None),
                    temperature=temperature,
                    max_tokens=min(max_tokens, 900),
                    keys=keys
                )
                if bullets and bullets.strip():
                    appended_block = "\n".join([
                        "",
                        "Core Competencies",  # neutral/safe section for unevidenced-but-required terms
                        bullets.strip(),
                        ""
                    ])

            # Step 5: Combine tailored text + optional bullets, load into editor (unsaved)
            final_txt = (tailored_txt + appended_block).strip()
            st.session_state["tailored_text"]  = final_txt
            st.session_state["tailored_edit"]  = final_txt
            st.session_state["tailored_saved"] = False

            # UX feedback
            if missing_after_tailor and appended_block:
                st.success(f"Tailored resume generated. Added ATS-friendly bullets for {len(missing_after_tailor)} still-missing keywords. Review, then click Save before exporting.")
            else:
                st.success("Tailored resume generated. All optimizer keywords are covered. Review, then click Save before exporting.")

        except Exception as e:
            st.error(str(e))


    # Editor + Save
    if "tailored_edit" not in st.session_state:
        st.session_state["tailored_edit"] = st.session_state.get("tailored_text", "")
    edited_text = st.text_area("Tailored resume (plain text)", key="tailored_edit", height=420)

    if st.button("💾 Save"):
        st.session_state["tailored_text"] = (edited_text or "").strip()
        st.session_state["tailored_saved"] = True
        st.success("Saved. Exports will use your edited text.")

    # -----------------
    # ATS Scan (AI) + local reconciliation
    # -----------------
    st.divider()
    st.subheader("ATS Scan (Keyword Coverage vs Final Resume)")

    if st.button("Run ATS analysis on edited resume"):
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

            # Reconcile deterministically (token/variant aware + compacted match)
            token_re = re.compile(r"[A-Za-z0-9#+.]+")
            def _canon(s: str) -> str:
                return " ".join(t.lower() for t in token_re.findall(s or ""))

            ranked = [(it.get("term") or "").strip() for it in (kw_obj_now.get("keywords") or []) if (it.get("term") or "").strip()]
            gaps = list(kw_obj_now.get("missing") or [])

            seen_terms, ordered_terms = set(), []
            for t in (ranked + gaps):
                c = _canon(t)
                if c and c not in seen_terms:
                    seen_terms.add(c); ordered_terms.append(t)

            # Variants map
            variants_map = {}
            for it in (kw_obj_now.get("keywords") or []):
                term = (it.get("term") or "").strip()
                if term:
                    variants_map[_canon(term)] = [v.strip() for v in (it.get("variants") or []) if v and v.strip()]

            final_tokens = [t.lower() for t in token_re.findall(final_text)]
            final_compact = "".join(final_tokens)

            def _tok_seq(s: str):
                return [t.lower() for t in token_re.findall(s or "")]

            present, missing, coverage = [], [], []
            for term in ordered_terms:
                cand_list = [term] + (variants_map.get(_canon(term), []))
                found = False
                found_pos = -1
                found_len = 0

                for cand in cand_list:
                    tt = _tok_seq(cand)
                    if not tt:
                        continue

                    # A) exact token-sequence match
                    L = len(tt)
                    for i in range(0, len(final_tokens) - L + 1):
                        if final_tokens[i:i+L] == tt:
                            found, found_pos, found_len = True, i, L
                            break
                    if found:
                        break

                    # B) compacted form (CI/CD -> cicd)
                    t_comp = "".join(tt)
                    if t_comp and t_comp in final_compact:
                        found, found_pos, found_len = True, -1, L
                        break

                    # C) single-word plural/singular toggle for simple drift (avoid short acronyms)
                    if L == 1 and len(tt[0]) > 3:
                        base = tt[0]
                        alt = base[:-1] if base.endswith("s") else base + "s"
                        if base in final_tokens or alt in final_tokens:
                            found, found_pos, found_len = True, -1, 1
                            break

                if found:
                    # evidence (best-effort)
                    if found_pos >= 0:
                        spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9#+.]+", final_text)]
                        s = spans[found_pos][1] if 0 <= found_pos < len(spans) else 0
                        e_idx = min(found_pos + max(1, found_len) - 1, len(spans) - 1)
                        e = spans[e_idx][2] if spans else s
                        s = max(0, s - 20); e = min(len(final_text), e + 20)
                        ev = final_text[s:e].replace("\n", " ").strip()
                    else:
                        ev = ""  # compact/toggled match; snippet not locatable
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

    # -----------------
    # Export (Saved text only)
    # -----------------
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
