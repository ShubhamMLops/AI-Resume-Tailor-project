import json
from typing import Dict, Any

TEMPLATE_INSTRUCTIONS = """
You are an expert resume writer. Rewrite the user's resume to best match the Job Description (JD),
but DO NOT invent experience, companies, dates, titles, or metrics that are not already present.
You may rephrase, reorder, condense, and highlight relevant content. You may merge similar bullet
points and surface matching keywords only when they are truly supported by the user's resume.

Return STRICT JSON with this schema:
{
  "name": "Full Name (from resume, if present)",
  "contact": {
    "email": "...",
    "phone": "...",
    "location": "...",
    "links": ["...","..."]
  },
  "summary": "2-4 line professional summary tailored to JD, truthful",
  "skills": ["keyword1", "keyword2", "..."],
  "experience": [
    {
      "company": "...",
      "title": "...",
      "location": "...",
      "start": "MMM YYYY",
      "end": "MMM YYYY or Present",
      "bullets": [
        "impactful bullet #1 aligned to JD, truthful",
        "impactful bullet #2 ... (use action verb + scope + tools + outcome)",
        "..."
      ]
    }
  ],
  "projects": [
    {
      "name": "...",
      "role": "...",
      "bullets": ["...", "..."]
    }
  ],
  "education": [
    {
      "school": "...",
      "degree": "...",
      "year": "YYYY",
      "details": ["optional bullet", "optional bullet"]
    }
  ],
  "certifications": ["optional", "array"],
  "extras": {
    "awards": ["optional","array"],
    "volunteering": ["optional","array"]
  }
}

Rules:
- Only include items supported by the original resume.
- Prefer concise, results-oriented bullets (use metrics ONLY if present).
- Max 6-10 bullets per role, 4-8 bullets per project.
- Keep skills list de-duplicated and JD-aligned where truthful.
- Dates must match original resume.
"""

def _to_system_prompt(jd_text: str, resume_text: str) -> str:
    return (
        TEMPLATE_INSTRUCTIONS +
        "\n\n---\nJOB DESCRIPTION (verbatim):\n" + jd_text[:20000] +
        "\n\n---\nORIGINAL RESUME (verbatim):\n" + resume_text[:20000]
    )

def _parse_json(s: str) -> Dict[str, Any]:
    # attempt strict parse, else fallback from fenced code blocks
    try:
        return json.loads(s)
    except Exception:
        # extract first {...} block
        import re
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            return json.loads(m.group(0))
        raise

def generate_tailored_json(provider: str, chat_model: str, api_key: str, jd_text: str, resume_text: str) -> Dict[str, Any]:
    prompt = _to_system_prompt(jd_text, resume_text)

    if provider.lower().startswith("openai"):
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=chat_model,
            temperature=0.2,
            messages=[
                {"role":"system","content":"You return only JSON unless asked otherwise."},
                {"role":"user","content":prompt}
            ]
        )
        content = resp.choices[0].message.content
        return _parse_json(content)

    else:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(chat_model)
        resp = model.generate_content(prompt, generation_config={"temperature":0.2})
        return _parse_json(resp.text)
