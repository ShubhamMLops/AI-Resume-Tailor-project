# 🚀 AI Resume Tailor (v8 Final)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An intelligent **Streamlit** application that analyzes and tailors resumes against job descriptions using powerful AI models.
This tool ensures your resume is **ATS-friendly**, **keyword-optimized**, and **professionally exportable** in both DOCX and PDF formats.

---

## ✨ Why This Project?

Tailoring a resume for every job can be repetitive and time-consuming.
This tool automates the process by combining **LLM intelligence** with ATS checks, ensuring your resume is always aligned with the target job description — **with Core Competencies enhanced automatically**.

---

## 📋 Key Features

### 🔬 Analysis & Insights

* **Contact Extraction**: Auto-detects name, email, phone, LinkedIn, GitHub (regex + AI refinement).
* **ATS Score**: AI-powered keyword coverage vs final resume.
* **Readability (FRE)**: Evaluates readability using the Flesch Reading Ease score.
* **Warnings & Gaps**: Flags issues like short resumes, first-person pronouns, or missing keywords.
* **Domain-Aware Keyword Extraction**: Detects role/domain, then extracts **tools, frameworks, and languages** tailored to that job.

### 🧩 Keyword Sentence Generator (ATS-Friendly)

* Generates **Core Competencies–only** keyword sentences.
* Uses **Top Keywords (ranked) + Gaps**, compares with resume, and generates only new/needed bullets.
* Sentences are **technically accurate**, **contextually relevant**, and **ATS-optimized**.
* Fully editable before integration, with a **Save** option.

### 🤖 AI-Powered Tailoring

* **Multiple Backends**: Works with **Google Gemini**, **OpenAI GPT**, and **Anthropic Claude**.
* **Resume Integration**: Blends your resume + saved keyword sentences seamlessly, avoiding duplication.
* **Strict Expert Guidance**: AI behaves like a professional resume writer, ensuring output looks **native, structured, and polished**.

### 📄 Professional Document Export

* **DOCX Export**: Clean, editable `.docx` with colored section headings.
* **PDF Export**: High-quality `.pdf` with consistent formatting.
* **Consistent Styling**: Headings in **#1F4E79** (deep blue), bullets as `•`.

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)
* **Backend**: Python
* **DOCX Generation**: `python-docx`
* **PDF Generation**: `reportlab`
* **AI Models**: `openai`, `google-generativeai`, `anthropic`
* **Text Parsing**: `pdfminer.six`, `PyPDF2`

---

## 🚀 Getting Started

### Prerequisites

* Python **3.9+**
* At least one API key (Gemini, OpenAI, or Anthropic)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ai-resume-tailor-v8-final.git
   cd ai-resume-tailor-v8-final
   ```

2. Create and activate a virtual environment:

   ```bash
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

5. Open your browser at [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Configuration

### 🔧 Parameters Explained

When running the AI models, two important parameters can be tuned in the sidebar:

#### **1. Temperature**
- Controls the **creativity vs. consistency** of AI responses.
- **Low values (0.0 – 0.3)** → More **deterministic and precise**. Best for extracting keywords, ATS analysis, or when you want repeatable results.
- **Medium values (0.4 – 0.7)** → Balanced. Useful for resume tailoring where some rewriting is needed but accuracy is still important.
- **High values (0.8 – 1.0)** → More **creative and varied** outputs. Can be useful for brainstorming summaries, but may introduce unnecessary fluff.

👉 **Recommendation:** Keep it between **0.2 – 0.5** for resume tasks.

---

#### **2. Max Tokens**
- Defines the **maximum length** of the AI’s response (in tokens).
- 1 token ≈ ¾ of a word (e.g., `256 tokens ≈ 190 words`).
- If too **low** → the output may get **cut off** mid-sentence or miss details.
- If too **high** → the model may generate overly long text and waste credits.

👉 **Recommendation:**
- **500–800 tokens** for keyword extraction & analysis.  
- **1000–1500 tokens** for full resume tailoring and document rewriting.  

---

⚡ **Quick Rule of Thumb:**
- **Temperature = controls style** (creative vs accurate).  
- **Max Tokens = controls size** (how much text you’ll get back).  


All configuration is handled in the **sidebar**:

* **API Keys**: Paste your key for OpenAI, Gemini, or Anthropic.
* **Provider & Model**: Choose your provider and optionally set a model name.
* **Parameters**: Adjust `temperature` and `max tokens` for generation.

---

## 🗺️ Project Structure

```
.
├── app.py              # Streamlit app (UI + workflow)
├── pipeline.py         # Core logic: analysis, tailoring, keyword generation
├── utils.py            # Export helpers for DOCX & PDF
├── ai/
│   ├── providers.py    # Wrapper classes for OpenAI, Gemini, Anthropic
│   ├── selector.py     # Provider selection logic
│   ├── matcher.py      # Keyword gap & ATS matching functions
│   └── prompts.py      # Centralized system & user prompts
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🤝 Contributing

Contributions are welcome!
To contribute:

1. Fork this repo
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
