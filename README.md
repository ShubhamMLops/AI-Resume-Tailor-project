# 🚀 AI Resume Tailor (v8 Final)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An intelligent **Streamlit** application that analyzes and tailors resumes against job descriptions using powerful AI models.  
This tool ensures your resume is **ATS-friendly**, **keyword-optimized**, and **professionally exportable** in both DOCX and PDF formats.

<br>

*Screenshot: The analysis dashboard showing contacts, ATS checks, and tailoring options.*

---

## ✨ Why This Project?

Tailoring a resume for every job can be repetitive and time-consuming.  
This tool automates the process by combining **LLM intelligence** with ATS checks, ensuring your resume is always aligned with the target job description.

---

## 📋 Key Features

### 🔬 Analysis & Insights
- **Contact Extraction**: Auto-detects name, email, phone, LinkedIn, GitHub (regex + AI refinement).
- **ATS Score**: Quick compatibility check with Applicant Tracking Systems.
- **Readability (FRE)**: Evaluates readability using the Flesch Reading Ease score.
- **Warnings & Gaps**: Flags issues like short resumes, first-person pronouns, or missing keywords.

### 🤖 AI-Powered Tailoring
- **Multiple Backends**: Works with **Google Gemini**, **OpenAI GPT**, and **Anthropic Claude**.
- **Strict JSON Parsing**: Ensures clean structure with consistent sections.
- **Keyword Weaving**: Naturally integrates job description keywords into your resume sections.
- **Fallback to Plain-Text**: Always produces output, even if JSON parse fails.

### 📄 Professional Document Export
- **DOCX Export**: Clean, editable `.docx` with colored section headings.
- **PDF Export**: High-quality `.pdf` with the same professional formatting.
- **Consistent Styling**: Headings in **#1F4E79** (deep blue), bullets as `•`.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Backend**: Python  
- **DOCX Generation**: `python-docx`  
- **PDF Generation**: `reportlab`  
- **AI Models**: `openai`, `google-generativeai`, `anthropic`  
- **Text Parsing**: `pdfminer.six`, `PyPDF2`  

---

## 🚀 Getting Started

### Prerequisites
- Python **3.9+**
- At least one API key (Gemini, OpenAI, or Anthropic)

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

All configuration is handled in the **sidebar**:
- **API Keys**: Paste your key for OpenAI, Gemini, or Anthropic.
- **Provider & Model**: Choose your provider and optionally set a model name.
- **Parameters**: Adjust `temperature` and `max tokens`.

---

## 🗺️ Project Structure

```
.
├── app.py              # Streamlit app (UI + workflow)
├── pipeline.py         # Core logic: analysis, tailoring, JSON handling
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
