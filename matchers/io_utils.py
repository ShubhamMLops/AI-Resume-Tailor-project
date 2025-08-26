import io, docx2txt
from pypdf import PdfReader

def read_txt(upload):
    return upload.read().decode("utf-8", errors="ignore")

def read_pdf(upload):
    data = io.BytesIO(upload.read())
    reader = PdfReader(data)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(upload):
    data = io.BytesIO(upload.read())
    # docx2txt expects a path, but can read BytesIO via temp write:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
        tmp.write(data.getvalue()); tmp.flush()
        return docx2txt.process(tmp.name)

def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if name.endswith(".docx"):
        return read_docx(uploaded_file)
    if name.endswith(".txt"):
        return read_txt(uploaded_file)
    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")
