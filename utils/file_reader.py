import pdfplumber


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def read_uploaded_file(uploaded_file) -> str:
    """
    Поддерживает .txt и .pdf.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        import io
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    else:
        return uploaded_file.read().decode("utf-8")
