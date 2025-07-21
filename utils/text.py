from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_files(files):
    text = ""

    for file in files:
        filename = file.name.lower()

        if filename.endswith(".pdf"):
            pdf = PdfReader(file)
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content

        elif filename.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif filename.endswith(".txt"):
            text += file.read().decode("utf-8")  # para garantir compatibilidade

    return text if text.strip() else None


def create_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return text_splitter.split_text(text)
