from pathlib import Path
from pypdf import PdfReader
from docx import Document
import csv


# Supported Document Extensions
supported_extensions = {
    ".pdf",
    ".docx",
    ".csv",
    ".txt"
}


def extract_pdf(file_path):
    """Extract text from a pdf document"""
    reader = PdfReader(file_path)
    pages = []
    
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "page_number": page_number,
                "text": text
            })
    return pages


def extract_docx(file_path):
    """Extract text from a text word document"""
    document = Document(file_path)
    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )
    return [{
        "page_number": None,
        "text": text
    }]


def extract_txt(file_path):
    """Extract text from every plain text"""
    text = Path(file_path).read_text(encoding="utf-8")
    return [{
        "page_number": None,
        "text": text
    }]


def extract_csv(file_path):
    """Extract structured csv records as text"""
    records = []
    with open(file_path, "r", encoding="utf-8", newline="")as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=1):
            record_text = "\n".join({
                f"{key}: {value}"
                for key, value in row.items()
            })
            records.append({
                "page_number": None,
                "text": f"Record{row_number} \n {record_text}"
            })
    return records


def load_document(file_path):
    """Identify a document type and extracts its contents"""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        extracted_content = extract_pdf(file_path)
    elif extension == ".docx":
        extracted_content = extract_docx(file_path)
    elif extension == ".txt":
        extracted_content = extract_txt(file_path)
    elif extension == ".csv":
        extracted_content = extract_csv(file_path)
    else:
        raise ValueError(f"Unsupported document type: {extension}")
    return extracted_content


def discover_documents(directory):
    """Recursively discover supported documents"""
    directory = Path(directory)
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            yield file_path       
