from typing import List
from logging import log
import io
import os
from pypdf import PdfReader, PdfWriter
from google.cloud import documentai
from app.config import settings
from bs4 import BeautifulSoup
from unstructured.partition.auto import partition

# === TEXT PARSING ==============
def parse_text(file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            raise e

# === PDF PARSING ===============
client = documentai.DocumentProcessorServiceClient()
MAX_PAGES_PER_REQUEST = 15
def process_document_chunk(image_content: bytes, name: str) -> str:
    # raw document --> request --> process document --> result
    raw_document = documentai.RawDocument(
        content=image_content,
        mime_type="application/pdf"
    )

    request = documentai.ProcessRequest(
        name = name,
        raw_document = raw_document
    )

    result = client.process_document(request=request)

    return result.document.text
def parse_pdf(file_path:str):
        print(f"Parsing PDF, filename = {file_path}")
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            print(f"Total pages: {total_pages}")

            name = client.processor_path(
                settings.PROJECT_ID,
                settings.GCP_DOC_AI_LOCATION,
                settings.GCP_DOC_AI_PROCESSOR_ID
            )

            full_text = ""

            if total_pages <= MAX_PAGES_PER_REQUEST:
                with open(file_path, "rb") as f:
                    image_content = f.read() #gives the output in bytes
                full_text = process_document_chunk(image_content, name)
            else:
                print(f"PDF exceeded {MAX_PAGES_PER_REQUEST} pages. Splitting document chunks...")

                for i in range(0, total_pages, MAX_PAGES_PER_REQUEST):
                    writer = PdfWriter()
                    chunk_end = min(i + MAX_PAGES_PER_REQUEST, total_pages)

                    for page_num in range(i, chunk_end):
                        writer.add_page(reader.pages[page_num])

                    with io.BytesIO() as bytes_stream:
                        writer.write(bytes_stream)
                        chunk_bytes = bytes_stream.getvalue()

                        chunk_text = process_document_chunk(chunk_bytes, name)
                        full_text += chunk_text + "\n"

            if not full_text.strip():
                print(f"Document AI returned empty text for {file_path}")
            else:
                print(f"Document AI successfully parsed {len(full_text)} characters")

            return full_text

        except Exception as e:
            raise e

# === HTML PARSING ==============
def parse_html(file_path:str):
        print(f"HTML Parsing, filename = {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            soup = BeautifulSoup(content, "html.parser")

            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()

            text = soup.get_text(separator= "\n")

            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_clean = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text_clean
        except Exception as e:
            raise e

# === OFFICE PARSING ===========
def parse_office(file_path:str):
    try:
        elements = partition(filename = file_path)
        full_text = "\n".join([str(e) for e in elements])
        return full_text
    except Exception as e:
        raise e

# === Chunking =================
def chunk_text(text:str, chunk_size:int = 1500):

        paragraphs  = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            # Handle paragraphs larger than chunk_size
            if len(p) > chunk_size:
                # Save the current chunk first
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split the large paragraph
                for i in range(0, len(p), chunk_size):
                    chunks.append(p[i:i + chunk_size])

                continue

            # Check if paragraph fits in current chunk
            if not current_chunk:
                current_chunk = p
            elif len(current_chunk) + len("\n\n") + len(p) <= chunk_size:
                current_chunk += "\n\n" + p
            else:
                chunks.append(current_chunk)
                current_chunk = p

        # Append the remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        chunks_num = len(chunks)

        return chunks_num

def count_chunks(data: str, count: int = 0) -> int:
    """
    Traverse the `data` folder (expects subfolders like 'noisy' and 'true'),
    parse each supported file, chunk it, and return the total chunk count.
    """
    for root, _, files in os.walk(data):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = filename.lower().split(".")[-1]

            try:
                if ext == "pdf":
                    full_text = parse_pdf(file_path)
                elif ext in ("html", "htm"):
                    full_text = parse_html(file_path)
                elif ext == "txt":
                    full_text = parse_text(file_path)
                elif ext in ("docx", "pptx"):
                    full_text = parse_office(file_path)
                else:
                    print(f"Skipping unsupported file type: {filename}")
                    continue

                if not full_text or not full_text.strip():
                    print(f"No text extracted from {filename}")
                    continue

                num_chunks = chunk_text(full_text)
                print(f"{file_path}: {num_chunks} chunks")
                count += num_chunks

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    return count

total_count = count_chunks(r"D:\projects\enterpriserag\data")
print(f"Total chunks: {total_count}")