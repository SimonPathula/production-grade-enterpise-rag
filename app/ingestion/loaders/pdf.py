from logging import log
import io
import logfire
from pypdf import PdfReader, PdfWriter
from google.cloud import documentai
from app.config import settings

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
    with logfire.span("Parsing PDF", filename = file_path):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            logfire.info(f"Total pages: {total_pages}")

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
                logfire.info(f"PDF exceeded {MAX_PAGES_PER_REQUEST} pages. Splitting document chunks...")

                for i in range(0, total_pages, MAX_PAGES_PER_REQUEST):
                    writer = PdfWriter()
                    chunk_end = min(i + MAX_PAGES_PER_REQUEST, total_pages)

                    for page_num in range(i, chunk_end):
                        writer.add_page(reader.pages[page_num])

                    with io.BytesIO() as bytes_stream:
                        writer.write(bytes_stream)
                        chunk_bytes = bytes_stream.getvalue()

                    with logfire.span(f"Processing pages {i + 1} to {chunk_end}"):
                        chunk_text = process_document_chunk(chunk_bytes, name)
                        full_text += chunk_text + "\n"

            if not full_text.strip():
                logfire.error(f"Document AI returned empty text for {file_path}")
            else:
                logfire.info(f"Document AI successfully parsed {len(full_text)} characters")

            return full_text

        except Exception as e:
            logfire.error(f"Document AI parse Failed: {e}")
            logfire.info("💡 Ensure the Processor ID is correct and the API is enabled.")
            raise e