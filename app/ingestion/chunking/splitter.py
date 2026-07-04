import logfire
from typing import List

def chunk_text(text:str, chunk_size:int = 1500) -> List[str]:
    with logfire.span("Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

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

        logfire.info(f"Generated {len(chunks)} chunks")
        return chunks
