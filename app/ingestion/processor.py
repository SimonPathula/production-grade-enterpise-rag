import os
import sys
import uuid
import json
import time
import tempfile
import logfire
import vertexai

from typing import List
from fastapi import FastAPI, Request, BackgroundTasks
from google.cloud import storage
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.office import parse_office
from app.ingestion.loaders.text import parse_text
from app.services.retrieval.embedding import embed_texts
from app.ingestion.chunking.splitter import chunk_text

# === Client Configuration ====

logfire.configure(service_name="enterprise-ingestion-service")
vertexai.init(project= settings.PROJECT_ID, location= settings.LOCATION)
storage_client = storage.Client(project=settings.PROJECT_ID)
qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout= 60)

# FastAPI
app = FastAPI(title="RAG Ingestion Service")

TOTAL_CHUNKS = 0
TOTAL_EMBEDDINGS = 0

@app.get("/")
def health():
    return {"status": "ok", "service": "RAG Ingestion", "mode": "cloud"}

def upload_to_gcs(data, bucket_name:str, destination:str, is_json: bool = False):
    with logfire.span("Google Cloud Storage Upload", bucket=bucket_name, blob=destination):
        try:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination)

            if is_json:
                blob.upload_from_string(json.dumps(data), content_type="application/json")
            else:
                blob.upload_from_filename(data)

            logfire.info(f"Uploaded to {bucket_name}/{destination}")
        except Exception as e:
            logfire.error(f"GCS Upload failed: {e}")
            raise e

def upsert_in_batches(points, batch_size=64, retries=3):
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        for attempt in range(retries):
            try:
                qdrant_client.upsert(collection_name=settings.QDRANT_COLLECTION, points=batch, wait=True)
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                logfire.warning(f"Upsert batch failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)

def process_file(file_path:str, filename:str, source_type:str, skip_raw_upload: bool = False):
    with logfire.span("Processing File", file_path= file_path, filename= filename, source= source_type, cloud_mode=skip_raw_upload):
        try:
            raw_gcs_path = f"{source_type}/{filename}"
            if not skip_raw_upload:
                upload_to_gcs(file_path, settings.RAW_BUCKET, raw_gcs_path)
            else:
                logfire.info(f"Skipping RAW upload (cloud mode) — file already at gs://{settings.RAW_BUCKET}/{raw_gcs_path}")

            ext = filename.lower().split(".")[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename}")
                return

            chunks = chunk_text(full_text)
            if not chunks:
                return

            global TOTAL_CHUNKS
            TOTAL_CHUNKS += len(chunks)

            processed_data = {"filename":filename, "chunks":chunks, "source_type":source_type}
            processed_gcs_path = f"{source_type}/{filename}.json"
            upload_to_gcs(processed_data, settings.PROCESSED_BUCKET, processed_gcs_path, is_json=True)

            with logfire.span("Vectorizing and Indexing"):
                embeddings = embed_texts(chunks)
                global TOTAL_EMBEDDINGS
                TOTAL_EMBEDDINGS += len(embeddings)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                            "raw_gcs_path": f"gs://{settings.RAW_BUCKET}/{raw_gcs_path}",
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]
                upsert_in_batches(points)
                logfire.info(f"Indexed {len(points)} points to Qdrant from '{filename}'")   

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")



def universal_ingestion(base_dir: str, explicit_source_type: str | None, wipe: bool = False):
    with logfire.span("Universal ingestion started", base_directory= base_dir):
        if wipe:
            with logfire.span("Wiping collection"):
                print("URL:", settings.QDRANT_URL)
                print("Collection:", settings.QDRANT_COLLECTION)
                print("Exists:", qdrant_client.collection_exists(settings.QDRANT_COLLECTION))
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection {settings.QDRANT_COLLECTION} deleted")

        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=768,   # e.g. 768 for text-embedding-004, 1536 for OpenAI ada
                    distance=models.Distance.COSINE,
                ),
            )

        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name   = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = "true" if "true" in base_name else "noisy" if "noisy" in base_name else "general"
            logfire.info(f"No subdirectories — processing {base_dir} as '{source_type}'")
            _process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = "true" if "true" in subdir.lower() else "noisy" if "noisy" in subdir.lower() else subdir
                _process_directory(os.path.join(base_dir, subdir), source_type)

def _process_directory(dir_path: str, source_type: str):
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"🔍 Found {len(files)} files")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type, skip_raw_upload=False)

if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv
    clean_args     = [a for a in sys.argv if a != "--wipe"]
    target_dir     = clean_args[1] if len(clean_args) > 1 else "data"
    explicit_type  = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: Path {target_dir} does not exist.")
        sys.exit(1)

    universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    print(f"Total chunks created: {TOTAL_CHUNKS}")
    print(f"Total embeddings generated: {TOTAL_EMBEDDINGS}")
    logfire.info(
    f"Universal Ingestion Job Completed | "
    f"Chunks: {TOTAL_CHUNKS} | Embeddings: {TOTAL_EMBEDDINGS}"
    )