import os
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    PointStruct,
)
import ollama
import tiktoken
from pypdf import PdfReader  # For extracting text from PDF documents

# Configuration
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "documents_collection"
DOCS_DIR = "eval/sample_documents"  # Directory where your policy/payment plan files are stored

qdrant_client = QdrantClient(url=QDRANT_URL)

# Fetch embedding size dynamically
sample_emb = ollama.embed(model=EMBED_MODEL, input="test")["embeddings"][0]
vec_size = len(sample_emb)

# Ensure collection exists with HNSW & INT8 Scalar Quantization configured
qdrant_client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
    
    # Configure HNSW Index Parameters
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100,
    ),
    
    # Configure Scalar Quantization (INT8 - 4x RAM savings)
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    ),
)


def deterministic_uuid(doc_filename: str, chunk_index: int) -> str:
    """Generates a deterministic UUIDv5 string based on document name and chunk index.
    
    This ensures re-running the script updates existing points without creating duplicates.
    """
    seed_string = f"{doc_filename}-chunk-{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_string))


def chunk_text_by_tokens(
    text: str, max_tokens: int = 400, overlap: int = 40, encoding_name: str = "cl100k_base"
) -> list[str]:
    """Splits document text into chunks using exact token counts and sliding-window overlap."""
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)

    if not tokens:
        return []

    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))

        if i + max_tokens >= len(tokens):
            break
        i += max_tokens - overlap

    return chunks


def extract_text_from_file(file_path: str) -> str:
    """Extracts raw text content from PDF, HTML, or TXT files."""
    ext = Path(file_path).suffix.lower()
    
    if ext == ".pdf":
        reader = PdfReader(file_path)
        text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n\n".join(text_pages)
        
    elif ext in [".html", ".htm", ".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
            
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def infer_metadata_from_filename(filename: str) -> dict:
    """Extracts categorical metadata (e.g., Location, Document Type) from the filename.
    
    Example: 'dubai_purchase_policy.pdf' -> location: 'Dubai', doc_type: 'purchase_policy'
    """
    fname_lower = filename.lower()
    
    # Infer Location
    if "dubai" in fname_lower:
        location = "Dubai"
    elif "alex" in fname_lower:
        location = "Alexandria"
    elif "cairo" in fname_lower or "cairo" in fname_lower:
        location = "Cairo"
    elif "riyadh" in fname_lower or "riyadh" in fname_lower:
        location = "Riyadh"
    elif "jeddah" in fname_lower or "jeddah" in fname_lower:
        location = "Jeddah"
    elif "doha" in fname_lower or "doha" in fname_lower:
        location = "Doha"
    else:
        location = "General"
        
    # Infer Document Type
    if "policy" in fname_lower:
        doc_type = "purchase_policy"
    elif "plan" in fname_lower:
        doc_type = "payment_plan"
    else:
        doc_type = "general_document"
        
    return {
        "filename": filename,
        "location": location,
        "doc_type": doc_type,
    }


def upsert_document(file_path: str):
    """Processes a single file, chunks it, embeds chunks via Ollama, and upserts to Qdrant."""
    filename = Path(file_path).name
    raw_text = extract_text_from_file(file_path)
    
    if not raw_text.strip():
        print(f"Skipping empty file: {filename}")
        return

    # Extract metadata attributes for metadata filtering
    file_metadata = infer_metadata_from_filename(filename)
    
    # Chunk text
    chunks = chunk_text_by_tokens(raw_text, max_tokens=400, overlap=40)
    if not chunks:
        return

    # Prefix required by nomic-embed-text for document chunks
    formatted_chunks = [f"search_document: {chunk}" for chunk in chunks]

    # Generate embeddings in Ollama
    response = ollama.embed(model=EMBED_MODEL, input=formatted_chunks)
    embeddings = response["embeddings"]

    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        pid = deterministic_uuid(filename, i)
        payload = {
            **file_metadata,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "text_len": len(chunk),
            "text": chunk,  # Stores actual chunk text for RAG generation
        }
        points.append(PointStruct(id=pid, vector=emb, payload=payload))

    # Upsert to Qdrant in batches
    batch_size = 64
    for i in range(0, len(points), batch_size):
        qdrant_client.upsert(collection_name=COLLECTION, points=points[i : i + batch_size])
        
    print(f" Successfully ingested '{filename}' ({len(chunks)} chunks).")


if __name__ == "__main__":
    # Ensure directory exists or create sample folder
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created directory '{DOCS_DIR}'. Please place your PDF/HTML documents there.")
    
    # Process all files in the directory
    supported_extensions = {".pdf", ".html", ".htm", ".txt", ".md"}
    files_to_process = [
        os.path.join(DOCS_DIR, f) for f in os.listdir(DOCS_DIR)
        if Path(f).suffix.lower() in supported_extensions
    ]

    print(f"Found {len(files_to_process)} document(s) to process in '{DOCS_DIR}'...")
    for file_path in files_to_process:
        upsert_document(file_path)

    print("\nAll documents ingested successfully into Qdrant.")

# Quick Query Verification Example:
# query_text = "What is the down payment requirement for Dubai?"
# formatted_query = f"search_query: {query_text}"
# query_vector = ollama.embed(model=EMBED_MODEL, input=formatted_query)["embeddings"][0]
# 
# results = qdrant_client.query_points(
#     collection_name=COLLECTION,
#     query=query_vector,
#     limit=3,
#     search_params=SearchParams(hnsw_ef=128)
# )