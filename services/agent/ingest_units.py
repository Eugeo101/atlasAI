from qdrant_client import QdrantClient # VectorDB
from qdrant_client.models import (
    Distance,
    VectorParams,            # for embedding size parameter
    HnswConfigDiff,          # for HNSW index building parameters
    ScalarQuantization,      # for Scalar Quantization (SQ)
    ScalarQuantizationConfig,
    ScalarType,
    PointStruct,             # ds for saving data in index 
    SearchParams,            # for query-time ef (top-n candidates for each node)
)
import ollama # Embedding models
import tiktoken # for token-based chunking

QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "units_collection"

quadrant_client = QdrantClient(url=QDRANT_URL)

sample_emb = ollama.embed(model=EMBED_MODEL, input="test")["embeddings"][0]
vec_size = len(sample_emb)

# Ensure collection exists with HNSW & SQ configured
quadrant_client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),

    # Configure HNSW INDEX BUILDING (m & ef_construction)
    hnsw_config=HnswConfigDiff(
        m=16,             # Max edge connections per node (default: 16)
        ef_construct=100, # Candidate list size during index build (default: 100)
    ),

    # Configure SCALAR QUANTIZATION (SQ)
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8, # Converts float32 -> int8 (4x RAM savings)
            quantile=0.99,        # Excludes 1% outer bounds for accuracy
            always_ram=True,      # Keeps quantized vectors in RAM for fast search
        )
    ),
)

def deterministic_id(unit_id: int, chunk_index: int) -> str:
  """Generates a deterministic UUID string compatible with Qdrant point IDs."""
  # import uuid
  # seed_string = f"unit-{unit_id}-chunk-{chunk_index}"
  # pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_string)) # 'f3a1c8b2-5e60-5a3d-98e1-2c09871ab123'

  # Encodes unit 1, chunk 0 -> 10000
  # Encodes unit 1, chunk 1 -> 10001
  pid = (unit_id * 10000) + chunk_index
  return pid

def chunk_text_by_tokens(
    text: str, max_tokens: int = 400, overlap: int = 40, encoding_name: str = "cl100k_base"
) -> list[str]:
  """Splits text into chunks using exact token counts and sliding-window overlap."""
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

def upsert_unit(unit_id: int, metadata: dict, text: str):
  chunks = chunk_text_by_tokens(text, max_tokens=400, overlap=40) # 10% overlapping
  if not chunks:
    return

  # Nomic Embed requires "search_document: " prefix for document chunks
  formatted_chunks = [f"search_document: {chunk}" for chunk in chunks]

  # Generate embeddings in Ollama
  response = ollama.embed(model=EMBED_MODEL, input=formatted_chunks)
  embeddings = response["embeddings"]

  points = []
  for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    pid = deterministic_id(unit_id, i)
    payload = {
        **metadata,
        "unit_id": unit_id,
        "chunk_index": i,
        "text_len": len(chunk),
        "text": chunk,  # Store actual chunk text for RAG context
    }
    points.append(PointStruct(id=pid, vector=emb, payload=payload))

  # Upsert to Qdrant in batches
  batch_size = 64
  for i in range(0, len(points), batch_size):
    quadrant_client.upsert(collection_name=COLLECTION, points=points[i : i + batch_size])

# call db
def fetch_units_from_db():
    """Fetch all unit rows from the PostgreSQL database."""

    import psycopg2
    from psycopg2.extras import RealDictCursor

    # Connect to Postgres via mapped host port 5433 (or internal port 5432 inside Docker network)
    DB_HOST = "localhost"
    DB_PORT = "5433"
    DB_NAME = "realestate"
    DB_USER = "realestate"
    DB_PASS = "realestate"

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        cursor_factory=RealDictCursor,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, city, district, unit_type, price, bedrooms, bathrooms, area_sqm, is_available, description
                FROM units
            """)
            return cursor.fetchall()
    finally:
        conn.close()

if __name__ == "__main__":
    units = fetch_units_from_db()
    print(f"Fetched {len(units)} units from PostgreSQL. Beginning ingestion...")

    for unit in units:
        unit_id = unit["id"]

        # 1. Enriched Text: Combine location/type attributes with description for semantic context
        enriched_text = (
            f"City: {unit['city']} | "
            f"District: {unit['district']} | "
            f"Type: {unit['unit_type']} | "
            f"Description: {unit['description'] or ''}"
        )

        # 2. Metadata Payload: Store all categorical & numerical attributes for hard filtering
        metadata = {
            "city": unit["city"],
            "district": unit["district"],
            "unit_type": unit["unit_type"],
            "price": float(unit["price"]) if unit["price"] is not None else 0.0,
            "bedrooms": int(unit["bedrooms"]) if unit["bedrooms"] is not None else 0,
            "bathrooms": int(unit["bathrooms"]) if unit["bathrooms"] is not None else 0,
            "area_sqm": float(unit["area_sqm"]) if unit["area_sqm"] is not None else 0.0,
            "is_available": bool(unit["is_available"]),
        }

        # 3. Embed & Upsert
        upsert_unit(unit_id=unit_id, metadata=metadata, text=enriched_text)

    print("Ingestion completed successfully.")

# for query time
# quadrant_client.query_points(
#     collection_name=COLLECTION,
#     query=query_vector,
#     search_params=SearchParams(
#         hnsw_ef=128,  # HIGHLIGHT: Overrides default search candidate list size
#     ),
# )