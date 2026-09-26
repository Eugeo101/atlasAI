import json
import os
import time
import ollama
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range, SearchParams
from sentence_transformers import CrossEncoder

# System Prompt supporting dynamic multi-source citations
SYSTEM_PROMPT = (
    "You are an expert real-estate assistant. Answer concisely and accurately. "
    "Use ONLY the retrieved sources provided in the context.\n"
    "Citations rules:\n"
    "- For property listings, cite like [source:units:UNIT_ID].\n"
    "- For payment policies/plans, cite like [source:docs:FILENAME].\n"
    "If the sources do not contain enough information to answer, state clearly "
    "'I don't have enough information in the provided documents' and suggest next steps."
)

# Environment & Infrastructure Configuration
GATEWAY_URL = "http://localhost:8081"
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "")
QDRANT_URL = "http://localhost:6333"

UNITS_COLLECTION = "units_collection"
DOCS_COLLECTION = "documents_collection"

EMBED_MODEL = "nomic-embed-text"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Initialize Qdrant Client
qdrant_client = QdrantClient(url=QDRANT_URL)

_reranker_instance = None


def get_reranker() -> CrossEncoder:
    """Singleton loader for CrossEncoder."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(RERANKER_MODEL)
    return _reranker_instance


def extract_intent_and_filters(user_query: str) -> dict:
    """Single Gateway Call: Extracts intent target ("units", "documents", or "both")

    and parses category-specific metadata filters.
    """
    system_instructions = (
        "You are an intent and metadata parser for a real-estate QA system.\n"
        "Analyze the user query and extract intent target and filters into raw JSON ONLY.\n\n"
        "JSON Structure:\n"
        "{\n"
        '  "target_source": "units" | "documents" | "both",\n'
        '  "unit_filters": {\n'
        '    "city": string (e.g. "Dubai", "Alexandria", "Cairo", "Riyadh", "Jeddah", "Doha"),\n'
        '    "district": string,\n'
        '    "unit_type": string ("apartment", "villa", "studio", "townhouse", "office"),\n'
        '    "is_available": boolean,\n'
        '    "price": {"gte": num, "lte": num},\n'
        '    "bedrooms": num | {"gte": num, "lte": num},\n'
        '    "bathrooms": num | {"gte": num, "lte": num}\n'
        "  },\n"
        '  "doc_filters": {\n'
        '    "location": string ("Dubai", "Alexandria", "Cairo", "Riyadh", "Jeddah", "Doha"),\n'
        '    "doc_type": string ("purchase_policy", "payment_plan")\n'
        "  }\n"
        "}\n\n"
        "Routing Rules:\n"
        "- If asking for specific property listings, availability, or prices -> target_source: 'units'.\n"
        "- If asking about payment terms, down payments, installments, legal terms, or FAQs -> target_source: 'documents'.\n"
        "- If query involves both (e.g., 'Find apartments in Dubai and tell me the payment plan') -> target_source: 'both'. and doc_filters has {'location': 'Dubai', 'doc_type': 'payment_plan'}\n"
        "- Do NOT include empty/null filter fields."
    )

    prompt_messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": f'Query: "{user_query}"'},
    ]

    try:
        out, _ = call_vllm_via_gateway(
            prompt_messages=prompt_messages, temperature=0.1, max_tokens=512
        )
        raw_content = out["choices"][0]["message"]["content"].strip()

        if raw_content.startswith("```"):
            raw_content = raw_content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        extracted = json.loads(raw_content)

        # Default fallbacks
        target_source = extracted.get("target_source", "both")
        unit_filters = extracted.get("unit_filters", {})
        doc_filters = extracted.get("doc_filters", {})

        if "is_available" not in unit_filters and target_source in ["units", "both"]:
            unit_filters["is_available"] = True

        return {
            "target_source": target_source,
            "unit_filters": unit_filters,
            "doc_filters": doc_filters,
        }

    except Exception as e:
        print(f"Warning: Intent/Metadata extraction failed ({e}). Defaulting to searching both collections.")
        return {
            "target_source": "both",
            "unit_filters": {"is_available": True},
            "doc_filters": {},
        }


def build_qdrant_filter(filters: dict | None) -> Filter | None:
    """Converts a dictionary of metadata filters into a Qdrant Filter object."""
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        if isinstance(value, dict):
            conditions.append(FieldCondition(key=key, range=Range(**value)))
        elif isinstance(value, (str, int, float, bool)):
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(must=conditions) if conditions else None


def get_query_embedding(query: str) -> list[float]:
    """Generates embedding using Ollama nomic-embed-text with mandatory 'search_query:' prefix."""
    formatted_query = f"search_query: {query}"
    response = ollama.embed(model=EMBED_MODEL, input=formatted_query)
    return response["embeddings"][0]


def search_collection(collection_name: str, query_vector: list[float], filters: dict = None, top_k: int = 20, hnsw_ef: int = 128) -> list:
    """Performs pre-filtered vector search on a specific collection."""
    qfilter = build_qdrant_filter(filters)

    res = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=qfilter,
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=hnsw_ef),
    )
    
    # Tag hits with collection origin for downstream citation formatting
    for pt in res.points:
        pt.payload["_collection"] = collection_name
        
    return res.points


def rerank_hits(query: str, hits: list, top_k_rerank: int = 5) -> list:
    """Reranks candidate hits across both collections using a Cross-Encoder."""
    if not hits:
        return []

    reranker = get_reranker()
    pairs = [[query, h.payload.get("text", "")] for h in hits]
    scores = reranker.predict(pairs)

    for hit, score in zip(hits, scores):
        hit.score = float(score)

    hits_sorted = sorted(hits, key=lambda x: x.score, reverse=True)
    return hits_sorted[:top_k_rerank]

def run_reranker(top_k_rerank, user_query, hits):
    if search_params["use_reranker"] and hits:
        final_hits = rerank_hits(user_query, hits, top_k_rerank=top_k_rerank)
    else:
        final_hits = hits[:top_k_rerank]
    return final_hits

def assemble_prompt(user_query: str, hits: list) -> list:
    """Formats context blocks with distinct citation headers based on source collection."""
    context_blocks = []
    
    for h in hits:
        payload = h.payload or {}
        coll = payload.get("_collection", UNITS_COLLECTION)
        
        if coll == UNITS_COLLECTION:
            src_id = payload.get("unit_id") or h.id
            header = f"--- [source:units:{src_id} score:{h.score:.3f}] ---\n"
        else:
            fname = payload.get("filename", "document")
            chunk_idx = payload.get("chunk_index", 0)
            header = f"--- [source:docs:{fname} chunk:{chunk_idx} score:{h.score:.3f}] ---\n"

        ctx = header + (payload.get("text", "") or "")
        context_blocks.append(ctx)

    retrieved_text = "\n\n".join(context_blocks)
    
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"User Query: {user_query}\n\nRetrieved Sources:\n{retrieved_text}\n\nProvide a clear answer with citations.",
        },
    ]
    return prompt


def call_vllm_via_gateway(
    prompt_messages,
    model="Qwen/Qwen2.5-3B-Instruct-AWQ",
    max_tokens=512,
    temperature=0.2,
):
    """Routes generation requests to vLLM via Gateway."""
    endpoint = f"{GATEWAY_URL}/v1/chat"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GATEWAY_API_KEY}",
    }
    body = {
        "model": model,
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.time()
    response = requests.post(endpoint, headers=headers, json=body, timeout=60)
    latency_ms = int((time.time() - start) * 1000)

    response.raise_for_status()
    return response.json(), latency_ms


def handle_qa(
    user_query: str,
    search_params: dict = None,
):
    """Main Orchestrator: Single-Pass Intent & Metadata Extraction -> Multi-Collection Vector Search -> Cross-Encoder Reranking -> LLM Answer Generation."""

    # 1. Intent Routing & Metadata Filter Extraction (Single Gateway Call)
    parsed = extract_intent_and_filters(user_query)
    target_source = parsed["target_source"]
    unit_filters = parsed["unit_filters"]
    doc_filters = parsed["doc_filters"]

    # 2. Vector Retrieval across target collections
    query_vector = get_query_embedding(user_query)
    candidate_hits = []

    # Search Units Collection if targeted
    if target_source in ["units", "both"]:
        unit_hits = search_collection(
            collection_name=UNITS_COLLECTION,
            query_vector=query_vector,
            filters=unit_filters,
            top_k=search_params["top_k_retrieval"],
            hnsw_ef=search_params["hnsw_ef"],
        )
        # Unified Cross-Encoder Reranking
        unit_hits = run_reranker(search_params["top_k_rerank"], user_query, unit_hits)
        candidate_hits.extend(unit_hits)

    # Search Documents Collection if targeted
    if target_source in ["documents", "both"]:
        doc_hits = search_collection(
            collection_name=DOCS_COLLECTION,
            query_vector=query_vector,
            filters=doc_filters,
            top_k=search_params["top_k_retrieval"],
            hnsw_ef=search_params["hnsw_ef"],
        )
        # Unified Cross-Encoder Reranking
        doc_hits = run_reranker(search_params["top_k_rerank"], user_query, doc_hits)
        candidate_hits.extend(doc_hits)

    # 4. Prompt Assembly & Answer Generation
    prompt = assemble_prompt(user_query, candidate_hits)
    out, latency = call_vllm_via_gateway(prompt)

    content = out["choices"][0]["message"]["content"]
    
    # Format metadata source references
    sources_summary = []
    for h in candidate_hits:
        p = h.payload
        if p.get("_collection") == UNITS_COLLECTION:
            sources_summary.append({"type": "unit", "id": p.get("unit_id"), "score": round(h.score, 4)})
        else:
            sources_summary.append({"type": "document", "file": p.get("filename"), "chunk": p.get("chunk_index"), "score": round(h.score, 4)})

    return {
        "answer": content,
        "intent_routing": target_source,
        "applied_filters": {"unit_filters": unit_filters, "doc_filters": doc_filters},
        "sources": sources_summary,
        "latency_ms": latency,
    }

# Example Execution
if __name__ == "__main__":
    sample_query = input("Please enter your query: ") # Find me an apartment between 180,000 and 300,000 in Cairo and its payment plan
    search_params = {
        "top_k_retrieval": 25,  # Fetch top 25 vector candidate matches from Qdrant
        "top_k_rerank": 5,      # Rerank and pass top 5 most relevant chunks to LLM
        "hnsw_ef": 128,         # Dynamic search candidate size for HNSW
        "use_reranker": True,   # Toggle cross-encoder step on/off
    }

    # Parameters directly in handle_qa
    result = handle_qa(
        user_query=sample_query,
        search_params=search_params
    )
    print(json.dumps(result, indent=2))

"""
Query:
Find me an apartment between 180,000 and 300,000 in Cairo and its payment plan:

Answer:
Based on the provided sources, there is an apartment available in Cairo that fits your criteria: between 180,000 and 300,000. The apartment is located in Zamalek, Cairo and measures 162.8 sqm with 2 spacious bedrooms and 2 bathrooms. The current price is 272,000.00.

For the payment plan, you can refer to the payment policy document found in [source:docs:cairo_payment_plan.pdf]. According to the document, the payment plan is structured as follows:

- Reservation: 10% of the agreed unit price                                                                                           
- Signing: 30% of the agreed unit price                                                                                               
- Installments: 60% divided across 18 equal installments                                                                              
                                                                                                                                      
To apply the plan to this specific listing, you would calculate the reservation and installment amounts using the listed price of 272,000.00. Here’s how you would calculate:                                                                                               
                                                                                                                                      
- Reservation = 272,000.00 * 10% = 27,200.00                                                                                          
- Monthly installment = (272,000.00 * 60%) / 18 = 8,666.67                                                                            
                                                                                                                                      
Please note that the payment percentages are generated policy assumptions and not historical payment terms from the CSV. The plan does not include financing interest or government fees.
"""