import json
from typing import Any, Callable

import ollama
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range
from sentence_transformers import CrossEncoder
from pathlib import Path
from dotenv import load_dotenv
import os

target_dir = Path(__file__).resolve().parents[2]
load_dotenv(target_dir / '.env.example')

QDRANT_URL = os.getenv("QDRANT_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
UNITS_COLLECTION = os.getenv("UNITS_COLLECTION")
DOCUMENTS_COLLECTION = os.getenv("DOCUMENTS_COLLECTION")

qdrant = QdrantClient(url=QDRANT_URL)

_reranker = None
def get_reranker() -> CrossEncoder:
    """Load the reranker once, when a search tool first needs it."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_points(query: str, points: list, limit: int) -> list[dict]:
    """Rerank Qdrant candidates by query–chunk relevance."""
    points_with_text = [
        (point, (point.payload or {}).get("text", ""))
        for point in points
    ]
    points_with_text = [(point, text) for point, text in points_with_text if text.strip()]

    if not points_with_text:
        return []

    reranker = get_reranker()
    pairs = [[query, text] for _, text in points_with_text]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(points_with_text, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results = []
    for (point, text), rerank_score in ranked[:limit]:
        results.append({
            "payload": point.payload or {},
            "qdrant_score": float(point.score),
            "rerank_score": float(rerank_score),
            "text": text,
        })

    return results

class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UnitsSearchArguments(ToolArguments):
    query: str = Field(min_length=1, description="What the user is looking for.")
    city: str | None = None
    district: str | None = None
    unit_type: str | None = None
    min_price: float | None = Field(default=None, ge=0)
    max_price: float | None = Field(default=None, ge=0)
    min_bedrooms: int | None = Field(default=None, ge=0)
    max_bedrooms: int | None = Field(default=None, ge=0)
    limit: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def validate_ranges(self):
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price must not exceed max_price")

        if (
            self.min_bedrooms is not None
            and self.max_bedrooms is not None
            and self.min_bedrooms > self.max_bedrooms
        ):
            raise ValueError("min_bedrooms must not exceed max_bedrooms")

        return self


class DocumentsSearchArguments(ToolArguments):
    query: str = Field(min_length=1, description="The question or topic to search for.")
    doc_type: str | None = None
    location: str | None = None
    limit: int = Field(default=5, ge=1, le=15)


def embed_query(query: str) -> list[float]:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=f"search_query: {query}",
    )
    return response["embeddings"][0]


def make_unit_filter(args: UnitsSearchArguments) -> Filter | None:
    conditions = []

    for key in ("city", "district", "unit_type"):
        value = getattr(args, key)
        if value is not None:
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    if args.min_price is not None or args.max_price is not None:
        conditions.append(
            FieldCondition(
                key="price",
                range=Range(gte=args.min_price, lte=args.max_price),
            )
        )

    if args.min_bedrooms is not None or args.max_bedrooms is not None:
        conditions.append(
            FieldCondition(
                key="bedrooms",
                range=Range(gte=args.min_bedrooms, lte=args.max_bedrooms),
            )
        )

    return Filter(must=conditions) if conditions else None


def units_search(
    args: UnitsSearchArguments,
    *,
    reranker_limit: int,
) -> dict[str, Any]:
    result = qdrant.query_points(
        collection_name=UNITS_COLLECTION,
        query=embed_query(args.query),
        query_filter=make_unit_filter(args),
        limit=args.limit,
        with_payload=True,
    )
    return {
        "results": rerank_points(
            query=args.query,
            points=result.points,
            limit=reranker_limit,
        )
    }

def documents_search(
    args: DocumentsSearchArguments,
    *,
    reranker_limit: int,
) -> dict[str, Any]:
    conditions = []

    for key in ("doc_type", "location"):
        value = getattr(args, key)
        if value is not None:
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    result = qdrant.query_points(
        collection_name=DOCUMENTS_COLLECTION,
        query=embed_query(args.query),
        query_filter=Filter(must=conditions) if conditions else None,
        limit=args.limit,
        with_payload=True,
    )

    return {
            "results": rerank_points(
                query=args.query,
                points=result.points,
                limit=reranker_limit,
            )
    }


class ToolRegistry:
    def __init__(
        self,
        units_limit: int = 10,
        doc_limit: int = 5,
        reranker_limit: int = 3,
    ) -> None:
        if not 1 <= units_limit <= 20:
            raise ValueError("units_limit must be between 1 and 20")
        if not 1 <= doc_limit <= 15:
            raise ValueError("doc_limit must be between 1 and 15")
        if not 1 <= reranker_limit <= min(units_limit, doc_limit):
            raise ValueError(
                "reranker_limit must be positive and no greater than "
                "units_limit or doc_limit"
            )

        self.units_limit = units_limit
        self.doc_limit = doc_limit
        self.reranker_limit = reranker_limit
        self._tools: dict[str, tuple[type[BaseModel], Callable]] = {}

    def register(
        self,
        name: str,
        description: str,
        arguments_model: type[BaseModel],
        handler: Callable,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        self._tools[name] = (arguments_model, handler)
        self._descriptions[name] = description

    def execute(self, name: str, arguments_json: str) -> dict[str, Any]:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        arguments_model, handler = self._tools[name]
        arguments = arguments_model.model_validate_json(arguments_json)
        if name == "units_search":
            arguments = arguments.model_copy(
                update={"limit": min(arguments.limit, self.units_limit)}
            )
            return handler(arguments, reranker_limit=self.reranker_limit)
        if name == "documents_search":
            arguments = arguments.model_copy(
                update={"limit": min(arguments.limit, self.doc_limit)}
            )
            return handler(arguments, reranker_limit=self.reranker_limit)
        return handler(arguments)

    def openai_tool_definitions(self) -> list[dict[str, Any]]:
        definitions = []
        for name, (arguments_model, _) in self._tools.items():
            parameters = arguments_model.model_json_schema()
            search_limit = {
                "units_search": self.units_limit,
                "documents_search": self.doc_limit,
            }.get(name)
            if search_limit is not None:
                limit_schema = parameters.get("properties", {}).get("limit")
                if limit_schema is not None:
                    limit_schema["maximum"] = search_limit
                    limit_schema["default"] = search_limit

            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": self._descriptions[name],
                    "parameters": parameters,
                },
            })
        return definitions

    @property
    def _descriptions(self) -> dict[str, str]:
        if not hasattr(self, "_tool_descriptions"):
            self._tool_descriptions = {}
        return self._tool_descriptions

if __name__ == "__main__":
    tool_registry = ToolRegistry(
        units_limit=10,
        doc_limit=5,
        reranker_limit=3,
    )

    tool_registry.register(
        name="units_search",
        description="Search available real-estate units using the user's query and optional property filters.",
        arguments_model=UnitsSearchArguments,
        handler=units_search,
    )
    tool_registry.register(
        name="documents_search",
        description="Search property documents for either purchase policy or payment plan.",
        arguments_model=DocumentsSearchArguments,
        handler=documents_search,
    )

    definitions = tool_registry.openai_tool_definitions()
    print(json.dumps(definitions, indent=2))
    print(f"Registered {len(definitions)} tools.")