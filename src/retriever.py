import re

from .models import Document, Query


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def score_document(query: Query, document: Document) -> int:
    query_tokens = tokenize(query.text)
    document_tokens = tokenize(document.text)

    return len(query_tokens & document_tokens)


def retrieve(
    query: Query,
    documents: list[Document],
    top_k: int = 5,
) -> list[tuple[Document, int]]:
    scored_documents = [
        (document, score_document(query, document))
        for document in documents
    ]

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_documents[:top_k]

def recall_at_k(
    query: Query,
    retrieved_documents: list[tuple[Document, int]],
    k: int,
) -> float:
    retrieved_ids = {
        document.id
        for document, _ in retrieved_documents[:k]
    }

    relevant_ids = set(query.supporting_documents)

    if not relevant_ids:
        return 0.0

    return len(retrieved_ids & relevant_ids) / len(relevant_ids)