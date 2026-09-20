from abc import ABC, abstractmethod
import re

from .models import Document, Query


class Retriever(ABC):

    @abstractmethod
    def index(
        self,
        documents: list[Document],
    ) -> None:
        pass

    @abstractmethod
    def retrieve(
        self,
        query: Query,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        pass


class LexicalRetriever(Retriever):

    @staticmethod
    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))

    def index(
        self,
        documents: list[Document],
    ) -> None:
        self.documents = documents

    def score_document(
        self,
        query: Query,
        document: Document,
    ) -> float:
        query_tokens = self.tokenize(query.text)
        document_tokens = self.tokenize(document.text)

        return float(len(query_tokens & document_tokens))

    def retrieve(
        self,
        query: Query,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:

        scored_documents = [
            (
                document,
                self.score_document(query, document),
            )
            for document in self.documents
        ]

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents[:top_k]


def recall_at_k(
    query: Query,
    retrieved_documents: list[tuple[Document, float]],
    k: int,
) -> float:

    retrieved_ids = {
        document.id
        for document, _ in retrieved_documents[:k]
    }

    relevant_ids = set(query.supporting_documents)

    if not relevant_ids:
        return 0.0

    return len(
        retrieved_ids & relevant_ids
    ) / len(relevant_ids)

def parent_recall_at_k(query, retrieved_documents, k):
    retrieved_parent_ids = {
        document.metadata.get("parent_id", document.id)
        for document, _ in retrieved_documents[:k]
    }

    relevant_ids = set(query.supporting_documents)

    if not relevant_ids:
        return 0.0

    return len(retrieved_parent_ids & relevant_ids) / len(relevant_ids)