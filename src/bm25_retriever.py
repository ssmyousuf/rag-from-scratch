import math
import re

from .models import Document, Query
from .retriever import Retriever


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever(Retriever):

    def score_document(
        self,
        query: Query,
        document: Document,
        k1: float = 1.30,
        b: float = 1.8,
    ) -> float:

        query_tokens = tokenize(query.text)

        frequencies = self.term_frequencies[
            document.id
        ]

        document_length = self.document_lengths[
            document.id
        ]

        score = 0.0

        for token in query_tokens:

            term_frequency = frequencies.get(
                token,
                0,
            )

            if term_frequency == 0:
                continue

            idf = self.inverse_document_frequency(
                token
            )

            length_normalization = (
                1
                - b
                + b
                * (
                    document_length
                    / self.average_document_length
                )
            )

            term_score = (
                idf
                * (
                    term_frequency
                    * (k1 + 1)
                )
                / (
                    term_frequency
                    + k1
                    * length_normalization
                )
            )

            score += term_score

        return score

    def index(
        self,
        documents: list[Document],
    ) -> None:

        self.documents = documents

        self.document_count = len(documents)

        self.document_lengths = {}

        self.term_frequencies = {}

        for document in documents:

            tokens = tokenize(document.text)

            self.document_lengths[document.id] = len(tokens)

            frequencies = {}

            for token in tokens:
                frequencies[token] = (
                    frequencies.get(token, 0) + 1
                )

            self.term_frequencies[document.id] = frequencies

        self.average_document_length = (
            sum(self.document_lengths.values())
            / self.document_count
        )

        self.document_frequency = (
            self.build_document_frequency()
        )

    def build_document_frequency(self) -> dict[str, int]:

        document_frequency: dict[str, int] = {}

        for frequencies in self.term_frequencies.values():

            for token in frequencies:
                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        return document_frequency

    def inverse_document_frequency(
        self,
        token: str,
    ) -> float:

        document_frequency = (
            self.document_frequency.get(token, 0)
        )

        return math.log(
            1
            + (
                self.document_count
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )
    
    def retrieve(
        self,
        query: Query,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:

        scored_documents = []

        for document in self.documents:

            score = self.score_document(
                query,
                document,
            )

            scored_documents.append(
                (document, score)
            )

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents[:top_k]

if __name__ == "__main__":

    from datasets import load_dataset
    from .dataset import build_documents, build_query

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    example = dataset["validation"][0]

    documents = build_documents(example)

    retriever = BM25Retriever()

    retriever.index(documents)

    print(
        f"Documents: "
        f"{retriever.document_count}"
    )

    print(
        f"Average document length: "
        f"{retriever.average_document_length:.2f}"
    )

    document = documents[0]

    print(
        f"\nDocument: {document.title}"
    )

    print(
        f"Length: "
        f"{retriever.document_lengths[document.id]}"
    )

    print("\nTerm frequencies:")

    frequencies = retriever.term_frequencies[
        document.id
    ]

    for word, count in sorted(
        frequencies.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:

         print(f"{word}: {count}")

    print("\nTerm IDF:")

    for word in ["the", "film", "wood", "scott"]:

        print(
            f"{word}: "
            f"{retriever.inverse_document_frequency(word):.4f}"
        )

    query = build_query(example)

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    print(f"\nQuestion: {query.text}")
    print(f"Answer: {query.answer}")

    print("\nBM25 Results:")

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{document.title} "
            f"score={score:.4f}"
        )