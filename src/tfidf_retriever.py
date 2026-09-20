import re
import math
from .models import Document, Query
from .retriever import Retriever

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class TfidfRetriever(Retriever):

    def index(
        self,
        documents: list[Document],
    ) -> None:

        self.documents = documents

        self.vocabulary = self.build_vocabulary(
            documents
        )

        self.idf = self.inverse_document_frequency(
            documents
        )

        self.document_vectors = {
            document.id: self.vectorize_document(
                document,
                self.vocabulary,
                self.idf,
            )
            for document in documents
        }

    def build_vocabulary(
        self,
        documents: list[Document],
    ) -> dict[str, int]:

        vocabulary = {}

        for document in documents:
            tokens = tokenize(document.text)

            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)

        return vocabulary

    def term_frequency(
        self,
        document: Document,
    ) -> dict[str, float]:

        tokens = tokenize(document.text)

        if not tokens:
            return {}

        counts: dict[str, int] = {}

        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        total_tokens = len(tokens)

        return {
            token: count / total_tokens
            for token, count in counts.items()
        }

    def inverse_document_frequency(
        self,
        documents: list[Document],
    ) -> dict[str, float]:

        document_count = len(documents)

        document_frequency: dict[str, int] = {}

        for document in documents:
            tokens = set(tokenize(document.text))

            for token in tokens:
                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        return {
            token: math.log(document_count / frequency)
            for token, frequency in document_frequency.items()
        }

    def vectorize_document(
        self,
        document: Document,
        vocabulary: dict[str, int],
        idf: dict[str, float],
    ) -> list[float]:

        tf = self.term_frequency(document)

        vector = [0.0] * len(vocabulary)

        for token, frequency in tf.items():
            if token in vocabulary:
                index = vocabulary[token]
                vector[index] = frequency * idf[token]

        return vector

    def vectorize_query(
        self,
        query: Query,
        vocabulary: dict[str, int],
        idf: dict[str, float],
    ) -> list[float]:

        tokens = tokenize(query.text)

        counts: dict[str, int] = {}

        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        total_tokens = len(tokens)

        if total_tokens == 0:
            return [0.0] * len(vocabulary)

        vector = [0.0] * len(vocabulary)

        for token, count in counts.items():

            if token not in vocabulary:
                continue

            tf = count / total_tokens
            index = vocabulary[token]

            vector[index] = tf * idf[token]

        return vector

    def retrieve(
        self,
        query: Query,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:

        query_vector = self.vectorize_query(
            query,
            self.vocabulary,
            self.idf,
        )

        scored_documents = []

        for document in self.documents:

            document_vector = self.document_vectors[
                document.id
            ]

            similarity = cosine_similarity(
                query_vector,
                document_vector,
            )

            scored_documents.append(
                (document, similarity)
            )

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents[:top_k]

def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )
    
if __name__ == "__main__":
    from datasets import load_dataset
    from .dataset import build_documents, build_query

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    example = dataset["validation"][0]

    documents = build_documents(example)

    retriever = TfidfRetriever()

    vocabulary = retriever.build_vocabulary(documents)

    print(f"Vocabulary size: {len(vocabulary)}")
    print("\nFirst 20 vocabulary entries:")

    for word, index in list(vocabulary.items())[:20]:
        print(f"{index}: {word}")

    document = documents[0]

    tf = retriever.term_frequency(document)

    print(f"\nDocument: {document.title}")
    print(f"Number of unique terms: {len(tf)}")

    print("\nTop terms:")

    for word, frequency in sorted(
        tf.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        print(f"{word}: {frequency:.4f}")

    idf = retriever.inverse_document_frequency(documents)

    print("\nMost common terms:")

    for word, value in sorted(
        idf.items(),
        key=lambda item: item[1],
    )[:20]:
        print(f"{word}: {value:.4f}")

    print("\nMost distinctive terms:")

    for word, value in sorted(
        idf.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:20]:
        print(f"{word}: {value:.4f}")

    idf = retriever.inverse_document_frequency(documents)

    document = documents[0]

    vector = retriever.vectorize_document(
        document,
        vocabulary,
        idf,
    )

    print(f"\nDocument: {document.title}")
    print(f"Vector dimension: {len(vector)}")
    print(f"Non-zero values: {sum(1 for value in vector if value != 0)}")

    print("\nFirst 20 vector values:")

    for index, value in enumerate(vector[:20]):
        print(f"{index}: {value:.6f}")

    example = dataset["validation"][0]

    documents = build_documents(example)
    query = build_query(example)

    retriever = TfidfRetriever()

    results = retriever.retrieve(
        query,
        documents,
        top_k=5,
    )

    print(f"Question: {query.text}")
    print(f"Answer: {query.answer}")

    print("\nTF-IDF Results:")

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{document.title} "
            f"score={score:.4f}"
        )