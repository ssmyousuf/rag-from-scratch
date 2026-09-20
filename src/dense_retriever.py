from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from .models import Document, Query
from .retriever import Retriever
from datasets import load_dataset
from .dataset import build_corpus                     

MODEL_NAME = "all-MiniLM-L6-v2"


class DenseRetriever(Retriever):

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def index(
        self,
        documents: list[Document],
    ) -> None:

        self.documents = documents

        texts = [
            document.text
            for document in documents
        ]

        self.document_embeddings = (
            self.model.encode(
                texts,
                normalize_embeddings=True,
            )
        )

    def retrieve(
        self,
        query: Query,
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:

        query_embedding = self.model.encode(
            query.text,
            normalize_embeddings=True,
        )

        similarities = cos_sim(
            query_embedding,
            self.document_embeddings,
        )[0]

        scored_documents = [
            (
                document,
                float(similarities[index]),
            )
            for index, document in enumerate(
                self.documents
            )
        ]

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored_documents[:top_k]


if __name__ == "__main__":
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    examples = dataset["validation"].select(range(100))

    corpus = build_corpus(examples)

    print("Corpus size:", len(corpus))

    retriever = DenseRetriever()

    retriever.index(corpus.documents)

    print(
        "Indexed documents:",
        len(retriever.documents),
    )

    embeddings = retriever.document_embeddings

    print("Shape:", embeddings.shape)
    print("Dtype:", embeddings.dtype)
    print("Bytes:", embeddings.nbytes)
    print(
        "MB:",
        embeddings.nbytes / (1024 ** 2),
    )
    
# if __name__ == "__main__":

#     from datasets import load_dataset

#     from .dataset import (
#         build_documents,
#         build_query,
#     )

#     dataset = load_dataset(
#         "hotpotqa/hotpot_qa",
#         "distractor",
#     )

#     example = dataset["validation"][0]

#     documents = build_documents(example)
#     query = build_query(example)

#     retriever = DenseRetriever()

#     retriever.index(documents)

#     embeddings = retriever.document_embeddings

#     print("Shape:", embeddings.shape)
#     print("Dtype:", embeddings.dtype)
#     print("Bytes:", embeddings.nbytes)
#     print("MB:", embeddings.nbytes / (1024 ** 2))

#     results = retriever.retrieve(
#         query,
#         top_k=5,
#     )

#     print(f"Question: {query.text}")
#     print(f"Answer: {query.answer}")

#     print("\nDense Retrieval Results:")

#     for rank, (document, score) in enumerate(
#         results,
#         start=1,
#     ):
#         print(
#             f"{rank}. "
#             f"{document.title} "
#             f"score={score:.4f}"
#         )