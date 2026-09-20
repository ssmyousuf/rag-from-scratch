from datasets import load_dataset

from .chunker import chunk_document
from .dataset import build_corpus, build_query
from .dense_retriever import DenseRetriever
from .retriever import parent_recall_at_k


def evaluate(retriever, documents, examples):
    retriever.index(documents)
    print(f"Embedding shape: {retriever.document_embeddings.shape}")
    print(f"Embedding dtype: {retriever.document_embeddings.dtype}")
    print(
        f"Embedding memory: "
        f"{retriever.document_embeddings.nbytes / (1024 * 1024):.4f} MB"
    )
    k_values = [1, 3, 5, 10]
    recalls = {k: [] for k in k_values}

    for example in examples:
        query = build_query(example)
        results = retriever.retrieve(
            query,
            top_k=max(k_values),
        )

        for k in k_values:
            recalls[k].append(
                parent_recall_at_k(
                    query,
                    results,
                    k,
                )
            )

    return {
        k: sum(values) / len(values)
        for k, values in recalls.items()
    }


def main():
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    examples = dataset["validation"].select(range(100))
    corpus = build_corpus(examples)

    print(f"Documents: {len(corpus)}")
    print()

    retriever = DenseRetriever()

    # Document-level retrieval
    print("Document-level retrieval")
    document_results = evaluate(
        retriever,
        corpus.documents,
        examples,
    )

    print(document_results)
    print()

    # Chunk-level retrieval
    for chunk_size in [50, 100, 200]:
        chunks = []

        for document in corpus.documents:
            chunks.extend(
                chunk_document(
                    document,
                    chunk_size=chunk_size,
                )
            )

        print(f"Chunk size: {chunk_size}")
        print(f"Chunks: {len(chunks)}")

        chunk_results = evaluate(
            retriever,
            chunks,
            examples,
        )

        print(chunk_results)
        print()


if __name__ == "__main__":
    main()