import time

from .dataset import build_query
from .retriever import Retriever, recall_at_k


def benchmark_retriever(
    retriever: Retriever,
    corpus,
    examples,
    k_values: list[int] | None = None,
) -> dict:

    if k_values is None:
        k_values = [1, 3, 5, 10]

    # -------------------------
    # Index build benchmark
    # -------------------------
    start = time.perf_counter()

    retriever.index(corpus.documents)

    index_time = time.perf_counter() - start

    # -------------------------
    # Query benchmark
    # -------------------------
    recalls = {k: [] for k in k_values}

    query_start = time.perf_counter()

    for example in examples:
        query = build_query(example)

        results = retriever.retrieve(
            query,
            top_k=max(k_values),
        )

        for k in k_values:
            recalls[k].append(
                recall_at_k(query, results, k)
            )

    query_time = time.perf_counter() - query_start

    query_count = len(examples)

    return {
        "index_time": index_time,
        "query_time": query_time,
        "queries_per_second": query_count / query_time,
        "average_query_latency_ms": (
            query_time / query_count
        ) * 1000,
        "recall": {
            k: sum(values) / len(values)
            for k, values in recalls.items()
        },
    }

if __name__ == "__main__":
    from datasets import load_dataset

    from .corpus import Corpus
    from .dataset import build_corpus
    from .retriever import LexicalRetriever
    from .tfidf_retriever import TfidfRetriever
    from .bm25_retriever import BM25Retriever
    from .dense_retriever import DenseRetriever

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    examples = dataset["validation"].select(range(100))

    corpus = build_corpus(examples)

    retrievers = {
        "Lexical": LexicalRetriever(),
        "TF-IDF": TfidfRetriever(),
        "BM25": BM25Retriever(),
        "Dense": DenseRetriever(),
    }

    for name, retriever in retrievers.items():

        print(f"\n{'=' * 50}")
        print(name)
        print(f"{'=' * 50}")

        results = benchmark_retriever(
            retriever,
            corpus,
            examples,
        )

        print(
            f"Index time: "
            f"{results['index_time']:.4f} s"
        )

        print(
            f"Query time: "
            f"{results['query_time']:.4f} s"
        )

        print(
            f"Average latency: "
            f"{results['average_query_latency_ms']:.2f} ms"
        )

        print(
            f"Queries/sec: "
            f"{results['queries_per_second']:.2f}"
        )

        for k, recall in results["recall"].items():
            print(
                f"Recall@{k}: "
                f"{recall:.3f}"
            )