from datasets import load_dataset
from tqdm import tqdm
from .dataset import build_corpus
from .dataset import build_query
from .retriever import LexicalRetriever, recall_at_k, Retriever
from .tfidf_retriever import TfidfRetriever
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever

# def evaluate_retriever(
#     retriever: Retriever,
#     examples,
# ) -> dict[int, float]:

#     k_values = [1, 3, 5, 10]

#     recalls = {
#         k: []
#         for k in k_values
#     }

#     for example in tqdm(examples):

#         query = build_query(example)
#         documents = build_documents(example)

#         # Build the retrieval index once for this document set.
#         retriever.index(documents)

#         # Retrieve enough documents for the largest K.
#         results = retriever.retrieve(
#             query,
#             top_k=max(k_values),
#         )

#         for k in k_values:
#             recalls[k].append(
#                 recall_at_k(
#                     query,
#                     results,
#                     k,
#                 )
#             )

#     return {
#         k: sum(values) / len(values)
#         for k, values in recalls.items()
#     }

def evaluate_retriever(
    retriever: Retriever,
    corpus,
    examples,
) -> dict[int, float]:

    k_values = [1, 3, 5, 10]

    recalls = {k: [] for k in k_values}

    # Build index once
    retriever.index(corpus.documents)

    # Run all queries against the same index
    for example in tqdm(examples):
        query = build_query(example)

        results = retriever.retrieve(
            query,
            top_k=max(k_values),
        )

        for k in k_values:
            recalls[k].append(
                recall_at_k(query, results, k)
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

    retrievers = {
        "Lexical Retriever": LexicalRetriever(),
        "TF-IDF Retriever": TfidfRetriever(),
        "BM25 Retriever": BM25Retriever(),
        "Dense Retriever": DenseRetriever(),
    }

    corpus = build_corpus(examples)

    for name, retriever in retrievers.items():
        results = evaluate_retriever(
            retriever,
            corpus,
            examples,
        )

        print(name)

        for k, recall in results.items():
            print(f"Recall@{k}: {recall:.3f}")

if __name__ == "__main__":
    main()