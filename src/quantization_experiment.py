from datasets import load_dataset
from sentence_transformers.util import cos_sim
import numpy as np
from .dataset import build_corpus, build_query
from .dense_retriever import DenseRetriever
from .retriever import recall_at_k

from .int8_quantization import (
    quantize_embeddings,
    dequantize_embeddings,
    quantized_similarity,
)

def retrieve_with_embeddings(
    query_embedding,
    document_embeddings,
    documents,
    top_k=5,
):
    similarities = cos_sim(
        query_embedding,
        document_embeddings,
    )[0]

    scored_documents = [
        (document, float(similarities[index]))
        for index, document in enumerate(documents)
    ]

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_documents[:top_k]


def evaluate_embeddings(
    retriever,
    document_embeddings,
    examples,
):
    k_values = [1, 3, 5, 10]
    recalls = {k: [] for k in k_values}

    for example in examples:
        query = build_query(example)

        query_embedding = retriever.model.encode(
            query.text,
            normalize_embeddings=True,
        )

        results = retrieve_with_embeddings(
            query_embedding,
            document_embeddings,
            retriever.documents,
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

def retrieve_with_int8(
    query_embedding,
    quantized_documents,
    document_scales,
    documents,
    top_k=5,
):
    scores = quantized_similarity(
        query_embedding,
        quantized_documents,
        document_scales,
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    return [
        (
            documents[index],
            float(scores[index]),
        )
        for index in top_indices
    ]

def evaluate_int8(
    retriever,
    quantized_documents,
    document_scales,
    examples,
):
    k_values = [1, 3, 5, 10]

    recalls = {
        k: []
        for k in k_values
    }

    for example in examples:
        query = build_query(example)

        query_embedding = retriever.model.encode(
            query.text,
            normalize_embeddings=True,
        )

        results = retrieve_with_int8(
            query_embedding,
            quantized_documents,
            document_scales,
            retriever.documents,
            top_k=max(k_values),
        )

        for k in k_values:
            recalls[k].append(
                recall_at_k(
                    query,
                    results,
                    k,
                )
            )

    return {
        k: sum(values) / len(values)
        for k, values in recalls.items()
    }

def memory_mb(embeddings):
    return embeddings.nbytes / (1024 ** 2)


if __name__ == "__main__":

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    examples = dataset["validation"].select(range(100))

    corpus = build_corpus(examples)

    print("Corpus size:", len(corpus))

    retriever = DenseRetriever()

    # Build the normal FP32 embedding index
    retriever.index(corpus.documents)

    fp32_embeddings = retriever.document_embeddings

    # Create compressed FP16 copy
    fp16_embeddings = fp32_embeddings.astype(
        "float16"
    )

    print("\nEmbedding storage:")
    print(
        f"FP32: {memory_mb(fp32_embeddings):.4f} MB"
    )
    print(
        f"FP16: {memory_mb(fp16_embeddings):.4f} MB"
    )

    # -------------------------
    # FP32 evaluation
    # -------------------------

    print("\nFP32 Retrieval")

    fp32_recall = evaluate_embeddings(
        retriever,
        fp32_embeddings,
        examples,
    )

    for k, recall in fp32_recall.items():
        print(f"Recall@{k}: {recall:.3f}")

    # -------------------------
    # FP16 evaluation
    # -------------------------

    print("\nFP16 Retrieval")

    fp16_recall = evaluate_embeddings(
        retriever,
        fp16_embeddings,
        examples,
    )

    for k, recall in fp16_recall.items():
        print(f"Recall@{k}: {recall:.3f}")


    int8_embeddings, scales = quantize_embeddings(
        fp32_embeddings
    )

    dequantized_embeddings = dequantize_embeddings(
        int8_embeddings,
        scales,
    )

    print("\nINT8 Retrieval")

    int8_recall = evaluate_embeddings(
        retriever,
        dequantized_embeddings,
        examples,
    )

    for k, recall in int8_recall.items():
        print(f"Recall@{k}: {recall:.3f}")

    int8_embeddings, int8_scales = (
        quantize_embeddings(
            fp32_embeddings
        )
    )

    print(
        "INT8 memory:",
        int8_embeddings.nbytes / (1024 ** 2),
        "MB",
    )

    print(
        "INT8 scale memory:",
        int8_scales.nbytes / (1024 ** 2),
        "MB",
    )

    print("\nINT8 Direct Retrieval")

    int8_recall = evaluate_int8(
        retriever,
        int8_embeddings,
        int8_scales,
        examples,
    )

    for k, recall in int8_recall.items():
        print(
            f"Recall@{k}: {recall:.3f}"
        )