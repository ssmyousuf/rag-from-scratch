import numpy as np
from .dataset import build_query
from sentence_transformers.util import cos_sim

def quantize_int8(vector):
    max_abs = np.max(np.abs(vector))

    if max_abs == 0:
        return (
            np.zeros_like(vector, dtype=np.int8),
            1.0,
        )

    scale = max_abs / 127.0

    quantized = np.round(
        vector / scale
    ).clip(-127, 127).astype(np.int8)

    return quantized, scale


def dequantize_int8(
    quantized,
    scale,
):
    return quantized.astype(
        np.float32
    ) * scale


def quantize_embeddings(embeddings):
    quantized_embeddings = np.empty(
        embeddings.shape,
        dtype=np.int8,
    )

    scales = np.empty(
        embeddings.shape[0],
        dtype=np.float32,
    )

    for index, vector in enumerate(embeddings):
        quantized, scale = quantize_int8(vector)

        quantized_embeddings[index] = quantized
        scales[index] = scale

    return quantized_embeddings, scales


def dequantize_embeddings(
    quantized_embeddings,
    scales,
):
    return (
        quantized_embeddings.astype(
            np.float32
        )
        * scales[:, np.newaxis]
    )

def quantized_similarity(
    query_embedding,
    document_embeddings,
    document_scales,
):
    quantized_query, query_scale = quantize_int8(
        query_embedding
    )

    integer_scores = (
        document_embeddings.astype(np.int32)
        @ quantized_query.astype(np.int32)
    )

    scores = (
        integer_scores.astype(np.float32)
        * query_scale
        * document_scales
    )

    return scores

def int8_dot_product(
    query,
    documents,
):
    return documents @ query

def quantized_similarity(
    query_embedding,
    document_embeddings,
    document_scales,
):
    quantized_query, query_scale = (
        quantize_int8(query_embedding)
    )

    integer_scores = (
        document_embeddings.astype(np.int32)
        @ quantized_query.astype(np.int32)
    )

    scores = (
        integer_scores.astype(np.float32)
        * query_scale
        * document_scales
    )

    return scores

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

def quantize_embeddings(embeddings):
    quantized_embeddings = np.empty(
        embeddings.shape,
        dtype=np.int8,
    )

    scales = np.empty(
        embeddings.shape[0],
        dtype=np.float32,
    )

    for index, vector in enumerate(embeddings):
        quantized, scale = quantize_int8(vector)

        quantized_embeddings[index] = quantized
        scales[index] = scale

    return quantized_embeddings, scales


def dequantize_embeddings(
    quantized_embeddings,
    scales,
):
    return (
        quantized_embeddings.astype(
            np.float32
        )
        * scales[:, np.newaxis]
    )

if __name__ == "__main__":
    from datasets import load_dataset

    from .dataset import build_corpus
    from .dense_retriever import DenseRetriever

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    examples = dataset["validation"].select(range(100))

    corpus = build_corpus(examples)

    retriever = DenseRetriever()
    retriever.index(corpus.documents)

    embeddings = retriever.document_embeddings

    quantized, scales = quantize_embeddings(
        embeddings
    )

    reconstructed = dequantize_embeddings(
        quantized,
        scales,
    )

    print("Original:", embeddings.shape)
    print("Quantized:", quantized.shape)
    print("Scales:", scales.shape)

    print(
        "FP32 memory:",
        embeddings.nbytes / (1024 ** 2),
        "MB",
    )

    print(
        "INT8 memory:",
        quantized.nbytes / (1024 ** 2),
        "MB",
    )

    print(
        "Scale memory:",
        scales.nbytes / (1024 ** 2),
        "MB",
    )

    print(
        "Maximum reconstruction error:",
        np.max(
            np.abs(
                embeddings - reconstructed
            )
        ),
    )

    example = examples[0]
    query = build_query(example)

    query_embedding = retriever.model.encode(
        query.text,
        normalize_embeddings=True,
    )

    # FP32 reference scores
    fp32_scores = cos_sim(
        query_embedding,
        embeddings,
    )[0].numpy()

    # Direct INT8 scores
    int8_scores = quantized_similarity(
        query_embedding,
        quantized,
        scales,
    )

    for index in range(5):
        print(
            f"Document {index}: "
            f"FP32={fp32_scores[index]:.6f}, "
            f"INT8={int8_scores[index]:.6f}"
        )