from datasets import load_dataset
from tqdm import tqdm

from .dataset import build_documents, build_query
from .retriever import recall_at_k, retrieve


def main():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

    examples = dataset["validation"].select(range(100))

    recalls = {
        1: [],
        3: [],
        5: [],
        10: [],
    }

    for example in tqdm(examples, desc="Evaluating"):
        query = build_query(example)
        documents = build_documents(example)

        results = retrieve(
            query,
            documents,
            top_k=10,
        )

        for k in recalls:
            recalls[k].append(
                recall_at_k(query, results, k)
            )

    print("\nRetrieval Baseline")
    print("==================")

    for k, values in recalls.items():
        average = sum(values) / len(values)
        print(f"Recall@{k}: {average:.3f}")


if __name__ == "__main__":
    main()