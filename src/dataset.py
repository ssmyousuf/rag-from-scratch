from datasets import load_dataset

from .models import Document, Query
from .retriever import recall_at_k, LexicalRetriever

from .corpus import Corpus

def build_corpus(examples):
    all_documents = []

    for example in examples:
        all_documents.extend(build_documents(example))

    return Corpus(all_documents)

def build_documents(example):
    documents = []

    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]

    for title, sentence_list in zip(titles, sentences):
        text = " ".join(sentence_list)

        documents.append(
            Document(
                id=title,
                title=title,
                text=text,
                metadata={"source": "hotpotqa"},
            )
        )

    return documents


def build_query(example):
    supporting_documents = example["supporting_facts"]["title"]

    return Query(
        id=example["id"],
        text=example["question"],
        answer=example["answer"],
        supporting_documents=list(set(supporting_documents)),
    )


def main():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")

    example = dataset["train"][0]

    documents = build_documents(example)
    query = build_query(example)

    print(f"Question: {query.text}")
    print(f"Answer: {query.answer}")
    print(f"Supporting documents: {query.supporting_documents}")
    print(f"Number of documents: {len(documents)}")
    # retriever = LexicalRetriever();
    # results = retriever.retrieve(query, top_k=5)

    # print("\nTop retrieved documents:")

    # for rank, (document, score) in enumerate(results, start=1):
    #     print(f"{rank}. {document.title} - score={score}")

    # recall = recall_at_k(query, results, k=5)

    # print(f"\nRecall@5: {recall:.2f}")

    examples = dataset["validation"].select(range(100))

    corpus = build_corpus(examples)

    print("Total documents:", len(corpus))

    print(corpus.documents[:3])

if __name__ == "__main__":
    main()