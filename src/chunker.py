import re

from .models import Document

def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if sentence.strip()
    ]


def chunk_document(
    document: Document,
    chunk_size: int = 100,
    overlap: int = 0,
) -> list[Document]:
    sentences = split_sentences(document.text)

    chunks = []
    current_sentences = []
    current_word_count = 0
    chunk_index = 0

    start_index = 0

    while start_index < len(sentences):
        current_sentences = []
        current_word_count = 0

        index = start_index

        while index < len(sentences):
            sentence = sentences[index]
            sentence_word_count = len(sentence.split())

            if (
                current_sentences
                and current_word_count + sentence_word_count > chunk_size
            ):
                break

            current_sentences.append(sentence)
            current_word_count += sentence_word_count
            index += 1

        if not current_sentences:
            break

        chunks.append(
            Document(
                id=f"{document.id}::chunk_{chunk_index}",
                title=document.title,
                text=" ".join(current_sentences),
                metadata={
                    **document.metadata,
                    "parent_id": document.id,
                    "chunk_index": chunk_index,
                },
            )
        )

        chunk_index += 1

        if index >= len(sentences):
            break

        # No overlap: start immediately after the current chunk.
        if overlap <= 0:
            start_index = index
            continue

        # Move backwards by sentences until approximately
        # 'overlap' words are included.
        overlap_words = 0
        overlap_start = index

        while overlap_start > start_index:
            previous_sentence = sentences[overlap_start - 1]
            previous_word_count = len(previous_sentence.split())

            if overlap_words + previous_word_count > overlap:
                break

            overlap_words += previous_word_count
            overlap_start -= 1

        start_index = overlap_start

    return chunks

# if __name__ == "__main__":
#     from datasets import load_dataset

#     from .dataset import build_documents
#     from .chunker import chunk_document


#     dataset = load_dataset(
#         "hotpotqa/hotpot_qa",
#         "distractor",
#     )

#     example = dataset["validation"][0]

#     document = build_documents(example)[0]
#     print("Sentences:", len(split_sentences(document.text)))
#     chunks = chunk_document(
#         document,
#         chunk_size=50,
#         overlap=20,
#     )
#     print("Document:")
#     print(document.title)
#     print("Words:", len(document.text.split()))

#     print("\nChunks:", len(chunks))

#     for chunk in chunks:
#         print(
#             f"\n{chunk.id}"
#         )
#         print(
#             f"Words: {len(chunk.text.split())}"
#         )
#         print(chunk.text[:200])

if __name__ == "__main__":
    from datasets import load_dataset
    from .dataset import build_documents, build_query
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
    )

    example = dataset["validation"][10]

    documents = build_documents(example)

    for document in documents:
        sentences = split_sentences(document.text)

        if len(sentences) >= 6:
            print(f"Document: {document.title}")
            print(f"Sentences: {len(sentences)}")
            print(f"Words: {len(document.text.split())}")

            chunks = chunk_document(
                document,
                chunk_size=50,
                overlap=20,
            )

            print(f"Chunks: {len(chunks)}")

            for chunk in chunks:
                print()
                print(chunk.id)
                print(f"Words: {len(chunk.text.split())}")
                print(chunk.text)