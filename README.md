# RAG From Scratch

A hands-on implementation of Retrieval-Augmented Generation (RAG)
built from first principles.

The goal is to understand RAG internals before introducing frameworks
such as LangChain, LlamaIndex, FAISS, or vector databases.

## Goals

We will progressively build:

1. Dataset ingestion
2. Document representation
3. Chunking
4. Embeddings
5. Vector indexing
6. Similarity search
7. Retrieval evaluation
8. LLM-based generation
9. Hybrid retrieval
10. Reranking
11. Query rewriting
12. Context compression
13. Adaptive retrieval
14. Advanced RAG
15. Resource-efficient / edge RAG

## Dataset

Initial dataset:

- HotpotQA
- Source: Hugging Face

The dataset is downloaded programmatically and is not committed
to this repository.

## Technology

- Python
- uv
- NumPy
- Sentence Transformers
- Hugging Face Datasets

## Philosophy

We implement important RAG components ourselves first.

External frameworks and optimized libraries will be introduced later
to understand what they provide and when they are useful.

## Project Status

### Milestone 0
- [x] Repository created
- [ ] Project bootstrap

### Milestone 1
- [ ] Load HotpotQA
- [ ] Understand dataset structure
- [ ] Build corpus representation

### Milestone 2
- [ ] Implement chunking

### Milestone 3
- [ ] Implement embeddings

### Milestone 4
- [ ] Implement vector index

### Milestone 5
- [ ] Implement retrieval

### Milestone 6
- [ ] Implement retrieval evaluation