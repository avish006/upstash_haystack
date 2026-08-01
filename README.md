# upstash-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/upstash-haystack?color=blue&label=pypi)](https://pypi.org/project/upstash-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/upstash-haystack)](https://pypi.org/project/upstash-haystack)
[![CI](https://github.com/avish006/template-repo/actions/workflows/test.yml/badge.svg)](https://github.com/avish006/template-repo/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://spdx.org/licenses/Apache-2.0.html)

> **Upstash Vector** integration for [Haystack](https://haystack.deepset.ai/) — serverless, scalable vector search with zero infrastructure.

---

## Overview

`upstash-haystack` brings [Upstash Vector](https://upstash.com/vector) into the Haystack ecosystem. Upstash Vector is a serverless, pay-as-you-go vector database with a generous free tier — no servers to provision, no clusters to manage.

### Components

| Component | Description |
|---|---|
| `UpstashDocumentStore` | Full-featured document store backed by Upstash Vector |
| `UpstashEmbeddingRetriever` | Dense retrieval using cosine/dot-product similarity |
| `UpstashHybridRetriever` | Dense + sparse hybrid search via native Reciprocal Rank Fusion (RRF) |

---

## Installation

```bash
pip install upstash-haystack
```

---

## Quick Start

### 1. Create an Upstash Vector index

Sign up at [console.upstash.com](https://console.upstash.com/) and create a Vector index. Copy the **REST URL** and **REST Token** from the dashboard.

```bash
export UPSTASH_VECTOR_REST_URL="https://your-endpoint.upstash.io"
export UPSTASH_VECTOR_REST_TOKEN="your-token"
```

### 2. Dense (Embedding) Retrieval

```python
from haystack import Document, Pipeline
from haystack_integrations.document_stores.upstash import UpstashDocumentStore
from haystack_integrations.components.retrievers.upstash import UpstashEmbeddingRetriever

# Initialize the document store (reads credentials from env vars)
document_store = UpstashDocumentStore()

# Write documents with embeddings
docs = [
    Document(content="The capital of France is Paris.", embedding=[0.1, 0.2, ...]),
    Document(content="The capital of Germany is Berlin.", embedding=[0.4, 0.5, ...]),
]
document_store.write_documents(docs)

# Retrieve the top-k most similar documents
retriever = UpstashEmbeddingRetriever(document_store=document_store)
result = retriever.run(query_embedding=[0.1, 0.2, ...], top_k=1)
print(result["documents"])
```

### 3. Hybrid Retrieval (Dense + Sparse)

Upstash Vector natively supports hybrid search via Reciprocal Rank Fusion (RRF), combining dense and sparse signals for superior relevance.

```python
from haystack.dataclasses import SparseEmbedding
from haystack_integrations.components.retrievers.upstash import UpstashHybridRetriever

retriever = UpstashHybridRetriever(document_store=document_store)

result = retriever.run(
    query_embedding=[0.1, 0.2, ...],
    query_sparse_embedding=SparseEmbedding(indices=[0, 5, 12], values=[0.9, 0.4, 0.2]),
    top_k=5,
)
print(result["documents"])
```

### 4. Filtering

```python
# Equality filter
docs = document_store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "science"})

# AND operator
docs = document_store.filter_documents(
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.category", "operator": "==", "value": "science"},
            {"field": "meta.year", "operator": ">", "value": 2020},
        ],
    }
)
```

---

## Configuration

The document store is configured via environment variables or explicit `Secret` objects:

```python
from haystack.utils.auth import Secret
from haystack_integrations.document_stores.upstash import UpstashDocumentStore

store = UpstashDocumentStore(
    url=Secret.from_env_var("UPSTASH_VECTOR_REST_URL"),
    token=Secret.from_env_var("UPSTASH_VECTOR_REST_TOKEN"),
)
```

---

## Development

This project uses [Hatch](https://hatch.pypa.io/) for environment and dependency management.

```bash
# Format and lint
hatch run fmt

# Type checking
hatch run test:types

# Unit tests (mocked, no credentials needed)
hatch run test:unit

# Integration tests (requires live Upstash credentials)
export UPSTASH_VECTOR_REST_URL="..."
export UPSTASH_VECTOR_REST_TOKEN="..."
hatch run test:integration
```

---

## License

`upstash-haystack` is distributed under the terms of the [Apache 2.0](https://spdx.org/licenses/Apache-2.0.html) license.
