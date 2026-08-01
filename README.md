# upstash-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/upstash-haystack.svg)](https://pypi.org/project/upstash-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/upstash-haystack.svg)](https://pypi.org/project/upstash-haystack)
[![CI](https://github.com/avish006/upstash-haystack/actions/workflows/test.yml/badge.svg)](https://github.com/avish006/upstash-haystack/actions)

An integration of [Upstash Vector](https://upstash.com/vector) with [Haystack](https://haystack.deepset.ai/).

## Features

- **`UpstashDocumentStore`**: A scalable and serverless document store utilizing Upstash Vector.
- **`UpstashEmbeddingRetriever`**: A dense retriever using the Upstash Vector integration.
- **`UpstashHybridRetriever`**: A retriever that combines dense and sparse embeddings via Upstash Vector's native Reciprocal Rank Fusion (RRF) for superior search results.

## Installation

Install the package via pip:

```bash
pip install upstash-haystack
```

## Quick Start

### 1. Setup Upstash Vector
Create an Upstash Vector index on the [Upstash Console](https://console.upstash.com/). You will need the `UPSTASH_VECTOR_REST_URL` and `UPSTASH_VECTOR_REST_TOKEN` from the console.

Set them as environment variables:
```bash
export UPSTASH_VECTOR_REST_URL="https://your-endpoint.upstash.io"
export UPSTASH_VECTOR_REST_TOKEN="your-token"
```

### 2. Basic Embedding Retrieval

```python
from haystack import Document
from haystack_integrations.document_stores.upstash import UpstashDocumentStore
from haystack_integrations.components.retrievers.upstash import UpstashEmbeddingRetriever

# Initialize the Document Store
document_store = UpstashDocumentStore()

# Index some documents
docs = [
    Document(content="The capital of France is Paris.", embedding=[0.1, 0.2, 0.3]),
    Document(content="The capital of Germany is Berlin.", embedding=[0.4, 0.5, 0.6]),
]
document_store.write_documents(docs)

# Retrieve documents
retriever = UpstashEmbeddingRetriever(document_store=document_store)
result = retriever.run(query_embedding=[0.1, 0.2, 0.3], top_k=1)

print(result["documents"])
```

## Hybrid Retrieval

Upstash Vector supports hybrid search using Reciprocal Rank Fusion (RRF). This integration natively supports it.

```python
from haystack.dataclasses import SparseEmbedding
from haystack_integrations.components.retrievers.upstash import UpstashHybridRetriever

retriever = UpstashHybridRetriever(document_store=document_store)

sparse_query = SparseEmbedding(indices=[0, 2], values=[0.8, 0.2])
dense_query = [0.1, 0.2, 0.3]

result = retriever.run(query_embedding=dense_query, query_sparse_embedding=sparse_query, top_k=5)
```

## Development

To develop `upstash-haystack` locally, you need [Hatch](https://hatch.pypa.io/).

```bash
# Run formatters
hatch run fmt

# Run type checks
hatch run test:types

# Run unit tests
hatch run test:unit
```

## License

`upstash-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
