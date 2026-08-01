import os

from haystack.dataclasses import Document, SparseEmbedding

from haystack_integrations.components.retrievers.upstash import UpstashHybridRetriever
from haystack_integrations.document_stores.upstash import UpstashDocumentStore

# Ensure you have set UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN
# For this example, we assume they are present in the environment.


def main():
    if not os.environ.get("UPSTASH_VECTOR_REST_URL") or not os.environ.get("UPSTASH_VECTOR_REST_TOKEN"):
        print("Please set UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN")
        return

    # Initialize the Document Store
    document_store = UpstashDocumentStore()

    # Create dummy documents with both dense and sparse embeddings
    docs = [
        Document(
            content="Upstash provides a serverless vector database.",
            embedding=[0.1, 0.2, 0.3],
            sparse_embedding=SparseEmbedding(indices=[0, 1], values=[0.9, 0.1]),
        ),
        Document(
            content="Haystack is an open source AI framework.",
            embedding=[0.4, 0.5, 0.6],
            sparse_embedding=SparseEmbedding(indices=[2, 3], values=[0.8, 0.2]),
        ),
    ]

    print("Writing documents...")
    document_store.write_documents(docs)

    # Initialize the Hybrid Retriever
    retriever = UpstashHybridRetriever(document_store=document_store)

    # Perform a hybrid search query
    dense_query = [0.1, 0.2, 0.3]
    sparse_query = SparseEmbedding(indices=[0, 1], values=[0.5, 0.5])

    print("Running hybrid retrieval...")
    result = retriever.run(query_embedding=dense_query, query_sparse_embedding=sparse_query, top_k=2)

    print("\nRetrieval Results:")
    for i, doc in enumerate(result["documents"]):
        print(f"Result {i + 1}:")
        print(f"  Content: {doc.content}")
        print(f"  Score: {doc.score}")
        print(f"  ID: {doc.id}")


if __name__ == "__main__":
    main()
