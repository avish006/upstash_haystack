# SPDX-FileCopyrightText: 2026-present Avish Sinha <avishsinha10@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.upstash import UpstashDocumentStore

# ---------------------------------------------------------------------------
# Unit tests (mocked) -- no network needed
# ---------------------------------------------------------------------------


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_init_with_defaults(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")
    UpstashDocumentStore()
    mock_index.assert_called_once_with(url="http://test", token="test-token")


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_to_dict_from_dict(mock_index, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")
    store = UpstashDocumentStore()

    data = store.to_dict()
    assert data == {
        "type": "haystack_integrations.document_stores.upstash.document_store.UpstashDocumentStore",
        "init_parameters": {
            "url": {"env_vars": ["UPSTASH_VECTOR_REST_URL"], "strict": True, "type": "env_var"},
            "token": {"env_vars": ["UPSTASH_VECTOR_REST_TOKEN"], "strict": True, "type": "env_var"},
        },
    }

    restored = UpstashDocumentStore.from_dict(data)
    assert restored.url.resolve_value() == "http://test"


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_count_documents(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_info = MagicMock()
    mock_info.vector_count = 42
    mock_instance.info.return_value = mock_info
    mock_index.return_value = mock_instance

    store = UpstashDocumentStore()
    assert store.count_documents() == 42


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_write_documents(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance
    mock_instance.fetch.return_value = []

    store = UpstashDocumentStore()
    docs = [Document(id="1", content="test", embedding=[0.1, 0.2, 0.3])]
    store.write_documents(docs)

    mock_instance.upsert.assert_called_once()
    _args, kwargs = mock_instance.upsert.call_args
    assert len(kwargs["vectors"]) == 1
    assert kwargs["vectors"][0]["id"] == "1"
    assert kwargs["vectors"][0]["vector"] == [0.1, 0.2, 0.3]
    assert kwargs["vectors"][0]["data"] == "test"
    assert "metadata" in kwargs["vectors"][0]


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_write_documents_no_embedding(mock_index, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    store = UpstashDocumentStore()
    docs = [Document(id="1", content="test")]
    with pytest.raises(DocumentStoreError):
        store.write_documents(docs)


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_write_documents_duplicate_fail(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance
    mock_res = MagicMock()
    mock_res.id = "1"
    mock_instance.fetch.return_value = [mock_res]

    store = UpstashDocumentStore()
    docs = [Document(id="1", content="test", embedding=[0.1, 0.2, 0.3])]
    with pytest.raises(DuplicateDocumentError):
        store.write_documents(docs, policy=DuplicatePolicy.FAIL)


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_write_documents_duplicate_skip(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance
    mock_res = MagicMock()
    mock_res.id = "1"
    mock_instance.fetch.return_value = [mock_res]

    store = UpstashDocumentStore()
    docs = [Document(id="1", content="test", embedding=[0.1, 0.2, 0.3])]
    written = store.write_documents(docs, policy=DuplicatePolicy.SKIP)

    assert written == 0
    mock_instance.upsert.assert_not_called()


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_delete_documents(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance

    store = UpstashDocumentStore()
    store.delete_documents(["1", "2"])

    mock_instance.delete.assert_called_once_with(["1", "2"])


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_filter_documents(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_info = MagicMock()
    mock_info.dimension = 3
    mock_instance.info.return_value = mock_info

    mock_res = MagicMock()
    mock_res.id = "1"
    mock_res.data = "test"
    mock_res.vector = [0.1, 0.2, 0.3]
    mock_res.metadata = {"genre": "tech"}
    mock_instance.query.return_value = [mock_res]

    mock_index.return_value = mock_instance

    store = UpstashDocumentStore()
    docs = store.filter_documents(filters={"field": "meta.genre", "operator": "==", "value": "tech"})

    assert len(docs) == 1
    assert docs[0].id == "1"
    assert docs[0].content == "test"
    assert docs[0].meta["genre"] == "tech"

    _args, kwargs = mock_instance.query.call_args
    assert kwargs["vector"] == [1.0, 0.0, 0.0]
    assert kwargs["filter"] == "genre = 'tech'"
    assert kwargs["include_data"] is True


# ---------------------------------------------------------------------------
# Integration tests -- require live Upstash credentials
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    "UPSTASH_VECTOR_REST_URL" not in os.environ or "UPSTASH_VECTOR_REST_TOKEN" not in os.environ,
    reason="No UPSTASH_VECTOR_REST_URL or UPSTASH_VECTOR_REST_TOKEN provided",
)
class TestUpstashDocumentStore:
    """
    Full integration tests against a live Upstash Vector index.

    Upstash Vector requires every document to have an embedding whose
    dimension matches the index dimension. We fetch the dimension once per
    test session via store._index.info().dimension and build all embeddings
    from that. A short sleep is added after writes to allow Upstash to
    become consistent.
    """

    SLEEP = 1  # seconds to wait for Upstash eventual consistency

    @pytest.fixture(autouse=True)
    def document_store(self):
        store = UpstashDocumentStore()
        self._dim = store._index.info().dimension
        self._store = store

        # Clear the index before every test
        existing = store.filter_documents()
        if existing:
            store.delete_documents([d.id for d in existing])
            time.sleep(self.SLEEP)

        yield store

        # Clean up after every test
        existing = store.filter_documents()
        if existing:
            store.delete_documents([d.id for d in existing])

    def _emb(self):
        """Return a valid non-zero embedding for the current index."""
        return [0.1] * self._dim

    def _doc(self, **kwargs) -> Document:
        """Create a Document with a valid embedding."""
        kwargs.setdefault("embedding", self._emb())
        return Document(**kwargs)

    # ---- count ----

    def test_count_empty(self, document_store):
        assert document_store.count_documents() == 0

    def test_count_not_empty(self, document_store):
        document_store.write_documents([self._doc(id="c1", content="hello")])
        time.sleep(self.SLEEP)
        assert document_store.count_documents() == 1

    # ---- write ----

    def test_write_documents(self, document_store):
        doc = self._doc(id="w1", content="hello")
        assert document_store.write_documents([doc]) == 1
        time.sleep(self.SLEEP)
        assert document_store.count_documents() == 1

    def test_write_documents_duplicate_overwrite(self, document_store):
        document_store.write_documents([self._doc(id="dup", content="original")])
        time.sleep(self.SLEEP)
        document_store.write_documents([self._doc(id="dup", content="updated")], policy=DuplicatePolicy.OVERWRITE)
        time.sleep(self.SLEEP)
        result = document_store.filter_documents()
        assert len(result) == 1
        assert result[0].content == "updated"

    def test_write_documents_duplicate_skip(self, document_store):
        document_store.write_documents([self._doc(id="dup", content="original")])
        time.sleep(self.SLEEP)
        written = document_store.write_documents([self._doc(id="dup", content="should not overwrite")], policy=DuplicatePolicy.SKIP)
        assert written == 0
        time.sleep(self.SLEEP)
        result = document_store.filter_documents()
        assert result[0].content == "original"

    def test_write_documents_duplicate_fail(self, document_store):
        document_store.write_documents([self._doc(id="dup", content="original")])
        time.sleep(self.SLEEP)
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([self._doc(id="dup", content="should fail")], policy=DuplicatePolicy.FAIL)

    def test_write_documents_invalid_input(self, document_store):
        doc = Document(id="no-emb", content="no embedding")
        with pytest.raises(DocumentStoreError):
            document_store.write_documents([doc])

    # ---- delete ----

    def test_delete_documents(self, document_store):
        document_store.write_documents([self._doc(id="del-me", content="bye")])
        time.sleep(self.SLEEP)
        document_store.delete_documents(["del-me"])
        time.sleep(self.SLEEP)
        assert document_store.count_documents() == 0

    def test_delete_documents_empty_document_store(self, document_store):
        document_store.delete_documents(["non-existent"])
        assert document_store.count_documents() == 0

    def test_delete_documents_non_existing_document(self, document_store):
        document_store.write_documents([self._doc(id="real", content="real doc")])
        time.sleep(self.SLEEP)
        document_store.delete_documents(["does-not-exist"])
        time.sleep(self.SLEEP)
        assert document_store.count_documents() == 1

    # ---- filter ----

    def test_no_filters(self, document_store):
        document_store.write_documents([
            self._doc(id="a", content="doc a"),
            self._doc(id="b", content="doc b"),
        ])
        time.sleep(self.SLEEP)
        result = document_store.filter_documents()
        assert len(result) == 2

    def test_filter_by_meta_equal(self, document_store):
        document_store.write_documents([
            self._doc(id="match", content="yes", meta={"category": "A"}),
            self._doc(id="no-match", content="no", meta={"category": "B"}),
        ])
        time.sleep(self.SLEEP)
        result = document_store.filter_documents(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert len(result) == 1
        assert result[0].id == "match"

    def test_filter_by_meta_not_equal(self, document_store):
        document_store.write_documents([
            self._doc(id="a", content="a", meta={"category": "A"}),
            self._doc(id="b", content="b", meta={"category": "B"}),
        ])
        time.sleep(self.SLEEP)
        result = document_store.filter_documents(
            filters={"field": "meta.category", "operator": "!=", "value": "A"}
        )
        assert len(result) == 1
        assert result[0].id == "b"

    def test_filter_by_meta_greater_than(self, document_store):
        document_store.write_documents([
            self._doc(id="low", content="low", meta={"score": 1}),
            self._doc(id="high", content="high", meta={"score": 10}),
        ])
        time.sleep(self.SLEEP)
        result = document_store.filter_documents(
            filters={"field": "meta.score", "operator": ">", "value": 5}
        )
        assert len(result) == 1
        assert result[0].id == "high"

    def test_filter_by_meta_in(self, document_store):
        document_store.write_documents([
            self._doc(id="cat-a", content="cat a", meta={"category": "A"}),
            self._doc(id="cat-b", content="cat b", meta={"category": "B"}),
            self._doc(id="cat-c", content="cat c", meta={"category": "C"}),
        ])
        time.sleep(self.SLEEP)
        result = document_store.filter_documents(
            filters={"field": "meta.category", "operator": "in", "value": ["A", "B"]}
        )
        assert len(result) == 2
        assert {d.id for d in result} == {"cat-a", "cat-b"}

    def test_filter_and_operator(self, document_store):
        document_store.write_documents([
            self._doc(id="ab", content="ab", meta={"category": "A", "score": 10}),
            self._doc(id="a-low", content="a low", meta={"category": "A", "score": 1}),
            self._doc(id="b-high", content="b high", meta={"category": "B", "score": 10}),
        ])
        time.sleep(self.SLEEP)
        result = document_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.score", "operator": ">", "value": 5},
                ],
            }
        )
        assert len(result) == 1
        assert result[0].id == "ab"

    def test_filter_or_operator(self, document_store):
        document_store.write_documents([
            self._doc(id="a", content="a", meta={"category": "A"}),
            self._doc(id="b", content="b", meta={"category": "B"}),
            self._doc(id="c", content="c", meta={"category": "C"}),
        ])
        time.sleep(self.SLEEP)
        result = document_store.filter_documents(
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "A"},
                    {"field": "meta.category", "operator": "==", "value": "B"},
                ],
            }
        )
        assert len(result) == 2
        assert {d.id for d in result} == {"a", "b"}
