"""SDK signature parity tests for mock Vertex AI modules.

These tests assert that our mocks expose the same interface as the real
vertexai SDK so that migration is a one-line import swap.
They do NOT require the real google-cloud-aiplatform package.
"""
from __future__ import annotations

import inspect

import pytest

from mocks.vertex_embedding import (
    MockTextEmbedding,
    MockTextEmbeddingModel,
    MockTextEmbeddingStatistics,
)
from mocks.vertex_generative import MockGenerationResponse, MockGenerativeModel


# ---------------------------------------------------------------------------
# TextEmbeddingModel signature parity
# ---------------------------------------------------------------------------


def test_mock_embedding_model_has_from_pretrained():
    assert hasattr(MockTextEmbeddingModel, "from_pretrained")
    assert callable(MockTextEmbeddingModel.from_pretrained)


def test_mock_embedding_model_from_pretrained_accepts_model_name():
    sig = inspect.signature(MockTextEmbeddingModel.from_pretrained)
    params = list(sig.parameters)
    assert "model_name" in params


def test_mock_embedding_model_has_get_embeddings():
    model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    assert hasattr(model, "get_embeddings")
    assert callable(model.get_embeddings)


def test_mock_embedding_get_embeddings_returns_list(fake_model):
    mock = MockTextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003", embedding_model=fake_model
    )
    results = mock.get_embeddings(["hello", "world"])
    assert isinstance(results, list)
    assert len(results) == 2


def test_mock_embedding_result_has_values_attribute(fake_model):
    mock = MockTextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003", embedding_model=fake_model
    )
    result = mock.get_embeddings(["test text"])
    assert hasattr(result[0], "values")
    assert isinstance(result[0].values, list)
    assert all(isinstance(v, float) for v in result[0].values)


def test_mock_embedding_result_has_statistics_attribute(fake_model):
    """Real SDK returns TextEmbedding.statistics.token_count."""
    mock = MockTextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003", embedding_model=fake_model
    )
    result = mock.get_embeddings(["cloud run autoscaling"])
    assert hasattr(result[0], "statistics")
    assert hasattr(result[0].statistics, "token_count")
    assert isinstance(result[0].statistics.token_count, int)
    assert result[0].statistics.token_count > 0


def test_mock_embedding_dimension_consistent(fake_model):
    mock = MockTextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003", embedding_model=fake_model
    )
    results = mock.get_embeddings(["a", "b", "c"])
    dims = [len(r.values) for r in results]
    assert len(set(dims)) == 1, "All embeddings must have the same dimension"


def test_mock_embedding_has_embed_documents(fake_model):
    """Mock must expose embed_documents() to satisfy EmbeddingModel Protocol."""
    mock = MockTextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003", embedding_model=fake_model
    )
    assert hasattr(mock, "embed_documents")
    result = mock.embed_documents(["cloud run autoscaling"])
    assert result.ndim == 2
    assert result.shape[0] == 1


def test_mock_embedding_has_embed_query(fake_model):
    """Mock must expose embed_query() to satisfy EmbeddingModel Protocol."""
    mock = MockTextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003", embedding_model=fake_model
    )
    assert hasattr(mock, "embed_query")
    result = mock.embed_query("pubsub flow control")
    assert result.ndim == 2
    assert result.shape[0] == 1


def test_mock_embedding_query_and_document_same_shape(fake_model):
    """embed_query() and embed_documents() must return the same dimension.

    gecko does not use asymmetric prefixes so both encode identically.
    """
    mock = MockTextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003", embedding_model=fake_model
    )
    doc_emb = mock.embed_documents(["pubsub flow control"])
    qry_emb = mock.embed_query("pubsub flow control")
    assert doc_emb.shape == qry_emb.shape


def test_mock_embedding_has_get_dimension(fake_model):
    mock = MockTextEmbeddingModel.from_pretrained(
        "textembedding-gecko@003", embedding_model=fake_model
    )
    assert hasattr(mock, "get_dimension")
    assert isinstance(mock.get_dimension(), int)
    assert mock.get_dimension() > 0


# ---------------------------------------------------------------------------
# GenerativeModel signature parity
# ---------------------------------------------------------------------------


def test_mock_generative_model_instantiable():
    model = MockGenerativeModel("gemini-1.5-flash", mode="good_expansion")
    assert model is not None


def test_mock_generative_model_has_generate_content():
    model = MockGenerativeModel("gemini-1.5-flash")
    assert hasattr(model, "generate_content")
    assert callable(model.generate_content)


def test_mock_generative_model_has_predict():
    """Mirrors older TextGenerationModel.predict() API."""
    model = MockGenerativeModel("text-bison@002")
    assert hasattr(model, "predict")
    assert callable(model.predict)


def test_mock_generative_generate_content_returns_text():
    model = MockGenerativeModel("gemini-1.5-flash", mode="good_expansion")
    response = model.generate_content("Query: PubSub flow control")
    assert hasattr(response, "text")
    assert isinstance(response.text, str)
    assert len(response.text) > 0


def test_mock_generative_predict_returns_text():
    model = MockGenerativeModel("text-bison@002", mode="null_expansion")
    response = model.predict("Query: cloud run scaling")
    assert hasattr(response, "text")
    assert isinstance(response.text, str)


def test_mock_generative_null_expansion_returns_original():
    """null_expansion must always return the original query — for any input."""
    model = MockGenerativeModel("gemini", mode="null_expansion")
    # Test with a query that IS in the EXPANSIONS dict
    q_in_dict = "How does the system handle peak load?"
    assert model.generate_content(f"Query: {q_in_dict}").text == q_in_dict
    # Test with a query that is NOT in the EXPANSIONS dict (generic fallback path)
    q_unlisted = "What is the maximum upload size for Cloud Storage?"
    assert model.generate_content(f"Query: {q_unlisted}").text == q_unlisted


def test_mock_generative_modes_produce_different_output():
    # Use q1 which IS in _EXPANSIONS dict — guarantees good != noisy
    query = "How does the system handle peak load?"
    prompt = f"Query: {query}"
    good = MockGenerativeModel("gemini", mode="good_expansion").generate_content(prompt).text
    noisy = MockGenerativeModel("gemini", mode="noisy_expansion").generate_content(prompt).text
    null = MockGenerativeModel("gemini", mode="null_expansion").generate_content(prompt).text

    assert good != noisy
    assert null == query  # null always returns the original unchanged
    assert good != null


def test_mock_generative_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        MockGenerativeModel("gemini", mode="bad_mode")


def test_from_pretrained_factory():
    model = MockGenerativeModel.from_pretrained("gemini-1.5-flash", mode="noisy_expansion")
    assert model.mode == "noisy_expansion"
