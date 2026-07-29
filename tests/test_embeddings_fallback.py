from main import fallback_embedding


def test_fallback_embedding_returns_expected_dimensions():
    vector = fallback_embedding("sample resume text", 8)
    assert len(vector) == 8
    assert all(isinstance(value, float) for value in vector)
