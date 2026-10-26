from app.services.claim_extractor import ClaimExtractor


def extract(text):
    claims, warnings = ClaimExtractor().extract(text)
    return [c.text for c in claims], warnings


def test_splits_coordinated_clauses_and_resolves_pronouns():
    texts, _ = extract("Albert Einstein was born in Ulm, Germany, and he won the Nobel Prize in Physics in 1921. "
                       "He died in Princeton in 1955.")
    assert "Albert Einstein was born in Ulm, Germany." in texts
    assert "Albert Einstein won the Nobel Prize in Physics in 1921." in texts
    assert "Albert Einstein died in Princeton in 1955." in texts


def test_shared_subject_copied_into_clause():
    texts, _ = extract("The Eiffel Tower is located in Paris and was completed in 1889.")
    assert texts == ["The Eiffel Tower is located in Paris.", "The Eiffel Tower was completed in 1889."]


def test_skips_non_factual_sentences():
    texts, _ = extract("Hello! How can I help you today? I think Paris is lovely. Please verify this.")
    assert texts == []


def test_limits_number_of_claims():
    text = " ".join(f"Paris had {i} museums in {1900 + i}." for i in range(20))
    claims, warnings = ClaimExtractor(max_claims=5).extract(text)
    assert len(claims) == 5
    assert any("only the first 5" in w for w in warnings)


def test_pronoun_subject_sentence_without_entities_is_resolved():
    claims, _ = ClaimExtractor().extract("Albert Einstein was born in Ulm. He developed the theory of relativity.")
    assert "Albert Einstein developed the theory of relativity." in [c.text for c in claims]


def test_topic_from_non_factual_sentence_resolves_later_pronoun():
    claims, _ = ClaimExtractor().extract("Python is a language. It was created by Guido van Rossum.")
    assert [c.text for c in claims] == ["Python was created by Guido van Rossum."]
