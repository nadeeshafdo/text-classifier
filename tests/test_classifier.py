"""
Tests for the text classifier.
"""

import pytest
from src.preprocessor import preprocess
from src.domains import DOMAINS


class TestPreprocessor:
    """Test the text preprocessing pipeline."""

    def test_lowercasing(self):
        result = preprocess("MACHINE LEARNING Is Amazing")
        assert "machine" in result
        assert "amazing" in result

    def test_url_removal(self):
        result = preprocess("Check https://example.com for more info")
        assert "https" not in result
        assert "example" not in result

    def test_email_removal(self):
        result = preprocess("Contact admin@university.edu for help")
        assert "admin" not in result
        assert "university" not in result

    def test_special_char_removal(self):
        result = preprocess("ML/AI is #1 in 2024!")
        assert "#" not in result
        assert "!" not in result

    def test_stopword_removal(self):
        result = preprocess("This is a system that uses the data")
        assert "this" not in result
        assert "the" not in result

    def test_lemmatization(self):
        result = preprocess("The systems are running processes")
        assert "system" in result
        assert "running" in result

    def test_empty_string(self):
        assert preprocess("") == ""
        assert preprocess("   ") == ""

    def test_none_input(self):
        assert preprocess(None) == ""

    def test_short_words_removed(self):
        """Words with 2 or fewer characters should be removed."""
        result = preprocess("I am an AI in ML")
        # Single/two char words should be gone
        assert " i " not in f" {result} "
        assert " am " not in f" {result} "
        assert " an " not in f" {result} "


class TestDomains:
    """Test domain label definitions."""

    def test_domain_count(self):
        assert len(DOMAINS) == 30

    def test_no_duplicates(self):
        assert len(DOMAINS) == len(set(DOMAINS))

    def test_all_strings(self):
        for domain in DOMAINS:
            assert isinstance(domain, str)
            assert len(domain) > 0


class TestTrainingPipeline:
    """Test the training data loading and model training."""

    def test_load_data(self):
        from src.train import load_data
        texts, labels = load_data()
        assert len(texts) > 0
        assert len(texts) == len(labels)
        # All labels should be lists
        for label_list in labels:
            assert isinstance(label_list, list)
            assert len(label_list) > 0

    def test_all_labels_valid(self):
        """All labels in training data should be valid domain names."""
        from src.train import load_data
        _, labels = load_data()
        all_labels = set(label for label_list in labels for label in label_list)
        invalid = all_labels - set(DOMAINS)
        assert not invalid, f"Invalid labels found in training data: {invalid}"


class TestPrediction:
    """Test prediction module (requires trained model)."""

    @pytest.fixture(autouse=True, scope="class")
    def ensure_model(self):
        """Train the model if not already present."""
        import os
        from src.train import MODEL_PATH, train
        if not os.path.exists(MODEL_PATH):
            train()

    def test_predict_returns_all_domains(self):
        from src.predict import predict
        results = predict("A machine learning system for healthcare diagnostics")
        assert len(results) == 30

    def test_predict_returns_sorted_by_confidence(self):
        from src.predict import predict
        results = predict("A deep learning model for image classification")
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_predict_confidence_range(self):
        from src.predict import predict
        results = predict("IoT sensors connected to cloud dashboard")
        for r in results:
            assert 0.0 <= r["confidence"] <= 1.0

    def test_predict_has_correct_keys(self):
        from src.predict import predict
        results = predict("Web application for e-commerce")
        for r in results:
            assert "domain" in r
            assert "confidence" in r

    def test_predict_empty_input(self):
        from src.predict import predict
        results = predict("")
        assert len(results) == 30
        assert all(r["confidence"] == 0.0 for r in results)

    def test_predict_relevant_domain_has_high_confidence(self):
        """Sanity check: ML-heavy text should score ML-related domains higher."""
        from src.predict import predict
        results = predict(
            "Training a random forest classifier to predict customer churn "
            "using feature engineering and cross-validation"
        )
        top_domains = [r["domain"] for r in results[:5]]
        assert "Machine Learning" in top_domains, (
            f"Expected 'Machine Learning' in top 5, got: {top_domains}"
        )
