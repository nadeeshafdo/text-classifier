# Text Classifier — Project Idea → Domain Mapping

A multi-label text classifier that takes a student's final year project idea (as a text paragraph) and predicts which academic domains it belongs to, each with a confidence score.

Designed to help final year students find the most relevant supervisor for their project.

## Domains

The model classifies into **30 atomic domains** including Machine Learning, Deep Learning, NLP, Computer Vision, IoT, Cybersecurity, Healthcare, Finance, and more. See [`src/domains.py`](src/domains.py) for the full list.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py train

# Classify a project idea
python main.py predict "An IoT-based smart irrigation system using soil sensors and machine learning"

# Interactive mode
python main.py predict --interactive
```

## Output Format

The model returns **all 30 domains** with confidence scores (0.0 – 1.0), sorted by confidence descending:

```
Domain                               Confidence
-----------------------------------------------
  IoT                                   92.34%  ██████████████████
  Machine Learning                      87.12%  █████████████████
  Agriculture                           71.45%  ██████████████
  ...
```

No threshold is baked in — your application decides the acceptance cutoff.

## Architecture

- **Vectorizer**: TF-IDF with unigrams + bigrams, sublinear TF
- **Classifier**: OneVsRestClassifier(CalibratedClassifierCV(LinearSVC))
- **Preprocessing**: NLTK-based pipeline (lowercase → URL/email removal → tokenize → stopword removal → lemmatize)
- **Training data**: 245 labeled examples across 30 domains

## Project Structure

```
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
├── training_data.csv        # Labeled training data
├── src/
│   ├── domains.py           # Domain label definitions
│   ├── preprocessor.py      # Text cleaning pipeline
│   ├── train.py             # Training pipeline
│   └── predict.py           # Inference module
├── models/                  # Saved models (generated)
├── tests/
│   └── test_classifier.py   # Unit tests
└── README.md
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

## API Usage

```python
from src.predict import predict

results = predict("Your project idea text here")
# Returns: [{"domain": "Machine Learning", "confidence": 0.91}, ...]

# Filter by your threshold
threshold = 0.5
relevant = [r for r in results if r["confidence"] >= threshold]
```
