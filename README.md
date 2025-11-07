# Text Classifier

A machine learning text classification system using Logistic Regression and TF-IDF vectorization.

## Project Structure

```
text-classifier/
├── main.py                    # Training script
├── predict.py                 # Prediction module
├── interface.py               # Command-line interface
├── web_interface.py           # Web interface (Flask)
├── training_data.csv          # Training dataset
├── requirements.txt           # Dependencies
├── text_classifier_model.pkl  # Trained model (generated)
└── tfidf_vectorizer.pkl       # Fitted vectorizer (generated)
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python main.py
```

This will train the model and save `text_classifier_model.pkl` and `tfidf_vectorizer.pkl`.

### Make Predictions

**Option 1: Python Script**

```python
from predict import predict_category

category, confidence = predict_category("Your text here")
print(f"Category: {category}, Confidence: {confidence:.2%}")
```

**Option 2: Command-Line Interface**

```bash
python interface.py
```

**Option 3: Web Interface**

```bash
python web_interface.py
```

Access at http://127.0.0.1:5000

## Model Configuration

**Algorithm:** Logistic Regression  
**Vectorization:** TF-IDF  
**Features:** Unigrams and bigrams, max 10,000 features  
**Train/Test Split:** 80/20

## API Endpoint

The web interface provides a REST API:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

Response:
```json
{
  "category": "predicted_category",
  "confidence": 0.85,
  "probabilities": {
    "category1": 0.85,
    "category2": 0.10,
    "category3": 0.05
  }
}
```

## Customization

### Change Algorithm

Edit `main.py`:

```python
# Logistic Regression (default)
clf = LogisticRegression(max_iter=1000)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

# Support Vector Machine
from sklearn.svm import SVC
clf = SVC(kernel='linear', probability=True)
```

### Tune Vectorizer

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),    # Unigrams and bigrams
    max_features=10000,    # Vocabulary size
    min_df=2,              # Minimum document frequency
    max_df=0.8             # Maximum document frequency
)
```

## Troubleshooting

**Model files not found**  
Run `python main.py` to train and save the model.

**Import errors**  
Ensure virtual environment is activated and dependencies are installed.

**Port already in use**  
Change port in `web_interface.py`: `app.run(port=5001)`

## License

MIT License
