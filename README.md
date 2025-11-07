# 🤖 Text Classifier

A machine learning project for classifying text into categories using Logistic Regression and TF-IDF vectorization.

## 📋 Project Structure

```
text-classifier/
├── main.py                    # Training script (trains and saves the model)
├── predict.py                 # Standalone prediction script
├── interface.py               # Command-line interface
├── web_interface.py          # Web-based interface (Flask)
├── training_data.csv         # Your training dataset
├── text_classifier_model.pkl # Saved trained model (generated)
└── tfidf_vectorizer.pkl      # Saved vectorizer (generated)
```

## 🚀 Quick Start Guide

### Step 1: Train and Save the Model

First, train your model and save it for later use:

```bash
python main.py
```

This will:
- Load and preprocess your training data
- Train a Logistic Regression classifier
- Display accuracy and evaluation metrics
- **Save the model** as `text_classifier_model.pkl`
- **Save the vectorizer** as `tfidf_vectorizer.pkl`

### Step 2: Use the Model with an Interface

After training, you have 3 options to use your model:

#### Option 1: Python Script (Simple)

```bash
python predict.py
```

Use this in your own code:
```python
from predict import predict_category

text = "Your text here"
category, confidence = predict_category(text)
print(f"Category: {category}, Confidence: {confidence:.2%}")
```

#### Option 2: Command-Line Interface

```bash
python interface.py
```

Interactive CLI where you can type text and get instant predictions.

#### Option 3: Web Interface (Recommended)

```bash
python web_interface.py
```

Then open your browser to: **http://127.0.0.1:5000**

Beautiful web interface with:
- Real-time text classification
- Confidence scores
- Probability visualization for all categories

## 📦 Installation

### Required Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn joblib flask
```

Or create a `requirements.txt`:

```txt
pandas
scikit-learn
matplotlib
seaborn
joblib
flask
```

Then install:
```bash
pip install -r requirements.txt
```

## 📊 Model Details

- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 
  - Unigrams and bigrams
  - Max 10,000 features
  - English stop words removed
- **Train/Test Split**: 80/20

## 🔧 Customization

### Changing the Model

Edit `main.py` to try different algorithms:

```python
# Current: Logistic Regression
clf = LogisticRegression(max_iter=1000)

# Try Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

# Try SVM
from sklearn.svm import SVC
clf = SVC(kernel='linear', probability=True)
```

### Tuning TF-IDF Parameters

```python
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),    # Change to (1, 3) for trigrams
    max_features=10000,    # Increase for more features
    min_df=2,              # Minimum document frequency
    max_df=0.8             # Maximum document frequency
)
```

## 📈 Model Evaluation

After training, you'll see:
- **Accuracy Score**: Overall correctness
- **Classification Report**: Precision, Recall, F1-Score per category
- **Confusion Matrix**: Visual heatmap of predictions

## 💾 Saved Model Files

The training process generates:
- `text_classifier_model.pkl`: The trained classifier
- `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer

**Both files are required** for making predictions!

## 🌐 API Endpoint (Web Interface)

When running `web_interface.py`, you can also use the API:

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

## 🔍 Troubleshooting

### Model files not found
```
❌ Error: Model files not found!
```
**Solution**: Run `python main.py` first to train and save the model.

### Import errors
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: Install dependencies: `pip install scikit-learn`

### Port already in use (Web Interface)
```
Address already in use
```
**Solution**: Change the port in `web_interface.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

## 📝 Next Steps

1. **Improve accuracy**: Try different algorithms, tune hyperparameters
2. **Add more data**: More training data = better performance
3. **Deploy**: Host your web interface on Heroku, AWS, or Azure
4. **Add features**: Implement batch prediction, API authentication, etc.

## 🤝 Contributing

Feel free to fork this project and make improvements!

## 📄 License

MIT License - feel free to use this for your projects!
