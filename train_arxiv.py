import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse
from pathlib import Path

def load_data(csv_path):
    """Load the prepared dataset."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples with {df['label'].nunique()} categories")
    return df

def prepare_data(df, test_size=0.2):
    """Split data into train and test sets."""
    X = df['text']
    y = df['label']
    
    print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test, max_features=10000):
    """Vectorize text using TF-IDF."""
    print(f"\nVectorizing text with TF-IDF (max_features={max_features})...")
    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        max_df=0.8
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Training matrix shape: {X_train_tfidf.shape}")
    
    return vectorizer, X_train_tfidf, X_test_tfidf

def train_model(X_train_tfidf, y_train, model_type='logistic'):
    """Train the classifier."""
    print(f"\nTraining {model_type} classifier...")
    
    if model_type == 'logistic':
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    clf.fit(X_train_tfidf, y_train)
    print("Training complete!")
    
    return clf

def evaluate_model(clf, X_test_tfidf, y_test, categories):
    """Evaluate the model and print metrics."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return y_pred

def plot_confusion_matrix(y_test, y_pred, categories, save_path='confusion_matrix.png'):
    """Create and save confusion matrix visualization."""
    print("\nGenerating confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred, labels=sorted(categories))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=sorted(categories),
        yticklabels=sorted(categories),
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('Actual Category', fontsize=12)
    plt.title('Confusion Matrix - arXiv Classification', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def save_models(clf, vectorizer, model_path='text_classifier_model.pkl', 
                vectorizer_path='tfidf_vectorizer.pkl'):
    """Save trained model and vectorizer."""
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

def test_prediction(clf, vectorizer, sample_text, categories):
    """Test prediction on sample text."""
    print("\n" + "="*70)
    print("SAMPLE PREDICTION")
    print("="*70)
    
    processed = vectorizer.transform([sample_text])
    prediction = clf.predict(processed)[0]
    probabilities = clf.predict_proba(processed)[0]
    
    print(f"\nInput text: {sample_text[:200]}...")
    print(f"\nPredicted category: {prediction}")
    print(f"Confidence: {probabilities.max():.4f} ({probabilities.max()*100:.2f}%)")
    
    print("\nTop 5 predictions:")
    top_5_idx = np.argsort(probabilities)[-5:][::-1]
    for idx in top_5_idx:
        category = clf.classes_[idx]
        prob = probabilities[idx]
        print(f"  {category}: {prob:.4f} ({prob*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Train arXiv text classifier')
    parser.add_argument('--input', type=str, default='training_data.csv',
                        help='Input CSV file')
    parser.add_argument('--model-output', type=str, default='text_classifier_model.pkl',
                        help='Output model file')
    parser.add_argument('--vectorizer-output', type=str, default='tfidf_vectorizer.pkl',
                        help='Output vectorizer file')
    parser.add_argument('--max-features', type=int, default=20000,
                        help='Maximum features for TF-IDF')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (0-1)')
    parser.add_argument('--plot-cm', action='store_true',
                        help='Generate confusion matrix plot')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("Please run prepare_arxiv_data.py first!")
        return
    
    # Load data
    df = load_data(args.input)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, test_size=args.test_size)
    
    # Vectorize
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(
        X_train, X_test, max_features=args.max_features
    )
    
    # Train
    clf = train_model(X_train_tfidf, y_train)
    
    # Evaluate
    categories = df['label'].unique()
    y_pred = evaluate_model(clf, X_test_tfidf, y_test, categories)
    
    # Plot confusion matrix if requested
    if args.plot_cm:
        plot_confusion_matrix(y_test, y_pred, categories)
    
    # Save models
    save_models(clf, vectorizer, args.model_output, args.vectorizer_output)
    
    # Test prediction with sample
    if len(df) > 0:
        sample_text = df.iloc[0]['text']
        test_prediction(clf, vectorizer, sample_text, categories)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nYou can now use the trained model with:")
    print(f"  - python predict.py")
    print(f"  - python interface.py")
    print(f"  - python web_interface.py")

if __name__ == "__main__":
    main()
