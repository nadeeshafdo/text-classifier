import pandas as pd

# Load the CSV dataset
df = pd.read_csv('training_data.csv')  # Save your generated CSV as this file

# Check the first few rows
print(df.head())

# Ensure correct column names
# Should be: 'text' and 'label'

from sklearn.model_selection import train_test_split

X = df['text']      # Text input
y = df['label']     # Category labels

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2),  # Use unigrams and bigrams
    max_features=10000   # Limit vocabulary size
)

# Fit on training data and transform both train and test
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression

# Initialize and train the model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_tfidf, y_train)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on test set
y_pred = clf.predict(X_test_tfidf)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['label'].unique()), yticklabels=sorted(df['label'].unique()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def predict_category(idea):
    # Preprocess and vectorize
    processed = vectorizer.transform([idea])
    # Predict
    prediction = clf.predict(processed)[0]
    confidence = clf.predict_proba(processed).max()  # Probability of top class
    return prediction, confidence

# Example usage
new_idea = "Building a mobile app to track daily water intake and remind users"
pred, conf = predict_category(new_idea)
print(f"Predicted Category: {pred} (Confidence: {conf:.2f})")

# Save the trained model and vectorizer
import joblib

print("\n📦 Saving model and vectorizer...")
joblib.dump(clf, 'text_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("✅ Model saved as 'text_classifier_model.pkl'")
print("✅ Vectorizer saved as 'tfidf_vectorizer.pkl'")
print("\nYou can now use these files in your interface!")

