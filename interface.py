import joblib
import sys

# Load the saved model and vectorizer
try:
    clf = joblib.load('text_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: Model files not found!")
    print("Please run 'python main.py' first to train and save the model.")
    sys.exit(1)

def predict_category(text):
    """Predict the category of a given text."""
    processed = vectorizer.transform([text])
    prediction = clf.predict(processed)[0]
    confidence = clf.predict_proba(processed).max()
    probabilities = clf.predict_proba(processed)[0]
    classes = clf.classes_
    
    return prediction, confidence, dict(zip(classes, probabilities))

def main():
    """Interactive command-line interface for text classification."""
    print("=" * 70)
    print("📝 TEXT CLASSIFIER - INTERACTIVE MODE")
    print("=" * 70)
    print("\nType your text to classify, or 'quit' to exit.\n")
    
    while True:
        # Get user input
        user_input = input("💬 Enter text: ").strip()
        
        # Check for exit command
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            print("⚠️  Please enter some text.\n")
            continue
        
        # Make prediction
        try:
            category, confidence, all_probs = predict_category(user_input)
            
            print("\n" + "-" * 70)
            print(f"📊 Predicted Category: {category}")
            print(f"💯 Confidence: {confidence:.2%}")
            print(f"\n📈 All Category Probabilities:")
            
            # Sort and display all probabilities
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            for cat, prob in sorted_probs:
                bar_length = int(prob * 30)  # Scale to 30 chars max
                bar = "█" * bar_length
                print(f"   {cat:20s} {bar:30s} {prob:.2%}")
            
            print("-" * 70 + "\n")
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}\n")

if __name__ == "__main__":
    main()
