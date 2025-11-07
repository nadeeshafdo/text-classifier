import joblib

# Load the saved model and vectorizer
print("📂 Loading model and vectorizer...")
clf = joblib.load('text_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
print("✅ Model loaded successfully!\n")

def predict_category(text):
    """
    Predict the category of a given text.
    
    Args:
        text (str): The text to classify
        
    Returns:
        tuple: (predicted_category, confidence_score)
    """
    # Vectorize the input text
    processed = vectorizer.transform([text])
    
    # Make prediction
    prediction = clf.predict(processed)[0]
    confidence = clf.predict_proba(processed).max()
    
    return prediction, confidence

def predict_with_all_probabilities(text):
    """
    Predict category with probabilities for all classes.
    
    Args:
        text (str): The text to classify
        
    Returns:
        dict: Dictionary with category names and their probabilities
    """
    processed = vectorizer.transform([text])
    prediction = clf.predict(processed)[0]
    probabilities = clf.predict_proba(processed)[0]
    classes = clf.classes_
    
    # Create a dictionary of all probabilities
    prob_dict = {class_name: prob for class_name, prob in zip(classes, probabilities)}
    
    return prediction, prob_dict

# Example usage
if __name__ == "__main__":
    # Test with some examples
    test_texts = [
        "Building a mobile app to track daily water intake and remind users",
        "Creating a machine learning model to predict stock prices",
        "Developing a website for online food delivery service"
    ]
    
    print("=" * 70)
    print("TEXT CLASSIFIER - PREDICTION DEMO")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Example {i}:")
        print(f"Text: {text[:80]}...")
        
        # Simple prediction
        category, confidence = predict_category(text)
        print(f"📊 Predicted Category: {category}")
        print(f"💯 Confidence: {confidence:.2%}")
        
        # Detailed probabilities
        _, all_probs = predict_with_all_probabilities(text)
        print(f"📈 All probabilities:")
        for cat, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat}: {prob:.2%}")
    
    print("\n" + "=" * 70)
    print("\n💡 You can now import this script in your interface:")
    print("   from predict import predict_category")
    print("   category, confidence = predict_category('your text here')")
