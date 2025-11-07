# Quick Start: Training with arXiv Dataset

You have downloaded the arXiv dataset (4.94 GB). Here's how to use it:

## Step 1: Locate Your Downloaded File

Find the file: `arxiv-metadata-oai-snapshot.json`

This file contains JSON data with one paper per line, including:
- Abstract (the text to classify)
- Categories (the labels)
- Other metadata

## Step 2: Prepare the Data

Run this command (replace the path with your actual file location):

```bash
python prepare_arxiv_data.py \
    --input /path/to/arxiv-metadata-oai-snapshot.json \
    --output training_data.csv \
    --max-categories 20 \
    --balance \
    --samples-per-category 1000
```

**What this does:**
- Reads the large JSON file
- Extracts abstracts (text) and categories (labels)
- Filters to keep only the top 20 most common categories
- Balances the dataset to have 1000 samples per category
- Saves as `training_data.csv` (ready for training)

**For faster testing, use:**
```bash
python prepare_arxiv_data.py \
    --input /path/to/arxiv-metadata-oai-snapshot.json \
    --max-samples 50000 \
    --max-categories 10 \
    --balance \
    --samples-per-category 500
```

## Step 3: Train the Model

After data preparation completes:

```bash
python train_arxiv.py --plot-cm
```

**What this does:**
- Loads the prepared `training_data.csv`
- Vectorizes text using TF-IDF
- Trains a Logistic Regression classifier
- Evaluates accuracy and metrics
- Saves the trained model files
- Creates a confusion matrix visualization

Expected training time: 5-15 minutes

## Step 4: Use Your Trained Model

Choose any interface:

**Web Interface (recommended):**
```bash
python web_interface.py
```
Then open http://127.0.0.1:5000 in your browser

**Command-Line:**
```bash
python interface.py
```

**Python Code:**
```python
from predict import predict_category

abstract = "Your scientific abstract here..."
category, confidence = predict_category(abstract)
print(f"Category: {category}, Confidence: {confidence:.2%}")
```

## Example Categories

You'll be classifying papers into categories like:
- cs.LG - Machine Learning
- cs.CV - Computer Vision
- hep-ph - High Energy Physics
- astro-ph - Astrophysics
- math.CO - Combinatorics
- And more...

## Full Example Workflow

```bash
# Activate virtual environment
source venv/bin/activate

# Prepare data (adjust path to your file)
python prepare_arxiv_data.py \
    --input ~/Downloads/arxiv-metadata-oai-snapshot.json \
    --max-categories 15 \
    --balance \
    --samples-per-category 1000

# Train model (takes ~10 minutes)
python train_arxiv.py --plot-cm

# Start web interface
python web_interface.py
```

## File Locations

After completing these steps, you'll have:
- `training_data.csv` - Prepared dataset (~2-20 MB depending on settings)
- `text_classifier_model.pkl` - Trained model (~50-100 MB)
- `tfidf_vectorizer.pkl` - Fitted vectorizer (~20-50 MB)
- `confusion_matrix.png` - Performance visualization (if --plot-cm used)

## Adjusting for Better Performance

**Want better accuracy?**
- Increase samples per category: `--samples-per-category 2000`
- Increase max features: `python train_arxiv.py --max-features 30000`

**Want faster processing?**
- Reduce categories: `--max-categories 10`
- Reduce samples: `--samples-per-category 500`
- Use max-samples for testing: `--max-samples 10000`

**Running out of memory?**
- Reduce max-features: `--max-features 10000`
- Reduce samples per category
- Process fewer categories

## Need Help?

Check the detailed guides:
- `ARXIV_GUIDE.md` - Complete arXiv dataset documentation
- `README.md` - General usage and API reference

## Verify Your Setup

Check that you have the JSON file:
```bash
ls -lh /path/to/arxiv-metadata-oai-snapshot.json
```

Should show a file around 4.94 GB in size.
