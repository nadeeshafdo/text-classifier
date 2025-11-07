# arXiv Dataset Training Guide

This guide explains how to train the text classifier using the arXiv dataset.

## Dataset Information

**Source:** https://www.kaggle.com/datasets/Cornell-University/arxiv  
**Size:** 4.94 GB  
**Format:** JSON (one paper per line)  
**Content:** Scientific papers with abstracts and categories

## Step-by-Step Instructions

### 1. Prepare the Dataset

Extract the downloaded dataset and locate the `arxiv-metadata-oai-snapshot.json` file.

Run the data preparation script:

```bash
# Basic usage: top 20 categories, balanced with 1000 samples each
python prepare_arxiv_data.py --input /path/to/arxiv-metadata-oai-snapshot.json --balance

# Custom configuration
python prepare_arxiv_data.py \
    --input /path/to/arxiv-metadata-oai-snapshot.json \
    --output training_data.csv \
    --max-categories 15 \
    --min-category-samples 500 \
    --balance \
    --samples-per-category 2000
```

**Parameters:**
- `--input`: Path to the JSON file (required)
- `--output`: Output CSV file (default: training_data.csv)
- `--max-samples`: Limit total samples for testing (optional)
- `--min-category-samples`: Minimum samples per category (default: 500)
- `--max-categories`: Number of top categories to keep (default: 20)
- `--balance`: Balance the dataset across categories
- `--samples-per-category`: Samples per category when balancing (default: 1000)

**Example for testing (faster):**
```bash
python prepare_arxiv_data.py \
    --input /path/to/arxiv-metadata-oai-snapshot.json \
    --max-samples 50000 \
    --max-categories 10 \
    --balance \
    --samples-per-category 500
```

### 2. Train the Model

After preparing the data:

```bash
# Basic training
python train_arxiv.py

# With custom parameters
python train_arxiv.py \
    --input training_data.csv \
    --max-features 20000 \
    --test-size 0.2 \
    --plot-cm
```

**Parameters:**
- `--input`: Input CSV file (default: training_data.csv)
- `--model-output`: Output model file (default: text_classifier_model.pkl)
- `--vectorizer-output`: Output vectorizer file (default: tfidf_vectorizer.pkl)
- `--max-features`: TF-IDF vocabulary size (default: 20000)
- `--test-size`: Test set proportion (default: 0.2)
- `--plot-cm`: Generate confusion matrix plot

### 3. Use the Trained Model

After training, use any of the interfaces:

```bash
# Command-line interface
python interface.py

# Web interface
python web_interface.py

# Python script
python predict.py
```

## Common arXiv Categories

The dataset includes various research categories:

**Computer Science:**
- cs.AI - Artificial Intelligence
- cs.CV - Computer Vision
- cs.LG - Machine Learning
- cs.CL - Computation and Language

**Physics:**
- hep-ph - High Energy Physics - Phenomenology
- astro-ph - Astrophysics
- quant-ph - Quantum Physics

**Mathematics:**
- math.CO - Combinatorics
- math.NT - Number Theory

**And many more...**

## Expected Performance

With a balanced dataset of 20 categories and 1000 samples each:
- Training time: 5-15 minutes (depending on hardware)
- Expected accuracy: 85-95%
- Model size: ~50-100 MB

## Tips for Better Results

1. **More data per category**: Increase `--samples-per-category` to 2000-5000
2. **More features**: Increase `--max-features` to 30000-50000
3. **Category selection**: Choose related categories for better distinction
4. **Try different models**: Edit train_arxiv.py to use RandomForest or SVM

## Troubleshooting

**Out of memory during training:**
- Reduce `--max-features`
- Reduce `--samples-per-category`
- Use fewer categories with `--max-categories`

**Low accuracy:**
- Ensure categories are well-balanced
- Increase training data
- Try different max_features values
- Consider using related categories only

**File not found:**
- Verify the path to arxiv-metadata-oai-snapshot.json
- Ensure training_data.csv was created successfully

## Example Workflow

```bash
# 1. Prepare data (10 categories, 1000 samples each)
python prepare_arxiv_data.py \
    --input ~/Downloads/arxiv-metadata-oai-snapshot.json \
    --max-categories 10 \
    --balance \
    --samples-per-category 1000

# 2. Train model
python train_arxiv.py --plot-cm

# 3. Test with web interface
python web_interface.py
```

## Dataset Statistics

After preparation, you'll see:
- Total papers processed
- Categories selected
- Samples per category
- Sample abstracts
- Text length statistics

This information helps you understand your training data and adjust parameters accordingly.
