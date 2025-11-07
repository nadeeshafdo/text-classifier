import json
import pandas as pd
from pathlib import Path
import argparse

def load_arxiv_json(json_file_path, max_samples=None, min_abstract_length=50):
    """
    Load and parse the arXiv JSON dataset.
    
    Args:
        json_file_path: Path to the arxiv-metadata-oai-snapshot.json file
        max_samples: Maximum number of samples to load (None for all)
        min_abstract_length: Minimum length of abstract to include
        
    Returns:
        pandas DataFrame with 'text' and 'label' columns
    """
    print(f"Loading arXiv dataset from: {json_file_path}")
    
    texts = []
    labels = []
    count = 0
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and count >= max_samples:
                break
                
            try:
                # Parse JSON line
                paper = json.loads(line)
                
                # Extract abstract and category
                abstract = paper.get('abstract', '').strip()
                categories = paper.get('categories', '').strip()
                
                # Skip if abstract is too short or missing
                if not abstract or len(abstract) < min_abstract_length:
                    continue
                
                # Skip if no category
                if not categories:
                    continue
                
                # Use primary category (first one listed)
                primary_category = categories.split()[0]
                
                # Clean abstract (remove extra whitespace)
                abstract = ' '.join(abstract.split())
                
                texts.append(abstract)
                labels.append(primary_category)
                count += 1
                
                if count % 10000 == 0:
                    print(f"Processed {count} papers...")
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    print(f"\nTotal papers loaded: {count}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    return df

def filter_categories(df, min_samples_per_category=100, max_categories=None):
    """
    Filter dataset to include only categories with sufficient samples.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        min_samples_per_category: Minimum number of samples required per category
        max_categories: Maximum number of categories to keep (most common)
        
    Returns:
        Filtered DataFrame
    """
    print("\nCategory distribution before filtering:")
    category_counts = df['label'].value_counts()
    print(f"Total categories: {len(category_counts)}")
    print(f"Total samples: {len(df)}")
    
    # Filter categories with enough samples
    valid_categories = category_counts[category_counts >= min_samples_per_category].index
    df_filtered = df[df['label'].isin(valid_categories)].copy()
    
    # Keep only top N categories if specified
    if max_categories:
        top_categories = category_counts.head(max_categories).index
        df_filtered = df_filtered[df_filtered['label'].isin(top_categories)].copy()
    
    print(f"\nCategory distribution after filtering:")
    print(f"Categories kept: {df_filtered['label'].nunique()}")
    print(f"Samples kept: {len(df_filtered)}")
    print("\nTop 10 categories:")
    print(df_filtered['label'].value_counts().head(10))
    
    return df_filtered

def balance_dataset(df, samples_per_category=None, method='undersample'):
    """
    Balance the dataset across categories.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        samples_per_category: Number of samples per category (None for auto)
        method: 'undersample' or 'limit'
        
    Returns:
        Balanced DataFrame
    """
    if samples_per_category is None:
        # Use median count
        samples_per_category = int(df['label'].value_counts().median())
    
    print(f"\nBalancing dataset with {samples_per_category} samples per category...")
    
    balanced_dfs = []
    for category in df['label'].unique():
        category_df = df[df['label'] == category]
        
        if len(category_df) >= samples_per_category:
            # Sample randomly
            sampled = category_df.sample(n=samples_per_category, random_state=42)
        else:
            # Keep all samples if less than desired
            sampled = category_df
        
        balanced_dfs.append(sampled)
    
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced dataset size: {len(df_balanced)}")
    print(f"Categories: {df_balanced['label'].nunique()}")
    
    return df_balanced

def main():
    parser = argparse.ArgumentParser(description='Prepare arXiv dataset for training')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to arxiv-metadata-oai-snapshot.json')
    parser.add_argument('--output', type=str, default='training_data.csv',
                        help='Output CSV file path')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to load (for testing)')
    parser.add_argument('--min-category-samples', type=int, default=500,
                        help='Minimum samples required per category')
    parser.add_argument('--max-categories', type=int, default=20,
                        help='Maximum number of categories to keep')
    parser.add_argument('--balance', action='store_true',
                        help='Balance dataset across categories')
    parser.add_argument('--samples-per-category', type=int, default=1000,
                        help='Samples per category when balancing')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Load data
    df = load_arxiv_json(
        args.input, 
        max_samples=args.max_samples,
        min_abstract_length=50
    )
    
    if len(df) == 0:
        print("Error: No data loaded!")
        return
    
    # Filter categories
    df = filter_categories(
        df,
        min_samples_per_category=args.min_category_samples,
        max_categories=args.max_categories
    )
    
    # Balance if requested
    if args.balance:
        df = balance_dataset(
            df,
            samples_per_category=args.samples_per_category
        )
    
    # Save to CSV
    print(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)
    print(f"Done! Dataset saved with {len(df)} samples and {df['label'].nunique()} categories.")
    
    # Show sample
    print("\nSample data:")
    print(df.head(2))
    print("\nAbstract lengths:")
    print(df['text'].str.len().describe())

if __name__ == "__main__":
    main()
