import pandas as pd
import os
from pathlib import Path

# Get project root (go up from scripts/)
script_dir = Path(__file__).parent
project_root = script_dir.parent
os.chdir(project_root)

# Read the parquet file
df = pd.read_parquet('data/manipulation-dataset/train.parquet')

print("Dataset shape:", df.shape)
print("\nLanguage distribution:")
print(df['lang'].value_counts())
print("\nManipulation distribution:")
print(df['manipulative'].value_counts())
print("\nCross-tabulation:")
print(pd.crosstab(df['lang'], df['manipulative']))

# Create test dataset with 10 examples per language per class
# That means: 10 UK manipulative, 10 UK non-manipulative, 10 RU manipulative, 10 RU non-manipulative
test_samples = []

for lang in ['uk', 'ru']:
    for manipulative in [True, False]:
        # Filter data
        filtered = df[(df['lang'] == lang) & (df['manipulative'] == manipulative)]

        # Sample 10 examples
        if len(filtered) >= 10:
            sample = filtered.sample(n=10, random_state=42)
        else:
            sample = filtered  # Take all if less than 10
            print(f"Warning: Only {len(filtered)} examples for lang={lang}, manipulative={manipulative}")

        test_samples.append(sample)

# Combine all samples
test_df = pd.concat(test_samples, ignore_index=True)

# Shuffle the dataset
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n\nTest dataset shape: {test_df.shape}")
print("\nTest dataset distribution:")
print(pd.crosstab(test_df['lang'], test_df['manipulative']))

# Create output directory if it doesn't exist
output_dir = 'data/test-dataset'
os.makedirs(output_dir, exist_ok=True)

# Save to CSV
output_path = os.path.join(output_dir, 'test.csv')
test_df.to_csv(output_path, index=False)

print(f"\n✅ Test dataset saved to: {output_path}")
print(f"\nTotal samples: {len(test_df)}")
print(f"Columns: {test_df.columns.tolist()}")
