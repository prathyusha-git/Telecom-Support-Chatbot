import pandas as pd
from transformers import BertTokenizer

# Load your dataset
df = pd.read_csv("train_dataset_20k_cleaned_final_fixed.csv")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Calculate token lengths without truncation
original_lengths = df["conversation"].dropna().apply(lambda x: len(tokenizer.encode(x, truncation=False)))

# Truncation analysis
num_truncated = (original_lengths > 512).sum()
total_samples = len(original_lengths)
percent_truncated = (num_truncated / total_samples) * 100

print(f"🔢 Total Sequences: {total_samples}")
print(f"⚠️ Truncated Sequences (>512): {num_truncated}")
print(f"📊 Truncation Percentage: {percent_truncated:.2f}%")
