import pandas as pd

# === Load dataset ===
df = pd.read_csv("train_cleaned_dynamic.csv")

# === Dimensions ===
print("📊 Dataset Dimensions:", df.shape)
print("Number of instances:", len(df))
print("Columns:", df.columns.tolist())
print("\n")

# === Basic descriptive statistics ===
summary = df.describe(include="all").transpose()
print("=== Descriptive Statistics ===")
print(summary)

# === Value counts per label (if intent_label column exists) ===
if "intent_label" in df.columns:
    label_counts = df["intent_label"].value_counts()
    print("\n=== Intent Label Distribution ===")
    print(label_counts)
    label_counts.to_csv("label_distribution.csv")

# === Text length stats (for client_text / agent_response if present) ===
for col in ["client_text", "agent_response"]:
    if col in df.columns:
        df[col + "_length"] = df[col].astype(str).apply(lambda x: len(x.split()))
        print(f"\n=== {col} length stats ===")
        print(df[col + "_length"].describe())

# === Save summary to CSV ===
summary.to_csv("phase1_dataset_summary.csv")
print("\n✅ Summary saved to phase1_dataset_summary.csv")
