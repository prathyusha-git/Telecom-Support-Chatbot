import pandas as pd
import torch
import time, os, psutil, pickle
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

# 🟡 Start wandb tracking
wandb.init(project="bert-intent-classification", name="bert_base_128_fixedbatch")

# 💾 Memory logging helper
def log_memory(stage):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[{stage}] 💾 Memory Used: {mem:.2f} MB")

# STEP 1: Load + clean datasets
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("val_data.csv")

train_df = train_df.dropna(subset=["client_text"])
val_df = val_df.dropna(subset=["client_text"])
train_df = train_df[train_df["client_text"].str.strip() != ""]
val_df = val_df[val_df["client_text"].str.strip() != ""]

# STEP 2: Encode labels
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["intent_label"])
val_df["label"] = label_encoder.transform(val_df["intent_label"])

train_df = train_df[["client_text", "label"]]
val_df = val_df[["client_text", "label"]]

# STEP 3: Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(batch):
    return tokenizer(batch["client_text"], padding=True, truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# STEP 4: Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# STEP 5: Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# STEP 6: TrainingArguments with wandb + CSV logging
training_args = TrainingArguments(
    output_dir="./bert_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="./logs",
    report_to="wandb",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# STEP 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# STEP 8: Train + track memory
log_memory("🔍 Before Training")
start = time.time()
trainer.train()
end = time.time()
log_memory("🔚 After Training")
print(f"⏱️ Training completed in {(end - start)/60:.2f} minutes")

# STEP 9: Save model & assets
model.save_pretrained("./bert_model_clean")
tokenizer.save_pretrained("./bert_model_clean")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ BERT training complete + assets saved + wandb & CSV logging active.")
