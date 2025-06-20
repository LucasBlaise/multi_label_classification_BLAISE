import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from tqdm import tqdm

#%% Parameters

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 1
MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-5

#%% Load data

dataset = load_dataset("adsabs/SciX_UAT_keywords")
train_data = dataset["train"]
val_data = dataset["val"]

def get_texts_and_labels(data):
    texts = [f"{t} {a}" for t, a in zip(data["title"], data["abstract"])]
    labels = data["verified_uat_labels"]
    return texts, labels

train_data = train_data.filter(lambda x: x.get("verified_uat_labels") is not None and len(x["verified_uat_labels"]) > 0)
val_data = val_data.filter(lambda x: x.get("verified_uat_labels") is not None and len(x["verified_uat_labels"]) > 0)

def is_valid(labels):
    return isinstance(labels, list) and len(labels) > 0

train_data = train_data.filter(lambda x: is_valid(x["verified_uat_labels"]))
val_data = val_data.filter(lambda x: is_valid(x["verified_uat_labels"]))

#%% Preprocessing

train_texts, train_labels = get_texts_and_labels(train_data)
val_texts, val_labels = get_texts_and_labels(val_data)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)
y_val = mlb.transform(val_labels)
num_labels = len(mlb.classes_)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

#%% Dataset

class SciXDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_dataset = SciXDataset(train_texts, y_train, tokenizer, MAX_LEN)
val_dataset = SciXDataset(val_texts, y_val, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type="multi_label_classification")
model.to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)

#%% Training

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

#%% Evaluation

model.eval()
y_preds, y_true = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].cpu().numpy()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = torch.sigmoid(outputs.logits).cpu().numpy()

        y_preds.extend(logits)
        y_true.extend(labels)

y_pred_bin = (np.array(y_preds) >= 0.2).astype(int)

print("F1 Score (macro):", f1_score(y_true, y_pred_bin, average='macro'))
print("Accuracy (samples):", accuracy_score(y_true, y_pred_bin))
print(classification_report(y_true, y_pred_bin, target_names=mlb.classes_))
