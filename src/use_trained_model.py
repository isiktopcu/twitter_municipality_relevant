import pymongo 
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import dateutil
from torch.nn import DataParallel


#MONGODB 
batch_size = 1024
max_seq_length = 512
device = torch.device("cuda")
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["politus_twitter"]
tweet_col = db["tweets"]
start_date = dateutil.parser.parse("01-01-2023", dayfirst=True)
end_date = dateutil.parser.parse("31-12-2023", dayfirst=True)
query = {"municipal_relevant": None, "municipal": {"$ne":[]}, "date":{"$gte":start_date, "$lte":end_date}}
print("Counting number of tweets...")
tweet_count = tweet_col.count_documents(query)
print(f"Number of tweets: {tweet_count}")
tweets_to_predict = tweet_col.find(query, ["_id", "text"])

model_path = '/home/itopcu/twitter_municipality_relevant/models/best_models/municipal_best_model.pth'
tokenizer_path = 'dbmdz/bert-base-turkish-128k-cased'
model = AutoModelForSequenceClassification.from_pretrained(tokenizer_path, num_labels=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)

model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def preprocess(text): # Preprocess text (username and link placeholders)
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class CSVTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=max_seq_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length, padding='max_length', return_attention_mask=True, truncation=True)
        return torch.tensor(encoded['input_ids']).to(device), torch.tensor(encoded['attention_mask']).to(device)


def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for input_ids, attention_mask in data_loader:
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).squeeze()
            preds = (probs > 0.5).long().cpu().numpy()
            predictions.extend(preds)
    return predictions


curr_batch = []
for i, tweet in enumerate(tweets_to_predict):
    if i % 10_000 == 0:
        print(f"{i:,}/{tweet_count:,} | {i/tweet_count*100:.2f}%")
    id_str = tweet["_id"]
    text = preprocess(tweet["text"])

    if len(text) > 0:
        curr_batch.append({"_id": id_str, "text": text})

    if len(curr_batch) == batch_size:
        texts = [d["text"] for d in curr_batch]
        csv_dataset = CSVTextDataset(texts, tokenizer)
        csv_loader = DataLoader(csv_dataset, batch_size=batch_size)
        predictions = predict(model, csv_loader)
        predictions = [int(pred) for pred in predictions]
        for pred_idx, pred in enumerate(predictions):
            curr_d = curr_batch[pred_idx]
            tweet_col.update_one({"_id": curr_d["_id"]}, {"$set": {"municipal_relevant": pred}})

        curr_batch = []

if len(curr_batch) != 0:
    texts = [d["text"] for d in curr_batch]
    csv_dataset = CSVTextDataset(texts, tokenizer)
    csv_loader = DataLoader(csv_dataset, batch_size=batch_size)
    predictions = predict(model, csv_loader)
    predictions = [int(pred) for pred in predictions]
    for pred_idx, pred in enumerate(predictions):
        curr_d = curr_batch[pred_idx]
        tweet_col.update_one({"_id": curr_d["_id"]}, {"$set": {"municipal_relevant": pred}})


