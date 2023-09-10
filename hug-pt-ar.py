# from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "CAMeL-Lab/bert-base-arabic-camelbert-ca-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
X_train_german = ["هذا اعجبني", "انا لا احب ذلك", "أنا بخير", "أنا لست بخير"]
batch = tokenizer(
    X_train_german, padding=True, truncation=True, max_length=512, return_tensors="pt"
)
with torch.no_grad():
    outputs = model(**batch)
    label_ids = torch.argmax(outputs.logits, dim=1)
    print(label_ids)
    labels = [model.config.id2label[label_id] for label_id in label_ids.tolist()]
    print(labels)
