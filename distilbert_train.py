import pandas as pd
import json
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

# Load the JSON dataset
with open('data/data.json', 'r') as f:
    data = json.load(f)

# Convert JSON data to a pandas DataFrame
df = pd.DataFrame(data)
df['nodes'] = df['nodes'].apply(lambda x: ','.join(x))  # Concatenate nodes into a single string

# Initialize the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Preprocess the data for the tokenizer
def preprocess_data(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128)

# Convert DataFrame to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Use MultiLabelBinarizer to convert labels to binary format
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(df['nodes'].str.split(','))

# Create a custom Dataset class for PyTorch
class NodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Prepare encodings and labels for the dataset
encodings = {
    'input_ids': tokenized_dataset['input_ids'],
    'attention_mask': tokenized_dataset['attention_mask']
}
labels = binary_labels

# Initialize the training dataset
train_dataset = NodeDataset(encodings=encodings, labels=labels)

# Function to compute F1 score for the model evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).int().numpy()
    f1 = f1_score(labels, preds, average='weighted')
    return {'f1': f1}

# Load the DistilBert model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=binary_labels.shape[1])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',             # Output directory for model predictions and checkpoints
    num_train_epochs=100,               # Number of training epochs
    per_device_train_batch_size=4,      # Batch size for training
    per_device_eval_batch_size=4,       # Batch size for evaluation
    warmup_steps=500,                   # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                  # Weight decay for optimization
    logging_dir='./logs',               # Directory for storing logs
    logging_steps=10,                   # Log every 10 steps
    evaluation_strategy="epoch"         # Evaluate model after each epoch
)

# Initialize the Trainer with the model, training arguments, datasets, and metric computation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model, tokenizer, and MultiLabelBinarizer classes
model.save_pretrained('./model_distilbert_action_extractor')
tokenizer.save_pretrained('./model_distilbert_action_extractor')
mlb_classes_path = './model_distilbert_action_extractor/mlb_classes.npy'
np.save(mlb_classes_path, mlb.classes_)
