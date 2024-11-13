import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load and prepare data
class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Load in data from csv and split
def load_data(file_path, tokenizer, max_len, batch_size):
    df = pd.read_csv(file_path)
    df_sampled = df.sample(frac=1)
    df_sampled['label'] = df_sampled['Category'].apply(lambda x: 1 if x == 'spam' else 0)

    texts = df_sampled['Message'].values
    labels = df_sampled['label'].values

    split_idx = int(len(df_sampled) * 0.9)
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    train_dataset = SpamDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = SpamDataset(val_texts, val_labels, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


# Display heatmap confusion matrix
def show_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(8, 6))
    hmap = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                       xticklabels=class_names, yticklabels=class_names)
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')  # Slightly rotate bottom labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()


# Training loop
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        correct_preds = 0
        total_preds = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        print(f"Epoch {epoch + 1} - Training Loss: {avg_train_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")
        print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f} seconds")

    return train_losses, train_accuracies


# Evaluate and call visuals
def evaluate_model(model, val_loader, device):
    model.eval()
    correct_preds = 0
    total_preds = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_accuracy = correct_preds / total_preds
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    class_names = ['ham', 'spam']  # List class names in the same order as they are represented in the confusion matrix
    show_confusion_matrix(conf_matrix, class_names)

    print(classification_report(all_labels, all_preds, target_names=class_names, labels=[0, 1], zero_division=0))

    return val_accuracy


# Plot validation accuracy, train accuracy, and loss
def plot_metrics(train_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Configuration
    file_path = 'spam.csv'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    batch_size = 16
    num_epochs = 3
    learning_rate = 2e-5
    weight_decay = 0.01

    # Load data
    train_loader, val_loader = load_data(file_path, tokenizer, max_len, batch_size)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train model
    train_losses, train_accuracies = train_model(model, train_loader, val_loader, optimizer, scheduler,
                                                 device, num_epochs)

    # Evaluate model after training
    val_accuracy = evaluate_model(model, val_loader, device)

    # Plot metrics
    plot_metrics(train_losses, train_accuracies, [val_accuracy] * len(train_losses))


if __name__ == "__main__":
    main()
