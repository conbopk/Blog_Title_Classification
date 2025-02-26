import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from huggingface_hub import HfApi, upload_file

# Kiểm tra và sử dụng GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. Data Analysis
def analyze_data(train_df):
    print("Data Analysis:")
    print(train_df.head())
    print("\nData Info:")
    print(train_df.info())

    # Thống kê phân phối nhãn
    print("\nLabel Distribution:")
    print(train_df['label_numeric'].value_counts())

    # Hiển thị biểu đồ phân phối nhãn
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label_numeric', data=train_df)
    plt.title("Label Distribution")
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig('label_distribution.png')
    plt.close()

    # Thống kê độ dài tiêu đề
    train_df['title_length'] = train_df['title'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['title_length'], bins=50)
    plt.title("Title Length Distribution")
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.savefig('title_length_distribution.png')
    plt.close()

    return train_df


# 2. Data Preprocessing
def preprocess_text(text):
    # Chuyển về chữ thường
    text = text.lower()

    # Loại bỏ các ký tự đặc biệt
    text = re.sub(r'[^\w\s]', '', text)

    # Loại bỏ số
    text = re.sub(r'\d+', '', text)

    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_data(train_df):
    print("Preprocessing data...")
    train_df['processed_title'] = train_df['title'].apply(preprocess_text)

    # Hiển thị một vài ví dụ đã tiền xử lý
    print("\nPreprocessed examples:")
    print(train_df[['title', 'processed_title']].head())

    return train_df


# 3. Build vocabulary and tokenize
def build_vocab(texts, max_vocab_size=10000):
    print(f"Building vocabulary (max size: {max_vocab_size})...")
    all_words = []
    for text in texts:
        words = text.split()
        all_words.extend(words)

    # Đếm tần suất của từ
    word_counts = Counter(all_words)
    print(f"Total unique words: {len(word_counts)}")

    # Lấy max_vocab_size từ phổ biến nhất
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(max_vocab_size - 2)]

    # Tạo từ điển word -> index
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    print(f"Final vocabulary size: {len(vocab)}")
    return vocab, word_to_idx


def text_to_indices(text, word_to_idx, max_length=50):
    words = text.split()
    indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]

    # Cắt hoặc pad để đạt được độ dài cố định
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [word_to_idx['<PAD>']] * (max_length - len(indices))

    return indices


# 4. Dataset class
class TitleDataset(Dataset):
    def __init__(self, indices, labels=None):
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.tensor(self.indices[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return torch.tensor(self.indices[idx], dtype=torch.long)


# 5. GRU Model
class AttentionGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)

        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch size, sentence length]
        embedded = self.embedding(x)
        # embedded shape: [batch size, sentence length, embedding dim]

        # GRU output
        outputs, hidden = self.gru(embedded)
        # outputs shape: [batch size, sentence length, hidden dim * 2]

        # Attention scores
        attention_scores = self.attention(outputs).squeeze(2)
        # attention_scores shape: [batch size, sentence length]

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(2)
        # attention_weights shape: [batch size, sentence length, 1]

        # Apply attention weights to GRU outputs
        context_vector = torch.sum(outputs * attention_weights, dim=1)
        # context_vector shape: [batch size, hidden dim * 2]

        # Apply dropout and classify
        output = self.dropout(context_vector)
        return self.fc(output)


# 6. Training function
def train(model, iterator, optimizer, criterion):
    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for batch in tqdm(iterator, desc="Training", leave=True, colour='green'):
        optimizer.zero_grad()

        text, labels = batch
        text = text.to(device)
        labels = labels.to(device)

        predictions = model(text)

        loss = criterion(predictions, labels)

        acc = (predictions.argmax(1) == labels).float().mean()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 7. Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=True, colour='blue'):
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device)

            predictions = model(text)

            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Main function
def main():
    print("Starting Blog Title Classification...")

    # Đọc dữ liệu
    try:
        train_df = pd.read_csv('train_set.csv')
        print(f"Loaded training data with {len(train_df)} samples")
    except FileNotFoundError:
        print("Error: Could not find training data file 'train_set.csv'")
        return

    # Phân tích dữ liệu
    train_df = analyze_data(train_df)

    # Tiền xử lý dữ liệu
    train_df = preprocess_data(train_df)

    # Xây dựng từ điển
    vocab, word_to_idx = build_vocab(train_df['processed_title'])

    # Chuyển đổi văn bản sang chỉ số
    print("Converting text to indices...")
    train_df['indices'] = train_df['processed_title'].apply(lambda x: text_to_indices(x, word_to_idx))

    # Chia dữ liệu huấn luyện và validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['indices'].tolist(),
        train_df['label_numeric'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=train_df['label_numeric']
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    # Tạo dataset và dataloader
    train_dataset = TitleDataset(X_train, y_train)
    val_dataset = TitleDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Định nghĩa mô hình
    VOCAB_SIZE = len(vocab)
    EMBEDDING_DIM = 200  # Tăng kích thước embedding
    HIDDEN_DIM = 256 # Tăng kích thước hidden layer
    OUTPUT_DIM = len(train_df['label_numeric'].unique())
    N_LAYERS = 6
    DROPOUT = 0.5

    model = AttentionGRU(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    model = model.to(device)

    # Định nghĩa loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training mô hình
    N_EPOCHS = 30
    best_val_loss = float('inf')

    print(f"Starting training for {N_EPOCHS} epochs...")
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best-model.pt')
            print(f"New best model saved with validation loss: {val_loss:.3f}")

        print(f'Epoch: {epoch + 1:02}/{N_EPOCHS}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}')
        print(f'\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}')

    # Đẩy model lên Hugging Face
    repo_id = "ThanhNguyen111/Blog-Title-Classification"
    upload_file(
        path_or_fileobj='best-model.pt',  # File model cần upload
        path_in_repo="best-model.pt",  # Tên file sau khi upload
        repo_id=repo_id,  # Repo Hugging Face đã tạo
        repo_type="model",  # Loại repo (mặc định là model)
    )

    print(f"✅ Model uploaded to https://huggingface.co/{repo_id}")

    # Đánh giá mô hình tốt nhất
    print("\nEvaluating the best model...")
    model.load_state_dict(torch.load('best-model.pt'))
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f'Best Model - Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.2f}')

    # Create Confusion Matrix and Classification Report
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for batch in val_loader:
            text, labels = batch
            text = text.to(device)

            predictions = model(text)
            predicted_classes = predictions.argmax(1)

            y_pred.extend(predicted_classes.cpu().numpy())
            y_true.extend(labels.numpy())

    # Confusion Matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Báo cáo phân loại
    report = classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)

    # Dự đoán trên tập test
    try:
        test_df = pd.read_csv('test_set_public.csv')
        print(f"Loaded test data with {len(test_df)} samples")
    except FileNotFoundError:
        print("Error: Could not find test data file 'test_set_public.csv'")
        return

    # Tiền xử lý dữ liệu test
    test_df['processed_title'] = test_df['title'].apply(preprocess_text)
    test_df['indices'] = test_df['processed_title'].apply(lambda x: text_to_indices(x, word_to_idx))

    # Tạo dataset và dataloader cho tập test
    test_dataset = TitleDataset(test_df['indices'].tolist())
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Dự đoán trên tập test
    print("Predicting on test data...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=True, colour='red'):
            batch = batch.to(device)
            output = model(batch)
            predicted_classes = output.argmax(1)
            predictions.extend(predicted_classes.cpu().numpy())

    # Thêm dự đoán vào DataFrame
    test_df['label_numeric'] = predictions

    # Tạo file submission (QUAN TRỌNG: Chỉ bao gồm id và label_numeric)
    submission_df = test_df[['_id', 'label_numeric']]
    submission_df.rename(columns={'_id': 'id'}, inplace=True)
    submission_df.to_csv('your_submissions.csv', index=False)

    print(f"Submission file created with {len(submission_df)} predictions")
    print("First 5 predictions:", submission_df.head())


# Run the main function
if __name__ == '__main__':
    main()