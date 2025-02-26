import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from vocab_tokenizer import *

#Dataset
class TitleDataset(Dataset):
    def __init__(self, indices, labels):
        super().__init__()
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return torch.tensor(self.indices[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


#Model
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                          bidirectional=True, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch size, sentence length]
        embedded = self.embedding(x)
        # embedded shape: [batch size, sentence length, embedding dim]

        output, hidden = self.gru(embedded)
        # output shape: [batch size, sentence length, hidden dim * 2]
        # hidden shape: [n layers * 2, batch size, hidden dim]

        # Sử dụng hidden state cuối cùng từ cả chiều xuôi và ngược
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        # hidden shape: [batch size, hidden dim * 2]

        return self.fc(hidden)

# Chia dữ liệu train và validation
x_train, x_val, y_train, y_val = train_test_split(
    train_df['indices'].tolist(),
    train_df['label_numeric'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=train_df['label_numeric']
)

# Create Dataset
train_dateset = TitleDataset(x_train, y_train)
val_dateset = TitleDataset(x_val, y_val)

# Create Dataloader
train_loader = DataLoader(train_dateset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dateset, batch_size=64)

# Define Model
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = len(train_df['label_numeric'].unique())
N_LAYER = 2
DROPOUT = 0.5

model = GRUClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYER, DROPOUT)


# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


