from Training import *
import torch

#Load test dataset
test_df = pd.read_csv('test_set_public.csv')

#Preprocess test dataset
test_df['processed_title'] = test_df['title'].apply(preprocess_text)
test_df['indices'] = test_df['processed_title'].apply(lambda x: text_to_indices(x, word_to_idx))

#Create Dataset and Dataloader
class TestDataset(Dataset):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return torch.tensor(self.indices[idx], dtype=torch.long)


test_dataset = TestDataset(test_df['indices'].tolist())
test_loader = DataLoader(test_dataset, batch_size=64)


# Dự đoán trên tập test
model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        output = model(batch)
        predicted_classes = output.argmax(1)
        predictions.extend(predicted_classes.cpu().numpy())


# Thêm dự đoán vào DataFrame
test_df['label_numeric'] = predictions

# Tạo file submission
submission_df = test_df[['_id', 'title','label_numeric']]
submission_df.rename(columns={'_id': 'id'}, inplace=True)
submission_df.to_csv('my_submission.csv', index=False)



