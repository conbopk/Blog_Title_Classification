import re
from Data_analysis import *

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

# Apply preprocessing
train_df['processed_title'] = train_df['title'].apply(preprocess_text)

if __name__=='__main__':
    print(train_df[['title', 'processed_title']].head())
