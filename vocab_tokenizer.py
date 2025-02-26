from collections import Counter
from data_preprocessing import *


# Build_vocab
def build_vocab(texts, max_vocab_size=10000):
    all_words = []
    for text in texts:
        words = text.split()
        all_words.extend(words)

    # Đếm tần suất của từ
    word_counts = Counter(all_words)

    # Lấy max_vocab_size từ phổ biến nhất
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(max_vocab_size - 2)]

    # Tạo từ điển word -> index
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    return vocab, word_to_idx


# Hàm chuyển văn bản thành chuỗi chỉ số
def text_to_indices(text, word_to_idx, max_length=30):
    words = text.split()
    indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]

    # Cắt hoặc pad để đạt được độ dài cố định
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [word_to_idx['<PAD>']] * (max_length - len(indices))

    return indices


# Xây dựng từ điển từ dữ liệu huấn luyện
vocab, word_to_idx = build_vocab(train_df['processed_title'])

#Apply convert
train_df['indices'] = train_df['processed_title'].apply(lambda x: text_to_indices(x, word_to_idx))



