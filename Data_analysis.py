#Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



train_df = pd.read_csv('train_set.csv')

if __name__=='__main__':
    print(train_df)

    print(train_df.info())

    # Thống kê phân phối nhãn
    print(train_df['label_numeric'].value_counts())
    print(train_df['label'].value_counts())

    # Thống kê phân phối nhãn
    plt.figure(figsize=(10,6))
    sns.countplot(x='label_numeric', data=train_df)
    plt.title("Label Distribution")
    plt.xlabel('Label')
    plt.ylabel('Quantity')
    plt.show()


