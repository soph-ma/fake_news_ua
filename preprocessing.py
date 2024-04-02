import pandas as pd
import nltk
import ast
import pymorphy2
from transformers import BertTokenizer

# setup
csvfiles_fake = ["data/hyser.csv", 
            "data/presentnews.csv", 
            "data/puer.csv", 
            "data/stopfake.csv", 
            "data/translated.csv", 
            "data/ukrlive.csv"
]
csvfiles_real = ["data/news.csv", "data/zepopo.csv"]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

morph = pymorphy2.MorphAnalyzer(lang='uk')
with open(r'C:\Users\matskovy\personal\fake_news\models\stopwords_ua_list.txt', "r", encoding="utf-8") as f: 
    stopwords = ast.literal_eval(f.read())


# functions
    
def merge_dfs(fake_news_files: list[str], real_news_files: list[str]) -> pd.DataFrame:
    merged_df = pd.DataFrame(columns=['text', 'label']) 
    for f in fake_news_files: 
        df = pd.read_csv(f, encoding="utf-8")
        for index, row in df.iterrows():
            text = row["text"]
            label = "fake"
            merged_df.loc[len(merged_df)] =  {
                'text': text,
                'label': label 
            } 
    for f in real_news_files: 
        df = pd.read_csv(f, encoding="utf-8")
        for index, row in df.iterrows():
            text = row["text"]
            label = "real"
            merged_df.loc[len(merged_df)] =  {
                'text': text,
                'label': label 
            } 
    return balance_data(merged_df)

def get_x(df: pd.DataFrame) -> list[str]: 
    x = []
    for index, row in df.iterrows():
        text = row["text"]
        x.append(text)
    return x

def get_y(df: pd.DataFrame) -> list[int]: # real = 0, fake = 1
    y = df["label"]
    nums = []
    for element in y: 
        if element == "real": nums.append(float(0))
        else: nums.append(float(1))
    return nums

def tokenize_x(x: list[str]) -> list[list[float]]: 
    dictionary = {}
    count = 0
    tokenized = []
    for text in x: 
        words = preprocess_text(text)
        for i, word in enumerate(words): 
            if word not in dictionary: 
                dictionary[word] = count
                words[i] = count
                count += 1
            else: 
                words[i] = dictionary[word]
        tokenized.append(words)
    return tokenized

def lemmatize_word(word):
    return morph.parse(word)[0].normal_form

def bert_tokenize(x: list[str]) -> list[list[int]]: 
    tokenized_x = []
    for text in x: 
        text = tokenizer.encode(str(text), 
                         add_special_tokens=True,
                         max_length=50, 
                         truncation=True, 
                         padding='max_length'
                         )
        tokenized_x.append(text)
    return tokenized_x


def preprocess_text(text: str, padding=50) -> list[str]: 
    # lemmatization and removing stop words
    text = str(text)
    words = [lemmatize_word(word) for word in text.split() if word.lower() not in stopwords]
    padded = words[:padding] + [0] * (padding - len(words))
    return padded

def balance_data(df): 
    num_fake = (df['label'] == 'fake').sum()
    num_real = (df['label'] == 'real').sum()

    if num_fake > num_real:
        fake_indices = df[df['label'] == 'fake'].index
        sampled_fake_indices = pd.DataFrame(fake_indices).sample(n=num_real, random_state=42).index
        df_balanced = df.loc[sampled_fake_indices.union(df[df['label'] == 'real'].index)]
    else:
        df_balanced = df
    return df_balanced

# df = merge_dfs(real_news_files=csvfiles_real, fake_news_files=csvfiles_fake)
# df.to_csv("data/data.csv")