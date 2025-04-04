import pandas as pd
import os
import matplotlib.pyplot as plt

import stocks_analysis
import open_ai_secret_key

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from openai import OpenAI 

def read_data(file_path = "D:\\datafiles\\acedemics\\sem_7\\machine_learning\\north_eastern\\supervised\\project\\data_news.csv"):
    df = pd.read_csv(file_path)
    print(df.head())
    print("Length of the dataframe: ", len(df))
    print("\n\n")
    return df

def pre_process_data(df):
    print("pre - processing the data ")
    
    i_count = len(df)
    df = df.drop_duplicates()
    f_count = len(df)
    duplicates_removed = i_count - f_count
    print("number of duplicates removed:  ",duplicates_removed)
    
    print("NUll values ")
    print(df.isnull().sum())
    df = df.dropna()
    
    df['date_time'] = pd.to_datetime(df['date_time'])
    print(df.head(3))
    
    return df

def plot_news_count(df, stocks):
    """ 
    plot the count of occurance of each stock
    """    
    filtered_df = df[df['Stock_symbol'].isin(stocks)]

    # Count occurrences of each company
    company_counts = filtered_df['Stock_symbol'].value_counts()

    # Plot the results
    plt.figure(figsize=(8, 5))
    company_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Company')
    plt.ylabel('Count')
    plt.title('Occurrence of Companies in Interested Stocks')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    


def make_cv_object(df):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Article_title'])
    return X

def make_tf_idf_object(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Article_title'])
    return X

def get_vector_embedding(text, model = "text-embedding-3-small"):
    client = OpenAI()
    client.api_key = open_ai_secret_key.secret_key
    
    responce = client.embeddings.create(
                input = [text],
                model = model
                )
    
    
    return responce.data[0].embedding

def use_opneAI_vector_embeddings(df, stocks, batch_size = 1, max_length = 1, file_path = "D:\\datafiles\\acedemics\\sem_7\\machine_learning\\north_eastern\\supervised\\project\\titles.csv"):
    filtered_df = df[df['Stock_symbol'].isin(stocks)]
    
    if len(stocks) >50:
        print("Too many stocks, please choose less than 50")
        return
    
    row_count = 0
    while row_count < len(filtered_df) and row_count < max_length:
        new_df = filtered_df[row_count:row_count + batch_size]
        new_df["embeddings"] = filtered_df["Article_title"].apply(get_vector_embedding)
        new_df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
        row_count += batch_size+1
        print(f"Saved {row_count} rows")
        
    return filtered_df