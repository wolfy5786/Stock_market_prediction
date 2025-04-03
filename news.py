import pandas as pd

def read_data(file_path = "D:\\datafiles\\acedemics\\sem_7\\machine_learning\\north_eastern\\supervised\\project\\data_news.csv"):
    df = pd.read_csv(file_path)
    print(df.head())
    print(len(df))
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
    df.dropna(inplace= True)
    
    df['date_time'] = pd.to_datetime(df['date_time'])
    print(df.head(3))
    
    return df