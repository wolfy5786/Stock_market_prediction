import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os

from pathlib import Path
from difflib import get_close_matches


stocks = ["AAPL","MFST","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","BRK,B","AVGO","LLY","XOM","UNH","MA","COST","PG","NFLX","JNJ","WMT","ABBV","HD","KO","MRK","ORCL","GE","DIS","PLTR","GS","QCOM","TXN","SPGI","CAT","AMD","UBER","MS","BLK","TMUS","SBUX","BX","LMT","CVS","CL","MSI","GD","PNC","COF"]

def ensure_files(stock, directory, file):
    print("stock: ",stock)
    print("looking for:",file)
    
    directory = Path(directory)
    if os.path.exists(file):
        return True
    else:
        print("FILE NOT FOUND, \n\n OPTIONS:\n Check the input stock symbol exist\n Check the corresponding csv file exist\n Manaully correct file name.")
        files = [f.name for f in directory.iterdir() if f.is_file()]
        closest_match = get_close_matches(stock + ".csv", files, n=10, cutoff=0.5)  

        if closest_match:
            print("Did you mean", closest_match)
            return False

def create_stock_directory(stocks, folder = "D:\\datafiles\\acedemics\\sem_7\\machine_learning\\north_eastern\\supervised\\project\\full_history\\full_history"):
    stocks_directories = {}
    stocks_not_found = [] 
    for stock in stocks:
        path = os.path.join(folder,stock + ".csv") 
        if ensure_files(stock ,folder, path):
            stocks_directories[stock] = path
        else: 
            stocks_not_found.append(stock)
            
            
        
    print("\n\nFollowing Stocks directories were not found: ", stocks_not_found)
    return stocks_directories, stocks_not_found
        

# stocks_directories, stocks_not_found = create_stock_directory(stocks)
# stocks = list(set(stocks).difference(set(stocks_not_found)))
# print("total count of stocks: ",len(stocks))

def pre_processing(data):
    print("null values : ")
    print(data.isnull().sum())
    print("null values dropped")
    data.dropna(inplace=True)
    
    print("\n duplicates count ")
    i_count = len(data)

    data = data.drop_duplicates()

    f_count = len(data)

    duplicates_removed = i_count - f_count
    print(duplicates_removed)
    
    return data

def stock_analysis(stock, paths):
    file_path = paths[stock]
    data = pd.read_csv(file_path)
    
    print("Analysis of stock: ",stock)
    data = pre_processing(data)
    
    data["date"] = pd.to_datetime(data["date"])

    #lets add some features to our data set
    data["net_change"] = data["close"] - data["open"]
    data["relative_net_change"] = 100 * data["net_change"]/data["open"]
    data["high_low_diff"] = data["high"] - data["low"]
    data["high_low_relative_change"] = 100 * data["high_low_diff"]/data["open"]
    
    
    
    print(data.info())
    print(data.describe())
    print(data.head(3))
    
    
    print("\n\n")
    return data
    
    
# temp = ["AMD"]
# data = {}
# for stock in temp:
#     data[stock] = stock_analysis(stock, stocks_directories)

# dataset = data["AMD"]
# dataset = dataset.sort_values(by = "date", ascending = True)   


# dataset["year_month"] = dataset["date"].dt.to_period("M")  # Convert to 'YYYY-MM' format


    
