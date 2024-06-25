import pandas as pd
from tqdm import tqdm
import numpy as np
import time
import pickle

# Path to your CSV file
file_path = '/home/mmunoz/GNN-Stock-Prediction/alpha/model/THGNN/FNSPID/Stock_news/nasdaq_exteral_data.csv'


stock_id = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"].to_list()



# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_filtered = df[df['Date'].dt.year >= 2016]
df_filtered['Stock_symbol'] = df_filtered['Stock_symbol'].fillna('Unknown')

# Group by 'Date' and 'Article_title', and aggregate the 'Stock_symbol' column
grouped_df = df_filtered.groupby(['Date', 'Article_title'])['Stock_symbol'].agg(lambda x: list(x.unique())).reset_index()
# Display the resulting DataFrame
print(grouped_df)


prev_date_num = 20
stock_trade_data= pd.read_hdf("trade_dates.h5", key="trade_dates")["trade_dates"]
dt=[]
for i in ['2016','2017','2018','2019','2020','2021','2022','2023']:
    for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        stock_m = stock_trade_data.loc[(stock_trade_data>=np.datetime64(i+'-'+j +"-01")) &(stock_trade_data<=np.datetime64(i+'-'+j)+pd.offsets.MonthEnd())]
        dt.append(stock_m.iloc[-1].strftime('%Y-%m-%d'))
print("---------------The last trading dates of each month are:-----------------")
print(dt)


for i in tqdm(range(len(dt))):
    # # Create an empty adjacency matrix
    adj_matrix = pd.DataFrame(0, index=stock_id, columns=stock_id)
    end_data = dt[i]
    start_data = stock_trade_data.iloc[stock_trade_data[stock_trade_data == end_data].index -(prev_date_num - 1)]
    start_data =  start_data.iloc[0].strftime('%Y-%m-%d')

    # Filter articles within the current month
    month_articles = grouped_df[(grouped_df['Date'] >= start_data) & (grouped_df['Date'] <= end_data)]
    
    # Iterate through each article and update the adjacency matrix
    for _, row in month_articles.iterrows():
        stocks = row['Stock_symbol']
        for idx1 in range(len(stocks)):
            for idx2 in range(idx1 + 1, len(stocks)):
                stock1 = stocks[idx1]
                stock2 = stocks[idx2]
                if stock1 in stock_id and stock2 in stock_id:
                    adj_matrix.loc[stock1, stock2] = 1
                    adj_matrix.loc[stock2, stock1] = 1

    adj_matrix.to_hdf("monthly_news/{}_{}.h5".format("adjacent_matrix", str(end_data)[:7]),key="graph")
print("Everything saved in /monthly_news!")


