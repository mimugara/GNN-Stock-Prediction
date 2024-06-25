
import yfinance as yf
from tqdm import tqdm
import requests
from fuzzywuzzy import process
import pandas as pd
import pywikibot
from SPARQLWrapper import SPARQLWrapper, JSON

df0 = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]


def find_ticker_by_name(search_name, df, threshold=90):
    # Use fuzzy matching to find the closest company name
    choices = df['Security'].tolist()
    best_match = process.extractOne(search_name, choices, score_cutoff=threshold)
    
    if best_match:
        matched_name = best_match[0]
        # Retrieve the ticker for the matched company name
        ticker = df[df['Security'] == matched_name]['Symbol'].iloc[0]
        return ticker, matched_name
    else:
        return None, None

def get_stockholders_graph():
    stock_id = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"].to_list()
    adj_matrix = pd.DataFrame(0, index=stock_id, columns=stock_id)

    # Iterate through each ticker, find institutional holders, and update adjacency matrix
    for ticker in tqdm(stock_id):
        try:
            stock = yf.Ticker(ticker)
            holders = stock.institutional_holders
            #print(holders)
            if holders is not None and 'Holder' in holders.columns:
                holders_list = holders['Holder'].to_list()
                #print(holders_list)
                # Find tickers for each holder and set adjacency matrix to 1 if found in stock_id
                for holder in holders_list:
                    holder_ticker = find_ticker_by_name(holder, df0)[0]
                    if holder_ticker:
                        if holder_ticker in stock_id:
                            adj_matrix.at[ticker, holder_ticker] = 1
                            adj_matrix.at[holder_ticker, ticker] = 1
                        else: 
                            raise Exception
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    adj_matrix.to_hdf("graph_constructors/{}_{}.h5".format("adjacent_matrix", "stockholders"),key="graph")