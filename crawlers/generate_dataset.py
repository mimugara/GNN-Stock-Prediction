import yfinance as yf
import pandas as pd



def pv(start_date, end_date):
    # Fetch the list of S&P 500 companies' symbols
    tickers = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"].to_list()
    print(tickers)
    # Download stock data for all required fields
    data = yf.download(tickers, start=start_date, end=end_date)
    print("Data download completed.")
    # Filling missing values
    data_filled = data.fillna(method='ffill')


    # Saving the data to an HDF5 file
    file_name = 'pv.h5'
    with pd.HDFStore(file_name, mode='w') as store:
        # Save each field under a separate key
        store.put('open', data['Open'], format='table', data_columns=True)
        store.put('high', data['High'], format='table', data_columns=True)
        store.put('low', data['Low'], format='table', data_columns=True)
        store.put('close', data['Close'], format='table', data_columns=True)
        store.put('volume', data['Volume'], format='table', data_columns=True)
        print(f"Data saved to {file_name} with keys for 'open', 'high', 'low', 'close', and 'volume'.")

    # Save the filled data in a separate file (optional)
    filled_file_name = 'pv_ffill.h5'
    with pd.HDFStore(filled_file_name, mode='w') as store:
        store.put('open', data_filled['Open'], format='table', data_columns=True)
        store.put('high', data_filled['High'], format='table', data_columns=True)
        store.put('low', data_filled['Low'], format='table', data_columns=True)
        store.put('close', data_filled['Close'], format='table', data_columns=True)
        store.put('volume', data_filled['Volume'], format='table', data_columns=True)
        print(f"Data with filled NA values saved to {filled_file_name} with keys for 'open_filled', 'high_filled', 'low_filled', 'close_filled', and 'volume_filled'.")


def stock_id():
    # Save the stock symbols to another HDF5 file
    tickers = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    symbols_file_name = 'stock_id.h5'
    symbols_key = 'stock_id'
    current_tickers_df =  tickers[0][['Symbol']]

    tickers[1]['Date'] = pd.to_datetime(tickers[1]['Date']['Date'])
    df_filtered = tickers[1][tickers[1]['Date']['Date'].dt.year >= 2015]
    print(df_filtered)
    old_tickers = pd.concat([df_filtered['Added']['Ticker'],df_filtered['Removed']['Ticker']],ignore_index=True).dropna().unique()
    old_tickers_df = pd.DataFrame(old_tickers, columns=['Symbol'])
    final_tickers = pd.concat([current_tickers_df, old_tickers_df], ignore_index=True).drop_duplicates()
    final_tickers = final_tickers.sort_values(by='Symbol').reset_index(drop=True)
    print(final_tickers)
    final_tickers.to_hdf(symbols_file_name, key=symbols_key, mode='w')
    print(f"Stock IDs saved to {symbols_file_name} under the key '{symbols_key}'.")

def univers():
    tickers = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    ide = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"].to_list()
    print(ide)
    trade_dates = pd.read_hdf("trade_dates.h5", key="trade_dates")

    df = pd.DataFrame(index = trade_dates['trade_dates'], columns=['Belongs'])
    df['Belongs'][-1] = pd.DataFrame(index = ide,columns= ['Active'])

    df['Belongs'][-1]['Active'] = False
    for t in tickers:
        if t in df['Belongs'][-1]['Active']: df['Belongs'][-1]['Active'][t] = True

    changes = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[1]
    added = changes['Added']['Ticker']
    # print(added)
    # print(changes)
    #print( df.index[::-1])
    # print(df['Belongs'].index[-1])
    idx = 0
    actual = '2024-05-09'
    prev = df['Belongs'].index[-1]


    for date in df.index[::-1]:
        if date == pd.to_datetime(actual): 
            continue
        df['Belongs'][date] = df['Belongs'][prev].copy()
        if date < pd.to_datetime(changes['Date']['Date'][idx]): 
            if not pd.isna(changes['Added']['Ticker'][idx]):
                df['Belongs'][date]['Active'][changes['Added']['Ticker'][idx]] = False
            if not pd.isna(changes['Removed']['Ticker'][idx]):
                df['Belongs'][date]['Active'][changes['Removed']['Ticker'][idx]] = True
            idx = idx +1
        prev = date

    df.to_hdf("univers.h5", key="univers", mode='w')
    


def save_trade_dates(start_date, end_date):
    # Fetch data for S&P 500 index as it's a reliable indicator of trading days in the US market
    data = yf.download('^GSPC', start=start_date, end=end_date)
    print(data)
    print(".....................")
    print(data.index)
    # Extract only the index which represents the trading dates
    trade_dates = data.index

    # Convert index to a DataFrame for easier storage
    trade_dates_df = pd.DataFrame(trade_dates.tolist(), columns=['trade_dates'])
    # Save the trade dates to an HDF5 file
    file_name = 'trade_dates.h5'
    key = 'trade_dates'
    trade_dates_df.to_hdf(file_name, key=key, mode='w')
    print(f"Trade dates saved to {file_name} under the key '{key}'.")



if __name__ == "__main__":
    start_date = '2014-01-01'
    actual = '2024-05-09'
    end_date = '2024-01-01'
    # save_trade_dates(start_date,actual)
    # stock_id()
    #univers()
    #pv(start_date,end_date)
    # opn = pd.read_hdf("pv.h5", key="open")
    # tickers = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"].to_list()
    # assert (opn.columns == tickers).all(), "Columns are not in the same order as tickers"
   
    