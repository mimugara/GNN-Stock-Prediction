import pandas as pd
from tqdm import tqdm

#https://www.quora.com/Where-can-I-find-a-full-list-of-all-companies-classified-under-GICS-Global-Industry-Classification-Standard-listing-their-industry-and-sub-sector

# Load stock symbols
stock_id = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"].to_list()

# Create an empty adjacency matrix
adj_matrix_industry = pd.DataFrame(0, index=stock_id, columns=stock_id)
adj_matrix_subindustry = pd.DataFrame(0, index=stock_id, columns=stock_id)

# Load the S&P 500 data from Wikipedia
df0 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

df1 = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[1]

# Map each symbol to its industry for faster lookup
symbol_to_industry = df0.set_index('Symbol')['GICS Sector'].to_dict()
symbol_to_subindustry = df0.set_index('Symbol')['GICS Sub-Industry'].to_dict()

# df1['Date'] = pd.to_datetime(df1['Date']['Date'])
# df_filtered = df1[df1['Date']['Date'].dt.year >= 2023]
# old_tickers = pd.concat([df_filtered['Added']['Ticker'],df_filtered['Removed']['Ticker']],ignore_index=True).dropna().unique()
# print(old_tickers)

#https://finviz.com/
symbol_to_industry['VNO'] = 'Real Estate'
symbol_to_subindustry['VNO'] = "Office REITs"

symbol_to_industry['SIVB'] = 'Financials'
symbol_to_subindustry['SIVB'] = "Diversified Banks"

symbol_to_industry['SBNY'] = 'Financials'
symbol_to_subindustry['SBNY'] = "Regional Banks"

symbol_to_industry['LUMN'] = 'Communication Services'
symbol_to_subindustry['LUMN'] = "Integrated Telecommunication Services"

symbol_to_industry['FRC'] = 'Financials'
symbol_to_subindustry['FRC'] = "Regional Banks"

symbol_to_industry['DISH'] = 'Communication Services'
symbol_to_subindustry['DISH'] = "Cable & Satellite"

symbol_to_industry['AAP'] = 'Consumer Discretionary'
symbol_to_subindustry['AAP'] = "Automotive Retail"

symbol_to_industry['NWL'] = 'Consumer Discretionary'
symbol_to_subindustry['NWL'] = "Housewares & Specialties"

symbol_to_industry['LNC'] = 'Financials'
symbol_to_subindustry['LNC'] = "Life & Health Insurance"

symbol_to_industry['DXC'] = 'Information Technology'
symbol_to_subindustry['DXC'] = "IT Consulting & Other Services"

symbol_to_industry['ATVI'] = 'Communication Services	'
symbol_to_subindustry['ATVI'] = "Interactive Media & Services"

symbol_to_industry['OGN'] = 'Health Care'
symbol_to_subindustry['OGN'] = "Pharmaceuticals"

symbol_to_industry['SEDG'] = 'Utilities'
symbol_to_subindustry['SEDG'] = "Renewable Electricity"

symbol_to_industry['ALK'] = 'Industrials'
symbol_to_subindustry['ALK'] = "Passenger Airlines"

symbol_to_industry['SEE'] = 'Materials'
symbol_to_subindustry['SEE'] = "Paper & Plastic Packaging Products & Materials"

symbol_to_industry['ZION'] = 'Financials'
symbol_to_subindustry['ZION'] = "Regional Banks"

symbol_to_industry['WHR'] = 'Consumer Discretionary'
symbol_to_subindustry['WHR'] = "Home Furnishings"

symbol_to_industry['VFC'] = 'Consumer Discretionary'
symbol_to_subindustry['VFC'] = "Apparel, Accessories & Luxury Goods"

symbol_to_industry['XRAY'] = 'Health Care'
symbol_to_subindustry['XRAY'] = "Health Care Supplies"

symbol_to_industry['PXD'] = 'Energy'
symbol_to_subindustry['PXD'] = "Oil & Gas Exploration & Production"


# Fill the adjacency matrix
for row in tqdm(stock_id):
    for col in stock_id:
        if row in symbol_to_industry and col in symbol_to_industry:
            # Set to 1 if both stocks are in the same industry
            if symbol_to_industry[row] == symbol_to_industry[col]:
                adj_matrix_industry.loc[row, col] = 1

        if row in symbol_to_subindustry and col in symbol_to_subindustry:
            # Set to 1 if both stocks are in the same industry
            if symbol_to_subindustry[row] == symbol_to_subindustry[col]:
                adj_matrix_subindustry.loc[row, col] = 1

# Save or display the adjacency matrix
adj_matrix_industry.to_hdf("graph_constructors/{}_{}.h5".format("adjacent_matrix", "industry"),key="graph")
adj_matrix_subindustry.to_hdf("graph_constructors/{}_{}.h5".format("adjacent_matrix", "subindustry"),key="graph")





