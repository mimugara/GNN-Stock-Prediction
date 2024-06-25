#https://stackoverflow.com/questions/48568094/wikidatasparql-lookup-a-company-based-on-its-ticker-symbol
#https://stackoverflow.com/questions/62396801/how-to-handle-too-many-requests-on-wikidata-using-sparqlwrapper
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import pandas as pd
import time

def get_wikidata_id_from_ticker(ticker, retry_delay=10, max_retries=100):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = f"""
    SELECT DISTINCT ?id WHERE {{
        ?id wdt:P31/wdt:P279* wd:Q4830453 .
        {{
            ?id wdt:P249 ?ticker . FILTER(LCASE(STR(?ticker)) = LCASE("{ticker}")) .
        }} UNION {{
            ?id p:P414 ?exchangesub .
            ?exchangesub pq:P249 ?ticker . FILTER(LCASE(STR(?ticker)) = LCASE("{ticker}")) .
        }} 
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    retries = 0
    while retries < max_retries:
        try:
            results = sparql.query().convert()
            if results["results"]["bindings"]:
                return results["results"]["bindings"][0]["id"]["value"].split('/')[-1]
            return None
        except Exception as e:
            time.sleep(retry_delay)  # Wait before retrying
            retries += 1
            retry_delay *= 2  # Exponential backoff

    return None  # Return None if all retries fail
    

def get_ticker_wikidataid_relation():
    # Example list of tickers
    tickers = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"].to_list()

    # Fetch Wikidata IDs and store in a dictionary
    wikidata_tickers = {}
    for ticker in tqdm(tickers):
        wikidata_id = get_wikidata_id_from_ticker(ticker)
        if wikidata_id:
            print(wikidata_id)
            wikidata_tickers[wikidata_id] = ticker

    # Convert the dictionary to a DataFrame
    df_wikidata_tickers = pd.DataFrame(list(wikidata_tickers.items()), columns=['WikidataID', 'Ticker'])

    df_wikidata_tickers.to_hdf('wikidata_tickers.h5', key='df', mode='w')

    print(df_wikidata_tickers)

def check_wikidata_relation(company_wikidata_id1, company_wikidata_id2, retry_delay=10, max_retries=100):
    # Initialize SPARQL client for Wikidata
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    
    # SPARQL query to check relationships between two companies
    query = f"""
        SELECT DISTINCT ?relationship ?product WHERE {{ 
            VALUES ?company1 {{ wd:{company_wikidata_id1} }}
            VALUES ?company2 {{ wd:{company_wikidata_id2} }}
            
            {{
                ?company1 ?relationship ?company2 .
                FILTER(?relationship IN (wdt:P127, wdt:P155, wdt:P156, wdt:P355, wdt:P749))
            }}
            UNION
            {{
                ?company1 wdt:P31 ?instanceProduct .
                ?company2 wdt:P366 ?instanceProduct .
                BIND(wdt:P366 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P31 ?instanceIndustry .
                ?company2 wdt:P452 ?instanceIndustry .
                BIND(wdt:P452 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P31 ?instanceProduct2 .
                ?company2 wdt:P1056 ?instanceProduct2 .
                BIND(wdt:P1056 AS ?relationship)
            }}
            UNION 
            {{
                ?company1 wdt:P112 ?founder .
                ?company2 wdt:P112 ?founder .
                BIND(wdt:P112 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P112 ?founderOwner .
                ?company2 wdt:P127 ?founderOwner .
                BIND(wdt:P127 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P112 ?founderCEO .
                ?company2 wdt:P169 ?founderCEO .
                BIND(wdt:P169 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P113 ?hub .
                ?company2 wdt:P113 ?hub .
                BIND(wdt:P113 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P114 ?alliance .
                ?company2 wdt:P114 ?alliance .
                BIND(wdt:P114 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P121 ?itemOperated .
                ?company2 wdt:P1056 ?itemOperated .
                BIND(wdt:P1056 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P121 ?item .
                ?company2 wdt:P121 ?item .
                BIND(wdt:P121 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P127 ?owner .
                ?company2 wdt:P112 ?owner .
                BIND(wdt:P112 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P127 ?owner2 .
                ?company2 wdt:P127 ?owner2 .
                BIND(wdt:P127 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P127 ?ownerCEO .
                ?company2 wdt:P169 ?ownerCEO .
                BIND(wdt:P169 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P127 ?ownerSubsidiary .
                ?company2 wdt:P355 ?ownerSubsidiary .
                BIND(wdt:P355 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P127 ?ownerParent .
                ?company2 wdt:P749 ?ownerParent .
                BIND(wdt:P749 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P127 ?ownerEntity .
                ?company2 wdt:P1830 ?ownerEntity .
                BIND(wdt:P1830 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P127 ?boardMember .
                ?company2 wdt:P3320 ?boardMember .
                BIND(wdt:P3320 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P155 ?previous .
                ?company2 wdt:P155 ?previous .
                BIND(wdt:P155 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P155 ?followsSub .
                ?company2 wdt:P355 ?followsSub .
                BIND(wdt:P355 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P166 ?award .
                ?company2 wdt:P166 ?award .
                BIND(wdt:P166 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P169 ?ceoFounder .
                ?company2 wdt:P112 ?ceoFounder .
                BIND(wdt:P112 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P169 ?ceoOwner .
                ?company2 wdt:P127 ?ceoOwner .
                BIND(wdt:P127 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P169 ?ceo .
                ?company2 wdt:P169 ?ceo .
                BIND(wdt:P169 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P169 ?ceoBoard .
                ?company2 wdt:P3320 ?ceoBoard .
                BIND(wdt:P3320 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P199 ?businessDivision .
                ?company2 wdt:P355 ?businessDivision .
                BIND(wdt:P355 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P306 ?os .
                ?company2 wdt:P1056 ?os .
                BIND(wdt:P1056 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P355 ?subsidiary .
                ?company2 wdt:P127 ?subsidiary .
                BIND(wdt:P127 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P355 ?subsidiaryFollows .
                ?company2 wdt:P155 ?subsidiaryFollows .
                BIND(wdt:P155 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P355 ?subsidiaryDivision .
                ?company2 wdt:P199 ?subsidiaryDivision .
                BIND(wdt:P199 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P355 ?subsidiarySame .
                ?company2 wdt:P355 ?subsidiarySame .
                BIND(wdt:P355 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P366 ?use .
                ?company2 wdt:P31 ?use .
                BIND(wdt:P31 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P400 ?platform .
                ?company2 wdt:P1056 ?platform .
                BIND(wdt:P1056 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P452 ?industryInstance .
                ?company2 wdt:P31 ?industryInstance .
                BIND(wdt:P31 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P452 ?industry .
                ?company2 wdt:P452 ?industry .
                BIND(wdt:P452 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P452 ?industryProduct .
                ?company2 wdt:P1056 ?industryProduct .
                BIND(wdt:P1056 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P452 ?industrySource .
                ?company2 wdt:P2770 ?industrySource .
                BIND(wdt:P2770 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P463 ?member .
                ?company2 wdt:P463 ?member .
                BIND(wdt:P463 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P749 ?parentOrg .
                ?company2 wdt:P127 ?parentOrg .
                BIND(wdt:P127 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P749 ?parentOwner .
                ?company2 wdt:P1830 ?parentOwner .
                BIND(wdt:P1830 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1056 ?productInstance .
                ?company2 wdt:P31 ?productInstance .
                BIND(wdt:P31 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1056 ?productItem .
                ?company2 wdt:P121 ?productItem .
                BIND(wdt:P121 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1056 ?productOS .
                ?company2 wdt:P306 ?productOS .
                BIND(wdt:P306 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1056 ?productPlatform .
                ?company2 wdt:P400 ?productPlatform .
                BIND(wdt:P400 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1056 ?productIndustry .
                ?company2 wdt:P452 ?productIndustry .
                BIND(wdt:P452 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1056 ?product .
                ?company2 wdt:P1056 ?product .
                BIND(wdt:P1056 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1344 ?participant .
                ?company2 wdt:P1344 ?participant .
                BIND(wdt:P1344 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1830 ?ownerOf .
                ?company2 wdt:P127 ?ownerOf .
                BIND(wdt:P127 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P1830 ?ownerParent .
                ?company2 wdt:P749 ?ownerParent .
                BIND(wdt:P749 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P2770 ?sourceIncome .
                ?company2 wdt:P452 ?sourceIncome .
                BIND(wdt:P452 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P3320 ?boardOwned .
                ?company2 wdt:P127 ?boardOwned .
                BIND(wdt:P127 AS ?relationship)
            }}
            UNION
            {{
                ?company1 wdt:P3320 ?boardCEO .
                ?company2 wdt:P169 ?boardCEO .
                BIND(wdt:P169 AS ?relationship)
            }}
        }}
    """
    retries = 0
    while retries < max_retries:
        try:
            sparql.method = 'POST'
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON) 
            # Perform the query
            results = sparql.query().convert()
            relationships = [result['relationship']['value'].split('/')[-1] for result in results['results']['bindings']]
            #print(results)
            return relationships
        except Exception as e:
            print("delay!")
            print(e)
            time.sleep(retry_delay)  # Wait before retrying
            retries += 1
            retry_delay *= 2  # Exponential backoff
    
    raise Exception("Too many requests!")

    


if __name__ == "__main__":
    # get_ticker_wikidataid_relation()
    
    # Load stock symbols
    stock_id = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"].to_list()

    # Create an empty adjacency matrix
    adj_matrix = pd.DataFrame(0, index=stock_id, columns=stock_id)
    stock_wiki_relation = pd.read_hdf("wikidata_tickers.h5", key="df")
    ticker_dict = stock_wiki_relation.set_index('WikidataID')['Ticker'].to_dict()

    wikidata_ids = list(ticker_dict.keys())

    for i, wikidata_id1 in tqdm(enumerate(wikidata_ids)):
        for wikidata_id2 in (wikidata_ids[i+1:]):
            relationships = check_wikidata_relation(wikidata_id1, wikidata_id2)
            if relationships:
                print(relationships)
                ticker1 = ticker_dict[wikidata_id1]
                ticker2 = ticker_dict[wikidata_id2]
                adj_matrix.loc[ticker1, ticker2] = 1
                adj_matrix.loc[ticker2, ticker1] = 1

    print(adj_matrix)
    adj_matrix.to_hdf("graph_constructors/{}_{}.h5".format("adjacent_matrix", "wikidataDefinitiu"),key="graph")

