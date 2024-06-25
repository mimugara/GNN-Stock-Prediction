import numpy as np
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import pandas as pd
import copy


def get_trade_dates(srt_date, end_date):
    trade_dates = pd.read_hdf("trade_dates.h5", key="trade_dates")
    df = trade_dates[(trade_dates >= srt_date) & (trade_dates <= end_date)]
    #print(df)
    return df.dropna()


def jaccard_similarity(A, B):
    # Flatten the matrices to compare as sets
    A_flat = A.flatten()
    B_flat = B.flatten()
    return jaccard_score(A_flat, B_flat)

def hamming_distance(A, B):
    # Flatten the matrices to compare element-wise
    A_flat = A.flatten()
    B_flat = B.flatten()
    return 1-hamming(A_flat, B_flat)

def cosine_sim(A, B):
    # Reshape matrices to 1D arrays
    A_flat = A.flatten().reshape(1, -1)
    B_flat = B.flatten().reshape(1, -1)
    return cosine_similarity(A_flat, B_flat)[0, 0]

def dice_coefficient(A, B):
    A_flat = A.flatten()
    B_flat = B.flatten()
    intersection = np.sum(A_flat * B_flat)
    return 2 * intersection / (np.sum(A_flat) + np.sum(B_flat))

def russell_rao_similarity(A, B):
    A_flat = A.flatten()
    B_flat = B.flatten()
    intersection = np.sum(A_flat * B_flat)
    n = len(A_flat)
    return intersection / n


def compare_graphs(graph_list, ideal_graphs):
    results = []
    for idx, graph in enumerate(graph_list):
        jaccard = jaccard_similarity(graph, ideal_graphs[idx])
        hamming_dist = hamming_distance(graph, ideal_graphs[idx])
        cosine_sim_val = cosine_sim(graph, ideal_graphs[idx])
        dice = dice_coefficient(graph, ideal_graphs[idx])
        russell_rao = russell_rao_similarity(graph, ideal_graphs[idx])
        edge = sum(sum(graph))
        results.append({
            'graph_index': idx,
            'jaccard_similarity': jaccard,
            'hamming_distance': hamming_dist,
            'cosine_similarity': cosine_sim_val,
            'dice': dice,
            'russel_rao': russell_rao,
            'edge': edge,
        })
    if len(graph_list) == 1:
        jaccard = jaccard_similarity(graph, ideal_graphs[1])
        hamming_dist = hamming_distance(graph, ideal_graphs[1])
        cosine_sim_val = cosine_sim(graph, ideal_graphs[1])
        dice = dice_coefficient(graph, ideal_graphs[1])
        russell_rao = russell_rao_similarity(graph, ideal_graphs[1])
        results.append({
            'graph_index': 1,
            'jaccard_similarity': jaccard,
            'hamming_distance': hamming_dist,
            'cosine_similarity': cosine_sim_val,
            'dice': dice,
            'russel_rao': russell_rao,
            'edge': edge,
        })
    return results

# Example usage:
# Define adjacency matrices for graphs (example with 3 nodes)

final_results = []
trade_dates = ['2023-02-01','2023-03-01','2023-04-01','2023-05-01','2023-06-01','2023-07-01','2023-08-01','2023-09-01','2023-10-01','2023-11-01','2023-12-01']
allElems = [("corr_annually_graph_data/adjacent_matrix_{}.h5",3, "correlacioAnual"),
                        ("corr_monthly_graph_data/adjacent_matrix_{}.h5",2, "correlacioMensual"),
                        ("monthly_news/adjacent_matrix_{}.h5", 2, "noticiesMensuals"),
                        ("graph_constructors/adjacent_matrix_industry.h5", 0, "industria"),
                        ("graph_constructors/adjacent_matrix_subindustry.h5", 0, "subindustria"),
                        ("graph_constructors/adjacent_matrix_wikidataDefinitiu.h5", 0, "wikidata"),
                        ("graph_constructors/adjacent_matrix_stockholders.h5", 0, "stockholders"),
                        ]

for g,m,name in allElems:

    if m == 0 or m == 3: n = 1
    else: n = 11
    jac = [0,0]
    ham = [0,0]
    cos = [0,0]
    rus = [0,0]
    dic = [0,0]
    edges = [0,0]
    
    for i in range(n):
        graph = []
        if m == 0:
            graph.append(pd.read_hdf(g, key="graph").values)
            ideal_graph = (pd.read_hdf("corr_annually_graph_data/adjacent_matrix_2023.h5", key="graph").values)
        else:
            if m == 2:
                ideal_graph = (pd.read_hdf(f"corr_monthly_graph_data/adjacent_matrix_{trade_dates[i][:7]}.h5", key="graph").values)
                previous_date = pd.to_datetime(trade_dates[i]) - pd.DateOffset(months=1)
                number_of_char = 7
            elif m == 3:
                ideal_graph = (pd.read_hdf("corr_annually_graph_data/adjacent_matrix_2023.h5", key="graph").values)
                previous_date = pd.to_datetime(trade_dates[i]) - pd.DateOffset(years=1)
                number_of_char = 4


            if g[:4] == "corr":
                corr_graf = pd.read_hdf(g.format(str(previous_date.strftime('%Y-%m-%d')[:number_of_char])), key="graph").values
                upstream = copy.deepcopy(corr_graf.squeeze())
                upstream[upstream >= 0.6] = 1
                upstream[upstream < 0.6] = 0
                downstream = copy.deepcopy(corr_graf.squeeze())
                downstream[downstream > -0.6] = 0
                downstream[downstream <= -0.6] = 1
                graph.append(upstream)
                graph.append(downstream)
            else: graph.append(pd.read_hdf(g.format(str(previous_date.strftime('%Y-%m-%d')[:number_of_char])), key="graph").values)


        ideal_upstream = copy.deepcopy(ideal_graph.squeeze())
        ideal_upstream[ideal_upstream >= 0.6] = 1
        ideal_upstream[ideal_upstream < 0.6] = 0
        ideal_downstream = copy.deepcopy(ideal_graph.squeeze())
        ideal_downstream[ideal_downstream > -0.6] = 0
        ideal_downstream[ideal_downstream <= -0.6] = 1
        ideal_graphs = [ideal_upstream,ideal_downstream]

        
        results = compare_graphs(graph, ideal_graphs)

        for result in results:
            jac[result['graph_index']] += result['jaccard_similarity']
            ham[result['graph_index']] += result['hamming_distance']
            cos[result['graph_index']] += result['cosine_similarity']
            dic[result['graph_index']] += result['dice']
            rus[result['graph_index']] += result['russel_rao']
            edges[result['graph_index']] += result['edge']
            
            with open("resultsGraphDistance.txt", "a") as file:
                file.write(f"{name}_____________ITERATION {i}____________\n")
                file.write(f"Graph {result['graph_index']} - Jaccard Similarity: {result['jaccard_similarity']:.4f}, \n"
                    f"Hamming Distance: {result['hamming_distance']:.4f}, Cosine Similarity: {result['cosine_similarity']:.4f}, \n" 
                    f"Russel similarity: {result['russel_rao']:.4f}, Dice similarity: {result['dice']:.4f}, Edges: {result['edge']} \n")
    
    with open("resultsGraphDistance.txt", "a") as file:
        file.write(f"*****Final results {name}: \n")
        file.write(f"Jac similarity: {[x / n for x in jac]}\n")
        file.write(f"Ham similarity: {[x / n for x in ham]}\n")
        file.write(f"Cos similarity: {[x / n for x in cos]}\n")
        file.write(f"Dice similarity: {[x / n for x in dic]}\n")
        file.write(f"Russel similarity: {[x / n for x in rus]}\n")
        file.write(f"Edges: {[x / n for x in edges]}\n")
    
    final_results.append((name, f"Jac similarity: {[x / n for x in jac]}", f"Ham similarity: {[x / n for x in ham]}", f"Cos similarity: {[x / n for x in cos]}", 
    f"Dice similarity: {[x / n for x in dic]}", f"Russel similarity: {[x / n for x in rus]}", f"Edges: {[x / n for x in edges]}"))


print("________________________________________________________________________")
print("FINAL RESULTS SUPER FINAL:")
print(final_results)
    





