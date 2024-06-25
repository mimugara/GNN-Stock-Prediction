import time
import pickle
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

feature_cols = ['high','low','close','open','volume']


def cal_pccs(x, y, n):
    sum_xy = np.sum(np.sum(x*y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x*x))
    sum_y2 = np.sum(np.sum(y*y))
    pcc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))
    return pcc

def calculate_pccs(xs, yss, n):
    result = []
    for name in yss:
        ys = yss[name]
        tmp_res = []
        for pos, x in enumerate(xs):
            y = ys[pos]
            tmp_res.append(cal_pccs(x, y, n))
        result.append(tmp_res)
    return np.mean(result, axis=1)

def stock_cor_matrix(ref_dict, codes, n, processes=1):
    if processes > 1:
        pool = mp.Pool(processes=processes)
        args_all = [(ref_dict[code], ref_dict, n) for code in codes]
        results = [pool.apply_async(calculate_pccs, args=args) for args in args_all]
        output = [o.get() for o in results]
        data = np.stack(output)
        return pd.DataFrame(data=data, index=codes, columns=codes)
    data = np.zeros([len(codes), len(codes)])
    for i in tqdm(range(len(codes))):
        data[i, :] = calculate_pccs(ref_dict[codes[i]], ref_dict, n)
    return pd.DataFrame(data=data, index=codes, columns=codes)



#prev_date_num Indicates the number of days in which stock correlation is calculated
prev_date_num = 20
stock_trade_data= pd.read_hdf("trade_dates.h5", key="trade_dates")["trade_dates"]
stock_id = pd.read_hdf("stock_id.h5", key="stock_id")

#dt is the last trading day of each month
dt=[]
for i in ['2021','2022','2023']:
    for j in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        stock_m = stock_trade_data.loc[(stock_trade_data>=np.datetime64(i+'-'+j +"-01")) &(stock_trade_data<=np.datetime64(i+'-'+j)+pd.offsets.MonthEnd())]
        dt.append(stock_m.iloc[-1].strftime('%Y-%m-%d'))
print("---------------The last trading dates of each month are:-----------------")
print(dt)

df = []
for feat in feature_cols:
    df.append(pd.read_hdf("pv.h5",key = feat))

for i in tqdm(range(len(dt))):
    end_data = dt[i]
    start_data = stock_trade_data.iloc[stock_trade_data[stock_trade_data == end_data].index -(prev_date_num - 1)]
    start_data =  start_data.iloc[0].strftime('%Y-%m-%d')
    test_tmp = {}
    for ide in stock_id["Symbol"]:
        y = []
        for k in range(len(feature_cols)):
            df1 = df[k]
            df1 = df1.loc[start_data:end_data]
            df1 = df1[ide]
            y.append(df1.to_list())
            
        y = np.array(y)
        if y.shape[1] == prev_date_num:
            test_tmp[ide] = y
        else: 
            raise ValueError(f"Expected {prev_date_num} days of data, but got {y.shape} for stock {ide}")

    t1 = time.time()
    result = stock_cor_matrix(test_tmp, list(test_tmp.keys()), prev_date_num, processes=5)
    result=result.fillna(0)
    for i in range(len(stock_id["Symbol"])):
        result.iloc[i,i]=1
    t2 = time.time()
    print('time cost', t2 - t1, 's')
    result.to_hdf("graph_data/{}_{}.h5".format("adjacent_matrix", str(end_data)[:7]),key="graph")
