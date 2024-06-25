import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import sys
import copy


#from alpha.config.config import *


class GraphDataset(Dataset):
    def __init__(self,
                 srt_date,
                 end_date,
                 label_df,
                 look_back_window,
                 factor_list,
                 universe_version,
                 graf,
                 mode
                 ):
        self.srt_date = srt_date
        self.end_date = end_date
        self.trade_dates = self.get_trade_dates(srt_date, end_date)
        self.label_df = label_df
        self.look_back_window = look_back_window
        self.factor_list = factor_list
        self.factor_data = self._load_data()
        self.universe_df = pd.read_hdf("univers.h5",key="univers")
        self.graf_constructor = graf
        self.mode = mode

    @staticmethod
    def get_trade_dates(srt_date, end_date):
        trade_dates = pd.read_hdf("trade_dates.h5", key="trade_dates")
        df = trade_dates[(trade_dates >= srt_date) & (trade_dates <= end_date)]
        #print(df)
        return df.dropna()

    def __len__(self):
        return len(self.trade_dates)

    def _load_data(self):
        arr_list = []
        for factor in self.factor_list:
            # df = pd.read_hdf(os.path.join(DATA_PATH, "Ashare_data/factor_data/{}.h5".format(factor)), key="v")
            df = pd.read_hdf("factor_data/{}.h5".format(factor), key=factor[-9:])
            df = df.ffill()
            df.index = pd.to_datetime(df.index)

            # srt_idx = self.trade_dates.index[0] - self.look_back_window + 1
            # end_idx = self.trade_dates.index[-1] + 1
            for i in range(len(df.index)):
                # print(df,self.trade_dates)
                # print(df.index[i],self.trade_dates['trade_dates'].iloc[0])
                # print(type(df.index[i]),type(self.trade_dates['trade_dates'].iloc[0]))
                if df.index[i] == self.trade_dates['trade_dates'].iloc[0]:
                    srt_idx = i - self.look_back_window + 1
                elif df.index[i] == self.trade_dates['trade_dates'].iloc[-1]:
                    end_idx = i + 1
                    break
                
            # print(srt_idx)
            # print(end_idx)
            assert(srt_idx >= 0)
            assert(end_idx >= 0)
            df = df.iloc[srt_idx:end_idx, :]
            arr_list.append(df.values)
        return np.stack(arr_list, axis=-1).transpose((1, 0, 2))

    def __getitem__(self, idx):
        data = self.factor_data[:, idx: idx + self.look_back_window, :]
        date = self.trade_dates.iloc[idx]

        graph = []
        for g,m in zip(self.graf_constructor,self.mode):
            if m == 0:
                graph.append(pd.read_hdf(g, key="graph"))
            else:
                if m == 1:
                    previous_date = self.trade_dates.iloc[idx-1]['trade_dates']
                    number_of_char = 10
                elif m == 2:
                    previous_date = date['trade_dates'] - pd.DateOffset(months=1)
                    number_of_char = 7
                elif m == 3:
                    previous_date = date['trade_dates'] - pd.DateOffset(years=1)
                    number_of_char = 4


                if g[:4] == "corr":
                    corr_graf = pd.read_hdf(g.format(str(previous_date.strftime('%Y-%m-%d')[:number_of_char])), key="graph")
                    upstream = copy.deepcopy(corr_graf.squeeze())
                    upstream[upstream >= 0.6] = 1
                    upstream[upstream < 0.6] = 0
                    downstream = copy.deepcopy(corr_graf.squeeze())
                    downstream[downstream > -0.6] = 0
                    downstream[downstream <= -0.6] = 1

                    graph.append(upstream)
                    graph.append(downstream)
                   
            
                else: graph.append(pd.read_hdf(g.format(str(previous_date.strftime('%Y-%m-%d')[:number_of_char])), key="graph"))
        
        stock_id = pd.read_hdf("stock_id.h5", key="stock_id")["Symbol"]
        label = self.label_df.loc[date]
        label = np.squeeze(label)

        d = date['trade_dates']
        is_universe = self.universe_df['Belongs'][d]['Active'].values
        is_nan_x = ~np.isnan(data).sum(-1).sum(-1).astype(bool)
        is_nan_y = ~np.isnan(label.values).astype(bool)
        mask =  is_nan_x & is_nan_y &is_universe


        # old_stdout = sys.stdout

        # log_file = open("message.log","w")

        # sys.stdout = log_file



    
        

        data2 = data[mask]
        #if len(data2) == 0: 
        #     print("-------Data before mask-----")
        #     with np.printoptions(threshold=np.inf):
        #         data[~np.isnan(data)] = 0
        #         print(data.shape)
        #         print(data)
        #         print("-------Label before mask-----")
        #         print(label)
        #         raise Exception("Sorry, no numbers below zero")
        
        stock_id = list(stock_id[mask])
        for i,g in enumerate(graph):
            graph[i] = g.loc[stock_id, stock_id].replace(np.nan, 0).values
            np.fill_diagonal(graph[i], 1)


        label = label[mask].values
        l = [date['trade_dates'].strftime('%Y-%m-%d') for _ in range(len(label))]
        
        # print("---------Data--------")
        # print(data2)

        # print("---------label--------")
        # print(label)

        # print("---------graph--------")
        # print(graph)
        # print("---------l--------")
        # print(l)

        # print("---------stock_id--------")
        # print(stock_id)

        # sys.stdout = old_stdout

        # log_file.close()
        #print(type(data2), type(label),type(graph),type(l),type(stock_id))
        return [data2, label, graph, l, stock_id]
        # return {
        #     "data": data,
        #     "label": label,
        #     "graph": graph,
        #     "date": [date for _ in range(len(label))],
        #     "stock_id": stock_id,
        # }
