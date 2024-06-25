import os
import optuna
from tqdm import tqdm
import pandas as pd
import math
import torch
import logging
import copy
import pickle
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import *
from model import *
from eval import *


PATH = "/home/mmunoz/GNN-Stock-Prediction/alpha/model/THGNN"

def set_logger(logger):
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=logger.name)

    logger.setLevel(logging.INFO)
    handler1.setLevel(logging.INFO)
    handler2.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def custom_collate_fn(batch):
    # Determine the minimum size in the batch
    min_size = min(item[0].shape[0] for item in batch)
    
    # Filter each instance to the minimum size
    filtered_batch = [
        (x[:min_size], y[:min_size], graph[:min_size, :min_size], date[:min_size], stock_id[:min_size]) 
        for x, y, graph, date, stock_id in batch
    ]

    # Separate filtered batch into individual components
    x_batch, y_batch, graph_batch, date_batch, stock_id_batch = zip(*filtered_batch)

    # Convert to tensors
    x_batch = torch.stack([torch.tensor(x) for x in x_batch])
    y_batch = torch.stack([torch.tensor(y) for y in y_batch])
    graph_batch = torch.stack([torch.tensor(graph) for graph in graph_batch])

    
    return x_batch, y_batch, graph_batch, date_batch, stock_id_batch


class THGNN_scheduler:
    def __init__(self,
                 name,
                 train_len,
                 valid_len,
                 look_back_window,
                 factor_list,
                 universe_version,
                 label_df,
                 batch_size,
                 hidden_size,
                 num_heads,
                 out_features,
                 num_layers,
                 lr,
                 weight_decay,
                 epochs,
                 max_patience
                 ):
        super(THGNN_scheduler).__init__()
        self.name = name

        os.makedirs(name, exist_ok=True)
        self.logger = logging.getLogger(os.path.join(name, "task.log"))
        self.logger = set_logger(self.logger)
        self.train_len = train_len
        self.valid_len = valid_len
        self.look_back_window = look_back_window
        self.label_df = label_df
        self.factor_list = factor_list
        self.universe_version = universe_version
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.out_features = out_features
        self.mum_layers = num_layers
        self.is_gpu = torch.cuda.is_available()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.loss_fn = torch.nn.MSELoss()
        self.max_patience = max_patience

    @staticmethod
    def get_date(date, num):
        trade_dates = pd.read_hdf("trade_dates.h5", key="trade_dates")
        idx = trade_dates[trade_dates <= date].last_valid_index()
        assert(idx-num >= 0)
        return trade_dates.iloc[idx-num]

    def train(self, srt_date, end_date, pathGraph, mode, nameGraph, identificador=0):
        train_srt_date = self.get_date(srt_date, self.train_len)
        valid_srt_date = self.get_date(srt_date, self.valid_len)
        train_end_date = self.get_date(valid_srt_date, self.look_back_window)
        valid_end_date = self.get_date(srt_date, self.look_back_window)
        print("----------- TRAINING DATES -----------------")
        print("Train start:", train_srt_date,"Train end:",train_end_date, "Valid start" , valid_srt_date, "Valid end", valid_end_date)

        train_dataset = GraphDataset(train_srt_date,
                                     train_end_date,
                                     self.label_df,
                                     self.look_back_window,
                                     self.factor_list,
                                     self.universe_version,
                                    pathGraph,
                                    mode,
                                     )
        valid_dataset = GraphDataset(valid_srt_date,
                                     valid_end_date,
                                     self.label_df,
                                     self.look_back_window,
                                     self.factor_list,
                                     self.universe_version,
                                    pathGraph,
                                    mode,)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        model = THGNN(len(self.factor_list), self.hidden_size, self.mum_layers, self.out_features, self.num_heads)
        if self.is_gpu:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_model = copy.deepcopy(model)
        best_metric = -np.inf
        num_patience = 0
        for i in range(self.epochs):
            train_loss, train_metric = self.train_epoch(model, train_dataloader, optimizer, "train")
            self.logger.info("EPOCH {}: LOSS {:.6f} | METRIC {:.3f}".format(i, train_loss, train_metric))
            valid_loss, valid_metric = self.train_epoch(model, valid_dataloader, optimizer, "valid")
            self.logger.info("EPOCH {}: LOSS {:.6f} | METRIC {:.3f}".format(i, valid_loss, valid_metric))
            if best_metric < valid_metric:
                num_patience = 0
                best_metric = valid_metric
                best_model = copy.deepcopy(model)
                self.logger.info("EPOCH {}: BEST METRIC {:.3f}".format(i, valid_metric))
            else:
                num_patience += 1
                self.logger.info("EPOCH {}: NUM PATIENCE {:.3f}".format(i, num_patience))
            if num_patience >= self.max_patience:
                break

        os.makedirs(self.name, exist_ok=True)
        if len(nameGraph) > 1:
            completeName = nameGraph[0]+nameGraph[1]
        else: completeName = nameGraph[0]
        with open(os.path.join(self.name, "model{}_{}_{}_{}.pkl").format(completeName,srt_date, end_date, identificador), "wb") as f:
            pickle.dump(best_model, f)
        return best_metric

    def train_epoch(self, model, loader, optimizer, mode):
        if mode == "train":
            model.train()
        elif mode == "valid":
            model.eval()
        total_loss = 0
        y_list = []
        y_pred_list = []
        stock_id_list = []
        date_list = []
        for x, y, graph, date, stock_id in (loader):
            if x.nelement() == 0:
                print("_---------------------------------------------------------------------")
                continue #to manage when the data is empty in the particular date (check mask application)
            x = (x.squeeze() - torch.mean(x.squeeze(), dim=0, keepdim=True)) / (torch.std(x.squeeze(), dim=0, keepdim=True) + 1e-6)
            y = (y.squeeze() - torch.mean(y.squeeze())) / (torch.std(y.squeeze()) + 1e-6)
            upstream = graph[0]
            downstream = graph[1]

            if self.is_gpu:
                x = x.squeeze().cuda()
                y = y.squeeze().cuda()
                upstream = upstream.cuda()
                downstream = downstream.cuda()
            y_pred, _ = model(x.float(), upstream.float(), downstream.float(), True)
            loss = self.loss_fn(y.float(), y_pred)
            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.data
            y_list.extend(y.squeeze().detach().cpu().numpy().tolist())
            y_pred_list.extend(y_pred.squeeze().detach().cpu().numpy().tolist())
            stock_id_list.extend(stock_id)
            date_list.extend(date)
        info_df = pd.DataFrame({
            "date": date_list,
            "stock_id": stock_id_list,
            "y": y_list,
            "y_pred": y_pred_list
        })
        info_df["date"] = info_df["date"].astype(str).str[2:-3]
        info_df["stock_id"] = info_df["stock_id"].astype(str).str[2:-3]
        ic = info_df.groupby("date").apply(lambda dd: dd[["y", "y_pred"]].corr().loc["y", "y_pred"],include_groups=False).mean()
        return total_loss/len(loader), ic

    def predict(self, srt_date, end_date, pathGraph, mode, nameGraph, identificador=0) :
        if len(nameGraph) > 1:
            completeName = nameGraph[0]+nameGraph[1]
        else: completeName = nameGraph[0]
        with open(os.path.join(PATH,self.name, "model{}_{}_{}_{}.pkl".format(completeName,srt_date, end_date, identificador)), "rb") as f:
            best_model = pickle.load(f)
        test_dataset = GraphDataset(srt_date,
                                    end_date,
                                    self.label_df,
                                    self.look_back_window,
                                    self.factor_list,
                                    self.universe_version,
                                    pathGraph,
                                    mode,
                                    )
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        if self.is_gpu:
            best_model.cuda()
        best_model.eval()
        total_loss = 0
        ret_list = []
        y_list = []
        y_pred_list = []
        stock_id_list = []
        date_list = []
        for x, y, graph, date, stock_id in (test_dataloader):
            if x.nelement() == 0: 
                print("_---------------------------------------------------------------------")
                continue
            x = (x.squeeze() - torch.mean(x.squeeze(), dim=0, keepdim=True)) / (torch.std(x.squeeze(), dim=0, keepdim=True) + 1e-6)
            y_ = (y.squeeze() - torch.mean(y.squeeze())) / (torch.std(y.squeeze()) + 1e-6)
            
            upstream = graph[0]
            downstream = graph[1]
            if self.is_gpu:
                x = x.squeeze().cuda()
                y_ = y_.squeeze().cuda()
                upstream = upstream.cuda()
                downstream = downstream.cuda()
            y_pred, _ = best_model(x.float(), upstream.float(), downstream.float(), True)
            loss = self.loss_fn(y_.float(), y_pred)

            total_loss += loss.data
            ret_list.extend(y.squeeze().cpu().numpy().tolist())
            y_list.extend(y_.squeeze().detach().cpu().numpy().tolist())
            y_pred_list.extend(y_pred.squeeze().detach().cpu().numpy().tolist())
            stock_id_list.extend(stock_id)
            date_list.extend(date)

        info_df = pd.DataFrame({
            "date": date_list,
            "stock_id": stock_id_list,
            "y": y_list,
            "y_pred": y_pred_list,
            "ret": ret_list
        })
        info_df["date"] = info_df["date"].astype(str).str[2:-3]
        info_df["stock_id"] = info_df["stock_id"].astype(str).str[2:-3]
        ic = info_df.groupby("date").apply(lambda dd: dd[["y", "y_pred"]].corr().loc["y", "y_pred"]).mean()
        info_df.to_csv(os.path.join(PATH,self.name, "info{}_{}_{}_{}.csv").format(completeName,srt_date, end_date, identificador))
        return total_loss/len(test_dataloader), ic, info_df

def objective(trial):
    # Suggest values for the hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [8,32,64])
    num_heads = trial.suggest_categorical('num_heads', [4, 8])
    out_features = trial.suggest_categorical('out_features', [8, 16])
    num_layers = trial.suggest_categorical('num_layers', [1, 2])
    lr = trial.suggest_categorical('lr', [1e-3, 1e-4])

    thgnn = THGNN_scheduler(
        name=f"THGNN_{trial.number}",
        train_len=252*5,
        valid_len=252,
        look_back_window=20,
        batch_size=1,
        factor_list = ['alphas_101_alpha_035',
               'alphas_101_alpha_038',
               'alphas_101_alpha_040',
               'alphas_101_alpha_043',
               'alphas_101_alpha_045',
               'alphas_101_alpha_049',
               'alphas_101_alpha_051',
               'alphas_101_alpha_053',
               'alphas_101_alpha_055',
               'alphas_101_alpha_060',
               'alphas_101_alpha_085',
               'alphas_101_alpha_001',
               'alphas_101_alpha_002',
               'alphas_101_alpha_003',
               'alphas_101_alpha_004',
               'alphas_101_alpha_006',
               'alphas_101_alpha_007',
               'alphas_101_alpha_008',
               'alphas_101_alpha_009',
               'alphas_101_alpha_010',
               'alphas_101_alpha_012',
               'alphas_101_alpha_013',
               'alphas_101_alpha_014',
               'alphas_101_alpha_015',
               'alphas_101_alpha_016',
               'alphas_101_alpha_017',
               'alphas_101_alpha_018',
               'alphas_101_alpha_019',
               'alphas_101_alpha_020',
               'alphas_101_alpha_021',
               'alphas_101_alpha_022',
               'alphas_101_alpha_023',
               'alphas_101_alpha_024',
               'alphas_101_alpha_026',
               'alphas_101_alpha_028',
               'alphas_101_alpha_029',
               'alphas_101_alpha_030',
               'alphas_101_alpha_031',
               'alphas_101_alpha_033',
               'alphas_101_alpha_034',
               'alphas_101_alpha_037',
               'alphas_101_alpha_039',
               'alphas_101_alpha_044',
               'alphas_101_alpha_046',
               'alphas_101_alpha_052',
               'alphas_101_alpha_054',
               'alphas_101_alpha_068'], 
        universe_version="zz800",
        label_df=opn_r,  
        hidden_size=hidden_size,
        num_heads=num_heads,
        out_features=out_features,
        num_layers=num_layers,
        lr=lr,
        weight_decay=0.0001,
        epochs=25,
        max_patience=10
    )
    n = 5
    count = 0
    for i in range(n):
        count += thgnn.train("2023-01-01", "2023-12-31")
        print("---------------------------------------")
        val_metric = count/n
    print("________________Trial Number: ", trial.number,flush = True)
    print("Val metric: ", val_metric,flush = True)
    print("Params: ",  hidden_size ,num_heads,
    out_features,
    num_layers,
    lr , flush = True)
    return val_metric





if __name__ == "__main__":
    # old_stdout = sys.stdout

    # log_file = open("message.log","w")

    # sys.stdout = log_file
    

    opn = pd.read_hdf("pv.h5", key="open")
    opn_r = opn.pct_change()
    opn_r = opn_r.shift(-2)
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100)

    # print("Best trial:")
    # trial = study.best_trial
    # print(f"  Value: {trial.value}")
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")
    thgnn = THGNN_scheduler(
                name="THGNN_0.0.1",
                train_len=252*5,
                valid_len=252,
                look_back_window=20,
                factor_list = ['alphas_101_alpha_035',
               'alphas_101_alpha_038',
               'alphas_101_alpha_040',
               'alphas_101_alpha_043',
               'alphas_101_alpha_045',
               'alphas_101_alpha_049',
               'alphas_101_alpha_051',
               'alphas_101_alpha_053',
               'alphas_101_alpha_055',
               'alphas_101_alpha_060',
               'alphas_101_alpha_085',
               'alphas_101_alpha_001',
               'alphas_101_alpha_002',
               'alphas_101_alpha_003',
               'alphas_101_alpha_004',
               'alphas_101_alpha_006',
               'alphas_101_alpha_007',
               'alphas_101_alpha_008',
               'alphas_101_alpha_009',
               'alphas_101_alpha_010',
               'alphas_101_alpha_012',
               'alphas_101_alpha_013',
               'alphas_101_alpha_014',
               'alphas_101_alpha_015',
               'alphas_101_alpha_016',
               'alphas_101_alpha_017',
               'alphas_101_alpha_018',
               'alphas_101_alpha_019',
               'alphas_101_alpha_020',
               'alphas_101_alpha_021',
               'alphas_101_alpha_022',
               'alphas_101_alpha_023',
               'alphas_101_alpha_024',
               'alphas_101_alpha_026',
               'alphas_101_alpha_028',
               'alphas_101_alpha_029',
               'alphas_101_alpha_030',
               'alphas_101_alpha_031',
               'alphas_101_alpha_033',
               'alphas_101_alpha_034',
               'alphas_101_alpha_037',
               'alphas_101_alpha_039',
               'alphas_101_alpha_044',
               'alphas_101_alpha_046',
               'alphas_101_alpha_052',
               'alphas_101_alpha_054',
               'alphas_101_alpha_068'],
                universe_version="zz800",
                label_df=opn_r,
                batch_size=1,
                hidden_size=8,
                num_heads=8,
                out_features=8,
                num_layers=2,
                lr=0.001,
                weight_decay=0.0001,
                epochs=100,
                max_patience=10)    
    
    # thgnn.train("2023-01-01", "2023-12-31")
    # loss_500, ic_500, df_500 = thgnn.predict("2023-01-01", "2023-12-31")
    # results = create_full_tear_sheet(df_500.set_index(["date", "stock_id"]))


    
    
    #MODE 0: graf
    #MODE 1: daily
    #MODE 2: monthly
    #MODE 3: yearly


    allElems = [("corr_annually_graph_data/adjacent_matrix_{}.h5",3, "correlacioAnual"),
                        ("corr_daily_graph_data/adjacent_matrix_{}.h5",1, "correlacioDiaria"),
                        ("corr_monthly_graph_data/adjacent_matrix_{}.h5",2, "correlacioMensual"),
                        ("monthly_news/adjacent_matrix_{}.h5", 2, "noticiesMensuals"),
                        ("graph_constructors/adjacent_matrix_industry.h5", 0, "industria"),
                        ("graph_constructors/adjacent_matrix_subindustry.h5", 0, "subindustria"),
                        ("graph_constructors/adjacent_matrix_wikidataDefinitiu.h5", 0, "wikidata"),
                        ("graph_constructors/adjacent_matrix_stockholders.h5", 0, "stockholders"),
                        ]

    elemsOneGraph = allElems[3:]
    elemsTwoGraphs = [[allElems[2]],[allElems[4],allElems[5]],[allElems[4],allElems[7]],]
                    #[allElems[0]],
    #                 [allElems[1]],
                    
                    # [allElems[3],allElems[4]],
                    # [allElems[3],allElems[5]],
                    # [allElems[3],allElems[6]],
                    # [allElems[3],allElems[7]],
                    
                    # [allElems[4],allElems[6]],
                    
                    # [allElems[5],allElems[6]],
                    # [allElems[5],allElems[7]],
                    # [allElems[6],allElems[7]],
                


    # for pathGraph, modeGraph, nameGraph in tqdm(elemsOneGraph):
    #     ic_mean = 0
    #     rank_ic_mean = 0
    #     arr =0
    #     av = 0
    #     sharpe = 0
    #     wr = 0
    #     mdd = 0
    #     n = 5
    
    #     for i in (range(n)):
    #         thgnn.train("2023-01-01", "2023-12-31", pathGraph, modeGraph, nameGraph, i)
    #         loss_500, ic_500, df_500 = thgnn.predict("2023-01-01", "2023-12-31", pathGraph, modeGraph, nameGraph, i)
    #         results = create_full_tear_sheet(df_500.set_index(["date", "stock_id"]))

    #         ic_mean += results["IC"]
    #         rank_ic_mean += results["Rank IC"]
    #         arr += results["ARR"]
    #         av += results["AV"]
    #         sharpe += results["Sharpe"]
    #         wr += results["WR"]
    #         mdd += results["MDD"]

    #         with open("resultsOneGraph.txt", "a") as file:
    #             file.write(f"{nameGraph}\n")
    #             file.write(f"Iteration {i}\n")
    #             file.write(str(results))
    #             file.write("\n\n")
    #             if i == n-1: file.write(f"Final results: IC: {ic_mean/n} Rank_IC: {rank_ic_mean/n} ARR: {arr/n} AV: {av/n} Sharpe: {sharpe/n} WR: {wr/n} MDD: {mdd/n}\n")
    #         print(results)

    for elem in tqdm(elemsTwoGraphs):
        pathGraph = []
        modeGraph = []
        nameGraph = []
        for e in elem:
            pathGraph.append(e[0]), modeGraph.append(e[1]), nameGraph.append(e[2])
        ic_mean = 0
        rank_ic_mean = 0
        arr =0
        av = 0
        sharpe = 0
        wr = 0
        mdd = 0
        n = 5
    
        for i in (range(n)):
            thgnn.train("2021-01-01", "2021-12-31", pathGraph, modeGraph, nameGraph, i)
            loss_500, ic_500, df_500 = thgnn.predict("2021-01-01", "2021-12-31", pathGraph, modeGraph, nameGraph, i)
            results = create_full_tear_sheet(df_500.set_index(["date", "stock_id"]))

            ic_mean += results["IC"]
            rank_ic_mean += results["Rank IC"]
            arr += results["ARR"]
            av += results["AV"]
            sharpe += results["Sharpe"]
            wr += results["WR"]
            mdd += results["MDD"]

            with open("TestingYears2021.txt", "a") as file:
                file.write(f"{nameGraph}\n")
                file.write(f"Iteration {i}\n")
                file.write(str(results))
                file.write("\n\n")
                if i == n-1: file.write(f"Final results: IC: {ic_mean/n} Rank_IC: {rank_ic_mean/n} ARR: {arr/n} AV: {av/n} Sharpe: {sharpe/n} WR: {wr/n} MDD: {mdd/n}\n")
            print(results)


    


    


