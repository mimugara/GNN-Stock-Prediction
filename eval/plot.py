import alphalens
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
from backtest import *


PATH = "/home/mmunoz/GNN-Stock-Prediction/alpha/model/THGNN/THGNN_0.0.1/infoValid_2020-01-01_2020-12-31.csv"

def plot_net_value(profit_series_list, labels, fig_size=(20, 5), title='Net Value Curve', x_label='Date', y_label='Net Value', line_width=2):
    """
    画净值曲线函数

    参数：
    profit_series_list: list，包含多个 pd.Series 类型的列表，每个 Series 代表一个净值序列
    fig_size: tuple，画布大小，默认为 (10, 6)
    title: str，图表标题，默认为 'Net Value Curve'
    x_label: str，x轴标签，默认为 'Date'
    y_label: str，y轴标签，默认为 'Net Value'

    返回：
    None
    """
    plt.figure(figsize=fig_size)
    # 遍历每个净值序列，计算其投资组合价值的时间序列数据，然后画出净值曲线
    for i, profit_series in enumerate(profit_series_list):
        net_value_series = profit_series.cumsum()
        plt.plot(net_value_series, label=labels[i], linewidth=line_width, marker="|")

    # 添加图例
    # 设置图表属性
    # plt.title(title)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    plt.legend()
    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.show()


if __name__ == "__main__":
    # ===================================pnl plot===============================================
    b = BackTest(1000000, 0.00, 0.00)

    thgnn = pd.read_csv(PATH, index_col=0)
    ret_thgnn = b.get_daily_pnl(thgnn, 0.1)
    ret_thgnn.index = pd.to_datetime(ret_thgnn.index, format="%Y-%m-%d")

    sp500 = thgnn.groupby("date")["ret"].mean().shift(1).fillna(0)
    sp500.index = pd.to_datetime(sp500.index, format="%Y-%m-%d")

    plot_net_value([ret_thgnn, sp500], ["THGNN", "SP500"])

    # ===================================signal plot===============================================
    # factor = thgnn[["date", "stock_id", "y_pred"]]
    # factor["date"] = pd.to_datetime(factor["date"], format="%Y%m%d")
    # factor.columns = ["date", "asset", "y_pred"]
    # factor = factor.set_index(["date", "asset"])["y_pred"]

    # ret = gru[["date", "stock_id", "ret"]]
    # ret["date"] = pd.to_datetime(ret["date"], format="%Y%m%d")
    # ret.columns = ["date", "asset", "ret"]
    # ret = ret.set_index(["date", "asset"])
    # ret.columns = ["1D"]

    # df = alphalens.utils.get_clean_factor(factor, ret, quantiles=5)
    # alphalens.tears.create_full_tear_sheet(df)

