import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

""""
过滤低方差特征
"""
def variance_demo():
    # 获取数据
    data = pd.read_csv("./resource/factor_returns.csv")
    data = data.iloc[:, 1:-2]

    print("原来的：", data.shape)

    # 实例化转化器类
    tranfer = VarianceThreshold(threshold=10)

    # 调用fit_transform
    data_new = tranfer.fit_transform(data)
    print(data_new)

    print("转化后的：", data_new.shape)

    # 计算2个变量之间的相关系数
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("相关系数1:", r1)

    r2 = pearsonr(data["revenue"], data["total_expense"])
    print("相关系数2:", r2)


if __name__ == "__main__":
    variance_demo()
