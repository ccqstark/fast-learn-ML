from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd

"""
归一化
"""
def min_max_demo():
    data = pd.read_csv("./resource/datingTestSet2.csv")
    data = data.iloc[:, :3]

    # 实例化一个转换器类
    # feature_range数据范围
    transfer = MinMaxScaler(feature_range=[2, 3])

    data_new = transfer.fit_transform(data)

    print(data_new)

"""
标准化
"""
def stand_demo():
    data = pd.read_csv("./resource/datingTestSet2.csv")
    data = data.iloc[:, :3]

    # 实例化一个转换器类
    transfer = StandardScaler()

    data_new = transfer.fit_transform(data)

    print(data_new)


if __name__ == '__main__':
    # min_max_demo();
    stand_demo()
