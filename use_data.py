from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def datasets_demo():
    """
    sklearn 数据集使用
    """

    # 获取数据集
    iris = load_iris()
    # print("鸢尾花数据集: \r\n", iris)
    # print("数据集描述: \r\n", iris["DESCR"])
    # print("查看特证名: \r\n", iris.feature_names)
    # print("查看特征值: \r\n", iris.data, iris.data.shape)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=666)
    print("训练集特征值: \r\n", x_train, x_train.shape)
    return None


if __name__ == '__main__':
    datasets_demo()
