from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knn_iris():
    """
    用KNN算法对鸢尾花进行分离
    :return:
    """
    # 1 获取数据
    iris = load_iris()

    # 2 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3 特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4 KNN算法预估模型
    estimator = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
    estimator.fit(x_train, y_train)

    # 5 模型评估
    # 方法1： 直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和与测试", y_test == y_predict)

    # 方法2： 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None


def knn_iris_gscv():
    """
    用KNN算法对鸢尾花进行分离, 并进行模型选择调优（网格搜索+交叉验证）
    :return:
    """
    # 1 获取数据
    iris = load_iris()

    # 2 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3 特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4 KNN算法预估模型
    estimator = KNeighborsClassifier(algorithm='auto')

    # 网格搜索+交叉验证
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    estimator.fit(x_train, y_train)

    # 5 模型评估
    # 方法1： 直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和与测试", y_test == y_predict)

    # 方法2： 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 最佳参数
    print(estimator.best_params_)
    # 最佳结果
    print(estimator.best_score_)
    # 最佳估计器
    print(estimator.best_estimator_)
    # 最佳验证结果
    print(estimator.cv_results_)

    return None


if __name__ == "__main__":
    # knn预测鸢尾花结果
    # knn_iris()

    knn_iris_gscv()
