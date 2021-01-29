# 字典特征抽取
from sklearn.feature_extraction import DictVectorizer

data = [
    {'city': '北京', 'temperature': 100},
    {'city': '上海', 'temperature': 60},
    {'city': '深圳', 'temperature': 30}
]

# 实例化一个转化器类, 默认返回sparse稀疏矩阵(稀疏矩阵将非0值按坐标表示出来)
transfer = DictVectorizer(sparse=False)
# 调用fit_transform()
data_new = transfer.fit_transform(data)

print(data_new)
print(transfer.get_feature_names())
