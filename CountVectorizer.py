# 文本特征抽取
from sklearn.feature_extraction.text import CountVectorizer

data = ["life is short,i like like python", "life is too long,i dislike python"]

transfer = CountVectorizer(stop_words=["is", "too"])

data_new = transfer.fit_transform(data)

print("data_new:\r\n", data_new.toarray(), transfer.get_feature_names())
