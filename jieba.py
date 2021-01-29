# jieba 分词
from sklearn.feature_extraction.text import CountVectorizer
import jieba


def cut_word(text):
    return list(jieba.cut(text))

# 分词后就可以做文本特征提取了
if __name__ == "__main__":
    data = cut_word("我爱北京天安门")
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    print(data_new.toarray(), "\r\n", transfer.get_feature_names())
