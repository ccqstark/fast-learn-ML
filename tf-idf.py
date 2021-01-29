# tf-idf 进行文本特征抽取
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


def cut_word(text):
    return " ".join(list(jieba.cut(text)))


# 分词后就可以做文本特征提取了
if __name__ == "__main__":

    data = ["一种还是一种今天很残酷,明天更残酷,后天很美好,但绝对大部分是死在明天晚上,所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的,这样当我们看到宇宙时,我们是在看它的过去",
            "如果只用一种方式了解某样事物,你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    transfer = TfidfVectorizer(stop_words=["一种", "所以"])

    data_final = transfer.fit_transform(data_new)

    print(data_final.toarray(), "\r\n", transfer.get_feature_names())
