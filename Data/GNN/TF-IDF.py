import math

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer


def cut_words():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")

    con3 = jieba.cut(
        "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")
    c1 = " ".join(con1)
    c2 = " ".join(con2)
    c3 = " ".join(con3)

    return c1, c2, c3


def tf_idf_sklearn():
    """
    中文特征化
    :return None
    """
    c1, c2, c3 = cut_words()
    print("c1:", c1)
    tf = TfidfVectorizer()
    data = tf.fit_transform([c1, c2, c3])
    print("特征：")
    print(tf.get_feature_names_out())
    print("特征的大小：")
    print(len(tf.get_feature_names_out()))
    print("词向量：")
    print(data.toarray())
    print("第一列词向量的个数:")
    print(len(data.toarray()[0]))
    return None


def tf_idf_python(corpus):
    weight_long = [eve.split() for eve in corpus]
    word_all = []
    for eve in weight_long:
        for x in eve:
            if len(x) > 1 or True:
                word_all.append(x)
    word_all = list(set(word_all))  # 集合去重词库
    # 开始计算tf-idf
    weight = [[] for i in corpus]
    weight_idf = [[] for i in corpus]
    for word in word_all:
        for i in range(len(corpus)):
            temp_list = corpus[i].split()
            n1 = temp_list.count(word)
            tf = n1
            n2 = len(corpus)
            n3 = 0
            for eve in corpus:
                temp_list_ = eve.split()
                if word in temp_list_:
                    n3 += 1
            idf = math.log(((n2 + 1) / (n3 + 1))) + 1
            weight_idf[i].append(idf)
            weight[i].append(tf * idf)
    print('词典为：')
    print(word_all)
    print('原始tf-idf值为：')
    for w in weight:
        print(w)
    # L2范式归一化过程
    l2_weight = [[] for i in range(len(corpus))]
    for e in range(len(weight)):
        all2plus = 0
        for ev in weight[e]:
            all2plus += ev ** 2
        for ev in weight[e]:
            l2_weight[e].append(ev / (all2plus ** 0.5))
    return l2_weight  # 返回最终结果


if __name__ == '__main__':
    # tf_idf()
    corpus = ["我 来到 中国 旅游", "中国 欢迎 你", "我 喜欢 来到 中国 天安门"]
    result_list = tf_idf_python(corpus)
    print('归一化后的tf-idf值为：')
    for weight in result_list:
        print(weight)
