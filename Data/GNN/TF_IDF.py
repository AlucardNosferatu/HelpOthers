import math

from tqdm import tqdm


def tf_idf_python(corpus, word_all=None):
    weight_long = [eve.split() for eve in corpus]
    if word_all is None:
        word_all = []
        for eve in weight_long:
            for x in eve:
                if len(x) > 1 or True:
                    word_all.append(x)
        word_all = list(set(word_all))  # 集合去重词库
    # 开始计算tf-idf
    weight = [[] for _ in corpus]
    weight_idf = [[] for _ in corpus]
    for word in tqdm(word_all):
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
    # print('词典为：')
    # print(word_all)
    # print('原始tf-idf值为：')
    # for w in weight:
    #     print(w)
    # L2范式归一化过程
    l2_weight = [[] for _ in range(len(corpus))]
    for text_index in range(len(weight)):
        all2plus = 0
        for word_weight in weight[text_index]:
            all2plus += word_weight ** 2
        for word_weight in weight[text_index]:
            if all2plus != 0:
                l2_weight[text_index].append(word_weight / (all2plus ** 0.5))
    return l2_weight, word_all


if __name__ == '__main__':
    # tf_idf()
    corpus_ = ["我 来到 中国 旅游", "中国 欢迎 你", "我 喜欢 来到 中国 天安门"]
    tf_idf, vocab = tf_idf_python(corpus_)
    print('归一化后的tf-idf值为：')
    for weight_ in tf_idf:
        print(vocab)
        print(weight_)
