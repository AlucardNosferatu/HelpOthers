import numpy as np
from snownlp import SnowNLP


def data_load_sentiments(data_path='../Data/chat.txt'):
    f = open(data_path, 'r', encoding='utf-8')
    lines = f.readlines()
    lines = [SnowNLP(doc=line.split('#')[1]).sentiments for line in lines]
    return np.array(lines)


def get_sentiment(speech_str):
    return SnowNLP(doc=speech_str).sentiments


if __name__ == '__main__':
    s = get_sentiment('你们的高端路由器实在是太棒啦！')
    print(s)
    s = get_sentiment('你们的高端路由器还行。')
    print(s)
    s = get_sentiment('你们的高端路由器就是一坨屎，一坨没有价值的电子垃圾。')
    print(s)
