import numpy as np

from Model.BertDNN.TopicsClassifier import onehot_mat
from Model.LSTM.LSTM import data_load_single_seq, pad_to_length, model_build


def enum_attempts(model, topics, sentiments, sentiment_levels=10):
    results_all = []
    all_attempts = []
    for i in range(7):
        current_topic = onehot_mat[i, :]
        results_topic = []
        for j in range(sentiment_levels):
            current_sentiment = j / sentiment_levels
            attempt_t = np.insert(topics, topics.shape[0], current_topic, axis=0)
            attempt_s = np.insert(sentiments, sentiments.shape[0], current_sentiment, axis=0)
            attempt_t, attempt_s = pad_to_length(attempt_t, attempt_s, 32)
            all_attempts.append([attempt_t, attempt_s])
            result = model.predict(
                x=[
                    np.expand_dims(attempt_t, axis=0),
                    np.expand_dims(attempt_s, axis=0)
                ],
                verbose=0
            )
            results_topic.append(result)
        results_all.append(results_topic)
    max_sentiment = [[sub_item[1][0] for sub_item in item] for item in results_all]
    max_sentiment = [[max(item), item.index(max(item))] for item in max_sentiment]
    max_sentiment = max_sentiment[
                        [item[0] for item in max_sentiment].index(
                            max([item[0] for item in max_sentiment])
                        )
                    ] + [
                        [item[0] for item in max_sentiment].index(
                            max([item[0] for item in max_sentiment])
                        )
                    ]
    max_sentiment_v = max_sentiment[0]
    max_sentiment_s = max_sentiment[1]
    max_sentiment_t = max_sentiment[2]
    return results_all, max_sentiment, max_sentiment_v, max_sentiment_s, max_sentiment_t, all_attempts


if __name__ == '__main__':
    _, _, t, s = data_load_single_seq(data_path='Data/chat.txt', pad_length=None, load_amount=None)
    for _ in range(4):
        mdl = model_build(force_new=False, model_path='Model/LSTM/NextPredictor.h5')
        ra, ms, msv, mss, mst, aa = enum_attempts(model=mdl, topics=t, sentiments=s)
        print(msv, mss, mst)
