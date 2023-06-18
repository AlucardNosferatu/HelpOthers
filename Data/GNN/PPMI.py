import nltk
from nltk import word_tokenize, BigramAssocMeasures, BigramCollocationFinder

nltk.download('punkt')


def pmi(
        text="this is a foo bar bar black sheep  foo bar bar black sheep foo bar bar black sheep shep bar bar black "
             "sentence",
        vocab=None
):
    words = word_tokenize(text)
    bi_gram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    pmi_list = []
    pair_dict = {}
    for row in finder.score_ngrams(bi_gram_measures.pmi):
        word_pair = row[0]
        if vocab is not None:
            if word_pair[0] in vocab and word_pair[1] in vocab:
                if word_pair[0] == word_pair[1]:
                    continue
                elif word_pair[1] in pair_dict.keys():
                    if word_pair[0] in pair_dict[word_pair[1]]:
                        continue
            else:
                continue
        pmi_list.append(row)
        if word_pair[0] not in pair_dict.keys():
            pair_dict.__setitem__(word_pair[0], [])
        pair_dict[word_pair[0]].append(word_pair[1])
    return pmi_list
