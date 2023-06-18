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
    for row in finder.score_ngrams(bi_gram_measures.pmi):
        if vocab is not None:
            word_pair = row[0]
            if word_pair[0] in vocab and word_pair[1] in vocab:
                if word_pair[0] == word_pair[1]:
                    continue
            else:
                continue
        pmi_list.append(row)
    return pmi_list
