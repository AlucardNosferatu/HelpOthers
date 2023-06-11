import keras_nlp
import numpy as np

tokenizer = keras_nlp.models.BertTokenizer.from_preset("bert_tiny_en_uncased")
# Tokenize some input.
vec = np.array(tokenizer("The quick brown fox tripped."))
# Detokenize some input.
tokenizer.detokenize(vec)
