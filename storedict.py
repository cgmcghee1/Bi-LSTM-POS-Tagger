import pickle
import numpy as np

embeddings_index = {}
with open("glove.6B.300d.txt") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ").astype(np.float64)
        embeddings_index[word] = coefs

pickle.dump(embeddings_index, open("embeddings.p", "wb"))
