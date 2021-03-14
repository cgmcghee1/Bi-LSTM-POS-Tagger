import numpy as np


def pad(X,y,tag_len):
    for i in range(100-len(X)):
        X.insert(0, np.zeros(300))
        y.insert(0,np.zeros(tag_len))

    return(X,y)
