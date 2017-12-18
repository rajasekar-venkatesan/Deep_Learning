import pandas as pd, numpy as np
from sklearn.preprocessing import OneHotEncoder

author_int_dict = {'EAP':0,'HPL':1,'MWS':2}

def load_train_test_data (num_samples=None):
    train_data = pd.read_csv('../data/train.csv')
    train_data['author'] = [author_int_dict[a] for a in train_data['author'].tolist()]
    test_data = pd.read_csv('../data/test.csv')
    return train_data[:num_samples],test_data[:num_samples]

def categorical_labeler (labels):
    labels = labels.reshape(-1, 1)
    #labels = OneHotEncoder().fit_transform(labels).todense()
    labels = np.array(labels, dtype=np.int64)
    return labels


if __name__ == '__main__':
    pass