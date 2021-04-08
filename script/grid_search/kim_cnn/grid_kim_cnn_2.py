import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split


from gensim.models import KeyedVectors
from gensim.test.utils import datapath


import sys
#"/home/jupyter/Cudazzo_Volpi/ReSt/"
root_project = "/Users/Alessandro/Dev/repos/ReSt/"
#root_project = "/home/jupyter/ReSt/"

sys.path.append(root_project)

from src.model_selection.gridsearch import gridsearch
from src.rest_cnn.kim_cnn import kim_cnn
from src.utils.utils import save_grid_result

from src.data.utils import load_csv_to_dict, set_unkmark_token
from src.data.word_embedding import get_index_key_association, get_int_seq, build_keras_embedding_matrix, get_data_to_emb


#PATH
dataset_dev_path           = root_project + "dataset/haspeede2/preprocessed/dev/dev.csv"
dataset_test_tweets_path   = root_project + "dataset/haspeede2/preprocessed/reference/reference_tweets.csv"
w2v_bin_path               = root_project + 'results/model/word2vec/twitter128.bin'

#load word2vec and embedding_matrix
w2v = KeyedVectors.load_word2vec_format(datapath(w2v_bin_path), binary=True)
index_to_key, key_to_index = get_index_key_association(w2v)
embedding_matrix, vocab_size = build_keras_embedding_matrix(w2v, index_to_key)

WORD_EMB_SIZE = 128
VOCAB_SIZE = vocab_size


#load dataset dictionary
dataset_dev = load_csv_to_dict(dataset_dev_path)
dataset_test_tweets = load_csv_to_dict(dataset_test_tweets_path)


def load_data(dataset_dict, w2v, key_to_index, embedding_matrix, max_text_len):
    #TODO: deve ritornare anche tutto il resto, extra, lemma, stem, ...
    senteces = dataset_dict["tokens"]
    X = dataset_dict["tokens"]
    X = set_unkmark_token(X, w2v)
    X = get_int_seq(X, key_to_index)
    X = pad_sequences(X, maxlen=max_text_len, padding='post', truncating='post')
    X = np.array(X)
    y = np.array(dataset_dict["stereotype"])
    return X, y

def to_emb(X):
    return np.array([ [ embedding_matrix[index_word] for index_word in sentence] for sentence in X])


#load dev/test
MAX_TEXT_LEN = 65

X, y = load_data(dataset_dev, w2v, key_to_index, embedding_matrix, MAX_TEXT_LEN)
X_e = to_emb(X)
X_test, y_test = load_data(dataset_test_tweets, w2v, key_to_index, embedding_matrix, MAX_TEXT_LEN)
X_test_e = to_emb(X_test)


param_grid_dict = {
    "filters": [256],
    "filter_sizes": [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
    "dropout":  [0.1, 0.5, 0.9],
    "hn": [64, 124, 512],
    "lr": [0.1, 0.01, 0.001, 0.0001],
}

result = gridsearch(param_grid_dict, kim_cnn, X_e, y, 5, 42, True, 1)
save_grid_result("result_kim_cnn_2.p", result)
