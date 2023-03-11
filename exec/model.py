MAX_TEXT_LEN = 65

import sys
import os
root_project = os.path.abspath(os.getcwd())
print(root_project)
print(os.path.abspath("results/model/word2vec/twitter128.bin"))
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath("results/model/word2vec/twitter128.bin"))
w2v_bin_path               = os.path.abspath("results/model/word2vec/twitter128.bin")

import numpy as np

from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath

from src.data.utils import set_unkmark_token, to_emb
from src.data.word_embedding import get_index_key_association, get_int_seq, build_keras_embedding_matrix
from src.data.preprocessing.preprocessor import PreProcessor

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.metrics import f1_macro

prp = PreProcessor()

#load word2vec and embedding_matrix
w2v = KeyedVectors.load_word2vec_format(datapath(w2v_bin_path), binary=True)
index_to_key, key_to_index = get_index_key_association(w2v)
embedding_matrix, vocab_size = build_keras_embedding_matrix(w2v, index_to_key)

dependencies = {
    'f1_macro': f1_macro
}
model = tf.keras.models.load_model(os.path.abspath("results/model/kcnn/kcnn.h5"), custom_objects=dependencies)

def preprocessing(text):
    preprocessed_input = prp.process_text(text)
    tokenized_pre_input = prp.get_token(preprocessed_input)
    return tokenized_pre_input

def to_embedding(input, w2v, key_to_index, embedding_matrix, max_text_len):
    #TODO: deve ritornare anche tutto il resto, extra, lemma, stem, ...
    X = input
    X = set_unkmark_token(X, w2v)
    X = get_int_seq(X, key_to_index)
    X = pad_sequences(X, maxlen=MAX_TEXT_LEN, padding='post', truncating='post')
    X = np.array([ [ embedding_matrix[index_word] for index_word in sentence] for sentence in X])
    return X


def get_prediction(text):
    preprocessed_text = preprocessing(text)
    X_e = to_embedding([preprocessed_text], w2v, key_to_index,embedding_matrix, MAX_TEXT_LEN)
    print(X_e)
    return model.predict(X_e)

#import tensorflow_addons as tfa


if __name__ == "__main__":
    result = get_prediction("@user Questo :) per dimostrare agli idioti buonisti di sinistra filoimmigrazionisti che nessuno vuole ")
    print(result)



