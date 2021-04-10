from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from tensorflow.keras.layers import Embedding

import ast
import pandas as pd
import numpy as np


def gensim_to_keras_embedding(model, train_embeddings=False):
    """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

    Parameters
    ----------
    train_embeddings : bool
        If False, the returned weights are frozen and stopped from being updated.
        If True, the weights can / will be further updated in Keras.

    Returns
    -------
    `keras.layers.Embedding`
        Embedding layer, to be used as input to deeper network layers.

    """
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array    
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
    )
    return layer


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


def build_w2v(data, size, window, min_count, sample, negative, hs, workers, fine_tuning_on = False, seed = False):
    print("Starting building w2v model")
    if seed:
        model = Word2Vec(size=size, window=window, min_count=min_count, sample=sample, negative=negative, hs=hs, workers=workers, callbacks=[callback()], seed = seed)
    else:
        model = Word2Vec(size=size, window=window, min_count=min_count, sample=sample, negative=negative, hs=hs, workers=workers, callbacks=[callback()])
    model.build_vocab(data)
    print("number of corpus: {}".format(model.corpus_count))
    #wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/home/jupyter/model/twitter128.bin"), binary=True)
    if fine_tuning_on:
        model.intersect_word2vec_format(datapath(fine_tuning_on), binary=True, lockf=0)
    print("n. vocab: {}".format(len(model.wv.vocab.keys())))
    return model

def train_w2v(model, data, max_epoch = 5):
    total_examples = model.corpus_count
    model.train(data, total_examples=total_examples, epochs=max_epoch, compute_loss=True)

def save_w2v_model(model, path):
    model.callbacks = ()
    model.save(path)

def save_w2v_kvectors(model, path):
    model.wv.save(path)

def load_w2v_model():
    Word2Vec.load()

def load_w2v_kvectors(path):
    KeyedVectors.load(path)

######################################
#   build_keras_embedding_matrix
####################################
def get_index_key_association(wv):
    key_to_index = {"<UNK>": 0}
    index_to_key = {0: "<UNK>"}
    for idx, word in enumerate(sorted(wv.vocab)):
        key_to_index[word]  = idx+1 # which row in `weights` corresponds to which word?
        index_to_key[idx+1] = word # which row in `weights` corresponds to which word?
    return index_to_key, key_to_index

def build_keras_embedding_matrix(wv, index_to_key=None):
    print('Vocab_size is {}'.format(len(wv.vocab)))
    vec_size = wv.vector_size
    vocab_size = len(wv.vocab) + 1 # plus the unknown word
    
    if index_to_key is None:
        index_to_key, _ = get_index_key_association(wv)
    # Create the embedding matrix where words are indexed alphabetically
    embedding_matrix = np.zeros(shape=(vocab_size, vec_size))
    for idx in index_to_key: 
        #jump the first, words not found in embedding int 0 and will be all-zeros
        if idx != 0:
            embedding_matrix[idx] = wv.get_vector(index_to_key[idx])

    print('Embedding_matrix with unk word loaded')
    print('Shape {}'.format(embedding_matrix.shape))
    return embedding_matrix, vocab_size


def get_int_seq(sentences, word_to_key):
    return [ [word_to_key[word] for word in sentence] for sentence in sentences]

######################################
#   POS
####################################
def get_index_key_pos_association(X):
    key_to_index = {"<UNK>": 0}
    index_to_key = {0: "<UNK>"}
    unique_pos = set([pos for words in X for pos in words])

    for idx, word in enumerate(sorted(unique_pos)):
        key_to_index[word]  = idx+1 # which row in `weights` corresponds to which word?
        index_to_key[idx+1] = word # which row in `weights` corresponds to which word?
    return index_to_key, key_to_index

def get_one_hot_pos(index_to_key_pos):
    index_to_onehot_pos = {}
    for idx in index_to_key_pos.keys():
        ohe = [0 for _ in range(len(index_to_key_pos.keys())-1)]
        if idx != 0:
            ohe[idx-1] = 1
        index_to_onehot_pos[idx]  = ohe # which row in `weights` corresponds to which word?
    return index_to_onehot_pos

######################################
#   manual
####################################
def sentence_to_emb(sentence, w2v, truncate = None, padding = False):
    pad_token = [0]*w2v.vector_size
    s_emb = [ w2v[word] if word in w2v.vocab else pad_token for word in sentence]
    if truncate is not None:
        s_emb = s_emb[:truncate] #truncate
    if padding:
        s_emb += [pad_token] * (truncate - len(s_emb))
    return np.array(s_emb)

def get_data_to_emb(data, w2v, truncate = None, padding = False):
    X = [sentence_to_emb(sentence, w2v, truncate, padding) for sentence in data]
    return np.array(X)

if __name__ == "__main__": 

    import sys
    #root_project = "/content/SaRaH/"
    root_project = "/Users/Alessandro/Dev/repos/SaRaH/"
    #root_project = "/home/jupyter/SaRaH/"
    sys.path.append(root_project)

    from src.data.utils import load_csv_to_dict

    dataset_path    = root_project+'dataset/haspeede2/preprocessed2/dev/dev.csv'
    w2v_bin_path    = root_project+'results/model/word2vec/twitter128.bin'

    dataset = load_csv_to_dict(dataset_path)
    

    new_set = set([word for words in dataset["tokens"] for word in words]) 
    print("Unique items in corpora are:", len(new_set))

    model = build_w2v(dataset["tokens"], 
                      size=128, 
                      window=5, 
                      min_count=1, 
                      sample=1e-4, 
                      negative=5, 
                      hs=0, 
                      workers=4, 
                      fine_tuning_on = w2v_bin_path, 
                      seed = 1)

    train_w2v(model, dataset["tokens"], max_epoch= 2000)
    
    print(model.wv.most_similar('boldrini'))

    save_w2v_model(model, "word2vec2.model")
    save_w2v_kvectors(model, "word2vec2.wordvectors")