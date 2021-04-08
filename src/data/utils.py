import ast
import csv
import pandas as pd
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from collections import defaultdict
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.data.word_embedding import get_int_seq

map_hashtag_delimiter = {"<hashtag>": "<", "</hashtag>": ">"}

list_features = ["lemma", "pos", "dep", "word_polarity", "tokens", "stem"]



def load_csv_to_df():
    pass

def load_csv_to_dict(path, sep = '\t', verbose = 0):
    """df : input a dataframe that contains a collumn "tokens" that is a string of a list of words
    return a list of lists for the word2vec
    """
    # data = defaultdict(list)
    # with open(path,'r') as data_csv: 
    #     for line in csv.DictReader(data_csv): 
    #         for k, v in line.items():
    #             data[k].append(v)

    dataset = pd.read_csv(path, sep=sep).to_dict('list')
    for k in list_features:
        if verbose:
            print("Converting data '{}' String in list of String".format(k))
        dataset[k] = [elem.split() for elem in dataset[k]]
    return dataset

def dtype(dict_dataset):
    for k in dict_dataset.keys():
        print("{} {}".format(k.ljust(20), type(dict_dataset[k][0])))

def str_to_list(string_list):
    """ It takes a string formatted in the following way:
            "['elem','elem','elem']"
        and return a list of string
    """
    try:
        list_of_string = ' '.join(ast.literal_eval(string_list)).split()
    except:
        print("wrong string list of string: {}".format(string_list))

    return list_of_string


def set_unkmark_token(sentences, vocab):
    sentences = [ [word if word in vocab else '<UNK>' for word in sentence]for sentence in sentences]
    return sentences

def dtype_transformation(dict_dataset, keys):
    for k in keys:
        print("Converting data '{}' in list of string".format(k))
        dict_dataset[k] = [str_to_list(elem) for elem in dict_dataset[k]]
    

def add_tokens_vector(w2v_path):
    wv_from_bin = KeyedVectors.load(datapath(w2v_path))
    map_hashtag_delimiter = {"<hashtag>": "<", "</hashtag>": ">"}
    dataset["tokens_vector"] = [ [wv_from_bin[elem] 
                                  if elem not in map_hashtag_delimiter 
                                  else wv_from_bin[map_hashtag_delimiter[elem]
                                  ] 
                                  for elem in elems] for elems in dataset["tokens"]]

def load_data(dataset_dict, w2v, key_to_index, key_to_index_pos, embedding_matrix, max_text_len):
    senteces = dataset_dict["tokens"]
    X = dataset_dict["tokens"]
    X = set_unkmark_token(X, w2v.vocab)
    X = get_int_seq(X, key_to_index)
    X = pad_sequences(X, maxlen=max_text_len, padding='post', truncating='post')
    X = np.array(X)
    
    X_pos = dataset_dict["pos"]
    X_pos = set_unkmark_token(X_pos, key_to_index_pos)
    X_pos = get_int_seq(X_pos, key_to_index_pos)
    X_pos = pad_sequences(X_pos, maxlen=max_text_len, padding='post', truncating='post')
    X_pos = np.array(X_pos)

    extra_feature = np.array([dataset_dict["hashtags"], dataset_dict["%CAPS-LOCK words"], dataset_dict["esclamations"], dataset_dict["questions"], dataset_dict["sentence_positive"], dataset_dict["sentence_negative"], dataset_dict["sentence_neutral"], dataset_dict["%bad_words"]]).T

    y = np.array(dataset_dict["stereotype"])
    return X, X_pos, extra_feature, y

def to_emb(X, embedding):
    return np.array([ [ embedding[index_word] for index_word in sentence] for sentence in X])

if __name__ == "__main__": 
    dataset_path = '/Users/Alessandro/Dev/repos/SaRaH/dataset/haspeede2/preprocessed/dev/dev.csv'
    w2v_path     = '/Users/Alessandro/Dev/repos/SaRaH/model/word2vec/word2vec.wordvectors'
    dataset = load_csv_to_dict(dataset_path)
    print(dataset.keys())
    # add_tokens_vector(w2v_path)

    # print(dataset["tokens"][0])
    # print(dataset["tokens_vector"][0])
    # print(dataset["target"][0])