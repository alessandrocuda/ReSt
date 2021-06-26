import sys, argparse, json

parser = argparse.ArgumentParser()
parser.add_argument("-c", action="store", dest="core", type=int)
parser.add_argument("-i", action="store", dest="input_file")
parser.add_argument("-o", action="store", dest="output_file")

results = parser.parse_args()
inputfile = results.input_file
outputfile = results.output_file
core = results.core

print()
print('Input file is ' + inputfile)
print('Output file is ' + outputfile)
print('Core {}'.format(core))
print()
# reading the data from the file
with open(inputfile) as f:
    data = f.read()
param_grid_dict = json.loads(data)
  
print("HP : ", param_grid_dict)

#"/home/jupyter/Cudazzo_Volpi/ReSt/"
#root_project = "/Users/Alessandro/Dev/repos/ReSt/"
#root_project = "/home/jupyter/ReSt/"
root_project = "../../../"
sys.path.append(root_project)

from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

from src.model_selection.gridsearch_multi import gridsearch
from src.rest_bilstm.a_bilstm import get_score
from src.utils.utils import save_grid_result
from src.data.utils import load_csv_to_dict, set_unkmark_token, load_data, to_emb
from src.data.word_embedding import get_index_key_association, build_keras_embedding_matrix, get_index_key_pos_association, get_one_hot_pos


#PATH
dataset_dev_path           = root_project + "dataset/haspeede2/preprocessed/dev/dev.csv"
dataset_test_tweets_path   = root_project + "dataset/haspeede2/preprocessed/reference/reference_tweets.csv"
w2v_bin_path               = root_project + 'results/model/word2vec/twitter128.bin'

#load dataset dictionary
dataset_dev = load_csv_to_dict(dataset_dev_path)
dataset_test_tweets = load_csv_to_dict(dataset_test_tweets_path)

#load word2vec and embedding_matrix
w2v = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
index_to_key, key_to_index = get_index_key_association(w2v)
embedding_matrix, vocab_size = build_keras_embedding_matrix(w2v, index_to_key)

WORD_EMB_SIZE = 128
VOCAB_SIZE = vocab_size

#load pos embedding
index_to_key_pos, key_to_index_pos = get_index_key_pos_association(X = dataset_dev["pos"])
index_to_onehot_pos = get_one_hot_pos(index_to_key_pos)


#load dev/test
MAX_TEXT_LEN = 65

X, X_pos, X_extra_feature, y = load_data(dataset_dev, w2v, key_to_index, key_to_index_pos, embedding_matrix, MAX_TEXT_LEN)
X_e = to_emb(X, embedding_matrix)
print(X_e.shape)
print(X_pos.shape)
data = {"text": X_e, "pos": X_pos, "extra": X_extra_feature, "target": y}
# grid search

result = gridsearch(param_grid_dict, get_score, data, 5, 42, True, core)
save_grid_result(outputfile, result)
