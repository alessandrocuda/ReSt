import pandas as pd
import numpy as np
from itertools import product
import joblib as joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import f1_score


#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def cross_val(get_score, data, kf, hyper_param):

    import tensorflow as tf    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
            
    folds_f1 = []
    folds_results = {}
    i = 0
    for train_index_fold, val_index_fold in kf.split(data["text"]):
        name_fold = "fold_{}".format(i)
        i+=1
        X_train = {"text": data["text"][train_index_fold], "pos": data["pos"][train_index_fold], "extra": data["extra"][train_index_fold]}
        Y_train = data["target"][train_index_fold]
        X_train, X_val_es, X_pos_train, X_pos_val_es, X_extra_feature_train, X_extra_feature_val_es, y_train, y_val_es = train_test_split(X_train["text"], X_train["pos"], X_train["extra"],   Y_train, test_size=0.09, random_state=128)
        
        #DATA
        X_train = {"text": X_train, "pos": X_pos_train, "extra": X_extra_feature_train}
        Y_train = y_train
        X_val_es = {"text": X_val_es, "pos": X_pos_val_es, "extra": X_extra_feature_val_es}
        Y_val_es = y_val_es
        X_val = {"text": data["text"][val_index_fold], "pos": data["pos"][val_index_fold], "extra": data["extra"][val_index_fold]}
        Y_val = data["target"][val_index_fold]

        folds_results[name_fold] = get_score(X_train, Y_train, X_val_es, Y_val_es, Y_val,  hyper_param)
        folds_f1.append(folds_results[name_fold]["val_f1"])

    print({"hyper_parm": hyper_param, "mean": np.mean(folds_f1), "std": np.std(folds_f1)})
    return {"hyper_parm": hyper_param, "mean": np.mean(folds_f1),"std": np.std(folds_f1), "folds": folds_results}

def gridsearch(grid, model, data, cv = 5, random_state = 42, shuffle = True, n_job = 1):
    param_grid = list(product_dict(**grid))

    kf = KFold(n_splits=cv, random_state=random_state, shuffle=shuffle)

    print(joblib.cpu_count())
    grid_result = Parallel(n_jobs = n_job)(delayed(cross_val)(model, data, kf, hyper_param) for hyper_param in tqdm(param_grid))
    return grid_result


