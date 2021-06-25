import pandas as pd
import numpy as np
from itertools import product
import joblib as joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import f1_score

from collections import defaultdict

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
            
    results_folds = defaultdict(list)

    for train_index_fold, val_index_fold in kf.split(data["text"]):

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

        results_fold = get_score(X_train, Y_train, X_val_es, Y_val_es, X_val, Y_val,  hyper_param)
        for key in results_fold:
            results_folds[key].append(results_fold[key])


    result_cross_val =  {"hyper_parm": hyper_param, 

                         "loss_mean": np.mean(results_folds["loss"]),
                         "loss_std": np.std(results_folds["loss"]), 
                         "f1_macro_mean": np.mean(results_folds["f1_macro"]),
                         "f1_macro_std": np.std(results_folds["f1_macro"]),

                         "val_loss_mean": np.mean(results_folds["val_loss"]),
                         "val_loss_std": np.std(results_folds["val_loss"]), 
                         "val_f1_macro_mean": np.mean(results_folds["val_f1_macro"]),
                         "val_f1_macro_std": np.std(results_folds["val_f1_macro"]),

                         "folds": results_folds}

    #print(result_cross_val)
    print("Val Macro F1: {}+/-{}".format(result_cross_val["val_f1_macro_mean"], result_cross_val["val_f1_macro_std"]))
    return result_cross_val

def gridsearch(grid, model, data, cv = 5, random_state = 42, shuffle = True, n_job = 1):
    param_grid = list(product_dict(**grid))

    kf = KFold(n_splits=cv, random_state=random_state, shuffle=shuffle)

    print(joblib.cpu_count())
    grid_result = Parallel(n_jobs = n_job)(delayed(cross_val)(model, data, kf, hyper_param) for hyper_param in tqdm(param_grid))
    return grid_result


