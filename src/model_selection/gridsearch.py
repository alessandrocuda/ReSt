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


def cross_val(nn_model, X, y, kf, hyper_param):
    import tensorflow as tf
    from src.utils.callbacks import ReturnBestEarlyStopping
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)
    results_folds = { "loss": [],
                          "val_loss": [],
                          "f1_macro": [],
                          "val_f1_macro": [] 
    }

    folds_f1 = []
    folds_results = {}
    i = 0
    for train_index_fold, val_index_fold in kf.split(X):
        name_fold = "fold_{}".format(i)
        i+=1
        X_train = X[train_index_fold]
        Y_train = y[train_index_fold]

        X_val = X[val_index_fold]
        Y_val = y[val_index_fold]

        X_train, X_val_es, Y_train, Y_val_es = train_test_split(X_train, Y_train, test_size=0.09, random_state=128)

        #print(hyper_param)
        input_shape = (X_train[0].shape[0], X_train[0].shape[1],)
        model = nn_model(input_shape, **hyper_param)
        best_callback = ReturnBestEarlyStopping(monitor="val_loss", patience=50, verbose=0, mode="min", restore_best_weights=True)
        history = model.fit(X_train, Y_train, batch_size=64, epochs=200, validation_data=(X_val_es, Y_val_es), callbacks=[best_callback], verbose = 0)

        #evaluation
        y_pred = np.where(model.predict(X_train) >0.5, 1,0)
        f1      = f1_score(Y_train, y_pred, average="macro")
        y_pred = np.where(model.predict(X_val) >0.5,1,0)
        val_f1  = f1_score(Y_val, y_pred, average="macro")

        results_folds["loss"].append(model.evaluate(X_train, Y_train, verbose = 0))
        results_folds["val_loss"].append(model.evaluate(X_val, Y_val, verbose = 0))
        results_folds["f1_macro"].append(f1)
        results_folds["val_f1_macro"].append(val_f1)

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

    print(result_cross_val)
    return result_cross_val

def gridsearch(grid, model, X, y, cv = 5, random_state = 42, shuffle = True, n_job = 1):
    param_grid = list(product_dict(**grid))

    kf = KFold(n_splits=cv, random_state=random_state, shuffle=shuffle)

    print(joblib.cpu_count())
    grid_result = Parallel(n_jobs = n_job)(delayed(cross_val)(model, X, y, kf, hyper_param) for hyper_param in tqdm(param_grid))
    return grid_result


