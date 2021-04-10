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


def cross_val(nn_model, data, kf, hyper_param):
    #import tensorflow as tf
    from src.utils.callbacks import ReturnBestEarlyStopping
    
    #physical_devices = tf.config.list_physical_devices('GPU')
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

        #print(hyper_param)
        input_shape = (X_train["text"][0].shape[0], X_train["text"][0].shape[1],)
        model = nn_model(input_shape, **hyper_param)
        best_callback = ReturnBestEarlyStopping(monitor="val_loss", patience=50, verbose=0, mode="min", restore_best_weights=True)
        history = model.fit(X_train, Y_train, batch_size=64, epochs=200, validation_data=(X_val_es, Y_val_es), callbacks=[best_callback], verbose = 0)
        
        y_pred = np.where(model.predict(X_train) >0.5,1,0)
        f1      = f1_score(Y_train, y_pred, average="macro")
        y_pred = np.where(model.predict(X_val) >0.5,1,0)
        val_f1  = f1_score(Y_val, y_pred, average="macro")

        folds_f1.append(val_f1)
        folds_results[name_fold] = {"f1": f1, "val_f1": val_f1}


    print({"hyper_parm": hyper_param, "mean": np.mean(folds_f1), "std": np.std(folds_f1)})
    return {"hyper_parm": hyper_param, "mean": np.mean(folds_f1),"std": np.std(folds_f1), "folds": folds_results}

def gridsearch(grid, model, data, cv = 5, random_state = 42, shuffle = True, n_job = 1):
    param_grid = list(product_dict(**grid))

    kf = KFold(n_splits=cv, random_state=random_state, shuffle=shuffle)

    print(joblib.cpu_count())
    grid_result = Parallel(n_jobs = n_job)(delayed(cross_val)(model, data, kf, hyper_param) for hyper_param in tqdm(param_grid))
    return grid_result


