from tensorflow.keras.callbacks import EarlyStopping, Callback
import numpy as np
from sklearn.metrics import f1_score


class ReturnBestEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        super(ReturnBestEarlyStopping, self).__init__(**kwargs)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            if self.verbose > 0:
                print(f'\nEpoch {self.stopped_epoch + 1}: early stopping')
        elif self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)


class FCallback(Callback):
  
    def __init__(self, validation = (), verbose = 0):
        self.validation = validation
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.f1 = []
        self.val_f1 = []
    def on_epoch_end(self, epoch, logs=None):
        y_t =  self.validation[1]
        y_p =  np.where(self.model.predict(self.validation[0]) > 0.5, 1, 0)
        logs['val_f1'] =  f1_score(y_t, y_p, average='macro')
        if self.verbose >0:
          print("— val_f1: {}".format(logs['val_f1']))


class GapCallback(Callback):
  
    def __init__(self, train = (), validation = (), verbose = 0):
        self.train = train
        self.validation = validation
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.gap = []
        self.val_gap = []
    def on_epoch_end(self, epoch, logs=None):
        y_t =  self.validation[1]
        y_p =  np.where(self.model.predict(self.validation[0]) > 0.5, 1, 0)
        val_f1 = f1_score(y_t, y_p, average='macro')
        y_t =  self.train[1]
        y_p =  np.where(self.model.predict(self.train[0]) > 0.5, 1, 0)
        f1 = f1_score(y_t, y_p, average='macro')
        logs['val_gap_f1'] =  np.abs( f1 - val_f1)
        if self.verbose >0:
            print("— val_gap_f1: {}".format(logs['val_gap_f1']))