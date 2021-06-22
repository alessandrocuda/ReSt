"""Utils Moduel.
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np

def plotHistory(history, orientation = "horizontal"):
    """Plots the learning curve for the training and the validation
    for for a specific loss and accuracy.
    Parameters
    ----------
    history : dict
        It contains for each epoch the values of loss and macro f1 score for
        training and validation and the time.
    
    orientation : None, "horizontal", "vertical"
        Indicates the orientation of the two plots.
    """
    pos_train = (0,0)
    if orientation == "horizontal":
        figsize = (12, 4)
        figdims = (1, 2)
        pos_val = (0, 1)
    elif orientation == "vertical":
        figsize = (7, 7)
        figdims = (2, 1)
        pos_val = (1, 0)
    else:
        raise Exception('Wrong value for orientation par.')

    fig = plt.figure(figsize=figsize) 
    fig_dims = figdims
    plt.subplot2grid(fig_dims, pos_train)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"], linestyle='--')
    plt.title('Binary cross-entropy')
    plt.ylabel("Binary cross-entropy")
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Training', 'Val'], loc='upper right', fontsize='large')

    plt.subplot2grid(fig_dims, pos_val)
    plt.plot(history['f1_macro'])
    plt.plot(history['val_f1_macro'], linestyle='--')
    plt.title('Macro F1 score')
    plt.ylabel('Macro F1 score')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Training', 'Val'], loc='lower right', fontsize='large')
    plt.tight_layout()
    plt.show()

def plotLoss(history):
    """Plots the learning curve for the training and the validation
    for for MSE.
    Parameters
    ----------
    history : dict
        It contains for each epoch the values of mse, mee and accuracy for 
        training and validation and the time.
    """
    plt.plot(history["loss"])
    plt.plot(history["val_loss"], linestyle='--')
    plt.title('Binary cross-entropy')
    plt.ylabel("Binary cross-entropy")
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Training', 'Val'], loc='upper right', fontsize='large')
    plt.tight_layout()
    plt.show()

def plotF1_macro(history):
    """Plots the learning curve for the training and the validation
    for for MSE.
    Parameters
    ----------
    history : dict
        It contains for each epoch the values of mse, mee and accuracy for 
        training and validation and the time.
    """
    plt.plot(history['f1_macro'])
    plt.plot(history['val_f1_macro'], linestyle='--')
    plt.title('Macro F1 score')
    plt.ylabel('Macro F1 score')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Training', 'Val'], loc='lower right', fontsize='large')
    plt.tight_layout()
    plt.show()

def save_grid_result(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def save_data_p(data, filename):
    """Serialize and save the past object on file.
    Parameters
    ----------
    data : object
        Data to serialize.
    filename : string
        Specifies the name of the file to save to.
    """   
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data_p(filename):
    """Deserialize and load an object from a specific path.
    Parameters
    ----------
    filename : string
        Specifies the name of the file to load to.
    Returns
    -------
    objct
        the deserialized object returns
    """   
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data