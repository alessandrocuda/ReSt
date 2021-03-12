import pickle

def save_grid_result(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)