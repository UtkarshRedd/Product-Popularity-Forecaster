## Import Statements

import os
import json
import pickle
import sys
import forecasting.app.data.config as cfg


def save_dict_as_json(dic, file_name=None, file_dir=None):
    if file_dir:
        file_name = os.path.join(file_dir, file_name)
    file = open(file_name, "w")
    json.dump(dic, file)
    file.close()

    return


def save_dict_to_file(dic, file_name=None, dir_name=None):
    if dir_name:
        file_name = os.path.join(dir_name, file_name)
    f = open(file_name, 'w')
    f.write(str(dic))
    f.close()


def save_dict_as_pickle(dic, file_name=None, dir_name=None):
    if dir_name:
        file_name = os.path.join(dir_name, file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(dic, f)

    return


def load_dict_from_pickle(file_name=None, dir_name=None):
    if dir_name:
        file_name = os.path.join(dir_name, file_name)
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        return data
    except:
        return None


def save_trained_model(model, file_name=None, dir_path=None):
    '''
    saves a fitted model as pickel file

    '''
    if dir_path:
        file_name = os.path.join(dir_path, file_name)
    with open(file_name, 'wb') as files:
        pickle.dump(model, files)

    return


def create_directory(root_path, directory_name):
    '''
    creates a directory at a specified location with given name
    '''
    directory = directory_name
    parent_dir = root_path
    path = os.path.join(parent_dir, directory)

    '''
    if the directory exists already it will return the path else will create a new one and return the path
    '''
    try:
        os.mkdir(path)
    except:
        return path
    return path


def load_trained_model(model_dir_path=None, l3_file_name=None, l2_file_name=None, l1_file_name=None,
                       l3_forecast=False, l2_forecast=False, l1_forecast=False):
    '''
    loads a trained model and uses for prediction
    '''
    if l3_file_name in os.listdir(model_dir_path[cfg.l3_col]):
        file_name, l3_forecast = os.path.join(model_dir_path[cfg.l3_col], l3_file_name), True

    elif l2_file_name in os.listdir(model_dir_path[cfg.l2_col]):
        file_name, l2_forecast = os.path.join(model_dir_path[cfg.l2_col], l2_file_name), True

    elif l1_file_name in os.listdir(model_dir_path[cfg.l1_col]):
        file_name, l1_forecast = os.path.join(model_dir_path[cfg.l1_col], l1_file_name), True

    else:
        pass

    with open(file_name, 'rb') as f:
        loaded_model = pickle.load(f)

    return loaded_model, l3_forecast, l2_forecast, l1_forecast

