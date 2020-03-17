#!/usr/bin/env python3
""" Functions save & load config
 saves a model’s configuration in
JSON format:
network is the model whose configuration should be saved
filename is the path of the file that the configuration should be saved to
Returns: None
loads a model with a specific configuration:
filename is the path of the file containing the model’s configuration in
JSON format
Returns: the loaded model
"""
import tensorflow.keras as K


def save_config(network, filename):
    """FUnction save_config"""
    network_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


def load_config(filename):
    """FUnction load_config"""
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.model_from_json(loaded_model_json)
    return(loaded_model)
