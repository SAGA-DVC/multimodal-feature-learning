import os
import ml_collections

from config.config_dvc_train import load_config as load_config_train
from config.config_dvc_test import load_config as load_config_test

def load_config():

    cfg = ml_collections.ConfigDict()

    cfg.is_train = True

    if cfg.is_train:
        return load_config_train()
    else:
        return load_config_test()