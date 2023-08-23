
import os
import sys
import yaml
import torch
import logging
import numpy as np
import pandas as pd
from typing import *
import torch.nn as nn
from datetime import datetime
import pytorch_lightning as pl
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


from src.logger import logging

def read_yaml_file(file_path:str) -> dict:
    
    try:
        with open(file=file_path,mode='rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logging.info(f"-- Error in read_yaml_file: {e}")
        return dict()
    



        





