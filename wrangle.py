# create a .py file with my acquire and prepare module and define a function to wrangle data calling functions from modules.
import numpy as np
import pandas as pd

from acquire import get_telco_data
from prepare import clean_telco_data

def wrangle_data():
    """
    This function wrangles the data by acquiring and preparing it.

    Returns:
        Pandas DataFrame of train, val, and test subsets
    """

    df = clean_telco_data()
    
    return train, val, test