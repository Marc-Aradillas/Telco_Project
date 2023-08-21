# create a .py file with my acquire and prepare module and define a function to wrangle data calling functions from modules.
from prepare import clean_telco_data

def wrangle_telco():
    """
    This function wrangles the data by acquiring and preparing it.

    Returns:
        Pandas DataFrame of train, val, and test subsets
    """

    df = clean_telco_data()
    
    return df