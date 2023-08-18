# imported libs for prep functions
import os
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer # may need for categorical abnormal values
from sklearn.model_selection import train_test_split

# custom import 
from acquire import get_telco_data



#------------PREP TELCO DATA FUNCTION-------------------

def prep_telco(df):
    '''
    Defined function to drop columns for the preparation phase and regularized the
    total_charges column in the data set. I dropped files that were unneccessary for
    EDA which follows in the next step in the DS Pipeline.
    '''
    # Replace various types of missing values with NaN
    missing_values = ["", " ", "NA", "N/A", "nan", "NaN", "null", "None"]
    df['total_charges'] = df['total_charges'].replace(missing_values, np.nan)
    
    # Convert column to numeric
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    
    # Calculate mode
    df_mode_value = df['total_charges'].mean()
    
    # Fill NaN values with mode
    df['total_charges'].fillna(df_mode_value, inplace=True)
    
    return df.drop(columns = ['customer_id', 'payment_type_id', 'contract_type_id', 'internet_service_type_id'])


    
#--------------DROP COLUMNS FUNCTION------------------------

def drop_cols(df):
    '''
    Defined function to drop columns in working dataframe.
    '''
    return df.drop(columns = [])

    
#-----------DATA SPLIT FUNCTION------------------------------


# 20% test, 80% train validate
# then of the 80% train validate: 30% validate, 70% train.

def train_val_test(df, strat, seed = 42):
    '''
    This function splits the data into different percentages to be used for exploratory data analysis
    and modeling.
    '''
    
    train, test = train_test_split(df, test_size = 0.2, random_state=seed, stratify=df[strat])
    
    train, val = train_test_split(train, test_size = 0.3, random_state=seed, stratify=train[strat])
    
    return train, val, test
    

    
#------------CLEAN TELCO DATA FUNCTION---------------------------------

# Defined function to implement cleaning of data
def clean_telco_data():
    '''
    A function to be called in wrangle.py to have data retrieved and cleaned
    to display a ready-to-use dataframe.
    '''
    
    df = get_telco_data()

    df = prep_telco(df)

    df = drop_cols(df)
    
    train, val, test = train_val_test(df, 'churn')
  
    return train, val, test
