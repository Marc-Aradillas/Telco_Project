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
    

# #--------------IMPUTATION FUNCTION-----------------------

# def impute_vals(train, val, test):
#     '''
#     Defined function to impute values for our train, val, and test subsets of my splitted data.
#     '''
#     # example:
#     '''
#     town_mode = train.embark_town.mode()
    
#     train.embark_town = train.embark_town.fillna(town_mode)
#     val.embark_town = val.embark_town.fillna(town_mode)
#     test.embark_town = test.embark_town.fillna(town_mode)
    
#     med_age = train.age.median()
    
#     train.age = train.age.fillna(med_age)
#     val.age = val.age.fillna(med_age)
#     test.age = test.age.fillna(med_age)
#     '''
#     return train, val, test

# #-----------GET DUMMIES FUNCTION----------------------------

# def dummies(df):
#     '''
#     # defined function to one-hot-encode categorical values in a column from the dataframe
#     '''
#     df = pd.get_dummies(df, columns = [''], drop_first = True)
    
#     df = pd.get_dummies(df)
    
#     return df

#-----------DATA SPLIT FUNCTION------------------------------


# 20% test, 80% train_validate
# then of the 80% train_validate: 30% validate, 70% train.

def train_val_test(df, strat, seed = 42):
    
    train, test = train_test_split(df, test_size = 0.2, random_state=seed, stratify=df[strat])
    
    train, val = train_test_split(train, test_size = 0.3, random_state=seed, stratify=train[strat])
    
    return train, val, test
    

#------------CLEAN TELCO DATA FUNCTION---------------------------------

# Defined function to implement cleaning of data
def clean_telco_data():
    
    df = get_telco_data()

    df = prep_telco(df)

    df = drop_cols(df)
    
    train, val, test = train_val_test(df, 'churn')
    
    # train, val, test = impute_vals(train, val, test)
    
    # train = dummies(train)
    
    # val = dummies(val)
    
    # test = dummies(test)
    
    return train, val, test
