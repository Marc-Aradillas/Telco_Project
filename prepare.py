# imported libs for prep functions
import os
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
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
    return df.drop(columns = ['customer_id', 'payment_type_id', 'contract_type_id', 'internet_service_type_id'])
    
    
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
    

#--------------IMPUTATION FUNCTION-----------------------

def impute_vals(train, val, test):
    '''
    Defined function to impute values for our train, val, and test subsets of my splitted data.
    '''


    # Impute 'internet_service_type' with mode
    internet_service_type_mode = train.internet_service_type.mode()[0]  # Get the mode value
    
    train.internet_service_type = train.internet_service_type.replace('None', internet_service_type_mode)
    val.internet_service_type = val.internet_service_type.replace('None', internet_service_type_mode)
    test.internet_service_type = test.internet_service_type.replace('None', internet_service_type_mode)
    
    # Impute 'total_charges' with mean
    missing_values = ["", " ", "NA", "N/A", "nan", "NaN", "null", "None"]
    train['total_charges'] = train['total_charges'].replace(missing_values, np.nan)
    val['total_charges'] = val['total_charges'].replace(missing_values, np.nan)
    test['total_charges'] = test['total_charges'].replace(missing_values, np.nan)
    
    train['total_charges'] = pd.to_numeric(train['total_charges'], errors='coerce')
    val['total_charges'] = pd.to_numeric(val['total_charges'], errors='coerce')
    test['total_charges'] = pd.to_numeric(test['total_charges'], errors='coerce')
    
    mean_total_charges = train['total_charges'].mean()
    
    train['total_charges'].fillna(mean_total_charges, inplace=True)
    val['total_charges'].fillna(mean_total_charges, inplace=True)
    test['total_charges'].fillna(mean_total_charges, inplace=True)

    dataframes = [train, val, test]  # List of DataFrames

    values_to_replace = ['No internet service', 'No phone service']
    
    # Iterate through each DataFrame
    for df in dataframes:
        # Iterate through each column in the DataFrame
        for column in df.columns:
            df[column] = df[column].replace(values_to_replace, 'No')

    return train, val, test

    
#-----------GET DUMMIES FUNCTION----------------------------

def dummies(df, columns_to_exclude=['churn', 'senior_citizen', 'tenure', 'monthly_charges', 'total_charges'], drop_first_columns=['gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing']):
    
    # Get dummies for all categorical columns except those in columns_to_exclude
    categorical_columns = [col for col in df.columns if col not in columns_to_exclude]
    
    # Apply get_dummies with drop_first for specific columns
    for col in categorical_columns:
        
        drop_first = col in drop_first_columns
        
        df = pd.get_dummies(df, columns=[col], drop_first=drop_first)

    
    return df



#-----------RENAME FUNCTION----------------------------

def rename(df):
    new_column_names = {
        'gender_Male': 'gender',
        'partner_Yes': 'partner',
        'dependents_Yes': 'dependents',
        'phone_service_Yes': 'phone',
        'multiple_lines_Yes': 'multiple_lines',
        'online_security_Yes': 'online_security',
        'online_backup_Yes': 'online_backup',
        'device_protection_Yes': 'device_protection',
        'tech_support_Yes': 'tech_support',
        'streaming_tv_Yes': 'streaming_tv',
        'streaming_movies_Yes': 'streaming_movies',
        'paperless_billing_Yes': 'paperless_billing',
        'internet_service_type_DSL': 'dsl',
        'internet_service_type_Fiber optic': 'fiber_optic',
        'contract_type_Month-to-month': 'month_to_month',
        'contract_type_One year': 'one_year',
        'contract_type_Two year': 'two_year',
        'payment_type_Bank transfer (automatic)': 'bank_transfer_payment',
        'payment_type_Credit card (automatic)': 'credit_card_payment',
        'payment_type_Electronic check' : 'electronic_payment',
        'payment_type_Mailed check' : 'mailed_payment'
        }
    
    df.rename(columns=new_column_names, inplace=True)
    return df

    
    
#------------CLEAN TELCO DATA FUNCTION---------------------------------

# Defined function to implement cleaning of data
def clean_telco_data():
    '''
    A function to be called in wrangle.py to have data retrieved and cleaned
    to display a ready-to-use dataframe.
    '''
    
    df = get_telco_data()

    df = prep_telco(df)
    
    train, val, test = train_val_test(df, 'churn')

    train, val, test = impute_vals(train, val, test)

    train = dummies(train)
    
    val = dummies(val)
    
    test = dummies(test)

    train = rename(train)
    
    val = rename(val)
    
    test = rename(test)
  
    return train, val, test
