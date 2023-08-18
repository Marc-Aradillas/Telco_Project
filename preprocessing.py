import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

from sklearn.impute import SimpleImputer
from wrangle import


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
