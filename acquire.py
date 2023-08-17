# imports for acquisition phase
import os
import numpy as np
import pandas as pd

from env import get_connect

def get_db_connect(database):
    '''
    wrapper function to make it easier to connect to a database.
    '''
    return get_connect(database)
    
def new_telco_data():
    
    '''
    Defined function to input query into MySQL to retrieve data from
    the telco_churn table form the Codeup database and reads in 
    dataframe using pandas function.
    '''
    query = '''
        SELECT *
        FROM customers
        JOIN internet_service_types USING (internet_service_type_id)
        JOIN contract_types USING (contract_type_id)
        JOIN payment_types USING (payment_type_id)
        ;
        '''
    
    df = pd.read_sql(query, get_connect('telco_churn'))
    print('\nTelco dataframe generated.')
    
    return df

def get_telco_data():

    '''
    Defined function used to retrieve/search for existing file in 
    computer operating system. os module enables search and retrieve
    actions for this function
    '''
    if os.path.isfile('telco.csv'):
        print('Found file')
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        print('Retrieving file...\n')
        # Read fresh data from db into a dataframe
        df = new_telco_data()
        # Cache data
        df.to_csv('telco.csv')
        print('\nFile retrieved.')
        
    return df