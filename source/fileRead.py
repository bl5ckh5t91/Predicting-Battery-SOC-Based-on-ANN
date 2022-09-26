# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:29:08 2021
@author: Debanjan

Import CSV file
"""
import pandas as pd
from glob import glob

def import_data():
    """
    Iterate through the csv files in the dataset and append into a single dataframe
    args: None
    returns: combined pandas dataframe df
    """
    df = pd.concat(map(pd.read_csv, glob('data/raw/*.csv')))
##    df['Time'] = pd.to_datetime(df['Time'], unit='ns')
##    df = df.set_index('Time')
##    df = df.sort_index()

    print('Battery data - Loading File....', df.head(), sep='\n')
    return df
