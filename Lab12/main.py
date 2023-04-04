#Spencer Kelly, 2023

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

def fileReader(file, years):
    '''Reads in a file with data and filters out all data but in the years specified, then returns the
    filtered dataset
    Args:
        file (string): file to be read in.
        years (list): list of years from which data is desired'''

    #read in file
    df = pd.read_csv(file, comment='#', index_col=False)
    #filter out undesired years of data
    new_df = df[df['year'].isin(years)]
    return new_df

def detrender(dataset):
    '''Detrends the data by fitting a 1st order polynomial using scipy.signal.detrend function to fit a linear function
    to the data
    Args:
        dataset (pandas dataframe): the dataset to be detrended'''

def main():
    data = fileReader('co2_mm_mlo.csv', range(1981,1991))
    plt.plot(data['decimal date'], data['average'], '+', markersize = 4)
    plt.ylim([335, 360])
    plt.show()

main()