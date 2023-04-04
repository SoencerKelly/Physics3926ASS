#Spencer Kelly, 2023

import pandas as pd
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import scipy.signal


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


def detrender(dataframe):
    '''Detrends the data by fitting a 1st order polynomial using numpy.polyfit function to fit a linear function
    to the data. And returns the residual values between the initial data and the linear trend
    Args:
        dataset (pandas dataframe): the dataset to be detrended'''

    #find the coefficients of a polynomial of 1st degree
    polycoeffs = np.polyfit(dataframe['decimal date'], dataframe['average'], 1)
    #create the appropriate linear trendline
    trendline = np.polyval(polycoeffs, dataframe['decimal date'])
    #generate residual data
    residualData = dataframe['average'] - trendline

    residualDataFrame = pd.DataFrame({'decimal date': dataframe['decimal date'], 'residual': residualData})


    return residualDataFrame

def sinefunc(t, A = 5, T = 1, phi = 0):

    return A * np.sin(2*np.pi*(t/T) + phi)

def main():
    #read in the file
    data = fileReader('co2_mm_mlo.csv', range(1981,1991))
    # plot the date vs average data to reproduce figure 5.1 in text
    plt.plot(data['decimal date'], data['average'], '+', markersize = 4)
    plt.ylim([335, 360])
    plt.xlabel('Year')
    plt.ylabel('CO2 (ppm)')
    plt.show()
    #detrend the data
    residualDataFrame = detrender(data)

    #plot the original data and the residual data
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(data['decimal date'], data['average'], '+', markersize=4)
    ax1.set_title('Initial Data')
    ax2.plot(residualDataFrame['decimal date'], residualDataFrame['residual'], '-', markersize = 4)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO2 (ppm)')
    ax2.set_title('Residual Values')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('CO2 difference value (ppm)')
    ax1.set_ylim([335, 360])
    plt.show()

    #For part 2, we will use sin function and continue to plug in values for the parameters until we fit as best we can the residuals with the sin graph
    plt.plot(residualDataFrame['decimal date'], residualDataFrame['residual'], '-', markersize = 4)
    plt.plot(residualDataFrame['decimal date'], sinefunc(residualDataFrame['decimal date'], phi = -0.5, T = 1, A = 3.2))
    plt.xlim([1980, 1991])
    plt.show()

    #through trial and error we came up with the above sin parameters as our best guess.
    #Now we will test how our guess of T = 1 yr holds up with numpy.fft's guess for the period.
    #Since the measurements are taken every month, and the x axis is in years, we will use distance between points of 1/12
    freq = np.fft.fftfreq(residualDataFrame['decimal date'].size, d=1/12)
    print(freq)
    period = 1/freq



main()