import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def main():
    myData = pd.read_csv('PeakSepDataSet.txt', sep='\s+', skiprows=1, names= ['Reference', 'Peak Seperation', 'Flux Shell', 'System Viewing', 'Useless'])
    #delete the last column of the dataframe
    myData.drop('Useless', axis=1, inplace=True)
    print(myData)
    ScattaPlotta(myData)
    _2dPlotta(myData)

def ScattaPlotta(DataFrame):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(xs = DataFrame['Peak Seperation'], ys = DataFrame['Flux Shell'], zs = DataFrame['System Viewing'])
    ax.view_init(elev=27, azim=32)
    ax.set_xlim([0, 500])
    ax.set_ylim([0.9, 2])
    ax.set_zlim(0, 90)
    plt.show()

def _2dPlotta(DataFrame):
    sns.scatterplot(data= DataFrame, x = 'Peak Seperation', y='Flux Shell', hue='System Viewing', palette='gray_r')
    plt.show()
main()