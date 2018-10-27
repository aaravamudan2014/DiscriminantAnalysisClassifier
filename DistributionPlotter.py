
#read elements
#plot pairwise with legend
#leave one out cross validation function
#function to get mu and correlation matrix in a case basis


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import csv

def plotPairwiseScatterPlot(xName, yName, dataframe):
    sns.scatterplot(x=xName, y=yName, hue="species",
                      data=dataframe)
    plt.show()

dataframe = []
with open("iris.data.txt", 'rt') as csvfile:
     iris = csv.reader(csvfile, delimiter=',', quotechar='|')
     dataframe = pd.DataFrame(list(iris))
dataframe.columns = ['sep_length','sep_width','pet_length','pet_width','species']
plotPairwiseScatterPlot("pet_length","pet_width",dataframe)


IrisReduced = dataframe.copy()
IrisReduced.columns = ['sep_length', 'sep_width', 'pet_length', 'pet_width', 'species']
IrisReduced.drop(['sep_length'], axis=1, inplace=True)
IrisReduced.drop(['sep_width'], axis=1, inplace=True)
IrisReduced[['pet_length', 'pet_width']] = IrisReduced[['pet_length', 'pet_width']].astype(float)

print (IrisReduced)

