

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.mlab

def plotDecisionRegionQDA(sampleMean, CV_, training, data):
    all_res = []
    minX, maxX, minY, maxY = 0., 8., 0., 8.

    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, 100, endpoint=True)
    y = np.linspace(minY, maxY, 100, endpoint=True)
    cnt = 0
    for i in range(len(x) - 1):
        for k in range(len(y) - 1):
            classIndex = 0
            logLikelihood = float("inf")
            for j in range(len(data.species.unique())):
                Nk = training.groupby('species').size()[data.species.unique()[j]]
                newLikelihood = logLikelihoodCalc([x[i],y[k]], sampleMean[j], CV_[j], Nk)
                if (newLikelihood < logLikelihood):
                    logLikelihood = newLikelihood
                    classIndex = j
            test = [x[i],y[k], data.species.unique()[classIndex]]
            test = pd.DataFrame([test], columns=['pet_length', 'pet_width', 'species'])
            cnt  =cnt + 1
            all_res.append(test)
    plotPairwiseScatterPlot(pd.concat(all_res))

def plotDecisionRegionLDA(sampleMean, CV_, training, data):
    all_res = []
    minX, maxX, minY, maxY = 0., 8., 0., 8.

    # create one-dimensional arrays for x and y
    x = np.linspace(minX, maxX, 100, endpoint=True)
    y = np.linspace(minY, maxY, 100, endpoint=True)
    cnt = 0
    for i in range(len(x) - 1):
        for k in range(len(y) - 1):
            classIndex = 0
            logLikelihood = float("inf")
            for j in range(len(data.species.unique())):
                Nk = training.groupby('species').size()[data.species.unique()[j]]
                newLikelihood = logLikelihoodCalc([x[i],y[k]], sampleMean[j], CV_, Nk)
                if (newLikelihood < logLikelihood):
                    logLikelihood = newLikelihood
                    classIndex = j
            test = [x[i],y[k], data.species.unique()[classIndex]]
            test = pd.DataFrame([test], columns=['pet_length', 'pet_width', 'species'])
            cnt  =cnt + 1
            all_res.append(test)
    plotPairwiseScatterPlot(pd.concat(all_res))


def plotPairwiseScatterPlot(dataframe):
    sns.scatterplot(x="pet_length", y="pet_width", hue="species",
                      data=dataframe)



def plotDecisionRegion(C, mu ):
    x = np.arange(-5, 10.0, 0.005)
    y = np.arange(-5, 10.0, 0.005)
    X, Y = np.meshgrid(x, y)
    Z = matplotlib.mlab.bivariate_normal(X, Y, C[0, 0], C[1, 1], mu[0], mu[1], (C[0,1]*C[1,0]))
    plt.contour(X, Y, Z, levels = 10)


def Mahalonobis(x,C):
    return x.dot(np.linalg.inv(C)).dot(np.transpose(x))

#-log likelihood
def logLikelihoodCalc(data,mu, C, Nk):
    return (1/2 * np.log(np.linalg.det(C)) + np.log(Mahalonobis(data-mu,C)) - np.log(Nk))

def generateSampleMean(data):
    mean = [ data["pet_length"].mean(),data["pet_width"].mean()]
    return mean
def generateClassSampleMeans(data):
    classMeans = []
    classes = data.species.unique()
    for species in classes:
        classFrame = data.loc[data['species'] == species]
        classMeans.append(generateSampleMean(classFrame))
    return np.asarray(classMeans)

def generateClassCovarianceMatrix(data, mu):
    classCVs = []
    classes = data.species.unique()
    j=0
    mu = np.asarray(mu)
    for species in classes:
        classFrame = data.loc[data['species'] == species]
        classCVs.append(generatePooledCovariance(classFrame,mu[j]))
        j = j+1

    return np.asarray(classCVs)


def generatePooledCovariance(data, mean):
    data = pd.DataFrame(data)
    C  = np.ndarray(shape=(2,2), dtype=float, order='F')
    mean = np.asarray(mean)
    data.reset_index(inplace=True, drop = True)
    for i in range(len(data.index)):
        x_ = [data["pet_length"][i],data["pet_width"][i]]
        for j in  range(len(data.species.unique())):
            if data.species.unique()[j] == data["species"][i]:
                C = C + 1/len(data.index)*np.outer((x_-mean[j]),np.transpose(x_-mean[j]))
    return C

def LOOCVQDA(data, mode):
    LOOCV_Error = 0.0;
    data = data.reset_index()
    for i in range(len(data.index)):
        training = data.copy(deep=True)
        training.drop(training.index[i], inplace=True)
        test = data.iloc[i]
        classIndex = 0
        logLikelihood = float("inf")
        sampleMean = generateClassSampleMeans(training)
        CV_ = generateClassCovarianceMatrix(training, sampleMean)
        for j in range(CV_.shape[0]):
            if mode > 0:
                CV_[j]= np.diag(np.diag(CV_[j]))
                if mode > 1:
                    CV_[j] = (np.trace(CV_[j]) / CV_[j].shape[1]) * \
                          np.identity(CV_[j].shape[1])
        for j in range(len(data.species.unique())):
            Nk = training.groupby('species').size()[data.species.unique()[j]]
            newLikelihood = logLikelihoodCalc([test['pet_length'], test['pet_width']], sampleMean[j], CV_[j], Nk)
            if (newLikelihood < logLikelihood):
                logLikelihood = newLikelihood
                classIndex = j
        if (i == len(data.index) - 1):
            #plotPairwiseScatterPlot(training)
            plotDecisionRegionQDA(sampleMean, CV_, training, data)
        if data.species.unique()[classIndex] != test["species"]:
            LOOCV_Error = LOOCV_Error + 1 / len(data.index);

    return LOOCV_Error


def LOOCVLDA(data, mode):
    LOOCV_Error = 0.0;
    data = data.reset_index()
    for i in range(len(data.index)):
        training = data.copy(deep=True)
        training.drop(training.index[i], inplace=True)
        test = data.iloc[i]
        classIndex = 0
        logLikelihood = float("inf")
        sampleMean = generateClassSampleMeans(training)
        CV_ = generatePooledCovariance(training,sampleMean)

        if mode > 0:
            CV_ = np.diag(np.diag(CV_))
            if mode > 1:
                CV_ =( np.trace(CV_) / CV_.shape[0]) * \
                     np.identity(CV_.shape[0])
        for j in range(len(data.species.unique())):
            #if(i == len(data.index) - 1):
                #plotDecisionRegion(CV_, sampleMean[j])
            Nk =training.groupby('species').size()[data.species.unique()[j]]
            newLikelihood = logLikelihoodCalc([test['pet_length'],test['pet_width']],sampleMean[j], CV_, Nk)
            if( newLikelihood < logLikelihood):
                logLikelihood = newLikelihood
                classIndex = j
        if (i == len(data.index) - 1):
            plotPairwiseScatterPlot(training)
            plotDecisionRegionLDA(sampleMean, CV_, training, data)
        if data.species.unique()[classIndex] != test["species"]:
            LOOCV_Error = LOOCV_Error + 1/len(data.index);

    return LOOCV_Error




with open("iris.data.txt", 'rt') as csvfile:
     iris = csv.reader(csvfile, delimiter=',', quotechar='|')
     IrisReduced = pd.DataFrame(list(iris))
IrisReduced.columns = ['sep_length','sep_width','pet_length','pet_width','species']
IrisReduced.drop(['sep_length'], axis = 1, inplace = True)
IrisReduced.drop(['sep_width'], axis = 1, inplace = True)
IrisReduced[['pet_length','pet_width']] = IrisReduced[['pet_length','pet_width']].astype(float)

#LDA mode: 0-general, 1-independent, 2- isotropic
print("Error for LDA: general case")
print(LOOCVLDA(IrisReduced, 1))
#print("Error for LDA: independent case")
#print(LOOCVLDA(IrisReduced, 1))
#print("Error for LDA: isotropic case")
#print(LOOCVLDA(IrisReduced, 2))


#QDA mode: 0-general, 1-independent, 2- isotropic
#print("Error for QDA: general case")
#print(LOOCVQDA(IrisReduced, 0))
#print("Error for QDA: in independent case")
#print(LOOCVQDA(IrisReduced, 1))
#print("Error for QDA: isotropic case")
#print(LOOCVQDA(IrisReduced, 2))
plt.show()


