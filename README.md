# DiscriminantAnalysisClassifier

This repository contains the code for the LDA and QDA classifier on the iris dataset (obtained here: https://archive.ics.uci.edu/ml/datasets/iris).
Three cases of LDA and QDA were investigated:
1. General Case 
2. Independent case (Covariance matrix is a diagonal matrix)
3. Isotropic case (Covariance matrix is a 

# Code
DistributionPLotter.py is used to plot pairwise feature plots.
Classifier contains all 6 cases of the classifiers.

# Results obtained
The best performing models for the iris dataset (along with errors [as predicates]) are as follows:
LDA-ISOTROPIC - 0.04
QDA-ISOTRPIC - 0.04
LDA-GENERAL - 0.053
