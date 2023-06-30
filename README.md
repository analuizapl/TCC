# Performance Analysis of DiCE - an Algorithm for Interpretability Based on Counterfactual Explanations

This study focuses on a detailed analysis of the DiCE algorithm applied to various databases, each differing in the number of attributes and instances. The primary objective is to uncover the strengths and weaknesses of the DiCE tool, assessing its effectiveness in enhancing decision transparency.

# DICE

You can find the algorithm's documentation in
https://pypi.org/project/dice-ml/

# Databases
My analysis used the following databases
Name| Link 
--- | --- 
Banana | [Link](https://www.kaggle.com/datasets/saranchandar/standard-classification-banana-dataset)

# Notes
This code assumes that the input dataset contains the target variable (outcome) column, and the features used for model training are all other columns.
The counterfactual examples are generated for the test data (x_test) using a RandomForestClassifier model.
Ensure that the required datasets are present in the same directory as the code or provide the appropriate file paths in the code.
Use the PySimpleGUI window to select the dataset file and interact with the DiCE code.
