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
Haberman | [Link](https://www.kaggle.com/datasets/gilsousa/habermans-survival-data-set)         
Banknote | [Link](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
Phoneme    | [Link](https://datahub.io/machine-learning/phoneme)      
Mammographic masses | [Link](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)
Room   Occupancy | [Link](https://www.kaggle.com/datasets/sachinsharma1123/room-occupancy)          
Liver disorders(BUPA) | [Link](https://networkrepository.com/liver-disorders-bupa.php)
Vertebral Column | [Link](http://archive.ics.uci.edu/ml/datasets/vertebral+column)
Monk | [Link](https://archive.ics.uci.edu/ml/datasets/MONK's+Problems)           
Diabetes | [Link](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)  
Employee   Future | [Link](https://www.kaggle.com/datasets/tejashvi14/employee-future-prediction)        
Breast   Cancer Coimbra | [Link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra)
Compass | [Link](https://github.com/adebayoj/fairml/blob/master/doc/example\_notebooks/propublica\_data\_for\_fairml.csv)
Indian   Liver Patient | [Link](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))     
Magic | [Link](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope)    
Phishing   Website | [Link](https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector?select=phishing.csv)
Heart   Failure | [Link](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)        
Heart Attack | [Link](https://www.kaggle.com/datasets/nareshbhat/health-care-data-set-on-heart-attack-possibility)
Eye   State Classification EEG  | [Link](https://www.kaggle.com/datasets/robikscube/eye-state-classification-eeg-dataset)
Australian | [Link](https://www.kaggle.com/datasets/aayushkandpal/credit-card-details-australian-bank)
Lung   Cancer | [Link](https://www.kaggle.com/datasets/imkrkannan/lung-cancer-dataset-by-staceyinrobert)
German   credit  | [Link](https://github.com/riccotti/LORE/tree/master/datasets)
Price | [Link](https://www.kaggle.com/code/vikramb/mobile-price-prediction/notebook)
Twonorm  | [Link]({https://www.openml.org/search?type=data&sort=runs&id=1507&status=active)
KC2 | [Link](https://datahub.io/machine-learning/kc2)
Seismic | [Link](https://datahub.io/machine-learning/seismic-bumps)
Home   Equity | [Link](https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc)
Default of credit card | [Link](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
Smoking  | Link
Titanic | [Link](https://www.kaggle.com/datasets/heptapod/titanic)
Breast   Cancer Wisconsin | [Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)      
Ionosphere | [Link](https://www.kaggle.com/datasets/jamieleech/ionosphere)
Churn | [Link](https://www.kaggle.com/code/avikpaul4u/telco-churn)
Horse colic | [Link](https://www.kaggle.com/datasets/uciml/horse-colic/code)
Biodeg | [Link]({https://www.kaggle.com/datasets/muhammetvarl/qsarbiodegradation)
Student | [Link](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set)
Tokyo | [Link](https://www.openml.org/data/download/4965306/tokyo1.arff)
Spambase | [Link](https://archive.ics.uci.edu/ml/datasets/spambase)
Sonar | [Link](https://github.com/jaredvasquez/RandomForest/blob/master/sonar.all-data.csv)
Mushrooms | [Link](https://archive.ics.uci.edu/ml/datasets/mushroom)
Splice | [Link](https://www.kaggle.com/datasets/muhammetvarl/splicejunction-gene-sequences-dataset)

# Output
The code generates counterfactual explanations in the form of a DataFrame, which is saved as "output.csv" in the same directory as the input dataset.
Additionally, the execution time is recorded and saved in a file named "executiontime.txt".

# Notes
This code assumes that the input dataset contains the target variable (outcome) column, and the features used for model training are all other columns.
The counterfactual examples are generated for the test data (x_test) using a RandomForestClassifier model.
Ensure that the required datasets are present in the same directory as the code or provide the appropriate file paths in the code.
Use the PySimpleGUI window to select the dataset file and interact with the DiCE code.
