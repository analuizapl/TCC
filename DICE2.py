import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import dice_ml
from dice_ml import Dice

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def prepareDataBase(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

menu_options = {
    1: 'Titanic',
    2: 'Adult',
    3: 'Option 3',
    4: 'Exit',
}

def print_menu():
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

def titanic():
    train = pd.read_csv('Titanic/train.csv')
    test = pd.read_csv('Titanic/test.csv')

    # verificando as dimensões do DataFrame
    print("Variáveis:\t{}\nEntradas:\t{}".format(train.shape[1], train.shape[0]))

    # identificar o tipo de cada variável
    #display(train.dtypes)

    print("Porcentagem valores faltantes")
    (train.isnull().sum() / train.shape[0]).sort_values(ascending=False)
    
    # salvar os índices dos datasets para recuperação posterior
    train_idx = train.shape[0]
    test_idx = test.shape[0]
    # concatenar treino e teste em um único DataFrame
    df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))

    df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # completar ou apagar valores faltantes nos datasets de treino e teste
    df_merged.isnull().sum()
    # age
    age_median = df_merged['Age'].median()
    df_merged['Age'].fillna(age_median, inplace=True)

    # fare
    fare_median = df_merged['Fare'].median()
    df_merged['Fare'].fillna(fare_median, inplace=True)

    # embarked
    embarked_top = df_merged['Embarked'].value_counts()[0]
    df_merged['Embarked'].fillna(embarked_top, inplace=True)
    df_merged['Embarked'] = df_merged['Embarked'].map({'C': 0, 'S': 1})

    # converter 'Sex' em 0 e 1
    df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})
    outcome_name_titanic = 'Survived'
    prepareDataBase(df_merged)
    dice(df_merged,outcome_name_titanic)

def adult():
    train = pd.read_csv('adult/adult_processada.csv')

    # verificando as dimensões do DataFrame
    print("Variáveis:\t{}\nEntradas:\t{}".format(train.shape[1], train.shape[0]))

    # identificar o tipo de cada variável
    #display(train.dtypes)

    print("Porcentagem valores faltantes")
    display(train.isnull().sum() / train.shape[0]).sort_values(ascending=False)
    '''
    # salvar os índices dos datasets para recuperação posterior
    train_idx = train.shape[0]
    # concatenar treino e teste em um único DataFrame2
    
    df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))

    df_merged.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # completar ou apagar valores faltantes nos datasets de treino e teste
    df_merged.isnull().sum()
    # age
    age_median = df_merged['Age'].median()
    df_merged['Age'].fillna(age_median, inplace=True)

    # fare
    fare_median = df_merged['Fare'].median()
    df_merged['Fare'].fillna(fare_median, inplace=True)

    # embarked
    embarked_top = df_merged['Embarked'].value_counts()[0]
    df_merged['Embarked'].fillna(embarked_top, inplace=True)
    df_merged['Embarked'] = df_merged['Embarked'].map({'C': 0, 'S': 1})

    # converter 'Sex' em 0 e 1
    df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})
    outcome_name_titanic = 'Survived'
    prepareDataBase(df_merged)
    dice(df_merged,outcome_name_titanic)
    '''
def dice(df,outcome_name):

    continuous_features = df.drop(outcome_name, axis=1).columns.tolist()
    target = df[outcome_name]

    # Split data into train and test
    datasetX = df.drop(outcome_name, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(datasetX,target, test_size=0.1,random_state=0,stratify=target)

    categorical_features = x_train.columns.difference(continuous_features)

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformations = ColumnTransformer(transformers=[('num', numeric_transformer, continuous_features), ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', transformations), ('classifier', RandomForestClassifier())])

    model = clf.fit(x_train, y_train)


    d = dice_ml.Data(dataframe=df,continuous_features=continuous_features,outcome_name=outcome_name)


    # We provide the type of model as a parameter (model_type)
    m = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')
    exp_genetic = Dice(d, m, method="genetic")

    # Single input
    query_instances = x_test[1:2]
    totalscfs = 10

    genetic = exp_genetic.generate_counterfactuals(query_instances, total_CFs= totalscfs, desired_class="opposite")

    genetic.visualize_as_dataframe(show_only_changes=True)


print_menu()
option = int(input('Enter your choice: '))
if option == 1:
  titanic()
elif option == 2:
  adult()
elif option == 3:
  print('Handle option \'Option 3\'')
elif option == 4:
  print('Thanks message before exiting')
  exit()
else:
  print('Invalid option. Please enter a number between 1 and 4.')




