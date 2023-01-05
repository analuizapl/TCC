
from cmath import nan
from codeop import CommandCompiler
from dataclasses import replace
from re import S
import dice_ml
from dice_ml import Dice
from dicechanged import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import json

import numpy as np
import pandas as pd
import datetime

import PySimpleGUI as sg

import io
def prepareskysurvey(df):
    df = df.replace("QSO",2)
    df = df.replace("GALAXY", 1)
    df = df.replace("STAR", 0)
    return df

def preparetdah(df):
    df = df.replace("Superior", 1)
    df = df.replace("Inferior", 0)
    return df

def preparevertabrae(df):
    df = df.replace("Abnormal", 1)
    df = df.replace("Normal", 0)
    return df

def preparetitanic(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def preparbreastcancer(df):
    df = df.replace("M", 0)
    df = df.replace("B", 1)
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float32)



def dice():


    df = pd.read_csv(path)

    if databasename == 'Skyserver_SQL2_27_2018 6_51_39 PM.csv':
        database= prepareskysurvey(df)
    elif databasename == 'tdah.csv':
        database= preparetdah(df)
    elif databasename == 'titanic.csv':
        database= preparetitanic(df)
    elif databasename == 'data.csv':
       database= preparbreastcancer(df)
    elif databasename =='tdah.csv':
        database= preparetdah(df)
    else:
        database = df


    #df.head()

    window['_LIST_'].update(list(database.columns))

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            break
        elif  event == 'Submit2':     # if a list item is chosen
            outcome_name = ''.join(values["_LIST_"])

            continuous_features_iris = database.drop(outcome_name, axis=1).columns.tolist()
            target = database[outcome_name]

            # Split data into train and test
            datasetX = database.drop(outcome_name, axis=1)
            x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                            target,
                                                            test_size=0.1,
                                                            random_state=0,
                                                            stratify=target)


            categorical_features = x_train.columns.difference(continuous_features_iris)

            # We create the preprocessing pipelines for both numeric and categorical data.
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            transformations = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, continuous_features_iris),
                    ('cat', categorical_transformer, categorical_features)])

            # Append classifier to preprocessing pipeline.
            # Now we have a full prediction pipeline.
            clf_iris = Pipeline(steps=[('preprocessor', transformations), ('classifier', RandomForestClassifier())])

            model_iris = clf_iris.fit(x_train, y_train)




            d_iris = dice_ml.Data(dataframe=database,
                                continuous_features=continuous_features_iris,
                                outcome_name=outcome_name)

            # We provide the type of model as a parameter (model_type)
            m_iris = dice_ml.Model(model=model_iris, backend="sklearn", model_type='classifier')
            exp_genetic_iris = Dice(d_iris, m_iris, method="random")#"genetic")

            # Single input
            query_instances_iris = x_test[1:2]
            totalscfs = int(values["quant"])-1
            start_time = datetime.datetime.now()
            genetic_iris = exp_genetic_iris.generate_counterfactuals(query_instances_iris, total_CFs=int(values["quant"]), desired_class="opposite")
            end_time = datetime.datetime.now()
            genetic_iris.visualize_as_dataframe(show_only_changes=True)
            teste = genetic_iris.cf_examples_list[0].final_cfs_df


            time_diff = (end_time - start_time)
            execution_time = (time_diff.total_seconds() * 1000)/  1000000

            with open( path.replace(".csv","") +"executiontime.txt", "w") as file:
                file.write('tempo de execução: '+str(execution_time)+' seconds')

            teste2 = genetic_iris.to_json()
            with open( path.replace(".csv","") +"ctf.json", "w") as file:
                    file.write(teste2)
            filename = path.replace(".csv","") +"ctf.json"

            jsonFile = open(filename, 'r')
            values = json.load(jsonFile)
            idValue = str(values['test_data'])
            idValue = idValue.replace("[[[","")
            idValue = idValue.replace("]]]","")
            df = pd.DataFrame([x.split(',') for x in idValue.split('\n')],columns = list(database.columns))
            df =  df.append([df]*int(totalscfs),ignore_index=True)

            teste.to_csv(path.replace(".csv","")+"contrafactuais.csv", sep='\t')
            df.to_csv(path.replace(".csv","")+"original.csv", sep='\t')

            df2 = pd.read_csv(path.replace(".csv","")+"contrafactuais.csv", sep='\t')
            df3 = pd.read_csv(path.replace(".csv","")+"original.csv", sep='\t')



            condicao = df2 != df3
            df4 = df2[condicao]

            vazio = np.nan
            df4 = df4.replace(vazio,'-')

            df4.to_csv(path.replace(".csv","")+"output.csv", encoding = 'utf-8-sig')




            window.close()






sg.theme('DarkPurple7')


layout = [
         [sg.Text("Choose a file: ", font=("Helvetica", 15)), sg.Input(size=(50, 20)), sg.FileBrowse(key="-IN-")],
         [sg.Button("Submit")],[sg.T("")],[sg.Text("Outcome name: ")],[sg.Listbox(list(), size=(40,10), enable_events=True, key='_LIST_')], [sg.Text("Quantidade contrafactuais"), sg.Input(key="quant")],
         [sg.Button("Criar", key = "Submit2")]]

###Building Window
window = sg.Window('DiCE', layout, size=(1000,500), resizable=True)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        break
    elif event == "Submit":
        path = values["-IN-"].replace("C:","")
        words = path.split('/')
        databasename = words[-1]
        dice()
