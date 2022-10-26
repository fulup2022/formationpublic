from fastapi import APIRouter, Response, File, UploadFile
from enum import Enum
from typing import Union
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import glob2
import os
from typing import List
import json
import sys
import time
import app.api.api_v1.specific_modules.specific_modules as specific_modules

sys.modules['specific_modules'] = specific_modules

router_model_multi_predictions = APIRouter()

class OutputModel(dict):
  def __init__(self, tag: str, predictions: pd.DataFrame):
    dict.__init__(self, tag=tag, predictions_in_pd_df_to_json=predictions.to_json(orient='index'))

class Answer():
  def __init__(self, outputs:List[OutputModel]):
    self.outputs = outputs

@router_model_multi_predictions.post("/")
def get_multi_predictions(file: UploadFile = File(...)):
  
  if not file.filename.endswith('.csv'):
    return {"Error": "The file must be a .csv"}    

  uploaded_filepath = 'app/api/api_v1/uploaded_files/' + file.filename.replace('.csv','') + time.strftime("%Y:%m:%d:%H:%M:%S").replace(':','') + '.csv' 
  print(uploaded_filepath)
  
  try:
    contents = 	file.file.read()
    print(file.filename)
    with open(uploaded_filepath, 'wb') as f:
      f.write(contents)      
  except Exception:
    return {"Error": "There was an error uploading the file"}
  finally:
    file.file.close()

  print(file.filename)

  try:
    source_df = pd.read_csv(uploaded_filepath)
  except Exception:
    return {"Error": "There was an error while reading the file"}

  print(source_df)

  if not set(['CustomerId', 'CreditScore','Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']).issubset(source_df.columns):
    return {"Error": "Some expected columns are missing in the file"}

  dataset = pd.DataFrame({'CreditScore':source_df['CreditScore'], 'Geography':source_df['Geography'], 'Gender':source_df['Gender'], 'Age':source_df['Age'], 'Tenure':source_df['Tenure'], 'Balance':source_df['Balance'], 'NumOfProducts':source_df['NumOfProducts'], 'HasCrCard':source_df['HasCrCard'], 'IsActiveMember':source_df['IsActiveMember'], 'EstimatedSalary':source_df['EstimatedSalary']})
  print(dataset)

  print(os.path.join('app/api/api_v1/trained_models/'))

  my_files = glob2.glob(os.path.join('app/api/api_v1/trained_models/', '*.sav'));

  print(my_files);

  list_of_models =  []

  for file in my_files:
    with open(file, 'rb') as f:
      model = pickle.load(f)
      list_of_models.append(model)

  answer = Answer([])

  for model in list_of_models:

    print('Tag = ' + str(model.tag))
    model_predictions_df = pd.DataFrame({'CustomerId' : source_df['CustomerId']})

    model_predictions = model.predict(dataset)
    print(model_predictions.shape) 
    model_predictions_df['Prediction'] = [str(element) for element in model_predictions]
    
    model_predictions_proba = model.predict_proba(dataset)
    print(model_predictions_proba.shape)
    model_predictions_df['Proba'] = [str(element) for element in model_predictions_proba]
    
    print(model_predictions_df)

    model_output = OutputModel(model.tag, model_predictions_df)

    answer.outputs.append(model_output)
    
    print(answer)
    
    print('---------')

  print(model_output)
    
  return Response(json.dumps(answer, default=lambda o: o.__dict__, indent=None, separators=None), media_type="application/json")

