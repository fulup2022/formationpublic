from fastapi import APIRouter, Response
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
import app.api.api_v1.specific_modules.specific_modules as specific_modules

sys.modules['specific_modules'] = specific_modules

router_model_one_prediction = APIRouter()

class OutputModel():
  def __init__(self, tag: str, isExitedPredicted: str, score: str):
    self.tag = tag
    self.isExitedPredicted = isExitedPredicted
    self.score = score

class Answer():
  def __init__(self, outputs:List[OutputModel]):
    self.outputs = outputs

class CountryName(str, Enum):
  country1 = 'France'
  country2 = 'Spain'
  country3 = 'Germany'

class Gender(str, Enum):
  gender1 = 'Female'
  gender2 = 'Male'

class HasCard(int, Enum):
  hascard1 = 0
  hascard2 = 1

class IsActiveMember(int, Enum):
  isactivemember1 = 0
  isactivemember2 = 1

class Client(BaseModel):
  customerId : int
  surname: str
  creditScore : int
  geography: CountryName
  gender: Gender
  age: int
  tenure: int
  balance: float
  numOfProducts: int
  hasCard: HasCard
  isActiveMember: IsActiveMember
  estimatedSalary: float  

@router_model_one_prediction.post("/")
async def get_one_prediction(client: Client):
  
  print('The current client is:' + str(client.customerId))
  dataset = pd.DataFrame({'CreditScore':[client.creditScore], 'Geography':[client.geography], 'Gender':[client.gender], 'Age':[client.age], 'Tenure':[client.tenure], 'Balance':[client.balance], 'NumOfProducts':[client.numOfProducts], 'HasCrCard':[client.hasCard], 'IsActiveMember':[client.isActiveMember], 'EstimatedSalary':[client.estimatedSalary]})
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
    model_predictions = model.predict(dataset)
    print(model_predictions)
    model_predictions_proba = model.predict_proba(dataset)
    print(model_predictions_proba.shape)
    model_output = OutputModel(model.tag, str(model_predictions[0]),str(model_predictions_proba))
    answer.outputs.append(model_output)
    print('---------')
    
  return Response(json.dumps(answer, default=lambda o: o.__dict__, indent=None, separators=None), media_type="application/json")

