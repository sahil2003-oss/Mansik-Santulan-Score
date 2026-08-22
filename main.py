import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('Mental_Health_Model.pkl')
app=FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]) 
# it is used to connect frontend (using HTML,CSS and JS) and backend (FastAPI)

top_countries=['Other','India','USA','Canada','Australia','UK','Germany','Mexico','Turkey','France']
@app.get('/')
def greet():
    return {'message':'Welcome to the Mental Health Score Predictor APP'}

class ResponseValidation(BaseModel): # it's a pydantic model responsible for response data validation
    prediction:float

class StudentData(BaseModel): # it's a pydantic model responsible for input data validation
    age:int=Field(...,ge=10,le=100)
    gender:Literal['Male','Female']
    country:str
    academic_level:Literal['Undergraduate','Graduate','High School']
    most_used_platform:Literal['Instagram','Facebook','Twitter','Snapchat','TikTok','LinkedIn','Youtube','WhatsApp','LINE','VKontakte','KakaoTalk','WeChat']
    purpose_of_use:Literal['Networking','Entertainment','Education','News']
    avg_daily_usage_hour:float=Field(...,ge=0,le=24)
    daily_unlocks:int=Field(...,ge=0,le=24)
    study_hours:float=Field(...,ge=0,le=24)
    physical_activity_hours:float=Field(...,ge=0,le=24)
    sleep_hours_per_night:float=Field(...,ge=0,le=24)
    stress_level:Literal['Medium','Low','High','Very High']

    
@app.post('/predict', response_model=ResponseValidation)
def predict(data:StudentData):
    country_group=data.country if data.country in top_countries else 'Other'
    input_data=pd.DataFrame([{
            'Age':data.age,
            'Gender':data.gender,
            'Country':data.country,
            'Grouped_Country':country_group,
            'Academic_Level':data.academic_level,
            'Most_Used_Platform':data.most_used_platform,
            'Purpose_Of_Use':data.purpose_of_use,
            'Avg_Daily_Usage_Hours':data.avg_daily_usage_hour,
            'Daily_Unlocks':data.daily_unlocks,
            'Study_Hours':data.study_hours,
            'Physical_Activity_Hours':data.physical_activity_hours,
            'Sleep_Hours_Per_Night':data.sleep_hours_per_night,
            'Stress_Level':data.stress_level
}])

    prediction=model.predict(input_data)[0]
    
    return ResponseValidation(prediction=round(float(prediction),3))
