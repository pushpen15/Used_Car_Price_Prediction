import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
import speech_recognition as sr
import pyttsx3

sns.set(rc={'figure.figsize':(11.7,8.27)})

#Reading CSV
df=pd.read_csv('C:/Users/goura/Desktop/car_price/car data1.csv')

#strucure of dataset
df.info()

#droping unwanted columns
inputs=df.drop(['Car_Name','Owner','Seller_Type'],axis='columns') 

#removing duplicate records
#inputs.drop_duplicates(keep='first',inplace=True)

#no. of missing values in each columns
#inputs.isnull().sum()

#variable year 
sns.regplot(x='Year',y='Selling_Price',scatter=True,
            fit_reg=False,data=inputs)

#working range 2000 and 2019

sns.boxplot(y=inputs['Selling_Price'])

#working range of data 
inputs=inputs[
    (inputs.Year<=2019)
    &(inputs.Year>=2000) ]


#kms driven vs price
sns.regplot(x='Kms_Driven',y='Selling_Price',scatter=True,
            fit_reg=False,data=inputs)

#fuel type vs price
sns.regplot(x='Fuel_Type',y='Selling_Price',scatter=True,                        
            fit_reg=False,data=inputs)

#transmission vs price
sns.regplot(x='Transmission',y='Selling_Price',scatter=True,
            fit_reg=False,data=inputs)


######################

target=df.Selling_Price
inputs

from sklearn.preprocessing import LabelEncoder
Numerics=LabelEncoder()

inputs['Fuel_Type_n']=Numerics.fit_transform(inputs['Fuel_Type'])
inputs['Transmission_n']=Numerics.fit_transform(inputs['Transmission'])
inputs

inputs_n=inputs.drop(['Fuel_Type','Transmission','Selling_Price'],axis='columns') 
inputs_n
model=linear_model.LinearRegression()
model.fit(inputs_n,target)
###1.year 2.recent price 3.kmsdriven 4.fuel type,2 for petrol 1 for disel 
##### 5 transmission 1 for manual o for automatic
pred=model.predict([[2015,11.60,339888,1,0]])
pred
 

mylis=[1,1,1,1,1]






for i in range(0,5):
    r = sr.Recognizer()
    with sr.Microphone() as mp:
        engine = pyttsx3.init()
        engine.runAndWait()
        engine.say("enter")
        r.adjust_for_ambient_noise(mp,duration=2)
        print('say')
        audio=r.listen(mp)
        mylis[i] = r.recognize_google(audio)
        print("got it",mylis[i])
        
for i in range(0,5):
    if(mylis[i]=="tu" or mylis[i]== "Tu"):
        mylis[i]=2
    elif(i==1):
        mylis[i]=float(mylis[i])
    else:
        mylis[i]=int(mylis[i])
        pred=model.predict([mylis])*100000
pred=round(pred[0])
pred

    
    
    

