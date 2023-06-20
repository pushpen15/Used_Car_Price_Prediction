# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:16:40 2021

@author: goura
"""

import pandas as pd
import numpy as np
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})

cars_data = pd.read_csv('car_price.csv')

#creating copy
cars = cars_data.copy()

#structure of dataset
cars.info()

#summerize the data
cars.describe()
pd.set_option('display.float_format',lambda x:'%.3f' % x)
cars.describe()

#to display maximum no of sets
pd.set_option('display.max_columns',500)
cars.describe()

#droping unwanted columns
col = ['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars = cars.drop(columns=col,axis=1)

#removing duplicate records
cars.drop_duplicates(keep='first',inplace=True)

#no.of missing values in each columns
cars.isnull().sum()

#variable year of registration
yearwise_count = cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration'] >2018)
sum(cars['yearOfRegistration'] <1950)

sns.regplot(x='yearOfRegistration',y='price',scatter=True,
            fit_reg=False,data=cars)
#working range 1950 and 2018

#variable price
price_count= cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)
#working range 100 and 150000

#variable powerps
power_count = cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,
            fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
#working range- 10 and 500

#======================================
#working range of data
#=========================================

cars = cars[
    (cars.yearOfRegistration <= 2018)
    &(cars.yearOfRegistration >=1950)
    &(cars.price >=100)
    &(cars.price <= 150000)
    &(cars.powerPS >= 10)
    &(cars.powerPS <= 500)
    ]

cars['monthOfRegistration']/=12

#creationg a new variable 'Age' by adding yearOfRegistration and monthOfregistration
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']= round(cars['Age'],2)
cars['Age'].describe()

#droping month and year column
cars = cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#age vs price
sns.regplot(x='Age',y='price',scatter=True,
            fit_reg=False,data=cars)

#powerPS vs price
sns.regplot(x='powerPS',y='price',scatter=True,
            fit_reg=False,data=cars)

#variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
# commercial => insiginificant

#variable offer type
cars['offerType'].value_counts()
sns.countplot(x= 'offerType',data=cars)
#ye bhi insiginificant hai

#variable ab test
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
#ye equally distributed hai

sns.boxplot(x='abtest',y='price',data=cars)
#price ko effect nhi karta isiliye insiginificant

#variable vehicle type
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
#effects price

#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)

#variable kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.boxplot(x='kilometer',y='price',data=cars)
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',scatter=True,
            fit_reg=False,data=cars)
#considered in modeling

#variable fueltype
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
#fuelType affects price

#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)

#variable notRepairedDamage
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)

#removing insiginificant variables
col=['seller','offerType','abtest']
cars= cars.drop(columns=col,axis=1)
cars_copy = cars.copy()

##########correlation
cars_select1 = cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

##ommiting missing values
cars_omit = cars.dropna(axis=0)

#converting the catagorical variables in dumies
cars_omit = pd.get_dummies(cars_omit,drop_first=True)

## importing nessesary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#model building with omitted data
#seprating input and output feature
x1 = cars_omit.drop(['price'],axis='columns',inplace=False)
y1 = cars_omit['price']

#ploting the variable price
prices = pd.DataFrame({"1. Before":y1,"2. After":np.log(y1)})
prices.hist()

#transforming price as a logerithemic value
y1= np.log(y1)

#splitting data into test and train
X_train,X_test,y_train,y_test = train_test_split(x1,y1,test_size=0.3,random_state = 3)
print(X_train.shape, X_test.shape,y_train.shape,y_test.shape)

#============BASELINE MODEL FOR OMMITED DATA ==================#

#find the mean for test data value
base_pred = np.mean(y_test)
print(base_pred) 

#repeating the same value till the length of test data

base_pred = np.repeat(base_pred,len(y_test)) 
 
#find the RMSE

base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
 
print(base_root_mean_square_error)

############linear regression with omited data

#setting intercept as true
lgr = LinearRegression(fit_intercept=True)

#model
model_lin1= lgr.fit(X_train,y_train)

#Predicting model as test set

cars_predictions_lin1 = lgr.predict(X_test)
print(lgr.predict())

#computing MSE and RMSE
lin_mse1 = mean_squared_error(y_test,cars_predictions_lin1)
lin_rmse1 = np.sqrt(lin_mse1)
print(lin_rmse1)

#R squared value
r2_lin_test1 = model_lin1.score(X_test,y_test)
r2_lin_train1 = model_lin1.score(X_train,y_train) 
print(r2_lin_test1,r2_lin_train1) 

#regression diagnostics - residual plot analysis
residuals1=y_test - cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1,y=residuals1,scatter=True,
            fit_reg=False,data=cars)
residuals1.describe()

#==============RANDOM FOREST WITH OMITED DATA========

#model parameters
rf = RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,
                           min_samples_leaf=4,random_state=1) 
#model
model_rf1 = rf.fit(X_train,y_train)

#predicting model on test set
cars_predictions_rf1 = rf.predict(X_test) 
 
#computing MSE and RMSE
rf_mse1 = mean_squared_error(y_test,cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)
 
 
#R squared value

r2_rf_test1 = model_rf1.score(X_test,y_test)
r2_rf_train1 = model_rf1.score(X_train,y_train) 
print(r2_rf_test1,r2_rf_train1)