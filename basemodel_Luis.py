# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:46:50 2020

@author: ldelaguila
"""

import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt  


os.chdir("C:/Users/ldelaguila/Documents/GitHub/Cholesk/m5-forecasting-accuracy")

#%% Lectura de datos
calendar=pd.read_csv('../m5-datos/calendar.csv', delimiter=",")
sales=pd.read_csv('../m5-datos/sales_train_validation.csv', delimiter=",")
submission=pd.read_csv('../m5-datos/sample_submission.csv', delimiter=",")
prices=pd.read_csv('../m5-datos/sell_prices.csv', delimiter=",")


#%% Creación TRAIN inicial

sales = sales.melt(id_vars=sales.iloc[:,list(range(0,6))].columns, var_name="d", value_name='Units').dropna(how='any').reset_index()
train = pd.merge(sales, calendar, on='d')
train = pd.merge(train, prices, on=['item_id', 'store_id', 'wm_yr_wk'])


#%% Guardado
train.to_csv('train_inicial.csv')

#%% Modificacion train

del(train['id'])
del(train['index'])
del(train['wm_yr_wk'])
del(train['d'])
del(train['weekday'])

train['item_id']=train['item_id'].map(lambda x: x[(len(x)-(x[::-1]).find("_")):len(x)])
train['dept_id']=train['dept_id'].map(lambda x: x[(len(x)-(x[::-1]).find("_")):len(x)])
train['store_id']=train['store_id'].map(lambda x: x[(len(x)-(x[::-1]).find("_")):len(x)])

train['day']=train.date.iloc[:].map(lambda x: datetime.strptime(x, '%Y-%m-%d').day)
del(train['date'])

train['event_name_1']=train['event_name_1'].fillna(0)
train['event_type_1']=train['event_type_1'].fillna(0)
train['event_name_2']=train['event_name_2'].fillna(0)
train['event_type_2']=train['event_type_2'].fillna(0)

#%% Analisis train
train2=train.iloc[1:15,:]

#train.to_csv('train_mod1.csv')
train=pd.read_csv('train_mod1.csv', delimiter=",")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
train.event_name_1 = label_encoder.fit_transform(train.event_name_1.map(lambda x: str(x)))
train.event_type_1 = label_encoder.fit_transform(train.event_type_1.map(lambda x: str(x)))
train.event_name_2 = label_encoder.fit_transform(train.event_name_2.map(lambda x: str(x)))
train.event_type_2 = label_encoder.fit_transform(train.event_type_2.map(lambda x: str(x)))
train['item_id']=train['item_id'].map(lambda x: int(x))
del(train_oh['Unnamed: 0'])

train_oh=pd.get_dummies(train)
#train_oh.to_csv('train_oh.csv')


#%% Gaussian distribution
os.chdir('C:/Users/laguila/Google Drive/ARC_KAGGLE/m5-datos')
train_oh=pd.read_csv('train_oh.csv', delimiter=",")
os.chdir("C:/Users/ldelaguila/Documents/GitHub/Cholesk/m5-forecasting-accuracy")
train2=train_oh.iloc[1:15,:]


# from sklearn import preprocessing
# gaussian_scaler = preprocessing.PowerTransformer(method='yeo-johnson', standardize=False)
# train_oh1 = gaussian_scaler.fit_transform(train_oh.iloc[0:23000000,])
# train_oh2 = gaussian_scaler.fit_transform(train_oh.iloc[23000001:train_oh.shape[0],])

X=train_oh.iloc[0:100,:].copy()
y=X['Units'].copy()
del(X['Units'])
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 123, shuffle = False)


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
pred = regr.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(check.real, check.pred)

check=pd.DataFrame({"x":np.arange(pred.shape[0]),"real": y_test, "pred":pred})
plt.scatter(check.iloc[:,0], check.iloc[:,1], color="black")
plt.scatter(check.iloc[:,0], check.iloc[:,2], color="red")
plt.show()

train2=train_oh.iloc[1:15,:]

pd.set_option('display.max_columns', len(train_oh))
print(train.head())

