# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:46:50 2020

@author: ldelaguila
"""

import os
import csv
import pandas as pd
import numpy as np


os.chdir("C:/Users/ldelaguila/Documents/GitHub/Cholesk/m5-forecasting-accuracy")

#%% Lectura de datos
calendar=pd.read_csv('../m5-datos/calendar.csv', delimiter=",")
sales=pd.read_csv('../m5-datos/sales_train_validation.csv', delimiter=",")
submission=pd.read_csv('../m5-datos/sample_submission.csv', delimiter=",")
prices=pd.read_csv('../m5-datos/sell_prices.csv', delimiter=",")

#%% Estadistica basica
prices.wm_yr_wk.describe()


#%% Creación TRAIN

sales = sales.melt(id_vars=sales.iloc[:,list(range(0,6))].columns, var_name="d", value_name='Units').dropna(how='any').reset_index()

train = pd.merge(sales, calendar, on='d')

train = pd.merge(train, prices, on=['item_id', 'store_id', 'wm_yr_wk'])

from datetime import datetime
from dateutil.parser import parse

#Pasar a fecha
#☺train['date'][0]
#fecha=datetime.strptime(train['date'][0], '%Y-%m-%d')

train['day']=np.zeros(train.shape[0],dtype='int')
train['day']=train.date.iloc[:].map(lambda x: datetime.strptime(x, '%Y-%m-%d').day)

pd.set_option('display.max_columns', len(train))
train.head()
