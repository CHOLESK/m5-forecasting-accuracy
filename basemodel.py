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
calendar=pd.read_csv('../m5-datos/calendar.csv', delimiter=",")
sales=pd.read_csv('../m5-datos/sales_train_validation.csv', delimiter=",", nrows = 20)
submission=pd.read_csv('../m5-datos/sample_submission.csv', delimiter=",")
prices=pd.read_csv('../m5-datos/sell_prices.csv', delimiter=",", nrows=20)
