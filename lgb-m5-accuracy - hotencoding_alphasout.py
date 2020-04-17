# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:18:18 2020

@author: PACO
"""


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
files = []
#for dirname, _, filenames in os.walk('/kaggle/input'):
for dirname, _, filenames in os.walk('C:/Users/laguila/Google Drive/ARC_KAGGLE/m5-datos'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
from dfply import *

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

pd.options.display.max_columns = 50

h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
fday

def create_dt(is_train = True, nrows = None, first_day = 1200, files = files):
    #prices
    prices = pd.read_csv(files[4], dtype = PRICE_DTYPES)
    # for col, col_dtype in PRICE_DTYPES.items():
    #     if col_dtype == "category":
    #         prices[col] = prices[col].cat.codes.astype("int16")
    #         prices[col] -= prices[col].min()
    #calendar        
    cal = pd.read_csv(files[0], dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])

    # for col, col_dtype in CAL_DTYPES.items():
    #     if col_dtype == "category":
    #         cal[col] = cal[col].cat.codes.astype("int16")
    #         cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    
    #validation
    dt = pd.read_csv(files[2], nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    if not is_train:
        for day in range(tr_last+1, tr_last+ 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                  id_vars = catcols,
                  value_vars = [col for col in dt.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    del(dt['weekday'])
    del(dt['wm_yr_wk'])
    dt['store_id'] = dt['store_id'].map(lambda x: x[-1:])
    dt['dept_id'] = dt['dept_id'].map(lambda x: x[-1:])
    dt['item_id'] = dt['item_id'].map(lambda x: x[-3:]).astype('int16')
    
    columns = [ 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in columns: 
        dt[col] = dt[col].cat.codes.astype("int16")
        dt[col] -= dt[col].min()

    return dt


def create_fea(dt):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    
    
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
            
          
    dt=pd.concat([dt, pd.get_dummies(dt.cat_id)], axis=1)
    dt=pd.concat([dt, pd.get_dummies(dt.state_id)], axis=1)
    
    dept_id = pd.get_dummies(dt.dept_id)
    idds = dept_id.columns
    dept_id.columns = [f"dept_{idd}" for idd in idds ]
    dt=pd.concat([dt, dept_id], axis=1)
    
    store_id = pd.get_dummies(dt.store_id)
    idds = store_id.columns
    store_id.columns = [f"store_{idd}" for idd in idds ]
    dt=pd.concat([dt, store_id], axis=1)
    
    dt.drop(["cat_id", "state_id", "dept_id", "store_id"], axis=1, inplace = True)

FIRST_DAY = 340 # If you want to load all the data set it to '1' -->  Great  memory overflow risk

df = create_dt(is_train=True, first_day= FIRST_DAY)
# df.shape

# df.head()
# df.info()

create_fea(df)
df.shape

df.dropna(inplace = True)
df.shape

#&cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales","d"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]

np.random.seed(767)

fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],
                 free_raw_data=False)

del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()

params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.073,
#         "sub_feature" : 0.8,
        "sub_row" : 0.73,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 1150,
    'num_leaves': 124,
    "min_data_in_leaf": 100,
}

m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20)
m_lgb.save_model("model.lgb")

alphas = [1.025, 1.023, 1.0175]
weights = [1/len(alphas)]*len(alphas)
sub = 0.

te = create_dt(False)
cols = [f"F{i}" for i in range(1,2)]

for tdelta in range(0,1):
   day = fday + timedelta(days=tdelta)
   print(tdelta, day)
   tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
   create_fea(tst)
   tst = tst.loc[tst.date == day , train_cols]
   te.loc[te.date == day, "sales"] = m_lgb.predict(tst) # magic multiplier by kyakovlev

for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

   te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h),
#                                                                           "id"].str.replace("validation$", "evaluation")
   te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
   te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
   te_sub.fillna(0., inplace = True)
   te_sub.sort_values("id", inplace = True)
   te_sub.reset_index(drop=True, inplace = True)
   te_sub.to_csv(f"submission_{icount}.csv",index=False)
   if icount == 0 :
       sub = te_sub
       sub[cols] *= weight*alpha
   else:
       sub[cols] += te_sub[cols]*weight*alpha
   print(icount, alpha, weight)


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission.csv",index=False)

sub.head(10)
sub.id.nunique(), sub["id"].str.contains("validation$").sum()
sub.shape