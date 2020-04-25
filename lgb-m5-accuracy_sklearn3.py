

import os
os.chdir('C:/Users/laguila/Google Drive/ARC_KAGGLE/m5')

# Any results you write to the current directory are saved as output.

from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
from dfply import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

pd.options.display.max_columns = 50

h = 28 
max_lags = 14
tr_last = 1913
fday = datetime(2016,4, 25) 
fday

def create_dt(is_train = True, nrows = None, first_day = 1200):
    #prices
    prices = pd.read_csv(os.getcwd()+"\\datos\\sell_prices.csv", dtype = PRICE_DTYPES)       
    cal = pd.read_csv(os.getcwd()+"\\datos\\calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    
    #validation
    dt = pd.read_csv(os.getcwd()+"\\datos\\sales_train_validation.csv", nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
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
    lags = [1, 7]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)
    dt = dt >> group_by(X.dept_id,X.store_id ,X.cat_id,X.state_id, X.date) >> summarize(Sales_1=X.lag_1.mean())  >> full_join(dt, by=["date", "dept_id","store_id" ,"cat_id","state_id"])
    dt["Sales_7"] = dt[["id", 'Sales_1']].groupby("id")['Sales_1'].transform(lambda x : x.rolling(7).mean())
    wins = [1, 7]
    for win in wins:
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
    
    
    dt=pd.concat([dt, pd.get_dummies(dt.cat_id)], axis=1)
    dt=pd.concat([dt, pd.get_dummies(dt.state_id)], axis=1)
    
    dept_id = pd.get_dummies(dt.dept_id)
    idds = dept_id.columns
    dept_id.columns = [f"dept_{idd}" for idd in idds ]
    
    store_id = pd.get_dummies(dt.store_id)
    idds = store_id.columns
    store_id.columns = [f"store_{idd}" for idd in idds ]
    
    dt=pd.concat([dt, store_id, dept_id], axis=1)
    
    dt.drop(["cat_id", "state_id", "dept_id", "store_id"], axis=1, inplace = True)
    
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        #"quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
    
    dt.drop(["d"], axis=1, inplace = True)

    return dt
            
    


      

FIRST_DAY = 500 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk

df = create_dt(is_train=True, first_day= FIRST_DAY)

df = create_fea(df)
df.dropna(inplace = True)
df.head()

cat_feats = ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "sales", "date"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X = df[train_cols]
y = df["sales"]

np.random.seed(767)

test_inds = np.random.choice(X.index.values, round(0.25*X.shape[0]), replace = False)
train_inds = np.setdiff1d(X.index.values, test_inds)

X_train = X.loc[train_inds,]
X_test = X.loc[test_inds,]
y_train = y.loc[train_inds]
y_test = y.loc[test_inds]


del df, X, y, test_inds,train_inds ; gc.collect()

#%% LGBM individual
#2.27085 original
# from time import time
# t = time()
# params = {
#     'num_threads': 8,
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'poisson',
#     'learning_rate': 0.1579020089834217,
#     'num_leaves': 2487, 
#     'min_data_in_leaf': 217,
#     'num_iteration': 1500, 
#     'max_bin': 208,
#     'verbose': 1,
#     'metric': "rmse",
#     'max_depth': 5
# }
# lgb_ = lgb.LGBMRegressor(**params)
# lgb_.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5) 
# print(t-time())

#%% Ranomized search

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform

# light = lgb.LGBMRegressor(num_iterations=15, objective = "poisson", silent = False, seed = 10)
# from scipy.stats import uniform
# params = {
#          "boosting_type" : ['gbdt', 'rf'],
#         #"objective" : ["poisson"],
#         "learning_rate" : uniform(loc=0.05, scale=0.5),
#         # 'num_leaves': [100, 120, 140],
#          'min_child_samples ': [10, 20, 30],
#         # "n_estimators" : [100, 120, 140]
#         # 'reg_alpha' : uniform(loc=0.05, scale=1),
#         # 'reg_lambda ' : uniform(loc=0.05, scale=1),
# }
# clf = RandomizedSearchCV(estimator = light, param_distributions = params, n_iter = 10, n_jobs=-1, cv = 2, verbose = 1)
# print("Training...")
# search3 = clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', early_stopping_rounds=5)

# search3.best_params_
#%% OPTIMIZACION PARAMETROS SKOPT

# define blackbox function
def f(x):
    print(x)
    params = {
        'num_threads': 8,
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'poisson',
        'learning_rate': x[0],
        'num_leaves': x[1], 
        'min_data_in_leaf': x[2],
        'num_iteration': x[3], 
        'max_bin': x[4],
        'verbose': 1,
        'metric': "rmse",
        'lambda_l2': x[5]
    }
    
    gbm = lgb.LGBMRegressor(**params)

    gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5) 
    
    print('score: ', mean_squared_error(gbm.predict(X_test), y_test))
    
    return mean_squared_error(gbm.predict(X_test), y_test)

# optimize params in these ranges
spaces = [
    (0.05, 0.10), #learning_rate.
    (100, 150), #num_leaves.
    (50, 150), #min_data_in_leaf
    (280, 300), #num_iteration
    (200, 220), #max_bin
    (0, 0.1) #max depth
    ]

# run optimization
from skopt import gp_minimize
res = gp_minimize(
    f, spaces,
    acq_func="EI",
    n_calls=20) # increase n_calls for more performance

# print tuned params
print(res.x)

# plot tuning process
from skopt.plots import plot_convergence
plot_convergence(res)

#%%OPTIMIZACION PARAMETROS OPTUNA
import optuna
def fit_lgbm(trial, X_train, y_train,  X_test, y_test, seed=None, cat_features=cat_feats):
    """Train Light GBM model"""
 
    params = {
    'num_threads': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'poisson',
    'learning_rate': trial.suggest_uniform('learning_rate', 0.05, 0.3),
    'num_leaves': trial.suggest_int('num_leaves', 100, 200), 
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 150),
    'num_iteration': 300, 
    'verbose': 1,
    'metric': "rmse",
    'lambda_l2': trial.suggest_uniform('learning_rate', 0, 0.1)
    }
   

    params['seed'] = 13

    early_stop = 5
    verbose_eval = 1

    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    d_valid = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_features)
    watchlist = [d_train, d_valid]

    print('training LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=100,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)

    # predictions
    y_pred_valid = model.predict(X_test, num_iteration=model.best_iteration)
    
    print('best_score', model.best_score)
    log = {'train/rmse': model.best_score['training']['rmse'],
           'valid/rmse': model.best_score['valid_1']['rmse']}
    return model, y_pred_valid, log

def objective(trial: Trial):
    # folds = 5
    # seed = 13
    # shuffle = False
    # kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)

    # X_train, y_train = create_X_y(train_df, target_meter=target_meter)
    # y_valid_pred_total = np.zeros(X_train.shape[0])
    # gc.collect()
    # print('target_meter', target_meter, X_train.shape)

    # cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
    # print('cat_features', cat_features)

    models = []
    # valid_score = 0
    # for train_idx, valid_idx in kf.split(X_train, y_train):
    #     train_data = X_train.iloc[train_idx,:], y_train[train_idx]
    #     valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

    #     print('train', len(train_idx), 'valid', len(valid_idx))
    # #     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
    model, y_pred_valid, log = fit_lgbm(trial, X_train, y_train,  X_test, y_test, num_rounds=2)
    #y_valid_pred_total[valid_idx] = y_pred_valid
    models.append(model)
    # gc.collect()
    valid_score = log["valid/rmse"]
    # if fast_check:
    #     break
    # valid_score /= len(models)
    # if return_info:
    #     return valid_score, models, y_pred_valid, y_train
    # else:
    return valid_score

study = optuna.create_study()
study.optimize(objective, n_trials=4)


print('Best trial: score {}, params {}'.format(study.best_trial.value, study.best_trial.params))

study.trials_dataframe()

import plotly
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_intermediate_values(study)
optuna.visualization.plot_slice(study)
optuna.visualization.plot_contour(study)
optuna.visualization.plot_parallel_coordinate(study)
#%% XGBoost individual
import xgboost as xgb
params2 = {
        "n_estimators" : 50,
        #"max_depth" :,
        "learning_rate" : 0.1,
        "verbosity" : 1,
        "booster" : "gblinear",
        "n_jobs" : -1,
        #"min_child_weight " :,
        'reg_alpha ': 0.01,
        'reg_lambda ' : 0,
        'random_state': 124
}
t = time()
xgb_ = xgb.XGBRegressor()
xgb_.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5) 
print(time()-t)
evals_result = xgb.evals_result()

#%%Voting regressor
xgb_ = xgb.XGBRegressor()
lgb_ = lgb.LGBMRegressor()
from sklearn.ensemble import VotingRegressor
vot = VotingRegressor([('xgb', xgb_), ('lgb', lgb_)])
vot.fit(X_train, y_train)

#%% Stacked model

estimators = [('xgb', xgb.XGBRegressor()),
              ('lgb', lgb.LGBMRegressor())]

from sklearn.ensemble import GradientBoostingRegressor #Para pegar todos juntos
from sklearn.ensemble import StackingRegressor

reg = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(random_state=42))

reg.fit(X_train, y_train)




#%% Ver las distribuciones de cada feature y transformaciones




#%%Preprocesamiento

pipelines
quitar variables poca importancia
quitar correlacionadas
lda/PCA/...
elastic net
stochastic gradient descent
svm



permutation importance and correlated features
validation and learning curve

#%% GUARDAR MODELOS

from joblib import dump, load
dump(lgb_, 'optlgb.joblib') 

#%% PREDICCION
te = create_dt(False)
cols = [f"F{i}" for i in range(1,29)]

for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        tst=create_fea(tst)
        tst = tst.loc[tst.date == day , train_cols]
        te.loc[te.date == day, "sales"] = lgb_.predict(tst) # magic multiplier by kyakovlev
        
        
te_check = create_dt(True)

for tdelta in range(0, 20):
        day = fday + timedelta(days=tdelta)-timedelta(days=20)
        print(tdelta, day)
        tst = te_check[(te_check.date >= day - timedelta(days=max_lags)) & (te_check.date <= day)].copy()
        tst=create_fea(tst)
        tst = tst.loc[tst.date == day , train_cols]
        te_check.loc[te_check.date == day, "sales_pred"] = lgb_.predict(tst) # magic multiplier by kyakovlev
te_check.dropna(inplace=True)
precios = te_check >> select(X.id, X.sales, X.date, X.sales_pred) >> mask(X.id == 'HOBBIES_1_001_CA_1_validation') 

plt.plot(precios.date, precios.sales, color="blue")
plt.plot(precios.date, precios.sales_pred, color="red")
plt.show()

#%% MAGIC MULTIPLIER KYAKOVLEV
alphas = [1.025, 1.023, 1.0175]
alphas = [1]
weights = [1/len(alphas)]*len(alphas)
sub = 0.


for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

   te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()
#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h),
#                                                                           "id"].str.replace("validation$", "evaluation")
   te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]
   te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()
   te_sub.fillna(0., inplace = True)
   te_sub.sort_values("id", inplace = True)
   te_sub.reset_index(drop=True, inplace = True)
   #te_sub.to_csv(f"submission_{icount}.csv",index=False)
   if icount == 0 :
       sub = te_sub
       sub[cols] *= weight*alpha
   else:
       sub[cols] += te_sub[cols]*weight*alpha
   print(icount, alpha, weight)


sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission_3.csv",index=False)

sub.head(10)
sub.id.nunique(), sub["id"].str.contains("validation$").sum()
sub.shape


