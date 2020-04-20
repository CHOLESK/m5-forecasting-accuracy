

#%% ===========================================================================
#                                 LUIS O PACO
# =============================================================================
global ejecucion
ejecucion="Paco"
# =============================================================================
#%% ===========================================================================
#                                   LIBRERIAS
# =============================================================================
import numpy as np
import pandas as pd
from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
import os
# =============================================================================
#%% ===========================================================================
#                                   FUNCIONES
# =============================================================================

os.chdir("C:/Users/PACO/Documents/GitHub/m5-forecasting-accuracy")
from Dfs_creation_paco import create_dt, create_fea

# =============================================================================
#%% ===========================================================================
#                                 SETTING INPUT
# =============================================================================
FIRST_DAY = 1000 
# =============================================================================
#%% ===========================================================================
#                             CREACIÓN DATAFRAME
# =============================================================================
df = create_dt(is_train=True, first_day= FIRST_DAY)

# =============================================================================
#%% ===========================================================================
#                             CREACIÓN INDICADORES
# =============================================================================
create_fea(df)
# =============================================================================
#%% ===========================================================================
#                                  NA.OMIT
# =============================================================================
df.dropna(inplace = True)
# =============================================================================
#%% ===========================================================================
#                              SELECCIÓN VARIABLES
# =============================================================================
cat_feats = ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "sales", "date",'rmean_1_28','rmean_7_1','rmean_28_7']
train_cols = df.columns[~df.columns.isin(useless_cols)]
X = df[train_cols]
y = df["sales"]
# =============================================================================
#%% ===========================================================================
#                                   DIVISIÓN TEST/TRAIN
# =============================================================================
np.random.seed(767)

test_inds = np.random.choice(X.index.values, round(0.25*X.shape[0]), replace = False)
train_inds = np.setdiff1d(X.index.values, test_inds)

X_train = X.loc[train_inds,]
X_test = X.loc[test_inds,]
y_train = y.loc[train_inds]
y_test = y.loc[test_inds]
# =============================================================================
#%% ===========================================================================
#                                  OPTIMIZACIÓN DE ESPACIO
# =============================================================================
del df, X, y, test_inds,train_inds ; gc.collect()
# =============================================================================

#%% LGBM

from time import time
t = time()
params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.073,
        "sub_row" : 0.73,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        'verbosity': 1,
        'num_iterations' : 11,
        'num_leaves': 124,
        "min_data_in_leaf": 100,
        "n_jobs" : -1
}
lgb_ = lgb.LGBMRegressor()
lgb_.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5) 
print(t-time())

#%% XGBoost
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


#%% Ranomized search

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

light = lgb.LGBMRegressor(num_iterations=15, objective = "poisson", silent = False, seed = 10)
from scipy.stats import uniform
params = {
         "boosting_type" : ['gbdt', 'rf'],
        #"objective" : ["poisson"],
        "learning_rate" : uniform(loc=0.05, scale=0.5),
        # 'num_leaves': [100, 120, 140],
         'min_child_samples ': [10, 20, 30],
        # "n_estimators" : [100, 120, 140]
        # 'reg_alpha' : uniform(loc=0.05, scale=1),
        # 'reg_lambda ' : uniform(loc=0.05, scale=1),
}
clf = RandomizedSearchCV(estimator = light, param_distributions = params, n_iter = 10, n_jobs=-1, cv = 2, verbose = 1)
print("Training...")
search3 = clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', early_stopping_rounds=5)

search3.best_params_

#%% Custom Score

#%%Preprocesamiento
Ver las distribuciones de cada feature
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
dump(vot, 'vot.joblib') 

#%% PREDICCION
te = create_dt(False)
cols = [f"F{i}" for i in range(1,29)]

for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        create_fea(tst)
        tst = tst.loc[tst.date == day , train_cols]
        te.loc[te.date == day, "sales"] = search3.predict(tst) # magic multiplier by kyakovlev


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
sub.to_csv("submission.csv",index=False)

sub.head(10)
sub.id.nunique(), sub["id"].str.contains("validation$").sum()
sub.shape


