

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
if ejecucion=="Luis":
    os.chdir("C:/Users/laguila/Documents/GitHub/m5-forecasting-accuracy")
    from Dfs_creation_luis import create_dt, create_fea
    os.chdir('C:/Users/laguila/Google Drive/ARC_KAGGLE/m5-datos/resultados')
else:
    os.chdir("C:/Users/PACO/Documents/GitHub/m5-forecasting-accuracy")
    from Dfs_creation_paco import create_dt, create_fea
    os.chdir("C:/Users/PACO/Google Drive (paco.noa.gut@gmail.com)/ARC_KAGGLE/m5")

# =============================================================================
#%% ===========================================================================
#                                 SETTING INPUT
# =============================================================================
FIRST_DAY = 350 
h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
fday
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
#                                  Guardado_datasets
# =============================================================================
cop=df.copy()
df.id.astype(str)

df = df.loc[df.id.str.contains(pat = 'CA_1')]
df = df.loc[df.id.str.contains(pat = 'HOBBIES_1')]

os.chdir("C:/Users/PACO/Desktop/datasets")
e=pd.DataFrame(df.shape).astype(str)
e=e.iloc[1,]
df.to_csv(("df_train_"+e[0]+".csv"),index=False)
# =============================================================================
#%% ===========================================================================
#                              SELECCIÓN VARIABLES
# =============================================================================
# cat_feats = ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
# useless_cols = ["id", "sales", "date","d",'rmean_1_28','rmean_7_1','rmean_28_7']
# train_cols = df.columns[~df.columns.isin(useless_cols)]


cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday", 'lag_1','rmean_1_1', 'rstd_1_1', 'rmean_7_1', 'rstd_7_1',
       'rmean_28_1', 'rstd_28_1', 'rmean_1_7', 'rstd_1_7',
       'rstd_7_7', 'rstd_28_7', 'rmean_1_28', 'rstd_1_28', 'rstd_7_28', 'rstd_28_28']
train_cols = df.columns[~df.columns.isin(useless_cols)]

X = df[train_cols]
y = df["sales"]
# =============================================================================
#%% ===========================================================================
#                                   DIVISIÓN TEST/TRAIN
# =============================================================================
np.random.seed(767)

test_total = np.random.choice(X.index.values, round(0.01*X.shape[0]), replace = False)
test_inds = np.setdiff1d(np.random.choice(X.index.values, round(0.05*X.shape[0]), replace = False),test_total)
train_inds = np.setdiff1d(np.setdiff1d(X.index.values, test_inds),test_total)

X_train = X.loc[train_inds,]
X_test = X.loc[test_inds,]
X_check = X.loc[test_total,]
y_train = y.loc[train_inds]
y_check = y.loc[test_total]
# =============================================================================
#%% ===========================================================================
#                                  OPTIMIZACIÓN DE ESPACIO
# =============================================================================
del df, X, y, test_inds,train_inds ; gc.collect()
# =============================================================================
#%%

# use this section if you want to customize optimization

# define blackbox function
def f(x):
    print(x)
    params = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'learning_rate': x[0],
        'num_leaves': x[1],
        'min_data_in_leaf': x[2],
        'num_iteration': x[3],
        'max_bin': x[4],
        'verbose': 1
    }
    
    gbm = lgb.train(params,
            train_data,
            num_boost_round=100,
            valid_sets=test_data,
            early_stopping_rounds=5)
            
    print(type(gbm.predict(X_test, num_iteration=gbm.best_iteration)[0]),type(up_test.astype(int)[0]))
    
    print('score: ', mean_squared_error(gbm.predict(X_test, num_iteration=gbm.best_iteration), up_test.astype(float)))
    
    return mean_squared_error(gbm.predict(X_test, num_iteration=gbm.best_iteration), up_test.astype(float))

# optimize params in these ranges
spaces = [
    (0.19, 0.20), #learning_rate
    (2450, 2600), #num_leaves
    (210, 230), #min_data_in_leaf
    (310, 330), #num_iteration
    (200, 220) #max_bin
    ]

# run optimization
from skopt import gp_minimize
res = gp_minimize(
    f, spaces,
    acq_func="EI",
    n_calls=10) # increase n_calls for more performance

# print tuned params
print(res.x)

# plot tuning process
from skopt.plots import plot_convergence
plot_convergence(res)



#%% LGBM

from time import time
t = time()
params = {
        'boosting_type': 'gbdt',
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.09602177025210422,
        "sub_row" : 0.73,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        'verbosity': 1,
        'num_iterations' : 11,
        'num_leaves': 124,
        "min_data_in_leaf": 100,
        "n_jobs" : -1,
        'min_child_samples ': 30
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

#%% Borrar d
X_train=X_train.drop(['d'], axis=1)
X_test=X_test.drop(['d'], axis=1)
#%% Ranomized search

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

light = lgb.LGBMRegressor(num_iterations=1020, objective = "poisson", silent = False, seed = 10)
from scipy.stats import uniform
params = {
         "boosting_type" : ['gbdt', 'rf'],
        #"objective" : ["poisson"],
        "learning_rate" : uniform(loc=0.05, scale=0.2),
        # 'num_leaves': [100, 120, 140],
         'min_child_samples ': [10, 30],
        # "n_estimators" : [100, 120, 140]
        # 'reg_alpha' : uniform(loc=0.05, scale=1),
        # 'reg_lambda ' : uniform(loc=0.05, scale=1),
}
clf = RandomizedSearchCV(estimator = light, param_distributions = params, n_iter = 17, n_jobs=-1, cv = 2, verbose = 1)
print("Training...")
search3 = clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', early_stopping_rounds=5)

search3.best_params_
# lgb.plot_importance(search3, importance_type='split', max_num_features=20)
# lgb.plot_importance(search3, importance_type='gain', max_num_features=20)
#%% Custom Score
import matplotlib.pylab as plt

prediccion=pd.DataFrame(search3.predict(X_check))

type(y_check)
type(prediccion)
df_c = pd.concat([y_check.reset_index(drop=True), prediccion], axis=1)

plt.plot(df_c.sales.iloc[0:5000,],color="red")
plt.plot(df_c.iloc[0:5000,1],color="blue")
plt.plot(abs(df_c.iloc[0:5000,1]-df_c.sales.iloc[0:5000,]),color="green")
#%%
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# #############################################################################
# Fit regression model
n_neighbors = 50

knn = neighbors.KNeighborsRegressor(n_neighbors)
y_ = knn.fit(X_train, y_train).predict(X_test)

plt.subplot(2, 1, i + 1)
plt.scatter(np.linspace(0, 199,200), y_test.iloc[0:200], color='darkorange', label='data')
plt.scatter(np.linspace(0, 199,200), y_[0:200], color='navy', label='prediction')
plt.axis('tight')
plt.legend()
plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors))


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_)

plt.tight_layout()
plt.show()
#%%
from sklearn import tree

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
y_=clf.predict(X_test)

plt.subplot(2, 1, i + 1)
plt.scatter(np.linspace(0, 19,20), y_test.iloc[0:20], color='darkorange', label='data')
plt.scatter(np.linspace(0, 19,20), y_[0:20], color='navy', label='prediction')
plt.axis('tight')
plt.legend()
plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors))


from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_)

plt.tight_layout()
plt.show()


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
        tst=tst.drop(['d'], axis=1)
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


