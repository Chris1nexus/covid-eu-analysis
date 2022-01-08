import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay, confusion_matrix, mean_absolute_error,plot_roc_curve, r2_score, f1_score, accuracy_score,auc, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

import imblearn
from imblearn.over_sampling import SMOTE

from collections import Counter
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import f_regression, SelectKBest, VarianceThreshold, SelectFromModel
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
import datetime as dt
from scipy import stats
from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib.cm
from imblearn.over_sampling import SMOTE
import imblearn.pipeline as imb_pipeline


import pandas as pd
import numpy as np
import scipy as sp
import sqlite3
import os


from analysis_lib.manager import plot_curve, DataManager, rf_init

import json
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

    

def save_json(file_path, dictionary):
     
        with io.open(file_path  , 'w', encoding='utf8') as outfile:
            str_ = json.dumps(dictionary,
                                          indent=4, 
                                          separators=(',', ': '), ensure_ascii=False)
            
            outfile.write(to_unicode(str_))

def fullname(o):
  return o.__module__ + "." + o.__class__.__name__
def export_pipeline(scikit_pipeline):
  """JSON export of a scikit-learn pipeline.
  
  Especially useful when paired with GridSearchCV, TPOT, etc.
    
  Example:
  best_model = GridSearchCV(
      some_pipeline,
      param_grid=some_tuning_parameters
  )
  best_model.fit(X=train_x, y=train_y)
  export_pipeline(best_model.best_estimator_)
  :param scikit_pipeline: a scikit-learn Pipeline object
  """
  steps_obj = {'steps':[]}
  for name, md in scikit_pipeline.steps:
      steps_obj['steps'].append({
          'name': name,
          'class_name': fullname(md),
          'params': md.get_params()
      })

  return steps_obj
"""
def train_runs(data_manager, 
                        
                        model_init_fn=rf_init,
                           num_runs=20,
                           k_smote=None
                          ):
        from tqdm import tqdm        

        DENOMINATOR_N_RUNS = num_runs
    
    
        X, y = data_manager.get_dataset()     
        longrun_fimp_dict = dict()
        for i in tqdm(range(num_runs)):

            if data_manager.prob_type == "regression":
                    selector_helper = RandomForestRegressor()
            else:
                    selector_helper = RandomForestClassifier()
            pipeline = imb_pipeline.Pipeline( steps= [('feature_selection', SelectFromModel(selector_helper )),
                                                      ('sampling', SMOTE(k_neighbors=k_smote)),
                                                  ("model",model_init_fn() )    ])
            
            pipeline.fit(X, y)
            '''
            gs = GridSearchCV(pipeline, param_grid = {'model__criterion':['gini'],
                                                'model__max_depth':[None],
                                                'model__max_features':['sqrt'],
                                                'model__min_samples_leaf':[1],
                                                'model__n_estimators':[100]},cv=StratifiedKFold(n_splits=10), 
                                      scoring=scorer)

            gs.fit(X, y)
  
            pipeline = gs.best_estimator_
            '''
            

            selected_features = X.columns[pipeline["feature_selection"].get_support()]    
            feature_importances = sorted(zip(selected_features, pipeline["model"].feature_importances_), key=lambda x: x[1], reverse=True )
            for feature, importance in feature_importances:
                if feature not in longrun_fimp_dict:
                    longrun_fimp_dict[feature] = [importance]
                else:
                    longrun_fimp_dict[feature].append(importance)
        res = dict()
        for feature, importances in  longrun_fimp_dict.items():
            res[feature] = (sum(importances)/DENOMINATOR_N_RUNS, len(importances))
        return res




def eval_robustness(data_manager,
                   model_init_fn,
                N_RUNS=20,
                    k_smote = None,
                    DEST_DIR = "./results",
                   figsize=(10,8)):
    def rf_clf_init():
        return RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_leaf=1)

    
        
    res = train_runs(data_manager, 
                        model_init_fn = model_init_fn,
                           num_runs=N_RUNS,
                           k_smote=k_smote
                          )
    
    
    
    RESULTS_DIR = os.path.join(DEST_DIR,"robustness")
    target = "Covid_Cases_Density"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    plt.figure(figsize=figsize)
    feature_importances = sorted(res.items(), key=lambda x: x[1][0])
    used_feature_names = [ item[0]for item in feature_importances]
    avg_importances = [ item[1][0] for item in feature_importances]
    num_times_feature_is_selected = [ item[1][1] for item in feature_importances]
    plt.title(F"Number of times that features have been selected over {N_RUNS} runs")
    plt.barh(used_feature_names,num_times_feature_is_selected)
    
    
    unique_id_string = f"_{data_manager.clf_type}_from_{data_manager.start_date}_to_{data_manager.end_date}"  

    
    
    plt.savefig(os.path.join(RESULTS_DIR,f"feature_selected_numtimes_in_{N_RUNS}" + unique_id_string), bbox_inches='tight', dpi=600)
    
    
    plt.figure(figsize=figsize )
    plt.title(F"Average importance over {N_RUNS} runs")
    plt.barh(used_feature_names,avg_importances)
    
    
    plt.savefig(os.path.join(RESULTS_DIR,f"average_feature_importances_in_{N_RUNS}"+ unique_id_string), bbox_inches='tight', dpi=600)
    
    return res

"""

  
def plot_lombardy_vs_all(data_manager,
                         figsize=(10,8),
                        RESULT_DIR = "./results/lombardy_vs_all"):
    
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    X,y = data_manager.get_dataset()
    for column in data_manager.X.columns:
        plt.figure(figsize=(10,5) )
        plt.hist(X[column].tolist(),bins=100)# [0 for _ in range(len(tm.df.Covid_Cases_Density.tolist()))])
        plt.axvline(X[X.index=="ITC4"][column].tolist()[0], c="red" )
        plt.ylabel("norm " + column)
        
        plt.savefig(os.path.join(RESULT_DIR,"standardized_" + column) )


def plot_class_distribution(data_manager,
                           figsize=(10,8),
                           RESULT_DIR = "./results"):
    plt.figure(figsize=figsize) 

    order_mapping_binary = {"below_than_avg":0, "higher_than_avg":1}
    order_mapping_multi = {"low":0, "medium":1, "high":2}
    order_mapping_extremes = {"low":0,  "high":1}

    mapper = { "binary": order_mapping_binary,
              "multi":order_mapping_multi,
              "extremes":order_mapping_extremes}
    
    
    

    current_mapper = mapper[data_manager.clf_type]

    
    X,y = data_manager.get_dataset()
    
    counter = Counter(y)
    sorted_counter = dict(sorted(counter.items(), key=lambda x: current_mapper[x[0]] ))

 
    plt.bar(x=sorted_counter.keys(), height=sorted_counter.values(), width=0.3)
    
    unique_id_string = f"_{data_manager.clf_type}_from_{data_manager.start_date}_to_{data_manager.end_date}"  

    
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    plt.savefig(os.path.join(RESULT_DIR, "class_distribution" + unique_id_string) )
    
    
    
def plot_sample_nans(data_manager,
                         figsize=(10,8),
                    RESULT_DIR = "./results"):
    plt.figure(figsize=figsize)
    plt.hist(data_manager.df.isna().sum(axis=1)/len(data_manager.df))
    plt.ylabel("frequency")
    plt.xlabel("NAN values % for each sample")
    
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    plt.savefig(os.path.join(RESULT_DIR, "sample_nans") )
    
    
def plot_feature_nans(data_manager,
                         figsize=(10,8),
                     RESULT_DIR = "./results"):
    plt.figure(figsize=figsize )
    plt.hist(data_manager.df.isna().sum(axis=0)/len(data_manager.df), bins=20)
    plt.ylabel("frequency")
    plt.xlabel("NAN values % for each feature")
    
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    plt.savefig(os.path.join(RESULT_DIR, "feature_nans"))
    
    
    
    
def plot_clustering_quality(data_manager,
                         figsize=(10,8),
                           RESULT_DIR = "./results"):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from matplotlib import pyplot as plt

    X =  data_manager.df["Covid_Cases_Density"].values.reshape(-1,1)

    X = np.where(np.isnan(X), X[ np.isnan(X)==False].mean(axis=0), X)

    distorsions = []
    scores = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.predict(X))
        scores.append(score)
        distorsions.append(kmeans.inertia_)

    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
        
    unique_id_string = f"_{data_manager.clf_type}_from_{data_manager.start_date}_to_{data_manager.end_date}"  
     
    fig = plt.figure(figsize=figsize )
    plt.plot(range(2, 20), distorsions)
    plt.grid(True)
    plt.title('Elbow curve (on y axis: Sum of squared distances of samples to their closest cluster center)')   
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    
    plt.savefig(os.path.join(RESULT_DIR, "elbow_curve" + unique_id_string))
    
    
    plt.figure(figsize=figsize)
    plt.scatter(range(2, 20), scores)
    plt.grid(True)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette coefficient")
    plt.ylim(0,1)
    plt.title('Silhouette score for number of cluster in [0,20]')  
    
    plt.savefig(os.path.join(RESULT_DIR, "silhouette_coefficient" + unique_id_string))
    
    

    
def plot_clustering_subdivision(data_manager,
                         figsize=(10,8),
                               RESULT_DIR = "./results"):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from matplotlib import pyplot as plt

    X =  data_manager.df["Covid_Cases_Density"].values.reshape(-1,1)

    X = np.where(np.isnan(X), X[ np.isnan(X)==False].mean(axis=0), X)

    distorsions = []
    scores = []
    
    if data_manager.clf_type == "binary":
        n_clusters = 2
    else:
        n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    c_labels = kmeans.predict(X)
    means = dict()
    for k in range(n_clusters):
        means[k] = X[c_labels == k].mean()

    sorted_mapping = list(sorted(means.items(), key=lambda x: x[1]))
    colors =["orange", "red", "darkred", "black"]

    unique_id_string = f"_{data_manager.clf_type}_from_{data_manager.start_date}_to_{data_manager.end_date}"  

    plt.figure(figsize=figsize )
    
    tag = "cases"
    if data_manager.response =="cumulativedeceased":
        tag = "deaths"
    elif data_manager.response =="cumulativerecovered":
        tag = "recovered"
    for mapper, color in zip(sorted_mapping , colors) :
        c_idx = mapper[0]
        
        if n_clusters==2:
            n_bins=20
        else:
            n_bins=40
        plt.hist(X[c_labels==c_idx]*10**5,  color=color, bins=n_bins )
        plt.xlabel(f"covid {tag} density \nper 100k inhabitants")
        plt.title(f"frequency of the density of {tag}")
        
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)        
    plt.savefig(os.path.join(RESULT_DIR,"clustering" + unique_id_string))

 