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



import pandas as pd
import numpy as np
import scipy as sp
import sqlite3
import os


from analysis_lib.manager import plot_curve, TrainingManager, rf_init





def eval_robustness(clf_type="binary",
                N_RUNS=20,
                    response="cumulativepositive",
                K_SMOTE = 8,
                N_SPLITS=10,
                    start_date="2020-01-20",
                 end_date="2020-08-20",
                    leave_one_out=False,
                    nuts2_to_remove = [], #alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']
                   figsize=(10,8),
                    clust_method='ML',
                    sqlite_file_path="./covid_at_lombardy.sqlite"):
    def rf_clf_init():
        return RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_leaf=1)


    
    if clf_type == "binary":
        n_clusters=2
    else:
        n_clusters=3
    training_manager = TrainingManager(sqlite_file_path=sqlite_file_path, 
                 prob_type="classification",
                                       response=response,
                 clf_type=clf_type, 
                 n_clusters=n_clusters,
                 nuts2_to_remove=nuts2_to_remove ,
                 start_date=start_date,
                 end_date=end_date,
                                      leave_one_out=leave_one_out,
                plot_roc=False,
                clust_method=clust_method,
                f1_score_on_minority=False) 
    res = training_manager.longrun_train_test(model_init_fn=rf_clf_init , metric=f1_score,
                                num_runs=N_RUNS,
                                n_splits=N_SPLITS,
                               k_smote=K_SMOTE)
    
    
    RESULTS_DIR = os.path.join("results","robustness")
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
    
    
    unique_id_string = f"_{training_manager.clf_type}_from_{training_manager.start_date}_to_{training_manager.end_date}"  

    
    
    plt.savefig(os.path.join(RESULTS_DIR,f"feature_selected_numtimes_in_{N_RUNS}" + unique_id_string), bbox_inches='tight', dpi=600)
    
    
    plt.figure(figsize=figsize )
    plt.title(F"Average importance over {N_RUNS} runs")
    plt.barh(used_feature_names,avg_importances)
    
    
    plt.savefig(os.path.join(RESULTS_DIR,f"average_feature_importances_in_{N_RUNS}"+ unique_id_string), bbox_inches='tight', dpi=600)
    
    return res, training_manager





def custom_dataset(training_manager, standardized=False ):
    selected_features = get_selected_features(training_manager)
    
    
    std_str = None
    if standardized:
        std_str = "standardized"
        X = training_manager.X
    else:
        std_str = "per100k"
        X = training_manager.interpretation_df
        

    X_selected = X[selected_features]
   
    # still, predictions have to be made on standardized data, as this is the one that is used to train the model
    X_pred = training_manager.X

    cols_dict = dict()
    preds = training_manager.pipeline.predict(X_pred)

    cols_dict["predictions"] = preds
    cols_dict["true_class"] = training_manager.y
    
    out_df = pd.DataFrame(cols_dict)

    RESULT_DIR = "predictions_dataset"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    
    
    unique_id_string = f"_{std_str}_{training_manager.clf_type}_from_{training_manager.start_date}_to_{training_manager.end_date}"  

    
    
    pd.concat([X, out_df], axis=1).to_csv(os.path.join(RESULT_DIR, "boxplot_analysis" + unique_id_string +".csv"))


# DEPRECATED
def custom_dataset_with_predictions(training_manager):
    selected_features = get_selected_features(training_manager)
    lombardy = training_manager.X[training_manager.X.index == "ITC4"][selected_features]

    X_test_selected = training_manager.X_test[selected_features]
    custom_X_test = pd.concat([X_test_selected, lombardy])


    cols_dict = dict()
    preds = training_manager.trained_model.predict(custom_X_test)

    cols_dict["predictions"] = preds
    cols_dict["true_class"] = pd.concat([training_manager.y_test, training_manager.y[training_manager.y.index == "ITC4"]]) 
   
    out_df = pd.DataFrame(cols_dict)

    RESULT_DIR = "predictions_dataset"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    
    unique_id_string = f"_{training_manager.clf_type}_from_{training_manager.start_date}_to_{training_manager.end_date}"  

    
    
    pd.concat([custom_X_test, out_df], axis=1).to_csv(os.path.join(RESULT_DIR, "boxplot_analysis" + unique_id_string +".csv"))


  
def plot_lombardy_vs_all(training_manager,
                         figsize=(10,8)):
    
    RESULT_DIR = "results/lombardy_vs_all"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    for column in training_manager.X.columns:
        plt.figure(figsize=(10,5) )
        plt.hist(training_manager.X[column].tolist(),bins=100)# [0 for _ in range(len(tm.df.Covid_Cases_Density.tolist()))])
        plt.axvline(training_manager.X[training_manager.X.index=="ITC4"][column].tolist()[0], c="red" )
        plt.ylabel("norm " + column)
        
        plt.savefig(os.path.join(RESULT_DIR,"standardized_" + column) )


def plot_class_distribution(training_manager,
                           figsize=(10,8)):
    plt.figure(figsize=figsize) 

    order_mapping_binary = {"below_than_avg":0, "higher_than_avg":1}
    order_mapping_multi = {"low":0, "medium":1, "high":2}
    order_mapping_extremes = {"low":0,  "high":1}

    mapper = { "binary": order_mapping_binary,
              "multi":order_mapping_multi,
              "extremes":order_mapping_extremes}
    
    
    

    current_mapper = mapper[training_manager.clf_type]

    counter = Counter(training_manager.y)
    sorted_counter = dict(sorted(counter.items(), key=lambda x: current_mapper[x[0]] ))

 
    plt.bar(x=sorted_counter.keys(), height=sorted_counter.values(), width=0.3)
    
    unique_id_string = f"_{training_manager.clf_type}_from_{training_manager.start_date}_to_{training_manager.end_date}"  

    
    RESULT_DIR = "results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    plt.savefig(os.path.join(RESULT_DIR, "class_distribution" + unique_id_string) )
    
    
    
def plot_sample_nans(training_manager,
                         figsize=(10,8)):
    plt.figure(figsize=figsize)
    plt.hist(training_manager.df.isna().sum(axis=1)/len(training_manager.df))
    plt.ylabel("frequency")
    plt.xlabel("NAN values % for each sample")
    
    RESULT_DIR = "results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    plt.savefig(os.path.join(RESULT_DIR, "sample_nans") )
    
    
def plot_feature_nans(training_manager,
                         figsize=(10,8)):
    plt.figure(figsize=figsize )
    plt.hist(training_manager.df.isna().sum(axis=0)/len(training_manager.df), bins=20)
    plt.ylabel("frequency")
    plt.xlabel("NAN values % for each feature")
    
    RESULT_DIR = "results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    plt.savefig(os.path.join(RESULT_DIR, "feature_nans"))
    
    
    
    
def plot_clustering_quality(training_manager,
                         figsize=(10,8)):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from matplotlib import pyplot as plt

    X =  training_manager.df["Covid_Cases_Density"].values.reshape(-1,1)

    X = np.where(np.isnan(X), X[ np.isnan(X)==False].mean(axis=0), X)

    distorsions = []
    scores = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.predict(X))
        scores.append(score)
        distorsions.append(kmeans.inertia_)

    RESULT_DIR = "results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
        
    unique_id_string = f"_{training_manager.clf_type}_from_{training_manager.start_date}_to_{training_manager.end_date}"  
     
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
    
    

    
def plot_clustering_subdivision(training_manager,
                         figsize=(10,8)):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from matplotlib import pyplot as plt

    X =  training_manager.df["Covid_Cases_Density"].values.reshape(-1,1)

    X = np.where(np.isnan(X), X[ np.isnan(X)==False].mean(axis=0), X)

    distorsions = []
    scores = []
    
    if training_manager.clf_type == "binary":
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

    unique_id_string = f"_{training_manager.clf_type}_from_{training_manager.start_date}_to_{training_manager.end_date}"  

    plt.figure(figsize=figsize )
    
    tag = "cases"
    if training_manager.response =="cumulativedeceased":
        tag = "deaths"
    elif training_manager.response =="cumulativerecovered":
        tag = "recovered"
    for mapper, color in zip(sorted_mapping , colors) :
        c_idx = mapper[0]
        
        if n_clusters==2:
            n_bins=200
        else:
            n_bins=400
        plt.hist(X[c_labels==c_idx],  color=color, bins=n_bins )
        plt.xlabel(f"covid {tag} density")
        plt.title(f"frequency of the density of {tag}")
        
    RESULT_DIR = "results"
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)        
    plt.savefig(os.path.join(RESULT_DIR,"clustering" + unique_id_string))

        
        
def train_test(model,
    parameter_grid=None,
    prob_type = "classification",
    clf_type = "binary",
    response="cumulativepositive",
    start_date="2020-01-20", #"DATE MUST BE FORMATTED AS "YEAR-MONTH-DAY"
    end_date="2020-08-20",
    k_smote = 9,
    test_size = 0.3,
               leave_one_out = True,
    nuts2_to_remove =[], # alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']
               
    clust_method='ML',
    sqlite_file_path="covid_at_lombardy.sqlite", 
    plot_roc=True, 
    f1_score_on_minority=False,
              manager_n_splits=10,
                cv_n_splits=10,
              cv_scoring=None):
    
    p_types = ["regression", "classification"]
    clf_types = [ "multi", "extremes", "binary"]
        
    assert prob_type in p_types, f'Error, found problem type = "{prob_type}" but available are {p_types}'
    assert clf_type in clf_types, f'Error, found classifier type = "{clf_type}" but available are {clf_types}'
    
    
    if prob_type == "classification":
        metric=f1_score
    else:
        metric=mean_absolute_error
       
    if parameter_grid is not None:
        if cv_scoring is not None:
            print(cv_scoring)
            model = GridSearchCV(model, param_grid=parameter_grid, cv=cv_n_splits, scoring=cv_scoring)
        else:
            model = GridSearchCV(model, param_grid=parameter_grid, cv=cv_n_splits)

        
    if clf_type == "binary":
        n_clusters=2
    else:
        # ignored if prob_type=="regression"
        n_clusters=3
        

    tm = TrainingManager(sqlite_file_path=sqlite_file_path,
                        clf_type=clf_type,
                         prob_type=prob_type,
                         response=response,
                         n_clusters=n_clusters,
                         nuts2_to_remove=nuts2_to_remove,
                        #wave=wave,
                        start_date=start_date,
                        end_date=end_date,
                         leave_one_out=leave_one_out,
                        plot_roc=plot_roc,
                         clust_method=clust_method,
                        f1_score_on_minority=f1_score_on_minority) 
    training_metrics, history = tm.train_test( model,
                                     metric=metric,
                                     n_splits=manager_n_splits,
                                     test_size=test_size,
                                    k_smote=k_smote)
    return training_metrics, history, tm
def get_selected_features(training_manager):
    selected_features = training_manager.X_train_full.columns[training_manager.pipeline["feature_selection"].get_support()]
    return selected_features
def print_metrics(training_metrics):
    print("Training metrics: ", training_metrics["Train"],training_metrics["Validation"], training_metrics["Test"] )
def display_roc_and_confusion_matrix(training_manager, training_metrics, figsize=(10,8)):
    
        assert training_manager.prob_type == "classification", f"Error: expected problem type 'classification' but found {training_manager.prob_type}"
        
        unique_id_string = f"_{training_manager.clf_type}_from_{training_manager.start_date}_to_{training_manager.end_date}"  
        
        RESULT_DIR = os.path.join("results", "model_results")
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)    
            
            
            
        
        if training_manager.plot_roc and (training_manager.n_clusters == 2 or training_manager.clf_type=="extremes"):
           
            plot_curve(training_metrics["tprs"],training_metrics["aucs"],\
                   training_metrics["mean_fpr"], training_metrics["ax"])
            training_metrics["fig"].savefig(os.path.join(RESULT_DIR, "roc_curve_" + unique_id_string))

        avg_cm = np.zeros(shape=training_metrics["cm"][0].shape)
        for cm in training_metrics["cm"]:
            avg_cm += cm
        avg_cm = avg_cm /len(training_metrics["cm"])   
        fig, ax = plt.subplots(figsize=figsize)
        avg_cm = np.around(avg_cm)
        ConfusionMatrixDisplay(avg_cm,
                               display_labels=training_manager.clusterid_to_categorical_mapping.values())\
                                .plot(ax=ax,
                                     cmap=plt.cm.Blues)
        
        fig.savefig(os.path.join(RESULT_DIR, "confusion_matrix_" + unique_id_string))
        return avg_cm