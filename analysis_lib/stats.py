import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay, confusion_matrix,\
    mean_absolute_error,plot_roc_curve, r2_score, f1_score, \
    accuracy_score,auc, roc_auc_score, roc_curve, RocCurveDisplay,\
    recall_score
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import dataframe_image as dfi
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# note: this has been suppressed after testing that pandas achieves the desired behavior
# it is documented online that such warning can wrongly appear sometimes, and this is the case
pd.options.mode.chained_assignment = None



        
class StatsLogger:
    def __init__(self, dest_dir_path):
        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)
            
        self.dest_dir_path = dest_dir_path
        
    
    def log_model_statistics(self, datamanager, 
                        model_init_fn,
                         P_VAL_THRESHOLD=0.05,
                         N_RUNS=20,
                            k_smote = 6,):


        
        per100k_dataset_path = datamanager.get_data_paths()["per100k_dataset"]
        
        violinplots_path = os.path.join(self.dest_dir_path, "violinplots")
        violinplots_sidebyside_path = os.path.join(self.dest_dir_path, "violinplots_sidebyside")
        fscores_path = os.path.join(self.dest_dir_path, "fscores")
        
        save_per_feature_distrib(datamanager, per100k_dataset_path, 
                                 dest_dir_path=violinplots_path)
        get_side_by_side_plots(datamanager, per100k_dataset_path, 
                               dest_dir_path=violinplots_sidebyside_path)
        f_test_features(datamanager, per100k_dataset_path, 
                        #feature_importance_dataset_file_path, 
                        THRESHOLD=P_VAL_THRESHOLD, 
                        dest_dir_path=fscores_path)    
            
    def get_experiment_stats(self, results_list):
        return get_experiment_stats(results_list, "models" , self.dest_dir_path)

            
          

def get_gridsearch_json(results_list):
    info_dict = dict()
    for dictionary in results_list:
        model_name = dictionary['model_name']
        cv_dict = dictionary['gs'].cv_results_
        updated_cv_dict = get_update( cv_dict)
        info_dict[model_name] = updated_cv_dict
    return info_dict
        
def get_update(dictionary):
    new_dict = dict()
    new_dict['params'] = dictionary['params']
    new_dict['mean_val_score'] = dictionary['mean_test_score'].tolist()
    new_dict['std_val_score'] = dictionary['std_test_score'].tolist()
    new_dict['rank_val_score'] = dictionary['rank_test_score'].tolist()
    return new_dict


def parse_gridsearch_to_csv(results_list, model_dir, dest_dir_path="./"):
    data = get_gridsearch_json(results_list)

    models_dict = dict()
    for model_name, model_cv_data in data.items():

        column_wise_data = dict()
        models_dict[model_name] = column_wise_data


        params = model_cv_data["params"]
        mean_test_scores = model_cv_data["mean_val_score"]
        std_test_scores = model_cv_data["std_val_score"]
        rank_test_scores = model_cv_data["rank_val_score"]

        column_wise_data["mean_val_score"] = model_cv_data["mean_val_score"]
        column_wise_data["std_val_score"] = model_cv_data["std_val_score"]
        column_wise_data["rank_val_score"] = model_cv_data["rank_val_score"]
    

        for param_set in params:
            for k,v in param_set.items():
                if k not in column_wise_data:
                    column_wise_data[k] = []

                column_wise_data[k].append(v)

    dir_path = os.path.join(dest_dir_path, model_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for model_name,  model_cv_data in models_dict.items():
        df = pd.DataFrame(model_cv_data)
        results_path = os.path.join(dir_path, model_name + ".csv")

        df.to_csv(results_path)
        



def get_files(dirpath):
    l = dict()
    for filename in os.listdir(dirpath):
        
        filepath = os.path.join(dirpath, filename)
        df = pd.read_csv(filepath, index_col=0)
        
        pos = filename.rindex(".")
        name = filename[:pos]
        l[name] = df
    return l


def get_best_model(wave_dirpath):
    dataframes_dict = get_files(wave_dirpath)
    l = []
    for model_name, df in dataframes_dict.items():
        best_config = df.sort_values("rank_val_score").iloc[0]  
        l.append( (model_name,best_config))    
        
    best_model_name = None
    best_model = None
    score = 0
    for model_name, best_config in l:
        if best_config["mean_val_score"] >= score:
            best_model_name = model_name
            score = best_config["mean_val_score"]
            best_model = best_config
        
    return best_model_name, best_model, score


def get_experiment_stats(gs_results_list, model_dir, dest_dir_path="./"):
    parse_gridsearch_to_csv(gs_results_list, model_dir, dest_dir_path=dest_dir_path)
    
    path = os.path.join(dest_dir_path, model_dir)
  
    model_name, model_cfg, metric = get_best_model(path)
    model_dataframes = get_files(path)
    
    
    return {'model_name':model_name, 
            'cfg':model_cfg, 
            'val_metric':metric,
           'stats_dataframes':model_dataframes}

            
            
            
            
            
            

def save_per_feature_distrib(datamanager, dataset_file_path, dest_dir_path="./violinplots"):
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)
    df = pd.read_csv(dataset_file_path)\
            .set_index("NUTS")
    #.drop(index=["FI20","ES64", "FRY5", "MT00", "LU00","ES63","BG32", "BG42"
    #                                          ])

    # FRY5 removed for anomalous number of vehicles per 100k abh, (600k, all others are below 200k)
    # FRY5 removed for anomalous utilised agricultural area 585005 while all others are below 200k
    # FI20 removed for anomalous number of hospital discharges (106,858 when all other regions are below 20k)
    # ES64 removed fro anomalous number of hospital discharges (over 50k  while all other regions are lower than 10k)
    # ES64 AND FI20 removed for anomalous number of health personnel 13650 and 29570 respectively (all other regions are below 5k)


    # ES63 removed for anomalous number of air passengers (around 20k when all others are below 10k)
    # MALTA AND LUXEMBOURG REMOVED FOR ANOMALOUS NUMBER OF HOURS WORKED per 100k
    # (LU00   250,456.747)
    # (MT00   304,749.211)
    # BG32,BG42 anomalous number of longterm care beds > 3000 per100k while all others are below that threshold
    #BG32   5,128.060
    #BG42   6,320.290


    population_dependent_features = datamanager.population_dependent_features

    TRUE_CLASS = "covid_severity"
    classes =  df[TRUE_CLASS].unique()


    columns_to_drop = ["covid_severity","covid_density_by100k", "Covid_Cases"]
    df_reduced = df.drop(columns=columns_to_drop)


    for column in df_reduced.columns:

        #print(column)
        groups_dict = dict()
        labels_pos = dict()
        groups_labels = []
        groups_data = []
        group_mapper = dict()

        i = 0
        for class_ in classes:
            groups_dict[column] = df[df[TRUE_CLASS] == class_ ][column].values

            groups_labels.append(class_)
            groups_data.append(df[df[TRUE_CLASS] == class_ ][column].values)

            group_mapper[class_] = df[df[TRUE_CLASS] == class_ ][column].values

            labels_pos[class_] = i
            i += 1

     
        plt.figure(figsize=(10,8))
       

        addition = column
        if column in population_dependent_features:
            addition = column + " per 100k inhabitants"
        plt.title(addition)

        order_mapping_binary = {"below_than_avg":0, "higher_than_avg":1}
        order_mapping_multi = {"low":0, "medium":1, "high":2}
        order_mapping_extremes = {"low":0,  "high":1}
        mapper = { "binary": order_mapping_binary,
                  "multi":order_mapping_multi,
                  "extremes":order_mapping_extremes}

        current_mapper = mapper["binary"]

        sorted_mapper = dict(sorted(group_mapper.items(), key=lambda x: current_mapper[x[0]] ))


        data = list(sorted_mapper.values())
        positions = [pos+1 for pos in range(len(sorted_mapper.keys()))]


        ax = plt.subplot(111)

        plt.violinplot(data, positions)

        ax.set_xticks(positions)
        ax.set_xticklabels(sorted_mapper.keys())

        value = df[df.index == "ITC4" ][column].values.tolist()[0]
      
        dot, = plt.plot(2,
                value, c='r', marker="o")
        plt.legend( [dot], ["lombardy"] )

        plt.savefig(os.path.join(dest_dir_path ,column) )
        
        plt.close()
        
    print(f"saved violin plots to {dest_dir_path}")
def get_side_by_side_plots(datamanager, dataset_file_path, dest_dir_path="./violinplots_sidebyside"):
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)
    df = pd.read_csv(dataset_file_path)\
            .set_index("NUTS")


    population_dependent_features = datamanager.population_dependent_features


    TRUE_CLASS = "covid_severity"
    classes =  df[TRUE_CLASS].unique()


    columns_to_drop = ["covid_severity","covid_density_by100k", "Covid_Cases"]
    df_reduced = df.drop(columns=columns_to_drop)
    
    analysis_df = df[df_reduced.columns.tolist()+ [TRUE_CLASS] ]
    analysis_df["representation"] = pd.Series( ["" for _ in range(len(analysis_df))], index=analysis_df.index)
    
    
    for column in df_reduced.columns:
        plt.figure(figsize=(10,7))
        addition = column
        if column in population_dependent_features:
            addition = column + " per 100k inhabitants"
        plt.title(addition)

        sns.violinplot(x="representation", y=column, data=analysis_df,hue=TRUE_CLASS,
                       hue_order=["below_than_avg", "higher_than_avg"], palette="muted",
                       cut=0,split=True)
        value = df[df.index == "ITC4" ][column].values.tolist()[0]
        
        dot, = plt.plot(0,
                value, c='r', marker="o")
        plt.legend( [dot], ["lombardy"] )
        plt.savefig(os.path.join(dest_dir_path ,column) )
        plt.close()
    print(f"saved violin plots side by side to {dest_dir_path}")


def f_test_features(datamanager, dataset_path, #feature_importance_dataset, 
                    THRESHOLD=0.05, dest_dir_path="./fscores"):
    pd.options.display.float_format = '{:,.3f}'.format

    if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)
    df = pd.read_csv(dataset_path)\
                .set_index("NUTS")

    columns_to_drop = ["covid_severity","covid_density_by100k", "Covid_Cases"]

    X = df[df.drop(columns=columns_to_drop).columns]
    y = df["covid_severity"]
    fstat, pvalue = f_classif(X,y)



    #importances_df = pd.read_csv(feature_importance_dataset)

    fstat_df = pd.DataFrame({"feature name": X.columns  ,"pvalue":pvalue,"F-statistic" : fstat }  ).set_index("feature name").sort_values(by="pvalue")

    #importances_df["avg importance (in % of total)"] = importances_df["avg importance"] / importances_df["avg importance"].sum()  *100
    #importances_df.drop(columns="avg importance", inplace=True)


    #importances_df["number of selections"].astype(np.uint)
    #result = importances_df.merge( fstat_df, on="feature name", how="outer").set_index("feature name")


    #pct_importances_dict = result["avg importance (in % of total)"].to_dict()

    #features_list = []
    #pct_importances = []
    #position = []

    #i = 1
    #for item in sorted(list(pct_importances_dict.items()), key=lambda x: x[1], reverse=True) :
    #    feature, pct_importance = item
    #    features_list.append(feature)
    #    #pct_importances.append(pct_importance)
    #    position.append(i)
    #    i += 1
    #ranking_df =pd.DataFrame( {"feature name": features_list, 
    #              #"feature importance ranking": position
    #                          }).set_index("feature name")
    #ranked_result_df = ranking_df.merge(result, on="feature name",how="outer")

    pd.set_option('display.precision',3)
    def style_negative(v, props=''):

        if pd.isna(v):
            return None
        return props if v < THRESHOLD else None

    def map_fn(v):
        opacity_val = 'opacity: 100%;'

        if pd.isna(v):
            return None

        if (v < 0.05):
            return opacity_val
        return None

    #ranked_result_df["number of selections"] = ranked_result_df["number of selections"].astype('Int64')
    #ranked_result_df.\
    s2 = fstat_df.sort_values(by='pvalue').\
            style.applymap(style_negative, props='color:red;',subset="pvalue")\
                  .applymap(map_fn, subset=["pvalue"])
    
    fscores_data_path = os.path.join(dest_dir_path, "fscores.csv")
    fscores_img_path = os.path.join(dest_dir_path, "fscores.png")
    
    fstat_df.to_csv(fscores_data_path)
    dfi.export(s2,fscores_img_path)
    print(f"saved fscores to {dest_dir_path}")
    return s2
    