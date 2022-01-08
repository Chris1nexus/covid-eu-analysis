import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold,LeaveOneOut, cross_val_score,StratifiedKFold
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay, confusion_matrix, mean_absolute_error,plot_roc_curve, r2_score, f1_score, accuracy_score,auc, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

import imblearn
from imblearn.over_sampling import SMOTE

import os
from collections import Counter
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestRegressor
import xgboost as xgb

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

from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib.cm

import copy

import pandas as pd
import numpy as np
import scipy as sp
import sqlite3


def plot_curve(tprs,aucs, mean_fpr, ax):
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic")
    ax.legend(loc="lower right")
    plt.show()

def rf_init():
    return RandomForestRegressor()

def percent_nans(column, dataframe):
    return len(dataframe[dataframe[column].isna()])/len(dataframe)



        

class DataManager(object):
    """
    ###target can be (lethality, survival, cum_positive_density, cum_deceased_density, cum_recovered_density)
    new target can be: lethality_rate (cumulated_deceased/cumulated_positives) , 
                    survival_rate (cumulated_recovered/cumulated_positives)
    """
    def __init__(self, sqlite_file_path="./covid_at_lombardy.sqlite", 
                 features=None,
                 prob_type="classification",
                 clf_type="multi", 
                 response="cumulativepositive",
                 n_clusters=3,
                 nuts2_to_remove=[],
                 start_date="2020-01-20",
                 end_date="2020-08-20",
                 clust_method='ML'):
        

        p_types = ["regression", "classification"]
        clf_types = [ "multi", "extremes", "binary"]
        avbl_targets = ["cumulativepositive", "cumulativedeceased","cumulativerecovered"]
        assert prob_type in p_types, f'Error, found problem type = "{prob_type}" but available are {p_types}'
        assert clf_type in clf_types, f'Error, found classifier type = "{clf_type}" but available are {clf_types}'
        assert response in avbl_targets, f'Error, found response type = "{response}" but available are {avbl_targets}'
        
        self.response = response
        self.prob_type = prob_type
        self.clf_type = clf_type
        self.n_clusters = n_clusters
        self.start_date = start_date
        

        self.end_date = end_date

        self.population_normalized = False
        self.clust_method = clust_method
        
        
        self.dest_folder = self.get_experiment_identifier()
        if not os.path.exists(self.dest_folder):
            os.makedirs(self.dest_folder)
        
        
        self.data_paths = dict()
        
     
        self.population_dependent_features = [
                "air_passengers.tsv",
                "nama_10r_2gdp.tsv",
                "farm_labour_force.tsv",
                "deaths.tsv",
                "hospital_discharges_resp_diseases_j00_to_j99_nuts2.tsv",
                "stock_of_vehicles_by_category_and_nuts2.tsv",
                "employment_thousand_hours_worked_nuts2.tsv",
                "students_enrolled_in_tertiary_education_by_education_level_programme_orientation_sex_and_nuts2.tsv",
                "health_personnel_by_nuts2.tsv",
                "pupils_and_students_enrolled_by_sex_age_and_nuts2.tsv",
                  "utilised_agricultural_area.tsv",]
        
        self.pct_features = ["available_hospital_beds_nuts2.tsv",
                              "participation_in_education_and_training.tsv",
            "causes_of_death_crude_death_rate_3year_average_by_nuts2.tsv",
            "longterm_care_beds_per_hundred_thousand_nuts2.tsv",
            "real_growth_rate_of_regional_gross_value_added_GVA_at_basic_prices_by_nuts2.tsv",
            "pop_density.tsv",
            "unemployment_rate_nuts2.tsv",
                             "early_leavers_from_education_and_training_by_sex_percentage_nuts2.tsv",
                             'young_people_neither_in_employment_nor_in_education_and_training_by_sex_NEET_RATE_nuts2.tsv'
        ]
        
        self.shorten_mapper = {'air_passengers': 'air passengers' ,
                        'available_hospital_beds_nuts2' : 'hospital beds',
                        'causes_of_death_crude_death_rate_3year_average_by_nuts2' :'crude death rate',
                        'compensation_of_employees_by_nuts2' : 'compensation of employees',
                        'deaths' : 'deaths',
                        'early_leavers_from_education_and_training_by_sex_percentage_nuts2' : 'pct leavers from education',
                        'employment_thousand_hours_worked_nuts2' : 'thousand hours worked',
                        'farm_labour_force' : 'farm labour force',
                        'health_personnel_by_nuts2' : 'health personnel',
                        'hospital_discharges_resp_diseases_j00_to_j99_nuts2' : 'hosp discharges resp diseases',
                        'life_expectancy' : 'life expectancy',
                        'longterm_care_beds_per_hundred_thousand_nuts2' : 'longterm care beds',
                        'nama_10r_2gdp' : 'GDP',
                        'participation_in_education_and_training' :'education and training' ,
                        'pop_density' : 'pop density',
                        'population_nuts2' : 'population',
                        'pupils_and_students_enrolled_by_sex_age_and_nuts2' : 'students enrolled',
                        'real_growth_rate_of_regional_gross_value_added_GVA_at_basic_prices_by_nuts2' : 'regional GWA',
                        'stock_of_vehicles_by_category_and_nuts2' : 'stock of vehicles',
                        'students_enrolled_in_tertiary_education_by_education_level_programme_orientation_sex_and_nuts2':'students tertiary ed',
                        'unemployment_rate_nuts2' : 'unemployment rate',
                        'utilised_agricultural_area':'utilised agricultural area',
                        'young_people_neither_in_employment_nor_in_education_and_training_by_sex_NEET_RATE_nuts2':'NEET rate'}

        
        
        # remove file extension from provided feature names
        self.population_dependent_features = [col.replace(".tsv","") for col in self.population_dependent_features]
        self.pct_features = [col.replace(".tsv","") for col in self.pct_features]
        
        # replace feature names with readable ones
        self.compensation =  self.shorten_mapper["compensation_of_employees_by_nuts2"]
        self.life_expectancy = self.shorten_mapper["life_expectancy"]
        self.population_dependent_features = [self.shorten_mapper[feat] for feat in self.population_dependent_features]
        self.pct_features = [self.shorten_mapper[feat] for feat in self.pct_features]
        
        
        self.covariates = self.population_dependent_features + \
                            self.pct_features +\
                            [self.compensation]+\
                        [self.life_expectancy] 
        
        self.sqlite_file_path = sqlite_file_path
        cnx = sqlite3.connect(sqlite_file_path)
        temp_X = pd.read_sql_query("SELECT * FROM covariates", cnx).pivot(index='NUTS', columns='Covariate', values='Value')
        temp_X = temp_X.rename(columns=self.shorten_mapper)
  
        temp_Y = pd.read_sql_query(f"select nuts, CAST(max({response}) as integer) as \"Covid_Cases\" from covid_cases where date between '{start_date}' and '{end_date}' group by nuts;", cnx)\
            .set_index("NUTS")
        
        # take the min to get the initial condition in the given interval (this minimum can be at different dates depending on when a given regin started logging their data)
        initial_condition = pd.read_sql_query(f"select nuts, CAST(min({response}) as integer) as \"Covid_Cases\" from covid_cases where date between '{start_date}' and '{end_date}' group by nuts;", cnx)\
            .set_index("NUTS")

        # SETUP THE SELECTED ROWS FOR Y AND X  #  has already been NUTS indexed
        self.temp_Y = (temp_Y - initial_condition)\
                .drop(nuts2_to_remove)
        
        
       
        # ignore errors meaning that if the item is not present, the 'error' is not raised
        # although it is  not an errror, just an absence of one of the values of the list in the index of the dataframe
        df = pd.merge(temp_X, self.temp_Y, left_on="NUTS",
                      right_index=True, right_on="NUTS",
                     )\
                .drop(nuts2_to_remove, errors="ignore")

        ml_dataset_path = os.path.join(self.dest_folder, f"ml_dataset_from_{start_date}_to_{end_date}.tsv")
        self.data_paths["ml_dataset"] = ml_dataset_path
        df.to_csv(ml_dataset_path)
        
        df["Covid_Cases_Density"] = df["Covid_Cases"] / df["population"]
        
        
        
        
        
        self.target_variable = "Covid_Cases_Density"
        
        
        if features is not None:
            X = df[features]
        else:
            X = df[self.covariates]
        self.df = df

        
        # all samples are selected initially
        selected_samples_series = pd.Series(data=[True for _ in range(len(df)) ], index=df.index )
 
        if self.prob_type == "regression":
            Y = df[self.target_variable] *100
        else:
            #"classification"
            res_response_target_name, res_response_series, subset_selected_samples_series = self.__clusterize_target( response_variable="Covid_Cases_Density",
                                                                                                                    n_clusters=n_clusters,
                                                                                                                    clust_method=self.clust_method)
            Y = pd.Series(res_response_series, name ="covid_severity")
            selected_samples_series = selected_samples_series & subset_selected_samples_series
            

    

        self.population_size_series = self.df["population"]
  
        self.selected_samples_series = selected_samples_series & (self.population_size_series.isnull() == False)
        self.X = X[self.selected_samples_series]
        self.y = Y[self.selected_samples_series]
        
        self.__prepare_datasets()

        
    def get_feature_mapper(self):
        return self.shorten_mapper
    
    def get_dataset(self):
        # todo: allow response variable selection (categorical{2,3,...} clusters, regression, )
        #       instead of selection of the problem type beforehand
        return self.X, self.y
    
    def get_train_test_split(self, test_size=0.3, shuffle=True):

        if self.prob_type =="classification":
            X_train_full, X_test, y_train_full, y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=shuffle, stratify=self.y)
        else:
            X_train_full, X_test, y_train_full, y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=shuffle)
            
            
        return X_train_full, X_test, y_train_full, y_test        
        
        
        
  
    def __clusterize_target(self, response_variable="Covid_Cases_Density",n_clusters=3, clust_method='ML'):

            X = self.df[response_variable]

            X = X.to_numpy().reshape(-1,1)
            X = np.where(np.isnan(X), X[ np.isnan(X)==False].mean(axis=0), X)
            
            if self.clf_type == "binary":
                cluster_labels = ["below_than_avg", "higher_than_avg"]
                n_clusters=2
            elif self.clf_type=="multi":
                if n_clusters == 3:
                    cluster_labels = ["low", "medium", "high"]
                else:
                    cluster_labels = [str(i) for i in range(n_clusters) ]
            else:
                cluster_labels = ["low", "medium", "high"]
                
            if clust_method == 'det':
                def evenly_spaced_clusters(n_clusters, N):
                    linspaced = np.linspace(0,1, n_clusters+1)*N
                    int_linspace = [int(val) for val in (linspaced)[1:] ]

                    i = 0
                    l = []
                    for j in range(N):

                        l.append(i)
                        if j >= int_linspace[i]:
                             i += 1
                    return l
                class ClustWrapper:
                    def __init__(self, labels):
                        self.labels_ = labels
            
                N = len(X)
                
                clusters = evenly_spaced_clusters(n_clusters, N )
                sorted_indices = np.argsort(X.flatten())
                cluster_to_samples = [0 for _ in range(N)  ]

                for pos, clust in zip(sorted_indices, clusters):
                    cluster_to_samples[pos] = clust
                    
                cluster = ClustWrapper(cluster_to_samples)
            else:
                kmeans = KMeans(n_clusters=n_clusters)
                cluster = kmeans.fit(X)

            from collections import Counter
            counter = Counter(cluster.labels_)
            clust_df = self.df.copy()
            clust_df["cluster_id"] = cluster.labels_
            # obtain the mapping from cluster id to mean feature value
            # this allows to sort the cluster ids from lowest to highest severity
            # since cluster ids are NOT necessarily assigned in increasing order of the underlying feature
            sorted_clust_id_by_mean_dict = clust_df.groupby(by="cluster_id")[response_variable].mean().sort_values().to_dict()

    
            clusterid_to_categorical_mapping = dict()
            categorical_to_clusterid_mapping = dict()
            
            self.clusterid_to_categorical_mapping = clusterid_to_categorical_mapping
            self.categorical_to_clusterid_mapping = categorical_to_clusterid_mapping 
            self.sorted_clust_id_by_mean_dict = sorted_clust_id_by_mean_dict
            
            mapper_old_to_ordered_clusterid = dict()
            ordinal_clusterid_to_label = dict()
            ordinal_clusterid_to_mean = dict()
            ordinal_label_to_ordered_clusterid = dict()
            new_index = 0
            for item, label in zip(sorted_clust_id_by_mean_dict.items(), cluster_labels):
                clusterid, feature_mean = item
                clusterid_to_categorical_mapping[clusterid] = label
                categorical_to_clusterid_mapping[label] = clusterid
                
                mapper_old_to_ordered_clusterid[clusterid] = new_index
                ordinal_clusterid_to_label[new_index] = label
                ordinal_clusterid_to_mean[new_index] = feature_mean
                ordinal_label_to_ordered_clusterid[label] = new_index
                new_index += 1
            
            self.clusterid_to_categorical_mapping = ordinal_clusterid_to_label
            self.categorical_to_clusterid_mapping = ordinal_label_to_ordered_clusterid 
            self.sorted_clust_id_by_mean_dict = ordinal_clusterid_to_mean
            
            self.mapper_old_to_ordered_clusterid = mapper_old_to_ordered_clusterid
            

            
            clust_df["cluster_id"] = clust_df["cluster_id"].map(mapper_old_to_ordered_clusterid) 
            clust_df[response_variable + "_severity"] = clust_df["cluster_id"].map(ordinal_clusterid_to_label)

            res_response_target_name = response_variable + "_severity"

            
            
            res_response_series = clust_df["cluster_id"].map(ordinal_clusterid_to_label)
            if self.clf_type == "extremes":
                selected_samples_series = res_response_series != "medium"
               
                med_clusterid = categorical_to_clusterid_mapping["medium"]
                del clusterid_to_categorical_mapping[med_clusterid]
                del categorical_to_clusterid_mapping["medium"]
            else:
                # select all
                selected_samples_series = res_response_series == res_response_series
                

            return res_response_target_name, res_response_series, selected_samples_series
    
    
    
    
    def __remove_population_dependency(self):
        if not self.population_normalized:
            for column in self.population_dependent_features:
                self.X[column] = self.X[column] / self.population_size_series
            
            # treated aside as a special case (since this number by100k inhabitants is of difficult interpretation)
            self.X[self.compensation] = self.X[self.compensation] / self.population_size_series
            
            self.population_normalized = True
    def __fillnans(self):
        
       
        X_copy = self.X.copy()
        
        
        X_copy["nuts0"] = X_copy.index.map(lambda x : x[:2]  )

        X_national_means = X_copy.groupby(by="nuts0").mean()

        nuts_aggregator_helper = pd.DataFrame(index=X_copy.index)
        nuts_aggregator_helper["nuts0"] = nuts_aggregator_helper.index.map(lambda x : x[:2]  )
        nuts_aggregator_df = nuts_aggregator_helper.merge(X_national_means,
                                                          left_on="nuts0", 
                                                          right_on="nuts0",
                                                          right_index=True)
       
        # pandas pops up a settingcopywarning, but this block has been tested(also by direct observation of the resulting self.X,
        # before reaching the line: self.X = self.X.fillna(self.X.mean(axis=0)). So the values have been correctly filled with the correct mean) 
        # and produces the expected
        # filling of the nans by the national mean.
        for column in self.X.columns:
            self.X[column] = self.X[column].fillna(nuts_aggregator_df[column], axis=0)
        self.national_means = nuts_aggregator_df
        
        # if there are still nans, fill with the mean of the feature
        self.X = self.X.fillna(self.X.mean(axis=0))
    def __standardize(self):

        self.X_mean = self.X.mean()
        self.X_std = self.X.std()
        self.X = (self.X - self.X_mean)/self.X_std
        
        Z_coord = 4
        # clip outliers
        for column in self.X.columns:
                sigma = self.X_std[column]
                self.X[column] = self.X[column].clip( lower = -Z_coord, 
                                                    upper = Z_coord)
    

    def __prepare_datasets(self):
        
        self.__extract_datasets()
        self.__fillnans()
        self.__remove_population_dependency()
        self.__standardize()

        
        
        
    def __extract_datasets(self):
          
        if not os.path.exists(self.dest_folder):
            os.makedirs(self.dest_folder)
        X = self.X.copy()
        X_copy = self.X.copy()
        
     
        X_copy["nuts0"] = X_copy.index.map(lambda x : x[:2]  )
        X_national_means = X_copy.groupby(by="nuts0").mean()
        nuts_aggregator_helper = pd.DataFrame(index=X_copy.index)
        nuts_aggregator_helper["nuts0"] = nuts_aggregator_helper.index.map(lambda x : x[:2]  )
        nuts_aggregator_df = nuts_aggregator_helper.merge(X_national_means,
                                                          left_on="nuts0", 
                                                          right_on="nuts0",
                                                          right_index=True)
        for column in X.columns:
            X[column] = X[column].fillna(nuts_aggregator_df[column], axis=0)
        national_means = nuts_aggregator_df
        # if there are still nans, fill with the mean of the feature
        X = X.fillna(self.X.mean(axis=0))
        
        

        for column in self.population_dependent_features:
                X[column] = (X[column] / self.population_size_series)
            
        # treated aside as a special case (since this number by100k inhabitants is of difficult interpretation)
        X[self.compensation] = X[self.compensation] / self.population_size_series *10**5

            
        
        y_df = pd.DataFrame(self.y )
        # make dataset for visualization (features maintain the semantic meaning but have to still be standardized)
        other_features = set(X.columns) - set(self.population_dependent_features) 
        # de standardize, then multiply by 100k the population_dependent_features
        pop_dep_by100k = X[self.population_dependent_features].copy()*10**5
        others_df = X[other_features].copy()
        
        density_by100k= pd.Series(self.df["Covid_Cases_Density"][self.selected_samples_series].copy()*10**5,name="covid_density_by100k" )
        
        pop_dep_by100k = pop_dep_by100k.rename(columns={feat: feat + ' per100k'  for feat in pop_dep_by100k.columns })
        others_df = others_df.rename(columns={self.compensation: self.compensation+ ' per100k' })
        
        per100k_df = pd.concat([ pop_dep_by100k, 
                                            others_df, 
                                           y_df, 
                                            density_by100k,
                                            self.df["Covid_Cases"][self.selected_samples_series]  ], axis=1)
        
        per100k_df_path = os.path.join(self.dest_folder,f"per100kdataset_from_{self.start_date}_to_{self.end_date}.csv")
        self.data_paths["per100k_dataset"] = per100k_df_path
        per100k_df.to_csv(per100k_df_path)
        
        
        
        
        
        X_mean = X.mean()
        X_std = X.std()
        X = (X - X_mean)/X_std
        Z_coord = 4
        # clip outliers
        for column in X.columns:
                sigma = X_std[column]
                X[column] = X[column].clip( lower = -Z_coord, 
                                                    upper = Z_coord)

        
        y_df = pd.DataFrame(self.y )
        # make the dataset that contains features and targets after preprocessing, normalization and standardization
        total_df = pd.concat([X,
                                   y_df, self.df["Covid_Cases_Density"][self.selected_samples_series]  ], axis=1)
        
        total_df_path = os.path.join(self.dest_folder,f"standardized_dataset_from_{self.start_date}_to_{self.end_date}.csv")
        self.data_paths["standardized_dataset"] = total_df_path
        total_df.to_csv(total_df_path)
    
    
    def get_experiment_identifier(self):
        return f"from_{self.start_date}_to_{self.end_date}__{self.response}__nclust_{self.n_clusters}_with_{self.clust_method}"
    def get_period(self):
        return {"start_date": self.start_date,
               "end_date": self.end_date}
    def get_data_paths(self):
        return copy.deepcopy(self.data_paths)