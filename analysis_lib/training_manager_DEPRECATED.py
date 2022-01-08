
class TrainingManager(object):
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
                #wave="1",
                 start_date="2020-01-20",
                 end_date="2020-08-20",
                  leave_one_out=False,
                 clust_method='ML',
                plot_roc=True,
                f1_score_on_minority=False):
        
        #assert target in ["cum_positive_density", "cum_deceased_density", "cum_recovered_density"], 'Error, target must be in ["cum_positive_density", "cum_deceased_density", "cum_recovered_density"]'
        #assert target in ["lethality_rate", "survival_rate"], 'Error, target must be in ["lethality_rate", "survival_rate"]'
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
        #self.wave = wave
        self.start_date = start_date
        
 
        self.leave_one_out = leave_one_out 
        self.end_date = end_date
        self.plot_roc = plot_roc
        self.f1_score_on_minority = f1_score_on_minority
        self.population_normalized = False
        
        
     
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
        self.compensation =  "compensation_of_employees_by_nuts2"
        self.life_expectancy = "life_expectancy"
        self.clust_method = clust_method
        self.population_dependent_features = [col.replace(".tsv","") for col in self.population_dependent_features]
        self.pct_features = [col.replace(".tsv","") for col in self.pct_features]
        
        
        self.covariates = self.population_dependent_features + \
                            self.pct_features +\
                            [self.compensation]+\
                        [self.life_expectancy] 
        
        self.sqlite_file_path = sqlite_file_path
        cnx = sqlite3.connect(sqlite_file_path)
        temp_X = pd.read_sql_query("SELECT * FROM covariates", cnx).pivot(index='NUTS', columns='Covariate', values='Value')
        
        #if wave == "1":
        #    temp_Y = pd.read_sql_query("select nuts, CAST(max(cumulativepositive) as integer) as \"Covid_Cases\" from covid_cases where date between '2020-01-30' and '2020-08-20' group by nuts;", cnx)
        #else:
        #    temp_Y = pd.read_sql_query("select nuts, CAST(max(cumulativepositive) as integer) as \"Covid_Cases\" from covid_cases where date between '2020-08-20' and '2021-02-20' group by nuts;", cnx)
        temp_Y = pd.read_sql_query(f"select nuts, CAST(max({response}) as integer) as \"Covid_Cases\" from covid_cases where date between '{start_date}' and '{end_date}' group by nuts;", cnx)\
            .set_index("NUTS")
        
        # take the min to get the initial condition in the given interval (this minimum can be at different dates depending on when a given regin started logging their data)
        initial_condition = pd.read_sql_query(f"select nuts, CAST(min({response}) as integer) as \"Covid_Cases\" from covid_cases where date between '{start_date}' and '{end_date}' group by nuts;", cnx)\
            .set_index("NUTS")

        # SETUP THE SELECTED ROWS FOR Y AND X  #  has already been NUTS indexed
        self.temp_Y = (temp_Y - initial_condition)\
                .drop(nuts2_to_remove)
        
        
        
        #print(nuts2_to_remove)
        # ignore errors meaning that if the item is not present, the 'error' is not raised
        # although it is  not an errror, just an absence of a value
        df = pd.merge(temp_X, self.temp_Y, left_on="NUTS",
                      right_index=True, right_on="NUTS",
                     )\
                .drop(nuts2_to_remove, errors="ignore")
                    # #.set_index("NUTS")\ 
                    # if right_index=True, that is taken as index of the resulting dataframe, THus no need to call set_index on the output
        
  
        df.to_csv(f"ml_dataset_all_from_{start_date}_to_{end_date}.tsv")
        df["Covid_Cases_Density"] = df["Covid_Cases"] / df["population_nuts2"]
        
        
        
        
        
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
            

    

        self.population_size_series = self.df["population_nuts2"]
  
        self.selected_samples_series = selected_samples_series & (self.population_size_series.isnull() == False)
        self.X = X[self.selected_samples_series]
        self.y = Y[self.selected_samples_series]
        
        
        
        self.shuffle = True
        self.test_size = 0.3
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
                        
                #sorted_indices = np.argsort(X)
                N = len(X)
                
                clusters = evenly_spaced_clusters(n_clusters, N )
                sorted_indices = np.argsort(X.flatten())
                cluster_to_samples = [0 for _ in range(N)  ]
                #print(sorted_indices)
                #print(clusters)
                for pos, clust in zip(sorted_indices, clusters):
                    cluster_to_samples[pos] = clust
                #print(cluster_to_samples)
                
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

            #print(sorted_clust_id_by_mean_dict)
            
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
            
            """
            print("initial: ", sorted_clust_id_by_mean_dict)
            print("init clid to cat: ",clusterid_to_categorical_mapping)
            print("init cat to clid: ",categorical_to_clusterid_mapping)
            
            print("clustid to ord cl: ", mapper_old_to_ordered_clusterid)
            print("ord cl to lab: ", ordinal_clusterid_to_label)
            print("ord clid to mean: ", ordinal_clusterid_to_mean)
            print("ord lab to ord clid: ", ordinal_label_to_ordered_clusterid)
            """
            
            clust_df["cluster_id"] = clust_df["cluster_id"].map(mapper_old_to_ordered_clusterid) 
            clust_df[response_variable + "_severity"] = clust_df["cluster_id"].map(ordinal_clusterid_to_label)
            #clust_df[response_variable + "_severity"] = clust_df["cluster_id"].map(clusterid_to_categorical_mapping)

            res_response_target_name = response_variable + "_severity"
            #res_response_series = clust_df["cluster_id"].map(clusterid_to_categorical_mapping)
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
        #self.X[self.population_dependent_features] = self.X[self.population_dependent_features] / self.X["population_nuts2.tsv"]
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
       
        for column in self.X.columns:
            self.X[column] = self.X[column].fillna(nuts_aggregator_df[column], axis=0)
        self.national_means = nuts_aggregator_df
        
        # if there are still nans, fill with the mean of the feature
        self.X = self.X.fillna(self.X.mean(axis=0))
    def __standardize(self):
        # could standardize after splitting for technicalities of data leak of the dataset statistics
        #self.X_mean = self.X.mean(axis=0)
        #self.X_std = self.X.std(axis=0)
        #self.X = (self.X - self.X.mean(axis=0))/ self.X.std(axis=0)
        self.X_mean = self.X.mean()
        self.X_std = self.X.std()
        self.X = (self.X - self.X_mean)/self.X_std
        
        Z_coord = 4
        # clip outliers
        for column in self.X.columns:
                sigma = self.X_std[column]
                self.X[column] = self.X[column].clip( lower = -Z_coord, 
                                                    upper = Z_coord)
        
    def __train_test_split(self, test_size=0.3, shuffle=True):
        self.test_size = test_size
        self.shuffle = shuffle
    
        
        if self.prob_type =="classification":
            self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=shuffle, stratify=self.y)
        else:
            self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=shuffle)
        
    
    def __set_model(self, model):
        """
        pass as parameter a scikit-learn like model that supports the scikit fit, predict interfaces 
        """
        self.model = model
        # threshold for the feature selection is computed as the mean of the feature importances 
        # this can be read from the docs: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
        
        if self.prob_type == "regression":
            selector_helper = RandomForestRegressor()
        else:
            selector_helper = RandomForestClassifier()
        self.pipeline = Pipeline( steps= [('feature_selection', SelectFromModel(selector_helper )),
                                          ("model",model)#RandomForestRegressor())#xgb.XGBRegressor(n_estimators=200, max_depth=5)),
                                                     ])
    def __set_cv(self, n_splits=10):
        self.n_splits = n_splits
        if self.prob_type =="classification":
            
            if self.leave_one_out:
                self.kf = LeaveOneOut()
            else:
                self.kf = StratifiedKFold(n_splits=n_splits)
        else:
            self.kf = KFold(n_splits=n_splits)
        
    def extract_datasets(self):
          
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
                X[column] = X[column] / self.population_size_series
            
        # treated aside as a special case (since this number by100k inhabitants is of difficult interpretation)
        X[self.compensation] = X[self.compensation] / self.population_size_series

            
        
        y_df = pd.DataFrame(self.y )
        # make dataset for visualization (features maintain the semantic meaning but have to still be standardized)
        other_features = set(X.columns) - set(self.population_dependent_features) 
        # de standardize, then multiply by 100k the population_dependent_features
        pop_dep_by100k = X[self.population_dependent_features].copy()*10**5
        others_df = X[other_features].copy()
        
        density_by100k= pd.Series(self.df["Covid_Cases_Density"][self.selected_samples_series].copy()*10**5,name="covid_density_by100k" )
        interpretation_df = pd.concat([ pop_dep_by100k, 
                                            others_df, 
                                           y_df, 
                                            density_by100k,
                                            self.df["Covid_Cases"][self.selected_samples_series]  ], axis=1)
        interpretation_df.to_csv(f"visual_interpretation_dataset_from_{self.start_date}_to_{self.end_date}.csv")
        
        
        
        
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
        total_df.to_csv(f"standardized_dataset_from_{self.start_date}_to_{self.end_date}.csv")
        
  
        
    def train_test(self,model, metric, test_size=0.3, shuffle=True,n_splits=10, k_smote=None):
        self.__fillnans()
        self.__remove_population_dependency()
        
        y_df = pd.DataFrame(self.y )
        # make dataset for visualization (features maintain the semantic meaning but have to still be standardized)
        other_features = set(self.X.columns) - set(self.population_dependent_features) 
        # de standardize, then multiply by 100k the population_dependent_features
        pop_dep_by100k = self.X[self.population_dependent_features].copy()*10**5
        others_df = self.X[other_features].copy()
        
        density_by100k= pd.Series(self.df["Covid_Cases_Density"][self.selected_samples_series].copy()*10**5,name="covid_density_by100k" )
        self.interpretation_df = pd.concat([ pop_dep_by100k, 
                                            others_df, 
                                           y_df, 
                                            density_by100k,
                                            self.df["Covid_Cases"][self.selected_samples_series]  ], axis=1)
        self.interpretation_df.to_csv(f"visual_interpretation_dataset_from_{self.start_date}_to_{self.end_date}.csv")
        
        
        self.__standardize()
        #for col in self.X.columns:
        #    print(col, " ",len(self.X[self.X[col].isnull()]) / len(self.X[col]) )
        self.__train_test_split(test_size=test_size, shuffle=shuffle)
        self.__set_model(model)
        self.__set_cv(n_splits=n_splits)
        
        
        y_df = pd.DataFrame(self.y )
        # make the dataset that contains features and targets after preprocessing, normalization and standardization
        self.total_df = pd.concat([self.X,
                                   y_df, self.df["Covid_Cases_Density"][self.selected_samples_series]  ], axis=1)
        self.total_df.to_csv(f"standardized_dataset_from_{self.start_date}_to_{self.end_date}.csv")
        
  


      
        tprs = None
        aucs = None
        mean_fpr =None

        fig, ax =None,None
        confusion_matrices = None
        
        #store roc curve data
        if self.plot_roc and self.prob_type == "classification" and (self.clf_type =="binary" or self.clf_type=="extremes" or self.n_clusters==2):
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            fig, ax = plt.subplots(figsize=(10,8))
     
        confusion_matrices = []
        
        
        train_metric = []
        valid_metric = []
        test_metric = []
        tot_train_score = 0
        tot_valid_score = 0
        tot_test_score = 0
   
        for cv_index, (train_index, valid_index) in enumerate(self.kf.split(self.X_train_full, self.y_train_full)):

            #print("TRAIN:", train_index, "TEST:", valid_index)
            X_train, X_valid = self.X_train_full.iloc[train_index], self.X_train_full.iloc[valid_index]
            y_train, y_valid = self.y_train_full.iloc[train_index], self.y_train_full.iloc[valid_index]
            
            if self.prob_type == "classification" and  k_smote is not None:
                oversample = SMOTE(k_neighbors=k_smote)
                X_train, y_train = oversample.fit_resample(X_train, y_train)
            
           
            self.pipeline.fit(X_train, y_train)

            
            y_train_preds = self.pipeline.predict(X_train)
            y_valid_preds = self.pipeline.predict(X_valid)
            y_test_preds = self.pipeline.predict(self.X_test)
            
            if self.prob_type == "classification":
                """
                if self.clf_type =="binary":
                    pos_clusterid = self.categorical_to_clusterid_mapping["higher_than_avg"]
                else:
                    pos_clusterid = self.categorical_to_clusterid_mapping["high"]
                pos_label = self.clusterid_to_categorical_mapping[pos_clusterid]
                pos_label=pos_label
                """
                if  self.plot_roc and (self.clf_type =="binary" or self.clf_type=="extremes" or self.n_clusters==2):

                    viz = plot_roc_curve(self.pipeline, X_valid, y_valid,
                                 name='ROC fold {}'.format(cv_index),
                                 alpha=0.3, lw=1, ax=ax)
                    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    aucs.append(viz.roc_auc)
                #elif self.n_clusters >=3 and  self.clf_type=="multi":
                #print(self.clusterid_to_categorical_mapping.values())
                confusion_matrices.append(confusion_matrix(self.y_test, y_test_preds,
                                                           labels=list(self.clusterid_to_categorical_mapping.values() ))  )

            curr_train_metric = 0
            curr_valid_metric = 0
            curr_test_metric = 0
            if metric == f1_score:
                
                
                if self.f1_score_on_minority and self.clf_type  == "extremes":
                    pos_clusterid = self.categorical_to_clusterid_mapping["high"]
                    pos_label = self.clusterid_to_categorical_mapping[pos_clusterid]
                    self.used_pos_label = pos_label
                    self.used_pos_label_id = pos_clusterid
                    # clustering can return categorical labels that do not follow an ordering, thus the mapping is required
                    curr_train_metric = metric(y_train, y_train_preds, average="binary", pos_label=pos_label)
                    curr_valid_metric = metric(y_valid, y_valid_preds, average="binary", pos_label=pos_label)
                    curr_test_metric = metric(self.y_test, y_test_preds, average="binary", pos_label=pos_label)
                elif self.f1_score_on_minority and  self.clf_type == "binary":
                    pos_clusterid = self.categorical_to_clusterid_mapping["higher_than_avg"]
                    pos_label = self.clusterid_to_categorical_mapping[pos_clusterid]
                    
                    self.used_pos_label = pos_label
                    self.used_pos_label_id = pos_clusterid
                    # clustering can return categorical labels that do not follow an ordering, thus the mapping is required
                    curr_train_metric = metric(y_train, y_train_preds, average="binary", pos_label="higher_than_avg")
                    curr_valid_metric = metric(y_valid, y_valid_preds, average="binary", pos_label="higher_than_avg")
                    curr_test_metric = metric(self.y_test, y_test_preds, average="binary", pos_label="higher_than_avg")
                else:
                
                    curr_train_metric = metric(y_train, y_train_preds, average="macro")
                    curr_valid_metric = metric(y_valid, y_valid_preds, average="macro")
                    curr_test_metric = metric(self.y_test, y_test_preds, average="macro")
                                         
            else:
                curr_train_metric = metric(y_train, y_train_preds)
                curr_valid_metric = metric(y_valid, y_valid_preds)
                curr_test_metric = metric(self.y_test, y_test_preds)
                
                
            tot_train_score += curr_train_metric
            tot_valid_score += curr_valid_metric
            tot_test_score += curr_test_metric

            train_metric.append(curr_train_metric)
            valid_metric.append(curr_valid_metric)
            test_metric.append(curr_test_metric)
                

        self.trained_model = self.pipeline["model"]
        
        return {"Train": tot_train_score/n_splits, 
                "Validation": tot_valid_score/n_splits,
                "Test": tot_test_score/n_splits,
                "tprs":tprs,"aucs": aucs,"ax": ax, "fig":fig, "mean_fpr":mean_fpr, "cm":confusion_matrices,
               }, {"Train_history": train_metric, 
                "Validation_history": valid_metric,
                "Test_history": test_metric}
    
    def longrun_train_test(self, model_init_fn=rf_init,
                           metric=mean_absolute_error,
                           num_runs=20,
                          test_size=0.3,
                           shuffle=True,
                           n_splits=10, 
                           k_smote=None
                          ):
        from tqdm import tqdm        

        DENOMINATOR_N_RUNS = num_runs
    
        longrun_fimp_dict = dict()
        for i in tqdm(range(num_runs)):
            self.train_test(model_init_fn(),
                            metric=metric,
                            test_size=test_size,
                            shuffle=shuffle,
                            n_splits=n_splits,
                            k_smote=k_smote)


            selected_features = self.X_train_full.columns[self.pipeline["feature_selection"].get_support()]    
            feature_importances = sorted(zip(selected_features, self.trained_model.feature_importances_), key=lambda x: x[1], reverse=True )
            for feature, importance in feature_importances:
                #print(feature, " ", importance)
                if feature not in longrun_fimp_dict:
                    longrun_fimp_dict[feature] = [importance]
                else:
                    longrun_fimp_dict[feature].append(importance)
        res = dict()
        for feature, importances in  longrun_fimp_dict.items():
            res[feature] = (sum(importances)/DENOMINATOR_N_RUNS, len(importances))
        return res#sorted(res.items(), key=lambda x: x[1][0], reverse=True)
