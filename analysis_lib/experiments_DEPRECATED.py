
from analysis_lib.utils import train_test
def binary_clf_experiment(model,
                         start_date="2020-02-20",
                    end_date="2020-08-20",
                          response="cumulativepositive",
                          parameter_grid=None,
                         k_smote = 9,
                         nuts2_to_remove =[],
                          plot_roc=True,
                          test_size=0.3,
                          leave_one_out=True,
                          f1_score_on_minority=True,
                          cv_scoring=None,
                          manager_n_splits=10,
                            cv_n_splits=10,
                          clust_method='ML',
                         sqlite_file_path="covid_at_lombardy.sqlite"):# alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']):
    
    
    training_metrics, history, training_manager = train_test(model,
                    parameter_grid=parameter_grid,
                    response=response,
                    prob_type = "classification",
                    clf_type = "binary",
                    #wave="1",# either '1' or '2'
                    start_date=start_date,
                    end_date=end_date,
                    k_smote = k_smote,
                    test_size = test_size,
                    leave_one_out=leave_one_out,
                    clust_method=clust_method,                               
                    nuts2_to_remove =nuts2_to_remove, # alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']
                    sqlite_file_path=sqlite_file_path,
                    plot_roc=plot_roc,
                    f1_score_on_minority=f1_score_on_minority,
                                                            manager_n_splits=manager_n_splits,
                                                            cv_n_splits=cv_n_splits,
                                                            cv_scoring=cv_scoring)
    return training_metrics, history, training_manager


def extremes_clf_experiment(model,
                         start_date="2020-02-20",
                    end_date="2020-08-20",
                            response="cumulativepositive",
                          parameter_grid=None,
                         k_smote = 9,
                         nuts2_to_remove =[],
                          plot_roc=True,
                            test_size=0.3,
                            leave_one_out=True,
                          f1_score_on_minority=True,
                            manager_n_splits=10,
                            cv_n_splits=10,
                            cv_scoring=None,
                            clust_method='ML',
                         sqlite_file_path="covid_at_lombardy.sqlite"):# alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']):
    
    
    training_metrics, history, training_manager = train_test(model,
                    parameter_grid=parameter_grid,
                    response=response,
                    prob_type = "classification",
                    clf_type = "extremes",
                    #wave="1",# either '1' or '2'
                    start_date=start_date,
                    end_date=end_date,
                    k_smote = k_smote,
                    test_size = test_size,
                    clust_method=clust_method,
                                                             leave_one_out=leave_one_out,
                    nuts2_to_remove =nuts2_to_remove, # alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']
                    sqlite_file_path=sqlite_file_path,
                    plot_roc=plot_roc,
                    f1_score_on_minority=f1_score_on_minority,
                                                            manager_n_splits=manager_n_splits,
                                                            cv_n_splits=cv_n_splits,
                                                            cv_scoring=cv_scoring)
    return training_metrics, history, training_manager
def triclass_clf_experiment(model,
                         start_date="2020-02-20",
                    end_date="2020-08-20",
                            response="cumulativepositive",
                          parameter_grid=None,
                         k_smote = 9,
                         nuts2_to_remove =[],
                            manager_n_splits=10,
                            cv_n_splits=10,
                            test_size=0.3,
                            leave_one_out=True,
                          plot_roc=False,
                          f1_score_on_minority=False,
                            cv_scoring=None,
                            clust_method='ML',
                         sqlite_file_path="covid_at_lombardy.sqlite"):# alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']):
    
    
    training_metrics, history, training_manager = train_test(model,
                    parameter_grid=parameter_grid,
                                                             response=response,
                    prob_type = "classification",
                    clf_type = "multi",
                    #wave="1",# either '1' or '2'
                    start_date=start_date,
                    end_date=end_date,
                    k_smote = k_smote,
                    test_size = test_size,
                    clust_method=clust_method,
                                                             leave_one_out=leave_one_out,
                    nuts2_to_remove =nuts2_to_remove, # alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']
                    sqlite_file_path=sqlite_file_path,
                    plot_roc=plot_roc,
                    f1_score_on_minority=f1_score_on_minority,
                                                            manager_n_splits=manager_n_splits,
                                                            cv_n_splits=cv_n_splits,
                                                            cv_scoring=cv_scoring)
    return training_metrics, history, training_manager

def regression_experiment(model,
                         start_date="2020-02-20",
                    end_date="2020-08-20",
                          response="cumulativepositive",
                          parameter_grid=None,
                         k_smote = 9,
                         nuts2_to_remove =[],
                          plot_roc=False,
                         leave_one_out=True,
                          test_size=0.3,
                          f1_score_on_minority=False,
                          manager_n_splits=10,
                            cv_n_splits=10,
                          cv_scoring=None,
                          clust_method='ML',
                         sqlite_file_path="covid_at_lombardy.sqlite"):# alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']):
    
    
    training_metrics, history, training_manager = train_test(model,
                    parameter_grid=parameter_grid,
                                                             response=response,
                    prob_type = "regression",
                    clf_type = "multi",
                    #wave="1",# either '1' or '2'
                    start_date=start_date,
                    end_date=end_date,
                    k_smote = k_smote,
                    test_size = test_size,
                                                            clust_method=clust_method,
                                                             leave_one_out=leave_one_out,
                    nuts2_to_remove =nuts2_to_remove, # alternative removal of sweden ['SE11', 'SE12', 'SE21', 'SE22', 'SE23', 'SE31', 'SE32', 'SE33']
                    sqlite_file_path=sqlite_file_path,
                    plot_roc=plot_roc,
                    f1_score_on_minority=f1_score_on_minority,
                                                             manager_n_splits=manager_n_splits,
                                                            cv_n_splits=cv_n_splits,
                                                            cv_scoring=cv_scoring)
    return training_metrics, history, training_manager