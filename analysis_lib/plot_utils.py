import json
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, ConfusionMatrixDisplay, confusion_matrix,\
    mean_absolute_error,plot_roc_curve, r2_score, f1_score, \
    accuracy_score,auc, roc_auc_score, roc_curve, RocCurveDisplay,\
    recall_score
import seaborn as sns
import ast

def by_classweight(x):
        if x != 'balanced':
            key = ast.literal_eval(x)['higher_than_avg']
            return key
        return 1000000
def index_by_classweight(x):
    
    transformed = []
    if isinstance(x, pd.Float64Index):
        return x
    for val in x:
        transformed.append(by_classweight( val) )

    return transformed
        
def plot_hpsearch_heatmap(hp_df, row_feat, col_feat, img_dirpath):
    
    columns = hp_df.columns.tolist()

    map_cols_to_readable = dict()
    for col in columns:
        cleaned_col = col.replace('model__', "").replace('_', ' ')
        map_cols_to_readable[col] = cleaned_col
    hp_df = hp_df.rename(columns=map_cols_to_readable)
    print('available parameter names:\n',hp_df.columns[3:].tolist())
    
    assert row_feat != 'class weight', 'Error, class weight is only supported as column feature'
    hp_pivoted = hp_df.pivot_table(index=row_feat, columns=col_feat, values='mean val score', aggfunc='mean')

    if col_feat == 'class weight':
        piv_columns = hp_pivoted.columns.tolist()
        map_piv_cols_to_readable = dict()
        for col in piv_columns:
            if col != 'balanced':
                mod_col = ast.literal_eval(col)
                map_piv_cols_to_readable[col] = f"{mod_col['higher_than_avg']}:{mod_col['below_than_avg']}"
            else:
                map_piv_cols_to_readable[col] = col
        sorted_items = sorted(hp_pivoted.columns.tolist(), key=by_classweight )
        hp_pivoted = hp_pivoted[sorted_items]
        hp_pivoted = hp_pivoted.rename(columns=map_piv_cols_to_readable)
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(hp_pivoted, cmap=sns.cm.rocket_r)
    if col_feat == 'class weight':
        plt.xlabel("class_weight (weight reported as high_risk:low_risk)")
        #ax.set(xlabel='class_weight (weight reported as high_risk:low_risk)')
    img_path = os.path.join(img_dirpath, f"{row_feat}_{col_feat}" +'.png')
    plt.savefig(img_path, dpi=300, bbox_inches = "tight")
    
    

def plot_curve(tprs,aucs, mean_fpr, ax, mean_curve_color='b',title=None, model_name=''):
    
    #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
    #        label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color=mean_curve_color,
            label= f'{model_name} ' + r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=mean_curve_color,#'grey', 
                    alpha=.1,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic" + (f": {title}" if title is not None else "") )
    ax.legend(loc="lower right",prop={'size': 12})
    #plt.show()


def plot_average_confmatr(confusion_matrices, datamanger, figsize=(10,8)):
    
        avg_cm = np.zeros(shape=confusion_matrices[0].shape)
        for cm in confusion_matrices:
            avg_cm += cm
        avg_cm = avg_cm /len(confusion_matrices)   
        fig, ax = plt.subplots(figsize=figsize)
        avg_cm = np.around(avg_cm)
        ConfusionMatrixDisplay(avg_cm,
                               display_labels=datamanger.clusterid_to_categorical_mapping.values())\
                                .plot(ax=ax,
                                     cmap=plt.cm.Blues)


        
        
def get_side_by_side_plots(datamanager, dataset_file_path, dest_dir_path="./violinplots_sidebyside"):
        if not os.path.exists(dest_dir_path):
            os.makedirs(dest_dir_path)
        df = pd.read_csv(dataset_file_path)\
                .set_index("NUTS")



        TRUE_CLASS = "covid_severity"
        classes =  df[TRUE_CLASS].unique()


        columns_to_drop = ["covid_severity","covid_density_by100k", "Covid_Cases"]
        df_reduced = df.drop(columns=columns_to_drop)

        analysis_df = df[df_reduced.columns.tolist()+ [TRUE_CLASS] ]
        #analysis_df["representation"] = pd.Series( ["" for _ in range(len(analysis_df))], index=analysis_df.index)
        
        
        X = analysis_df[df_reduced.columns.tolist()]
        y = df[TRUE_CLASS]
        
        X_mean = X.mean()
        X_std = X.std()
        X = (X - X_mean)/X_std
        
        Z_coord = 4
        # clip outliers
        for column in X.columns:
                sigma = X_std[column]
                X[column] = X[column].clip( lower = -Z_coord, 
                                                    upper = Z_coord)
        
        from sklearn.feature_selection import f_classif
        fstat, pvalue = f_classif(X,y)

        pval_sorted_columns = sorted(zip(X.columns.tolist(), pvalue.tolist()), 
                                     key=lambda x: x[1], 
                                     #reverse=True
                                    )
        #pval_sorted_columns = [val[0] for val in pval_sorted_columns]
        MAX_RANGE = 22
        for i in range(0, len(X.columns), MAX_RANGE):
            #print(X.columns[i:(i+MAX_RANGE)])
            col_subset = pval_sorted_columns [i:(i+MAX_RANGE)]

            feature_values = []
            feature_names = []
            classes = []
            for col, pvalue in col_subset:
                assert len(X[col].tolist()) == \
                len([ col for _ in range(len(X[col]))]) ==\
                len(y.tolist() )
                feature_values.extend(X[col].tolist() )
                feature_names.extend([ col + ('\n(p-value:%.3f)'%pvalue) for _ in range(len(X[col]))] )
                classes.extend( y.tolist() )


            #print(len(feature_values))
            #print(len(feature_names))
            #print(len(classes))
            newdf = pd.DataFrame({'value': feature_values,
                                 'feature': feature_names,
                                 'class': [ "high risk" if c == "higher_than_avg" else "reduced risk" for c in classes ]})
            sns.set(font_scale=1.1)
            plt.figure(figsize=(8,15))
            plt.title('standardized features ordered by increasing F-test pvalue\nfrom top to bottom')
            ax = sns.violinplot(y="feature", x="value", hue="class",
                                data=newdf, palette="muted", hue_order=["reduced risk", "high risk"], split=True)



        
        
        
        
        
def get_optimal_model(results, metric_name, sel_model_class='random_forest'):
    assert metric_name in results[0].keys(), f'Error: {metric_name} not found. Available keys are: {results[0].keys()} '
    score = 0
    best_model = None
    for model_dict in results:
        if sel_model_class == model_dict['model_name']:
            if model_dict[metric_name] > score:
                score = model_dict[metric_name]
                best_model = model_dict
    return best_model

def save_interpretations(results, datamanager, metric_name):
    X, y = datamanager.get_dataset()
    logger_path = os.path.join(datamanager.get_experiment_identifier())
    
    
    all_dataframes = []
    for model_dict in results:
        
        model_name = model_dict['model_name']
        
        selected_features = X.columns
        
        coefficients = None
        if model_dict['model_name'] == 'random_forest':
            coefficients = model_dict['gs']['model'].feature_importances_
        else:
            coefficients = model_dict['gs']['model'].coef_.flatten().tolist()
            
        sorted_fimps = sorted(zip(selected_features, coefficients),
                              key=lambda x: x[1],
                              reverse=True)


        curr_res_df = pd.DataFrame({'model_name': [ model_dict['model_name'] for x in range(len(coefficients)) ],
                      'feature': selected_features,
                      'coefficient':coefficients})

        all_dataframes.append(curr_res_df)
    save_all_path = os.path.join(logger_path, 'all_coef_dataframe.csv')
    all_df = pd.concat(all_dataframes, axis=0)
    all_df.to_csv(save_all_path)
        
        
        
    model_names = set([x['model_name'] for x in results])
    for model_name in model_names:
        best_model = get_optimal_model(results, metric_name,  sel_model_class=model_name)

        
        selected_features = X.columns
        
        coefficients = None
        if best_model['model_name'] == 'random_forest':
            coefficients = best_model['gs']['model'].feature_importances_
        else:
            coefficients = best_model['gs']['model'].coef_.flatten().tolist()
            
        sorted_fimps = sorted(zip(selected_features, coefficients),
                              key=lambda x: x[1],
                              reverse=True)


        
        sns.set(font_scale=1.2)
        plt.figure(figsize=(8,6))
        colors = ['g' if val >= 0 else 'r' for name, val in sorted_fimps]
        b = sns.barplot([x[1] for x in sorted_fimps], [x[0] for x in sorted_fimps], palette=colors)
        if best_model['model_name'] == 'random_forest':
            b.set_title(best_model['model_name'] + ' feature importances(non negative)', weight='bold')
        else:
            b.set_title(best_model['model_name'] + ' coefficients', weight='bold')
        
        save_image_path = os.path.join(logger_path, 'interp')
        save_coefs_path = os.path.join(logger_path, 'coef_dataframes')
        if not os.path.exists(save_image_path):
            os.makedirs(save_image_path)
            os.makedirs(save_coefs_path)
        plt.savefig(os.path.join(save_image_path, best_model['model_name']+'.png'), dpi=300, bbox_inches = "tight")
        plt.close()
        
        curr_res_df = pd.DataFrame({'model_name': [ best_model['model_name'] for x in range(len(coefficients)) ],
                      'feature': selected_features,
                      'coefficient':coefficients})
        save_coefs_path = os.path.join(logger_path, 'coef_dataframes', f'{best_model["model_name"]}.csv')
        curr_res_df.to_csv(save_coefs_path)
        
        __save_ci_plots(datamanager)
        __save_svm_logreg_rescaled_ci_plots(datamanager)
        
        
def __save_ci_plots(datamanager):
    df_file_path = os.path.join(datamanager.get_experiment_identifier(), 'all_coef_dataframe.csv')
    all_df = pd.read_csv(df_file_path, index_col=0)
    
    plots_dir = os.path.join(datamanager.get_experiment_identifier(), 'ci_coefs')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    #shorten_mapper = datamanager.get_feature_mapper()
    #all_df['feature'] = all_df['feature'].apply(lambda x: shorten_mapper[x])
         
    for model_name in all_df['model_name'].unique():
        curr_df = all_df[all_df['model_name'] ==model_name]

        plt.figure(figsize=(10,8))
        
        palette = sns.diverging_palette(10, 240, n=len(curr_df.groupby('feature')))
        df_func =  curr_df.groupby('feature').mean().sort_values('coefficient', ascending=False)['coefficient']
        
        #print(list(zip(df_func.to_dict().items(), palette)) )
        # each item is a tuple composed of two tuples: (k,v), (R,G,B)
        # the whole list is sorted by the absolute value V of the first tuple (the coefficient in the original dataframe)
        # this way colors are paired with the size of the bars
        abs_sorted_pairs = sorted(list(zip(df_func.to_dict().items(), palette)), 
                                  key=lambda x : abs(x[0][1] ), 
                                  reverse=True)
        abs_sorted_palette = [ item[1] for item in abs_sorted_pairs ]
        order_list = curr_df.groupby('feature').mean()['coefficient'].abs().sort_values(ascending=False).index.values
        
        if model_name == 'random_forest':
            n_to_consider = len(curr_df.groupby('feature'))
            # consider more colors, such that the black tones are skipped and the grey ones do not cover the estimated intervals  
            to_add = 10
            abs_sorted_palette = sns.color_palette("Greys", n_colors=n_to_consider+to_add )[::-1][to_add:]
        
        sns.barplot(data=curr_df, x='coefficient', y='feature',
                   capsize=0.3,
                    palette=abs_sorted_palette, #palette,#'RdBu',
                    order=order_list
                   )
        
        img_path = os.path.join(plots_dir, model_name +'.png')
        plt.savefig(img_path, dpi=300, bbox_inches = "tight")
        plt.close()
        
def __save_svm_logreg_rescaled_ci_plots(datamanager):
    df_file_path = os.path.join(datamanager.get_experiment_identifier(), 'all_coef_dataframe.csv')
    all_df = pd.read_csv(df_file_path)
    
    plots_dir = os.path.join(datamanager.get_experiment_identifier(), 'rescaled_ci_coefs')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    #if rescale:
    newres = all_df[all_df['model_name'] != 'random_forest' ]

    #shorten_mapper = datamanager.get_feature_mapper()
    #all_df['feature'] = all_df['feature'].apply(lambda x: shorten_mapper[x])
  
    for model_name in newres['model_name'].unique():

            coefs = newres[newres['model_name'] == model_name]['coefficient']

            coef_range = coefs.max() - coefs.min()
            
            newmax = 1
            newmin = -1


            newres.loc[newres['model_name'] == model_name, 'coefficient'] = coefs /max( abs(coefs.max()), abs( coefs.min()) )
            all_df.loc[all_df['model_name'] == model_name, 'coefficient'] = newres.loc[newres['model_name'] == model_name, 'coefficient'] 
            
    all_df = all_df[all_df['model_name'] != 'random_forest']
    for model_name in all_df['model_name'].unique():
        curr_df = all_df[all_df['model_name'] ==model_name]

        
        plt.figure(figsize=(10,8))
        
        palette = sns.diverging_palette(10, 240, n=len(curr_df.groupby('feature')))
        df_func =  curr_df.groupby('feature').mean().sort_values('coefficient', ascending=False)['coefficient']
        
        #print(list(zip(df_func.to_dict().items(), palette)) )
        # each item is a tuple composed of two tuples: (k,v), (R,G,B)
        # the whole list is sorted by the absolute value V of the first tuple (the coefficient in the original dataframe)
        # this way colors are paired with the size of the bars
        abs_sorted_pairs = sorted(list(zip(df_func.to_dict().items(), palette)), 
                                  key=lambda x : abs(x[0][1] ), 
                                  reverse=True)
        abs_sorted_palette = [ item[1] for item in abs_sorted_pairs ]
        order_list = curr_df.groupby('feature').mean()['coefficient'].abs().sort_values(ascending=False).index.values
        sns.barplot(data=all_df, x='coefficient', y='feature',
                   capsize=0.3, hue='model_name',
                    palette=abs_sorted_palette, #palette,#'RdBu',
                    order=order_list
                   )
        
        img_path = os.path.join(plots_dir, model_name +'.png')
        plt.savefig(img_path, dpi=300, bbox_inches = "tight")
        plt.close()
        
        