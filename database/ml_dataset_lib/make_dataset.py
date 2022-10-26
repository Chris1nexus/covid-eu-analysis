import pandas as pd
import os
import numpy as np
import argparse


def main(args):

    EUROSTAT_DIR = args.eurostat_datadir
    COVID_FILEPATH = args.covid_dataset

    covariate_dfs = []

    for directory, subdirs, files in os.walk(EUROSTAT_DIR):
        feature_files = [ (os.path.join(directory,file), file) for file in files if file.endswith(".tsv") ]
        [ covariate_dfs.append((filename, pd.read_csv(file_path, sep="\t").set_index("geo"))) for file_path,filename in feature_files]

    new_dfs = []
    for filename, df in covariate_dfs:
        new_df = pd.DataFrame()
        new_df[filename] = df.fillna(method="ffill", axis=1).iloc[:,-1]
        #new_dfs.append(df.fillna(method="ffill").iloc[:,-1:])
        new_dfs.append((filename,new_df))
    X_df = None
    spec = None
    for filename, df in new_dfs:
        if X_df is None:
            X_df = df
        else:
            #print(filename, len(X_df))
            #X_df = pd.merge(X_df, df, how="inner", left_index=True,right_index=True)
            X_df = X_df.join(df, how="outer")
           


    assert (X_df.index.value_counts() > 1).any() == False, "Error, dataset contain multiple rows for the same NUTS code (static covariates do not present such characteristic)"


    covid_df = pd.read_csv(COVID_FILEPATH,  index_col=0)



    """
    latest_covid_positive = covid_df.pivot_table(index="NUTS", columns="Date", values="CumulativePositive").fillna(method="ffill",axis=1).iloc[:,-1:]

    latest_covid_deceased = covid_df.pivot_table(index="NUTS", columns="Date", values="CumulativeDeceased").fillna(method="ffill",axis=1).iloc[:,-1:]
    
    
    latest_covid_recovered = covid_df.pivot_table(index="NUTS", columns="Date", values="CumulativeRecovered").fillna(method="ffill",axis=1).iloc[:,-1:]
    
    
    y_df = pd.DataFrame()
    y_df["CumulativePositive"] = latest_covid_positive.iloc[:,-1]
    y_df["CumulativeDeceased"] = latest_covid_deceased.iloc[:,-1]
    y_df["CumulativeRecovered"] = latest_covid_recovered.iloc[:,-1]

    dataset = X_df.join(y_df,  how="outer")


    dataset["cum_positive_density"] = dataset["CumulativePositive"] / dataset["population_nuts2.tsv"]
    dataset["cum_deceased_density"] = dataset["CumulativeDeceased"] / dataset["population_nuts2.tsv"]
    dataset["cum_recovered_density"] = dataset["CumulativeRecovered"] / dataset["population_nuts2.tsv"]
    """

    dataset = covid_df.join(X_df,  how="inner")


    dataset["cum_positive_density"] = covid_df["cum_positive"] / dataset["population_nuts2.tsv"]
    dataset["cum_deceased_density"] = covid_df["cum_deceased"] / dataset["population_nuts2.tsv"]
    dataset["cum_recovered_density"] = covid_df["cum_recovered"] / dataset["population_nuts2.tsv"]

    dataset.to_csv(args.out_filepath, sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create ML dataset from eurostat and covid data"
    )

    parser.add_argument(
        "--eurostat-datadir",
        type=str,
        help="directory that contains eurostat datasets",
    )
    parser.add_argument(
        "--covid-dataset",
        type=str,
 
        help="jrc covid dataset path",
    )
    parser.add_argument(
        "--out-filepath",
        type=str,
 
        help="path in which the output dataset is saved",
    )
 


    args = parser.parse_args()
    main(args)