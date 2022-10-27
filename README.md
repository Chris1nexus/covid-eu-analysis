# covid-eu-analysis
Multivariate analysis on the impact of socio-economic factors on the spread of covid-19 within European NUTS2 regions,
during the first and second wave of the epidemic.

## Setup

### Dataset

### Run the experiments

## Dataset
The dataset has been built in order to assess which socio-economic features of the analyzed NUTS2 European regions intrinsically posed each at greater risk of epidemic spread.
Such experimental setting would ideally serve the purpose of observing whether the epidemic spread, which occurred in Lombardy at the start of 2020, had been due to randomness or due to some intrinsic factors that are possibly shared by one or more European regions: those were majorly affected by the epidemic.

Hence, the dataset has been built by considering the NUTS2 European regions as the samples, each of which has been characterized according to a set of socio-economic variables.
The target variable has been engineered from the raw number of cases that had occurred in the first and second wave of the epidemic. 

The dataset consists NUTS2 European regions, each represented by:
- a set of variables gathered from the official [Eurostat data repository](https://ec.europa.eu/eurostat/web/main/data/database)
- coronavirus cases data has been collected from the [European Joint Reseach Center github repository](https://github.com/ec-jrc/COVID-19)
at the following [link](https://raw.githubusercontent.com/ec-jrc/COVID-19/master/data-by-region/jrc-covid-19-all-days-by-regions.csv)

The dataset has been processed in ordet to obtain a tabular formatted dataset which contains both predictors and the target variable, separately for the first and second wave of the epidemic.

The target variable is the categorical binary risk class obtained by considering two clusters of coronavirus cases density.

## Models
Interpretability has been the main driver behind the choice of each classification model selected for the analysis.
Hence, the following models have been considered:
- Logistic regression
- Random forest
- Linear svm

Hyper-parameter optimization has been carried out on each of these.


