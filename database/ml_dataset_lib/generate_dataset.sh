#!/bin/bash
cd ../eurostat_lib
bash ../eurostat_lib/run_eurostat_downloader.sh
cd ../covid_lib
python3 covid_jrc_downloader.py --start-date "1/2/2020" --end-date "20/8/2020"
cd ../ml_dataset_lib
python3 mapred_covid_cases.py
python3 make_dataset.py --eurostat-datadir ../eurostat_datasets --covid-dataset ../covid_mapreduce_output.csv --out-filepath ../ml_dataset.tsv
