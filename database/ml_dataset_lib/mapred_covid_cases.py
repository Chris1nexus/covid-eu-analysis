from pyspark import SparkConf, SparkContext
import sys
from  mapred_utils import CovidData, map0, reduce0, map1, reduce1 
import argparse


def main(args):

    
    #Create a configuration object and
    #set the name of the application
    conf = SparkConf().setAppName("Spark covid mapreduce")
    # Create a Spark Context object
    sc = SparkContext.getOrCreate(conf=conf)
    
    folderPath = args.covid_dataset#"../covid_jrc_1_aug_2021.csv"
    logFileRDD = sc.textFile(folderPath)
    
    
    res = logFileRDD.map(map0).filter(lambda item: item is not None)\
        .reduceByKey(reduce0)\
        .map(map1).filter(lambda item: item is not None)\
        .filter(lambda item: len(item[0]) ==4 )\
        .reduceByKey(reduce1)\
        .filter(lambda item: item[1].cum_positive > 0.)\
        .map(lambda item: item[1].as_row(item[0]))\
        .collect()

    with open(args.out_filepath, "w") as outfile:
        outfile.write("\n".join([CovidData.row_header()] + res))

        
    
    
    
  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="create ML dataset from eurostat and covid data"
    )


    parser.add_argument(
        "--covid-dataset",
        type=str,
        default="../covid_cases_jrc.csv",
        help="jrc covid dataset path",
    )
    parser.add_argument(
        "--out-filepath",
        type=str,
        default="../covid_mapreduce_output.csv",
        help="path in which the output dataset is saved",
    )
 


    args = parser.parse_args()
    main(args)