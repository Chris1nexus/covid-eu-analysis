import urllib.request
import argparse
import pandas as pd
from io import StringIO

def main(args):

    
    

    covid_data_url = "https://raw.githubusercontent.com/ec-jrc/COVID-19/master/data-by-region/jrc-covid-19-all-days-by-regions.csv"

    with urllib.request.urlopen(covid_data_url) as f:
        string = f.read().decode('utf-8')
    
    str_stream = StringIO(string)
    df = pd.read_csv(str_stream)
    
    
    # obtain START DATE comparable date format (from 1/2/2020 -> 2020-2-1)
    start_query_date = pd.to_datetime(args.start_date)
    start_parsed_dates = pd.to_datetime(df["Date"], infer_datetime_format=True)
    start_query_result = start_parsed_dates >=  start_query_date
    
    # obtain END DATE comparable date format (from 20/8/2020 -> 2020-8-20)
    end_query_date = pd.to_datetime(args.end_date)
    end_parsed_dates = pd.to_datetime(df["Date"], infer_datetime_format=True)
    end_query_result = end_parsed_dates <=  end_query_date
    df = df[end_query_result]
    
    
    df.to_csv(args.out_filepath,index=False)
    #with open(args.out_filepath, "w+",  encoding="utf-8") as outfile:
    #    outfile.write(string)
    
  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="download jrc covid cases"
    )


    parser.add_argument(
        "--out-filepath",
        type=str,
        default="../covid_cases_jrc.csv",
        help="path in which the output dataset is saved",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="1/2/2020",
        help="starting date of the period of the considered pandemic data (format is month/day/year\n\t default=1/2/2020 )",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="20/8/2020",
        help="end date of the period of the considered pandemic data  (format is month/day/year\n\t default=20/8/2020 )",
    )
 


    args = parser.parse_args()
    main(args)