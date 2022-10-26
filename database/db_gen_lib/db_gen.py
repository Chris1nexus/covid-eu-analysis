import pandas as pd
import sqlite3
from db_utils import DatasetsMerger
import argparse






def main(args):
    dm = DatasetsMerger(args.db_path, args.covid_data, args.euro_data, args.pol_data, db_folder=args.output_path)
    dm.merge()
    dm.save_to_sqlite()






if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DB merger"
    )

    parser.add_argument(
        "--covid-data",
        type=str,
        default=None,
        help="covid data FILE path",
    )
    parser.add_argument(
        "--euro-data",
        type=str,
        default=None,
        help="eurostat datasets FOLDER path",
    )
    parser.add_argument(
        "--pol-data",
        type=str,
        default=None,
        help="policy aggregated FILE path",
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="excel db FILE path",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="path where the sqlite db is generated",
    )




    args = parser.parse_args()
    main(args)