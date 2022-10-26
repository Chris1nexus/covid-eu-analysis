import pandas as pd
import sqlite3
import os
import csv

COVID_COLS = ["CumulativePositive", "CumulativeDeceased", "CumulativeRecovered", "CurrentlyPositive", "Hospitalized", "IntensiveCare"]
POLICIES_COLS = ["Curfew"]

class Policies(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = self._load_df()
            
    def _load_df(self):
        """
            Load and pre-process the policy file
        """
        return pd.read_csv(self.file_name)
    
    def extract_policy_from_iso_a2_code(self, nuts2_code, iso_a2_code):
        """
            Extract the policy from an ISO-A2 code
        """
        out_df = self.df[self.df["province"] == iso_a2_code][["date"] + POLICIES_COLS]
        out_df["NUTS"] = nuts2_code
        return out_df
    
class CovidCases(object):
    def __init__(self, covid_file_name, datasetMerger):
        self.covid_file_name = covid_file_name
        self.datasetMerger = datasetMerger
        self.df = self._load_df()
        
    def _load_df(self):
        """
            Load and pre-process the file with Covid cases
        """
        df = pd.read_csv(self.covid_file_name)
        df[COVID_COLS] = df[COVID_COLS].fillna(0)
        return df
    
    def _find_children_from_nuts_2(self, nuts2_code, dataset_merger):
        """
            From a nuts2_code, get all the children nuts3 codes
            params:
                nuts2_code: str
                
            return:
                list (str)
        """
        return dataset_merger.db_df[self.datasetMerger.db_df["NUTS"] == nuts2_code]["Covid (NUTS)"].to_list()
    
    def _aggregate_from_nuts_2(self, nuts2_code, dataset_merger):
        """
            Sum all covid cases from a nuts2_code aggregation
            params:
                nuts2_code: str
            return:
                DataFrame
        """
        
        # Call _find_children_from_nuts_2(self, nuts2_code)
        
        # Sum all
        covid_keys = self._find_children_from_nuts_2(nuts2_code, dataset_merger)
        return self.df[self.df["NUTS"].isin(covid_keys)].groupby(["Date"])[COVID_COLS].sum().reset_index()
    
    def get_covid_cases(self, covid_code, nuts2_code, dataset_merger):
        """
            Get all covid cases
            params:
                covid_code: str
                nuts2_code: str   
        """
        
        # If covid_code == nuts2_code, then just extract data from covid_code
        # Else, call _aggregate_from_nuts_2 
        out_df = None
        if covid_code == nuts2_code:
            out_df = self.df[self.df["NUTS"] == nuts2_code][["Date"] + COVID_COLS]
        else:
            #print("aaaaaaa: ", covid_code, " ", nuts2_code )
            out_df = self._aggregate_from_nuts_2(nuts2_code, dataset_merger)
        
        out_df["NUTS"] = nuts2_code
        return out_df
    
    
class Covariate(object):
    def __init__(self, file_path, file_type='xlsx', aggregation_method='sum'):
        """
            Covariate file
            params: 
                file_name: str
                file_type: str
                    among 'xlsx', 'tsv', 'csv'
                aggregation_method: str
                    among 'sum', 'popsum', 'avg'
        """
        self.file_path = file_path
        self.file_type = file_type
        self.aggregation_method = aggregation_method
        
        self.col_name = '.'.join([file_path,  aggregation_method])
        
        self.df = self._load_df()
        
    
    def _load_df(self):
        """
            Load the covariate data
            
            :return
                DataFrame
        """
        
        df = None
        # Check the file type and load from the according file type
        if self.file_type == "xlsx":
            df = self._load_excel()
        elif self.file_type == "csv":
            df = self._load_csv()
        elif self.file_type == "tsv":
            df = self._load_tsv()
            
        return self._compute_covariate_value(df)
        
    
    def _load_excel(self):
        """
            Load from an .xlsx file
            
            :return
                DataFrame
        """
        return pd.read_excel(self.file_path)
    
    def _load_csv(self):
        """
            Read from an .csv file
            
            :return
                DataFrame
        """
        return pd.read_csv(self.file_path)
    
    def _load_tsv(self):
        """
            Read from an .tsv file
            
            :return
                DataFrame
        """
        return pd.read_csv(self.file_path, sep='\t') 
    
    @staticmethod
    def _compute_covariate_value(df):
        """
            Compute the covariate value by coalescing the columns from right-most to left-most
            
            :return
                DataFrame
        """
        return df.assign(
            covariate_value=pd.to_numeric(df.iloc[:, ::-1].notnull().idxmax(1).pipe(
                lambda d: df.lookup(d.index, d.values)
            ), errors='coerce')
        )
    
    def extract_covariate(self, nuts_codes):
        """
            Extract the covariate value for given nuts_codes and a specified aggregartion method
            
            params:
                nuts_codes: list (str)
            
            return:
                DataFrame
        """
        
        if len(nuts_codes) > 1:
            return self.df[self.df.iloc[:, 0].isin(nuts_codes)]["covariate_value"].aggregate(self.aggregation_method)
        else:
            return self.df[self.df.iloc[:, 0] == nuts_codes[0]]["covariate_value"].values[0]

        
        
class DatasetsMerger(object):
    
    def __init__(self, db_filename, covid_file_name, eurostat_folder, policy_file_name, db_folder='./', db_sheet=3):
        """
            DatasetMerger
            
            params:
                db_file_name: str
                db_folder: str
        """
        self.db_filename = db_filename
        self.covid_file_name = covid_file_name 
        self.eurostat_folder = eurostat_folder
        self.policy_file_name = policy_file_name
        self.db_folder = db_folder
        self.db_sheet = db_sheet
        
        
        self.db_df = self._load_db_df()
        print(self.db_df.columns[:25])
        self.covariates = self._load_covariates()
        self.covid_cases = self._load_covid_cases()
        self.policies = self._load_policies()
        
        self._raw_data = {}
        
        
        
    
    def _load_db_df(self):
        """
            Load the DBFinale
            
            returns:
                DataFrame
        """
        return pd.read_excel(self.db_filename, sheet_name = self.db_sheet)
        #return pd.read_csv(self.db_file_name, sep="\t")
    def _load_covariates(self):
        """
            Load all the covariates from the db_df
            
            return:
                list (Covariate)
        """
        
        # Return a map of {col_name_cov_1: Covariate(), col_name_cov_2: Covariate(), ...}
        covs = {}
        #"""
        for covariate_info in list(self.db_df.columns.values)[8:]:
            if "inserire nome covariate" in covariate_info:
                continue
                
            try:
                cov_file_name = covariate_info.split(".")[0]
                cov_file_type = covariate_info.split(".")[1]
                cov_agg_method = covariate_info.split(".")[2]
                
                cov_file_path = os.path.join(self.eurostat_folder, cov_file_name + '.' + cov_file_type) 
                covs[covariate_info] = Covariate(cov_file_path, file_type=cov_file_type, aggregation_method=cov_agg_method)
            except Exception as e:
                print(covariate_info, " ",e)
                pass
        """    
        with open(self.db_filename) as fd:
            reader = csv.reader(fd, delimiter="\t", quotechar='"')
            for line in fd:
                print(line)
            next(reader, None)  # skip the headers
            for row in reader:
                print(row)
                cov_file_name = row[0]
                 
                cov_name = row[0].split(".")[0]
                cov_file_type = row[0].split(".")[-1]
                cov_agg_method = row[-1]
                
                cov_file_path = os.path.join(self.eurostat_folder, cov_file_name) 
                covs[cov_name] = Covariate(cov_file_path, file_type=cov_file_type, aggregation_method=cov_agg_method)
        """
        return covs
            
        
    
    def _load_covid_cases(self):
        """
            Load Covid Cases
            
            return:
                CovidCases
        """
        
        # Return CovidCases
        return CovidCases(self.covid_file_name, self)
        
    
    def _load_policies(self):
        """
            Load Policies
            
            return:
                Policiesd4f
        """
        
        # Return Policies
        return Policies(self.policy_file_name)
        
        
    def merge(self):
        """
            Merge the dataset
            
            return:
                SQLLite Database
        """
        all_policies = []
        all_cov_values = []
        all_covid_cases = []
        
        _already_seen_nuts = []
        
        # Loop over all the rows
        for index, r in self.db_df.iterrows():
            
            if r["NUTS"] in _already_seen_nuts:
                continue
            _already_seen_nuts.append(r["NUTS"])
                        
            covid_infos = self.covid_cases.get_covid_cases(r["Covid (NUTS)"], r["NUTS"], self)
            policies = self.policies.extract_policy_from_iso_a2_code(r["NUTS"], r["ISO_A2 (FOR NATIONAL POLICIES)"])
            
            all_policies.append(policies)
            all_covid_cases.append(covid_infos)
            
            for cov_col_name, cov in self.covariates.items():
                try:
                    nuts_codes = list(map(lambda e: e.strip(), r[cov_col_name].split('/')))
                    cov_value = cov.extract_covariate(nuts_codes)
                    all_cov_values.append([r['Key'], cov_col_name.split(".")[0], cov_value])
                except Exception:
                    all_cov_values.append([r['Key'], cov_col_name.split(".")[0], None])
                    
        self._raw_data["policies"] = all_policies
        self._raw_data["covariates"] = all_cov_values
        self._raw_data["covid"] = all_covid_cases
        
        
    def save_to_sqlite(self):
        """
            Save the data to SQLLite Format
        """
        
        to_store_covid = pd.concat(self._raw_data["covid"])
        to_store_covariates = pd.DataFrame(self._raw_data["covariates"], columns=["NUTS", "Covariate", "Value"])
        to_store_policies = pd.concat(self._raw_data["policies"])
        
        output_db_file_path = os.path.join(self.db_folder,"covid_at_lombardy.sqlite")
        conn = sqlite3.connect(output_db_file_path)
        to_store_covid.to_sql('covid_cases', conn, if_exists='replace', index=False)
        to_store_covariates.to_sql('covariates', conn, if_exists='replace', index=False)
        to_store_policies.to_sql('policies', conn, if_exists='replace', index=False)
        
                    
                    

          
        