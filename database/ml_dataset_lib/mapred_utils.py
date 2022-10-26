import numpy as np
def to_float(item):
    if len(item) > 0:
        item = float(item)
        return item
    return np.nan

# custom max fn since the default max returns always the first argument if either of the two is a np.nan
def max_no_nan(val1,val2):
    # must use np.isnan to check if a value is np.nan, checks like val1 is np.nan or val1 == np.nan fail in this multi process context
    if np.isnan(val1) and np.isnan(val2):
        return np.nan
    return max(get_val_or_0(val1) , get_val_or_0(val2))

def get_val_or_0(val):
    return val if val is not np.nan else 0.

def sum_aggr_nans(val1,val2):
    if np.isnan(val1) and np.isnan(val2):
        return np.nan
    return get_val_or_0(val1) + get_val_or_0(val2)
    
def map0(row):
    row_data = row.split(",")
    date = row_data[0]
    nuts = row_data[-1]
    # avoid mapping the header
    if "CumulativePositive" not in row:
        return nuts, (date, CovidData.parse(row))
    return None
def reduce0(item1, item2):
    date1, covid1 = item1
    date2, covid2 = item2
    
    if date1 >= date2:
        return date1, CovidData.time_aggregate(covid1, covid2)
    else:
        return date2, CovidData.time_aggregate(covid1, covid2)

    
def map1(item):
    nuts, (date, covid_data) = item
    #row_data = row.split(",")
    #date = row_data[0]
    #nuts = row_data[-1]
    
    if len(nuts)> 4:
        nuts = nuts[:4]
    return nuts, covid_data
    
def reduce1(covid1, covid2):
    return CovidData.aggregate(covid1, covid2)
    


class CovidData(object):
    def __init__(self, cum_positive, cum_deceased, cum_recovered, curr_positive, hospitalized, intensive_care,
                 iso3, country_name, EUCountry, EUCPMCountry, date):
        self.cum_positive, self.cum_deceased, self.cum_recovered = cum_positive, cum_deceased, cum_recovered
        self.curr_positive, self.hospitalized, self.intensive_care = curr_positive, hospitalized, intensive_care
        self.iso3, self.country_name, self.EUCountry, self.EUCPMCountry = iso3, country_name, EUCountry, EUCPMCountry
        self.date = date
        
    def aggregate(covid1, covid2):
        
        # there should be assert statemetns to verify that time-independent data are equal for both items (e.g. iso3 of the first == iso3 of the second)
        
        if covid1.date >= covid2.date:
            date = covid1.date
        else:
            date = covid2.date
            
        iso3 = covid1.iso3
        country_name = covid1.country_name
        
        cum_positive = sum_aggr_nans(covid1.cum_positive, covid2.cum_positive)
        cum_deceased = sum_aggr_nans(covid1.cum_deceased, covid2.cum_deceased)
        cum_recovered = sum_aggr_nans(covid1.cum_recovered, covid2.cum_recovered)
        curr_positive = sum_aggr_nans(covid1.curr_positive, covid2.curr_positive)
        hospitalized = sum_aggr_nans(covid1.hospitalized, covid2.hospitalized)
        intensive_care  = sum_aggr_nans(covid1.intensive_care, covid2.intensive_care)

        EUCountry = covid1.EUCountry
        EUCPMCountry = covid1.EUCPMCountry
    
        covid_res = CovidData(cum_positive, cum_deceased, cum_recovered, curr_positive, hospitalized, intensive_care,
                 iso3, country_name, EUCountry, EUCPMCountry,
                 date)
        return covid_res
    
    
    def time_aggregate(covid1, covid2):
        
        iso3 = covid1.iso3
        country_name = covid1.country_name
        EUCountry = covid1.EUCountry
        EUCPMCountry = covid1.EUCPMCountry
    
        # for cumulated values, take the highest value of the two, for others, the most recent is instead ok
        cum_positive = max_no_nan(covid1.cum_positive, covid2.cum_positive)
        
        cum_deceased = max_no_nan(covid1.cum_deceased, covid2.cum_deceased)
     
        cum_recovered = max_no_nan(covid1.cum_recovered, covid2.cum_recovered)
        if covid1.date >= covid2.date:
            date = covid1.date
            curr_positive = covid1.curr_positive 
            hospitalized = covid1.hospitalized 
            intensive_care  = covid1.intensive_care 
        else:
            date = covid2.date 
            curr_positive = covid2.curr_positive 
            hospitalized = covid2.hospitalized 
            intensive_care  = covid2.intensive_care 
            
            
        covid_res = CovidData(cum_positive, cum_deceased, cum_recovered, curr_positive, hospitalized, intensive_care,
                 iso3, country_name, EUCountry, EUCPMCountry,
                 date)
        return covid_res
    
    def parse(row):
        row_data = row.split(",")
        date = row_data[0]
        iso3 = row_data[1]
        country_name = row_data[2]
        
        region = row_data[3]
        
        
        cum_positive, cum_deceased, cum_recovered, curr_positive, hospitalized, intensive_care = [ to_float(item) for item in row_data[6:12] ]

        EUCountry = row_data[12]
        EUCPMCountry = row_data[13]
        
        covid = CovidData(cum_positive, cum_deceased, cum_recovered, curr_positive, hospitalized, intensive_care,
                 iso3, country_name, EUCountry, EUCPMCountry,
                 date)
        return covid
    
    def row_header():
        return ",".join(["nuts", "cum_positive", "cum_deceased", "cum_recovered",
                        "curr_positive", "hospitalized", "intensive_care",
                        "iso3", "country_name", "EUCountry", "EUCPMCountry", "date"])
    
    # date is not needed in the row representation (for the purpose of this work,
    # a coviddata item can be the aggregate of multiple dates, rendering this attribute useless, in a generic context)
    def as_row(self, nuts):
        return ",".join([nuts, str(self.cum_positive), str(self.cum_deceased), str(self.cum_recovered),
                        str(self.curr_positive), str(self.hospitalized), str(self.intensive_care),
                        self.iso3, self.country_name, self.EUCountry, self.EUCPMCountry, self.date])
        
        
        
        
        
class CovidInfo(object):
    def __init__(self, cum_positive, cum_deceased, cum_recovered, curr_positive, hospitalized, intensive_care,
                 iso3, country_name, EUCountry, EUCPMCountry):
        self.cum_positive, self.cum_deceased, self.cum_recovered = cum_positive, cum_deceased, cum_recovered
        self.curr_positive, self.hospitalized, self.intensive_care = curr_positive, hospitalized, intensive_care
        self.iso3, self.country_name, self.EUCountry, self.EUCPMCountry = iso3, country_name, EUCountry, EUCPMCountry
   
    
