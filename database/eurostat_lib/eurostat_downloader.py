import sys
import os
import urllib
import urllib.request
import gzip
import re
assert len(sys.argv) == 2+1, f"Error, expected command line inputs are:\n path_to_filemapper.tsv dest_path_of_the_downloaded_files\nBut received:\n {sys.argv}"
assert os.path.exists(sys.argv[1]), f"Error {sys.argv[1]} does not exist"

if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])
    
_, filemapper_path, dst_path = sys.argv


class DataObject:
    
    def __init__(self, filename, eurostat_fd, url):
        self.filename = filename
        self.eurostat_fd = eurostat_fd
        self.url = url
        self.dataset = None
    def load(self):        
        with urllib.request.urlopen(self.url) as f:
            gz_source = f.read()
            gzip_fd = gzip.GzipFile(fileobj=gz_source)
            self.dataset = gzip.decompress(gz_source).decode('utf-8')
    def clean(self):
        #print(self.dataset)
        
        cleaned_dataset = []
        line_cnt = 0
        for line in self.dataset.split("\n"):
            if len(line) >= 2:
                
                data = line.rstrip().split("\t")

                region_name = None
                geo_info = data[0]
                if line_cnt == 0:
                    region_name = "geo"
                else:
                    geo_info.split(",")[-1]
                    region_name = geo_info.split(",")[-1]
                clean_data = [ re.sub("[()a-zA-Z: ]", "", item)  for item in data[1:] ] 
                db_row = "\t".join([region_name] +clean_data )  
                
                cleaned_dataset.append(db_row)

                line_cnt += 1
        self.dataset = "\n".join(cleaned_dataset)
  
        # re.sub("[a-zA-Z ]", "", s)
        pass
    def save(self, filepath):
        filepath = os.path.join(filepath, self.filename)
        assert self.dataset is not None, f"Error: dataset must be loaded before saving it.\n Error at {filename}" 
        with open(filepath, "w") as out_file:
            out_file.write(self.dataset)


start_line = True
data_objects = []

with open(filemapper_path, encoding="cp1252") as f:
    i = 0
    for line in f:
        if start_line:
            start_line = False
        else:
            
            items = line.rstrip().split("\t")
            
            assert len(items) == 4, f"Error, {len(items)} elements have been found, expected 4.\n Error at line {i}(0 indexed):\n{line}"
            filename, eurostat_fd, url,agg_type = items
            obj = DataObject(filename, eurostat_fd, url)
            obj.load()
            obj.clean()
            data_objects.append(obj)
            print(f"Loaded: {filename}")
        i += 1
    
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
for data_object in data_objects:
    data_object.save(dst_path)    
    
    
