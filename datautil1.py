import os
import urllib3
import zipfile
import requests
import pandas as pd

urllib3.disable_warnings()
def download_data(url, file_path):
    r = requests.get(url, verify=False)
    with open(file_path, "wb") as f:
        f.write(r.content)

def unzip_data(zip_file, out_path):
    zf = zipfile.ZipFile(zip_file, "r")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for fn in zf.namelist():
        fp = out_path + "/" + fn.replace("/", "-")
        data = zf.read(fn)
        with open(fp, "wb") as f:
            f.write(data)
            
def extract_data(datafile, out_path):
    if datafile.endswith(".zip"):
        unzip_data(datafile, out_path)
    else:
        raise NotImplementedError(".{} is not currently supportted".format(datafile.split(".")[-1]))
    
data_url = "https://www.palmetto.clemson.edu/dsci/dataset/housing/california-housing-prices.zip"
local_data_dir = os.path.join(os.getcwd(), "datasets")
if not os.path.exists(local_data_dir):
    os.mkdir(local_data_dir)
local_compressed_datafile = os.path.join(local_data_dir, 'california-housing-prices.zip')
local_cached_datafile = os.path.join(local_data_dir, 'housing.csv')

def load_housing_data(data_file=os.path.join(local_data_dir, 'housing.csv')):
    if not os.path.exists(local_cached_datafile):
        if not os.path.exists(local_compressed_datafile):
            download_data(data_url, local_compressed_datafile)
        extract_data(local_compressed_datafile, local_data_dir)
    return pd.read_csv(local_cached_datafile)