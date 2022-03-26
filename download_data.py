import zipfile
from download import download

print("Downloading the data file. This could take a couple of minutes...")

url = "https://www.dropbox.com/s/anqe5pwawb0gpaw/data.zip?dl=0"

path = "./data/data.zip"

path = download(url=url, path=path, progressbar=True, replace=True)

print("File downloaded...")

print("Extracting data.zip in ./data/")

with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall('./data/')
    
print("Data extracted...")