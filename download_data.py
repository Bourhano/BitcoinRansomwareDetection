import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

# file source:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00526/data.zip
url = "https://www.dropbox.com/s/gpz7730z5ta1g1p/BitcoinHeistData.csv?dl=1"

public_path = "./data/public/"
private_path = "./data/"
file_name = 'BitcoinHeistData.csv'
file_path = os.path.join(private_path, file_name)

try:
    os.stat(private_path)
except OSError:
    os.mkdir(private_path)

try:
    os.stat(public_path)
except OSError:
    os.mkdir(public_path)

print("Downloading the data file. This could take a couple of minutes...")
r = requests.get(url, allow_redirects=True)
open(file_path, 'wb').write(r.content)
print("File downloaded...")

df = pd.read_csv(file_path)
df.loc[df['label'] != 'white', 'label'] = 1
df.loc[df['label'] == 'white', 'label'] = -1

Y = df[["label"]]
X = df.drop(columns=["label"])

X_public, X_hidden, Y_public, Y_hidden = train_test_split(
    X, Y, random_state=42, test_size=0.3, stratify=Y, shuffle=True
)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_public, Y_public, random_state=42, test_size=0.3,
    stratify=Y_public, shuffle=True
)

X_hidden.loc[:, 'label'] = Y_hidden
X_train.loc[:, 'label'] = Y_train
X_test.loc[:, 'label'] = Y_test

private_set = X_hidden
public_train = X_train
public_test = X_test

private_set.to_csv(os.path.join(private_path, "private.csv.gz"),
                   index=False, compression="gzip")
public_train.to_csv(os.path.join(public_path, "train.csv.gz"),
                    index=False, compression="gzip")
public_test.to_csv(os.path.join(public_path, "test.csv.gz"),
                   index=False, compression="gzip")

os.remove(file_path)
print("Data extracted...")
