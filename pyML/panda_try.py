import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import download as urlutils
import os
import hashlib


def func():
    pseries = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
    print pseries.head(5)


def get_housing_data(filepath, filename):
    full_path = os.path.join(filepath, filename)
    data = pd.read_csv(full_path)
    return data


def test_set_check(identifier, test_ratio, hash):
    return hash( np.int64( identifier)). digest()[-1] < (2**8) * test_ratio


def split_train_test_by_id(data, id_column, test_ratio, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))

    return data.loc[~in_test_set], data.loc[in_test_set]


# ==
def housing_analysis():
    download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    housing_path = "datasets/housing"
    housing_filename_tgz = 'housing.tgz'
    housing_filename_csv = 'housing.csv'

    # downlaod ...
    # urlutils.download(download_root, housing_path, housing_filename_tgz)

    data = get_housing_data(housing_path, housing_filename_csv)
    #print data.head()
    #print data['ocean_proximity'].value_counts()
    #print data.describe()
    #data.hist(bins=50, figsize=(20, 15))
    #plt.show()

    # adds an `index' column
    indexed_data = data.reset_index()

    no_test, test = split_train_test_by_id(indexed_data, 'index', 0.2)
    print "No test:", no_test
    print "\n"
    print "Test:", test


if __name__ == '__main__':
    housing_analysis()
