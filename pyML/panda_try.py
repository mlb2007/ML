import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import download as urlutils
import os
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

def func():
    pseries = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
    print pseries.head(5)


def get_housing_data(filepath, filename):
    full_path = os.path.join(filepath, filename)
    data = pd.read_csv(full_path)
    return data


# does not work, always returns False .. why?
def test_set_check(identifier, test_ratio, hash):
    t_val = hash(np.int64(identifier)).digest()[-1] < (256 * test_ratio)
    return t_val


def split_train_test_by_id(data, id_column, test_ratio, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))

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

    # print data.head()
    # print data['ocean_proximity'].value_counts()
    # print data.describe()
    # data.hist(bins=50, figsize=(20, 15))
    # plt.show()

    # # adds an `index' column and splits data into train and no-train set
    # # does not work ...
    # indexed_data = data.reset_index()
    # no_test, test = split_train_test_by_id(indexed_data, 'index', 0.2)
    # print "No test:", no_test
    # print "\n"
    # print "Test:", test
    
    # create a income category, bin the data into 5 bins (divide by 1.5)
    data['income_cat'] = np.ceil(data['median_income']/1.5)
    # Now bin the housing['income_cat']
    data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)
    # data['income_cat'].hist(bins=20, figsize=(10, 10))
    # plt.show()
    # now split corresponding to this binning using sci-kit
    split_obj = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split_obj.split(data, data['income_cat']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    print data['income_cat'].value_counts()/len(data)

    for _data in (strat_train_set, strat_test_set):
        _data.drop('income_cat', axis=1, inplace=True)

    # Visual analysis of data ...
    use_data = strat_train_set.copy()

    #plotdata = use_data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
    #plt.show(plotdata)

    #plotpopu = use_data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=use_data['population']/100,
    #                        c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True, label='population')
    #plt.show(plotpopu)

    # find all correlations ...
    #corr_matrix = use_data.corr()
    #print corr_matrix['median_house_value'].sort_values(ascending=False)

    # all scatter plots of correlation matrix
    #attribs = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    #ff = scatter_matrix(use_data[attribs])
    #plt.show(ff[0][0])

    #plotdata = use_data.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.7)
    #plt.show(plotdata)

    # create extra data that makes sense
    use_data['rooms_per_household'] = use_data['total_rooms']/use_data['households']
    use_data['bedrooms_fraction'] = use_data['total_bedrooms']/use_data['total_rooms']
    use_data['persons_per_household'] = use_data['population']/use_data['households']

    # correlation again
    corr_matrix = use_data.corr()
    print corr_matrix['median_house_value'].sort_values(ascending=False)

if __name__ == '__main__':
    housing_analysis()
