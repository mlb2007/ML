import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import download as urlutils
import os
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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


room_index, bedroom_index, population_index, household_index = 3, 4, 5, 6


class CustomAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args, **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, room_index]/X[:, household_index]
        pop_per_household = X[:, population_index]/X[:, household_index]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedroom_index]/X[:,room_index]
            return np.c_[X, rooms_per_household, pop_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, pop_per_household]


class PandaDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attributes].values

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

    # remove the labels (values, Y = f(X), where label means Y
    use_data = strat_train_set.drop("median_house_value", axis=1)
    use_data_labels = strat_train_set['median_house_value'].copy()

    # cleaning N/A and other missing data
    # use imputer
    imputer = Imputer(strategy='median')

    # drop non-numeric column
    use_data_only = use_data.drop('ocean_proximity', axis=1)

    imputer.fit(use_data_only)
    print imputer.statistics_

    npX = imputer.transform(use_data_only)
    transformed_user_data = pd.DataFrame(npX, columns=use_data_only.columns)

    # simple encoder ..
    #encoder = LabelEncoder()
    #use_data_oc = use_data['ocean_proximity']
    #use_data_oc_enc = encoder.fit_transform(use_data_oc)
    #print use_data_oc_enc
    #print encoder.classes_

    ## label binarizer
    encoder = LabelBinarizer()
    use_data_oc = use_data['ocean_proximity']
    use_data_oc_enc = encoder.fit_transform(use_data_oc)
    print use_data_oc_enc

    attr_adder = CustomAttributesAdder(add_bedrooms_per_room=False)
    data_extra_attribs = attr_adder.transform(use_data_only.values)
    #print data_extra_attribs

    attribs_selected = list(use_data_only)
    #print attribs_selected
    cat_attrib = ['ocean_proximity']

    #panda_to_np = PandaDataFrameSelector(attribs_selected)
    #panda_to_np_data = panda_to_np.transform(use_data_only)

    data_pipe = Pipeline([('selector', PandaDataFrameSelector(attribs_selected)),
                         ('imputer', Imputer(strategy='median')),
                         ('attribs_adder', CustomAttributesAdder()),
                          ('std_scaler', StandardScaler())
                        ])

    cat_pipe = Pipeline([('selector', PandaDataFrameSelector(cat_attrib)),
                         ('label_binarizer', LabelBinarizer())])

    full_pipeline = FeatureUnion(transformer_list=[('data_pipe', data_pipe),
                                                   ('cat_pipe', cat_pipe)])

    final_data = full_pipeline.fit_transform(use_data)
    print final_data.shape

    lin_reg = LinearRegression()
    lin_reg.fit(final_data, use_data_labels)

    some_data = use_data.iloc[:5]
    some_label = use_data_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print "Predictions:", lin_reg.predict(some_data_prepared)
    print "Labels:", list(some_label)

    housing_predicitons = lin_reg.predict(final_data)
    lin_mse = mean_squared_error(use_data_labels, housing_predicitons)
    lin_rmse = np.sqrt(lin_mse)
    print "RMSE:", lin_rmse

if __name__ == '__main__':
    housing_analysis()
