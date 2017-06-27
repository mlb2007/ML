import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.neighbors

def country_data(oecd_bli, gdp_pc):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_pc.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_pc.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_pc, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    sample_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
    # missing_data = full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[remove_indices]

    ## === PLT ===
    #sample_data.plot(kind='scatter', x="GDP per capita", y='Life satisfaction', figsize=(5,3))
    #plt.axis([0, 60000, 0, 10])
    #plt.show()
    ## ===========

    return sample_data


if __name__ == '__main__':
    oecd_bli = pd.read_csv('oecd_bli_2015.csv', thousands=",")
    gdp_pc = pd.read_csv('gdp_per_capita.csv', thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

    country_stats = country_data(oecd_bli, gdp_pc)

    X = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]

    # Visualize the data
    # country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
    # plt.show()

    # Select a linear model
    model = sklearn.linear_model.LinearRegression()

    # k-neighbors model
    model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

    # Train the model
    model.fit(X, y)

    # Make a prediction for Cyprus
    X_new = [[22587]]  # Cyprus' GDP per capita
    print(model.predict(X_new)) # outputs [[ 5.96242338]]

