import numpy as np
import pandas as pd


class preprocess():

    def importing_data(self):

        train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
        test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
        train_df = train_df.reindex(np.random.permutation(train_df.index))  # shuffle the training set

        return train_df,test_df

    def remove_uncorrelated(self,train_df,test_df):

        print(train_df.corr()["median_house_value"])

        # total bedrooms do not affect the prediction a lot so we are going to create a new column called bedroom per room
        train_df["bedroom_per_rooms"] = train_df["total_bedrooms"] / train_df["total_rooms"]
        test_df["bedroom_per_rooms"] = test_df["total_bedrooms"] / test_df["total_rooms"]

        # Therefore we will drop the columns which have low correlation values
        corr = train_df.corr()["median_high"]

        abs_corr = abs(corr)

        sorted_corr = abs_corr.sort_values()

        low_correlations = sorted_corr[sorted_corr < 0.05]

        low_correlations_indexes = list(low_correlations.index)

        for drop in low_correlations_indexes:
            train_df = train_df.drop(drop, axis=1)
            test_df = test_df.drop(drop, axis=1)

        train_features = train_df.drop("median_house_value", axis=1)
        train_labels = train_df["median_house_value"]
        test_features = train_df.drop("median_house_value", axis=1)
        test_labels = train_df["median_house_value"]

        return train_features,train_labels,test_features,test_labels

    def normalization(self,data):
        data_norm = (data - data.mean()) / data.std()

        return data_norm
