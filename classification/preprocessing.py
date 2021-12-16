import numpy as np
import pandas as pd


class preprocess():

    def importing_data(self):

        train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
        test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
        train_df = train_df.reindex(np.random.permutation(train_df.index))  # shuffle the training set

        print(train_df.corr()["median_house_value"])

        train_df["bedroom_per_rooms"] = train_df["total_bedrooms"] / train_df["total_rooms"]
        test_df["bedroom_per_rooms"] = test_df["total_bedrooms"] / test_df["total_rooms"]

        print(train_df.corr()["median_house_value"])

        threshold = 265000
        train_df["median_high"] = (train_df["median_house_value"] > threshold).astype(float)
        test_df["median_high"] = (test_df["median_house_value"] > threshold).astype(float)

        train_df = train_df.drop("total_bedrooms", axis=1)
        train_df = train_df.drop("population", axis=1)
        train_df = train_df.drop("median_house_value", axis=1)

        test_df = test_df.drop("total_bedrooms", axis=1)
        test_df = test_df.drop("population", axis=1)
        test_df = test_df.drop("median_house_value", axis=1)

        train_features = train_df.drop("median_high", axis=1)
        train_labels = train_df["median_high"]

        test_features = train_df.drop("median_high", axis=1)
        test_labels = train_df["median_high"]

        return train_features, train_labels, test_features, test_labels

    def normalization(self,data):

        data_norm = (data - data.mean()) / data.std()

        return data_norm
