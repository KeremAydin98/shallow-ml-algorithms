from preprocessing import preprocess
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
import numpy as np

pro = preprocess()

train_df,test_df = pro.importing_data()

train_features,train_labels,test_features,test_labels = pro.remove_uncorrelated(train_df,test_df)

train_features_norm = pro.normalization(train_features)
train_labels_norm = pro.normalization(train_labels)
test_features_norm = pro.normalization(test_features)
test_labels_norm = pro.normalization(test_labels)

def train_and_evaluate_model(model):

  model.fit(train_features_norm,train_labels_norm)
  test_pred = model.predict(test_features_norm)
  mse = mean_squared_error(test_labels_norm, test_pred)
  mse = np.sqrt(mse)

  print(model)
  print(mse)


reg = LinearRegression()
dct = DecisionTreeRegressor()
svr = SVR()

train_and_evaluate_model(reg)
train_and_evaluate_model(dct)
train_and_evaluate_model(svr)

voting_rgs = VotingRegressor(estimators=[('lr', reg), ('dt', dct), ('svr', svr)])

train_and_evaluate_model(voting_rgs)
