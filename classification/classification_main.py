from preprocessing import preprocess
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


pro = preprocess()

train_features, train_labels, test_features, test_labels = pro.importing_data()

train_features_norm = pro.normalization(train_features)
test_features_norm = pro.normalization(test_features)

def train_and_evaluate_model(model):
    model.fit(train_features_norm, train_labels)
    test_pred = model.predict(test_features_norm)

    recall = recall_score(test_labels, test_pred)
    precision = precision_score(test_labels, test_pred)

    F1score = 2 * recall * precision / (precision+recall)

    print(f"{model}")
    print(F1score)

logreg = LogisticRegression()
dct = DecisionTreeClassifier()
svc = SVC()

train_and_evaluate_model(logreg)
train_and_evaluate_model(dct)
train_and_evaluate_model(svc)

voting_clf = VotingClassifier(estimators=[('lr', logreg), ('dt', dct), ('svr', svc)])

train_and_evaluate_model(voting_clf)