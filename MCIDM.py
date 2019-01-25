import numpy as np
import pandas as pd
import statistics
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalanceCascade
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.metrics import geometric_mean_score

def prepare_data(data):
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    features = data.iloc[:, 0:-1].values.astype(float)
    labels = data.iloc[:, -1].values.astype(str)
    classes = np.unique(labels)
    return features, labels, classes

df = pd.read_csv('data/poker-8-9_vs_5.csv')
X,y,z = prepare_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

kf = StratifiedKFold(n_splits=10)
kf.get_n_splits(X, y)

bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=42)
brfc = BalancedRandomForestClassifier(max_depth=2, random_state=0)
eec = EasyEnsembleClassifier(base_estimator=DecisionTreeClassifier(random_state=0), random_state=42)
rbc = RUSBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0),random_state=0)

bbc_score = []
brfc_score = []
eec_score = []
rbc_score = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    bbc.fit(X_train, y_train)
    brfc.fit(X_train, y_train)
    eec.fit(X_train, y_train)
    rbc.fit(X_train, y_train)
    y_pred_bbc = bbc.predict(X_test)
    y_pred_brfc = brfc.predict(X_test)
    y_pred_eec = eec.predict(X_test)
    y_pred_rbc = rbc.predict(X_test)
    bbc_score.append(balanced_accuracy_score(y_test, y_pred_bbc))
    brfc_score.append(balanced_accuracy_score(y_test, y_pred_brfc))
    eec_score.append(balanced_accuracy_score(y_test, y_pred_eec))
    rbc_score.append(balanced_accuracy_score(y_test, y_pred_rbc))

print("\t Average score:\t\t Standard deviation:")
print("bbc\t", sum(bbc_score)/float(len(bbc_score)), "\t", statistics.stdev(bbc_score))
print("brfc\t", sum(brfc_score)/float(len(brfc_score)), "\t", statistics.stdev(brfc_score))
print("eec\t", sum(eec_score)/float(len(eec_score)), "\t", statistics.stdev(eec_score))
print("rbc\t", sum(rbc_score)/float(len(rbc_score)), "\t", statistics.stdev(rbc_score))
