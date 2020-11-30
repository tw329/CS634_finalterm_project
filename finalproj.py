import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

def plot_classification_report(cr, title='Classification report ', cmap=plt.cm.Blues):

    lines = cr.split('\n')
    labels = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        if len(t)==0:
            break
        labels.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        plotMat.append(v)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(labels))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    
def recall(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    result = TP/(TP+FN)
    return result
def precision(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    result = TP/(TP+FP)
    return result
def f1_score(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    result = 2*TP/(2*TP+FP+FN)
    return result
def acc(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    result = (TP+FN)/(TP+FP+FN+TN)
    return result
def err_rate(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    result = (FP+FN)/(TP+FP+FN+TN)
def TSS(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    result = TP/(TP+FN)-FP/(FP+TN)
    return result

def HSS(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    result = 2 * (TP*TN-FP*FN)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
    return result

def BACC(cm):
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    result = 0.5 * (TP/(TP+FN)+TN/(FP+TN))
    return result

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

pipe_lr = [Pipeline([('pca', PCA(n_components=5)), ('clf', svm.SVC(C=1, kernel='linear', gamma=0.001))]),
          Pipeline([('pca', PCA(n_components=5)), ('clf', LogisticRegression())]),
          Pipeline([('pca', PCA(n_components=5)), ('clf', RandomForestClassifier(max_depth=2))])]

for clf in pipe_lr:
    print(clf)
    kfold = KFold(n_splits=10)
    kfold.get_n_splits(X)
    reports = []
    cms = []
    for train, test in kfold.split(X):
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        cm = metrics.confusion_matrix(y[test], y_pred)
        cms.append(cm)
        plot_confusion_matrix(clf, X[test], y[test], cmap=plt.cm.Blues)
        plt.show()
        print("BACC:", BACC(cm))
        print("TSS:", TSS(cm))
        print("HSS:", HSS(cm))
        plot_classification_report(metrics.classification_report(y[test], y_pred))
    sum_cm = [[0, 0], [0, 0]]
    for i in range(len(cms)):
        sum_cm[0] = [x + y for x, y in zip(sum_cm[0], cms[i][0])]
        sum_cm[1] = [x + y for x, y in zip(sum_cm[1], cms[i][1])]
    avg_cm = [[0, 0], [0, 0]]
    avg_cm[0][0] = sum_cm[0][0]/10
    avg_cm[0][1] = sum_cm[0][1]/10
    avg_cm[1][0] = sum_cm[1][0]/10
    avg_cm[1][1] = sum_cm[1][1]/10
    print("average of all 10 folds:\n")
    print(avg_cm)
    print("Overall BACC:", BACC(avg_cm))
    print("Overall TSS:", TSS(avg_cm))
    print("Overall HSS:", HSS(avg_cm))
    print("Overall Recall:", recall(avg_cm))
    print("Overall Precision:", precision(avg_cm))
    print("Overall F1 measure:", f1_score(avg_cm))
    print("Acc:", acc(avg_cm))
    print("Error rate:", err_rate(avg_cm))