# CS634_finalterm_project

There will be two parts of my report.

First, I will talk about the data and the methods I use to finish the project. Second, I will show the result of all three classification algorithms.
The data I used is the breast cancer Wisconsin dataset. It includes 569 malignant tumor cell and benign tumor cell samples. Every data has 32 features. The first and second feature are the unique ID number and its diagnosis. So, what I did is reading the data directly from the website (https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data), and transform it to X (data) and y (labels). 

Since there are 30 features in the data, I use PCA (Principle Component Analysis) to reduce the dimension. The classification algorithms I used are logistic regression, random forest, and SVM. The parameters are following:
PCA(n_components=5)
svm.SVC(C=1, kernel='linear', gamma=0.001)
LogisticRegression()
RandomForestClassifier(max_depth=2)

As required, I used 10-fold cross validation, and use the classification report function and the confusion matrix to show the results. 
The package I mainly used is scikit-learn, which includes most of the functions I used, PCA, classifications, confusion matrix, and classification report. I used the matplotlib package to do the visualization.
