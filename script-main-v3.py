"""
Modeling and predicting datasets with ensemble methods
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,auc,cohen_kappa_score
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation, metrics,tree
import pandas as pd
import script-loaddata-v3
import time

def save_tofile(content,filename='/home/bigdatatech14/bigdatatech14/log/dt.log',mode='a'):
  file=open(filename,mode)
  filetime=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
  file.write(filetime+'\t'+str(content)+'\n')
  file.close()

save_tofile('=====PROGRAM STARTED AT '+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+'=====')

dftrain, dftest=script-loaddata-v3.load()
X_dev,X_test,Y_dev,Y_test=train_test_split(dftrain.drop('risk_flag',1),dftrain.risk_flag,test_size=0.20)

n_trees = 100
n_folds = 5

# Our level 0 classifiers
clfs = [
    RandomForestClassifier(n_estimators = n_trees, criterion = 'gini'),
    ExtraTreesClassifier(n_estimators = n_trees * 2, criterion = 'gini'),
    GradientBoostingClassifier(n_estimators = n_trees),
]

# Ready for cross validation
skf = list(StratifiedKFold(Y_dev, n_folds))

# Pre-allocate the data
blend_train = np.zeros((X_dev.shape[0], len(clfs))) # Number of training data x Number of classifiers
blend_test = np.zeros((X_test.shape[0], len(clfs))) # Number of testing data x Number of classifiers

print 'X_test.shape = %s' % (str(X_test.shape))
print 'blend_train.shape = %s' % (str(blend_train.shape))
print 'blend_test.shape = %s' % (str(blend_test.shape))

# For each classifier, we train the number of fold times (=len(skf))
for j, clf in enumerate(clfs):
    print 'Training classifier [%s]' % (j)
    blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
    for i, (train_index, cv_index) in enumerate(skf):
        print 'Fold [%s]' % (i)

        # This is the training and validation set
        X_train = X_dev.iloc[train_index]
        Y_train = Y_dev.iloc[train_index]
        X_cv = X_dev.iloc[cv_index]
        Y_cv = Y_dev.iloc[cv_index]

        clf.fit(X_train, Y_train)

        # This output will be the basis for our blended classifier to train against,
        # which is also the output of our classifiers
        blend_train[cv_index, j] = clf.predict(X_cv)
        blend_test_j[:, i] = clf.predict(X_test)
    # Take the mean of the predictions of the cross validation set
    blend_test[:, j] = blend_test_j.mean(1)

print 'Y_dev.shape = %s' % (Y_dev.shape)

# Start blending!
bclf = LogisticRegression()
bclf.fit(blend_train, Y_dev)

# Predict now
Y_test_predict = bclf.predict(blend_test)
score = metrics.accuracy_score(Y_test, Y_test_predict)
print 'Accuracy = %s' % (score)
