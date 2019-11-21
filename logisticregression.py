#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
plt.rc("font", size=12)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Load dataset
url = "https://raw.githubusercontent.com/TheFirstTimeLord/Data-Science/master/tosses.csv"
names = ['coin-height', 'puddle-width', 'coin-distance', 'drink-drip', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# simple run
LRR = []
LRR.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
results = []
names = []
for name, model in LRR:
	kfold = StratifiedKFold(n_splits=10, random_state=1)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

pyplot.boxplot(results, labels=names)
pyplot.title('Simple Logistic Regression')
pyplot.show()