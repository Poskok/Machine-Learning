#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:43:16 2018

@author: mguina
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

logreg = LogisticRegression()

data = LogisticRegression_TwoRegionscsv

X = data[:, 0].transpose()
y = data[:, 1].transpose()


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train = X_train.reshape((len(X_train), 1))
y_train = y_train.reshape((len(y_train), 1))

X_test = X_test.reshape((len(X_test), 1))
y_test = y_test.reshape((len(y_test), 1))

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train.ravel())

y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
        classifier.score(X_test, y_test)))

print(classification_report(y_test, y_pred))