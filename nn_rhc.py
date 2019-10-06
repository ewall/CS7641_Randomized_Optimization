#!/usr/bin/env python -W ignore::DeprecationWarning

#

import mlrose
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_files
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

SEED=1

# Load & preprocess datasets
X_train, y_train, X_test, y_test = load_svmlight_files(("data/htru_train.dat", "data/htru_test.dat"))
X_train = csr_matrix(X_train).todense()  # make dense matrices from sparse
X_test = csr_matrix(X_test).todense()
y_train[y_train < 0] = 0  # change -1's to zeros so sklearn recongizes this as binary
y_test[y_test < 0] = 0

# build model
nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[5], activation='sigmoid',
                                 algorithm='random_hill_climb', restarts=10,
                                 max_iters=10000, bias=False, is_classifier=True,
                                 learning_rate=0.1, early_stopping=True,
                                 clip_max=5, max_attempts=30, random_state=SEED)
nn_model1.fit(X_train, y_train)

# predict & assess on training data
y_train_pred = nn_model1.predict(X_train)
y_train_error = 1.0 - accuracy_score(y_train, y_train_pred)
y_train_kappa = cohen_kappa_score(y_train, y_train_pred)  #, labels=target_names)
y_train_confusion = confusion_matrix(y_train, y_train_pred)  #, labels=target_names)
print("# TRAIN DATA:\nerror=", y_train_error, "\nkappa=", y_train_kappa, "\nconfusion matrix=\n", y_train_confusion)

# predict & assess on test data
y_test_pred = nn_model1.predict(X_test)
y_test_error = 1.0 - accuracy_score(y_test, y_test_pred)
y_test_kappa = cohen_kappa_score(y_test, y_test_pred)  #, labels=target_names)
y_test_confusion = confusion_matrix(y_test, y_test_pred)  #, labels=target_names)
print("\n# TEST DATA:\nerror=", y_test_error, "\nkappa=", y_test_kappa, "\nconfusion matrix=\n", y_test_confusion)
