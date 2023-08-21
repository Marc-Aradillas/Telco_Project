# imports
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# custom imports
import wrangle as w


### VARIABLES ###

train, val, test = w.wrangle_telco()

### X and y ###

def xy_split(data):
    X = data.drop(columns=['churn'])
    y = data['churn']
    return X, y

    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    X_test, y_test = xy_split(test)

### Modeling and Metrics ###


################## Decision Tree ##########################

def train_and_evaluate_model(train, val, test, target_col='churn', max_depth=3, random_state=42):
    train = train.drop(columns='customer_id')
    val = val.drop(columns='customer_id')
    test = test.drop(columns='customer_id')
    
    X_train = train.drop(columns=target_col)
    y_train = train[target_col]

    X_val = val.drop(columns=target_col)
    y_val = val[target_col]

    # X_test = test.drop(columns=target_col)
    # y_test = test[target_col]

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)

    # Classification Report
    print(classification_report(y_train, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_train, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()

    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train) * 100))

    print('Accuracy of Decision Tree classifier on validation set: {:.2f}\n'
      .format(clf.score(X_val, y_val) * 100))

    # Calculate metrics
    accuracy = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
    recall = round(TP / (TP + FN) * 100, 2)
    precision = round(TP / (TP + FP) * 100, 2)
    f1_score = round((2 * (precision * recall) / (precision + recall)), 2)

    metrics_data = {
        'Metric': ['Accuracy', 'Recall', 'Precision', 'F1-Score'],
        'Value': [accuracy, recall, precision, f1_score]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    return clf, metrics_df



################### random forest function ###################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_evaluate_rf_with_grid_search(train, val, seed=42):
    train, val, test = w.wrangle_telco()
    train
    
    train = train.drop(columns='customer_id')
    val = val.drop(columns='customer_id')
    #test = test.drop(columns='customer_id')
    
    def xy_split(data):
        X = data.drop(columns=['churn'])
        y = data['churn']
        return X, y
    
    # Create X, y for train, validation, and test
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    #X_test, y_test = xy_split(test)
    
    # Random Forest model with default parameters
    rf_default = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
    rf_default.fit(X_train, y_train)

    # Accuracy scores
    train_acc = rf_default.score(X_train, y_train) * 100
    val_acc = rf_default.score(X_val, y_val) * 100
    
    print('Accuracy of Random Forest classifier on training set: {:.2f}%'.format(train_acc))
    print('Accuracy of Random Forest classifier on validation set: {:.2f}%'.format(val_acc))
    
    # Classification Report
    y_pred = rf_default.predict(X_val)
    print(classification_report(y_val, y_pred))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_val, y_pred)
    TN, FP, FN, TP = conf_matrix.ravel()
    
    # Grid Search for best parameters
    param_grid = {
        'max_depth': range(2, 22),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=seed), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Grid Search Results:")
    print("Best Parameters:", best_params)
    print("Best Score: {:.2f}%".format(best_score * 100))
    
    return rf_default, best_rf_model



################### log reg function ###################

from sklearn.linear_model import LogisticRegression

def evaluate_logistic_regression(train, val):
    # Drop customer_id column
    train = train.drop(columns='customer_id')
    val = val.drop(columns='customer_id')
    #test = test.drop(columns='customer_id')
    
    # Split into X and y
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    #X_test, y_test = xy_split(test)
    
    # Calculate the baseline
    baseline = (y_train == 0).mean() * 100
    
    # Initialize Logistic Regression model
    seed = 42
    logit = LogisticRegression(random_state=seed, max_iter=400, solver='liblinear', penalty='l2')
    
    # Train Logistic Regression model on training set
    logit.fit(X_train, y_train)
    
    # Evaluate on training set
    train_accuracy = logit.score(X_train, y_train) * 100
    
    # Print results
    print("Baseline is {:.2f}%".format(baseline))
    print("Logistic Regression using all features.")
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}%'.format(train_accuracy))
    
    # Evaluate on validation set
    val_accuracy = logit.score(X_val, y_val) * 100
    print("Baseline is {:.2f}%".format(baseline))
    print("Logistic Regression using all features.")
    print('Accuracy of Logistic Regression classifier on validation set: {:.2f}%'.format(val_accuracy))
    
    

### TEST ###
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def test_model():
    """
    The function loads, preprocesses, splits, trains a random forest model with specified parameters, 
    prints the accuracy score on the test data, and displays a bar plot of accuracy for each class.
    """
    # Load and preprocess your data using the appropriate functions from wrangle.py
    train, val, test = w.wrangle_telco()

    train = train.drop(columns='customer_id')
    val = val.drop(columns='customer_id')
    test = test.drop(columns='customer_id')

    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    X_test, y_test = xy_split(test)
    
    rf = RandomForestClassifier(max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)
    
    # Predict on test data
    y_pred = rf.predict(X_test)
    
    print('Random Forest','\n')
    print(f'Accuracy on test: {round(accuracy * 100, 2)}%','\n')
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create classification report
    cls_report = classification_report(y_test, y_pred)
    print('Classification Report:')
    print(cls_report)
    
    # Calculate accuracy for each class
    class_accuracy = [cm[i, i] / cm[i, :].sum() for i in range(cm.shape[0])]
    
    # Create a bar plot of class accuracy
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(range(len(class_accuracy))), y=class_accuracy)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Each Class')
    plt.xticks(list(range(len(class_accuracy))), ['Class 0', 'Class 1'])  # Replace with actual class labels
    plt.ylim(0, 1.0)
    plt.show()
