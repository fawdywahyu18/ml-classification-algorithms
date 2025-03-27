"""
Assignment 1 "Machine Learning and Data Mining"
@author: mwah0641 (Muhammad Fawyd Renardi Wahyu)
"""

# Import all libraries
from sklearn.model_selection import StratifiedKFold

# Ignore future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Load the rice dataset: rice-final2.csv
import pandas as pd

# Load dataset
file_path = "rice-final2.csv"  # Adjust the path if needed
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset Information:")
print(df.info())

# Display summary statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Display first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values per column:")
print(missing_values)

# Pre-process dataset
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Convert all feature columns to float
for col in df.columns[:-1]:  
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, coercing errors to NaN

# Fill missing values with column mean
imputer = SimpleImputer(strategy='mean')
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])  # Apply only to feature columns

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Convert class labels: "class1" -> 0, "class2" -> 1
df.iloc[:, -1] = df.iloc[:, -1].replace({"class1": 0, "class2": 1}).astype(int)

# Convert DataFrame to NumPy arrays
X = df.iloc[:, :-1].to_numpy()  # Features
y = df.iloc[:, -1].to_numpy()   # Class labels

# Define the modified print_data function
def print_data(X, y, n_rows=10):
    """Takes a numpy data array and target and prints the first n_rows.
    
    Arguments:
        X: numpy array of shape (n_examples, n_features)
        y: numpy array of shape (n_examples)
        n_rows: number of rows to print (default is 10)
    """
    for example_num in range(n_rows):
        # Print feature values formatted to 4 decimal places
        print(",".join("{:.4f}".format(feature) for feature in X[example_num]), end=",")
        # Print class label without decimal places
        print(y[example_num])

# Call print_data function with X and y
print_data(X, y, n_rows=10)

# Part 1: Cross-validation without parameter tuning

from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure y is an integer array
y = np.array(y, dtype=int)  # Convert y to integer type

# Define StratifiedKFold with 10 splits and random_state=0
cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.linear_model import LogisticRegression
# Function for Logistic Regression Classifier
def logregClassifier(X, y):
    """
    Trains and evaluates a Logistic Regression model using 10-fold stratified cross-validation.

    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.

    Returns:
        Mean cross-validation accuracy score.
    """
    model = LogisticRegression(random_state=0, max_iter=1000)  # Ensure convergence with more iterations
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

from sklearn.naive_bayes import GaussianNB

# Function for Naïve Bayes Classifier
def nbClassifier(X, y):
    """
    Trains and evaluates a Gaussian Naïve Bayes model using 10-fold stratified cross-validation.

    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.

    Returns:
        Mean cross-validation accuracy score.
    """
    model = GaussianNB()
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

from sklearn.tree import DecisionTreeClassifier

# Function for Decision Tree Classifier
def dtClassifier(X, y):
    """
    Trains and evaluates a Decision Tree model using 10-fold stratified cross-validation.
    
    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.

    Returns:
        Mean cross-validation accuracy score.
    """
    model = DecisionTreeClassifier(criterion="entropy", random_state=0)  # Using Information Gain (Entropy)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Function for Bagging with Decision Trees
def bagDTClassifier(X, y, n_estimators=50, max_samples=1.0, max_depth=None):
    """
    Trains and evaluates a Bagging model with Decision Trees using 10-fold stratified cross-validation.

    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.
        n_estimators: Number of base estimators (default=50).
        max_samples: Fraction of samples per estimator (default=1.0).
        max_depth: Maximum depth of each decision tree (default=None for unlimited).

    Returns:
        Mean cross-validation accuracy score.
    """
    base_tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=0)
    model = BaggingClassifier(base_tree, n_estimators=n_estimators, max_samples=max_samples, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

# Function for AdaBoost with Decision Trees
def adaDTClassifier(X, y, n_estimators=50, learning_rate=1.0, max_depth=1):
    """
    Trains and evaluates an AdaBoost model with Decision Trees using 10-fold stratified cross-validation.

    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.
        n_estimators: Number of weak learners (default=50).
        learning_rate: Learning rate (default=1.0).
        max_depth: Maximum depth of base decision tree (default=1).

    Returns:
        Mean cross-validation accuracy score.
    """
    base_tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=0)
    model = AdaBoostClassifier(base_tree, n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

# Function for Gradient Boosting
def gbClassifier(X, y, n_estimators=50, learning_rate=0.1):
    """
    Trains and evaluates a Gradient Boosting model using 10-fold stratified cross-validation.

    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.
        n_estimators: Number of boosting stages (default=50).
        learning_rate: Learning rate (default=0.1).

    Returns:
        Mean cross-validation accuracy score.
    """
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

# Parameters for Part 1:

# Bagging
bag_n_estimators = 50
bag_max_samples = 100
bag_max_depth = 5

# AdaBoost
ada_n_estimators = 50
ada_learning_rate = 0.5
ada_max_depth = 5

# Gradient Boosting
gb_n_estimators = 50
gb_learning_rate = 0.5

# Run classifiers and store results
logreg_score = logregClassifier(X, y)
nb_score = nbClassifier(X, y)
dt_score = dtClassifier(X, y)
bag_score = bagDTClassifier(X, y, n_estimators=bag_n_estimators, max_samples=0.8, max_depth=bag_max_depth)
ada_score = adaDTClassifier(X, y, n_estimators=ada_n_estimators, learning_rate=ada_learning_rate, max_depth=ada_max_depth)
gb_score = gbClassifier(X, y, n_estimators=gb_n_estimators, learning_rate=gb_learning_rate)

# Print results for each classifier in Part 1 to 4 decimal places
print(f"LogR average cross-validation accuracy: {logreg_score:.4f}")
print(f"NB average cross-validation accuracy: {nb_score:.4f}")
print(f"DT average cross-validation accuracy: {dt_score:.4f}")
print(f"Bagging average cross-validation accuracy: {bag_score:.4f}")
print(f"AdaBoost average cross-validation accuracy: {ada_score:.4f}")
print(f"GB average cross-validation accuracy: {gb_score:.4f}")


## Part 2: Cross-validation with parameter tuning
# Import necessary libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Define the stratified KFold for cross-validation
cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# KNN
k = [1, 3, 5, 7]
p = [1, 2]


def bestKNNClassifier(X, y):
    """
    Finds the best KNN classifier using grid search with 10-fold stratified cross-validation.
    
    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.
    
    Returns:
        best_k: Best value of k (n_neighbors).
        best_p: Best value of p (distance metric).
        best_cv_accuracy: Best cross-validation accuracy.
        test_accuracy: Test set accuracy.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    
    # Define the parameter grid for KNN
    param_grid = {'n_neighbors': k, 'p': p}
    
    # Initialize KNN classifier
    knn = KNeighborsClassifier()
    
    # Perform grid search with 10-fold stratified cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=cvKFold, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and cross-validation accuracy
    best_k = grid_search.best_params_['n_neighbors']
    best_p = grid_search.best_params_['p']
    best_cv_accuracy = grid_search.best_score_
    
    # Evaluate on the test set
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    return best_k, best_p, best_cv_accuracy, test_accuracy

# SVM
# You should use SVC from sklearn.svm with kernel set to 'rbf'
C = [0.01, 0.1, 1, 5] 
gamma = [0.01, 0.1, 1, 10]

def bestSVMClassifier(X, y):
    """
    Finds the best SVM classifier using grid search with 10-fold stratified cross-validation.
    
    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.
    
    Returns:
        best_C: Best value of C (regularization parameter).
        best_gamma: Best value of gamma (kernel coefficient).
        best_cv_accuracy: Best cross-validation accuracy.
        test_accuracy: Test set accuracy.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    
    # Define the parameter grid for SVM
    param_grid = {'C': C, 'gamma': gamma}
    
    # Initialize SVM classifier with RBF kernel
    svm = SVC(kernel='rbf', random_state=0)
    
    # Perform grid search with 10-fold stratified cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=cvKFold, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and cross-validation accuracy
    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    best_cv_accuracy = grid_search.best_score_
    
    # Evaluate on the test set
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    return best_C, best_gamma, best_cv_accuracy, test_accuracy

# Random Forest
# You should use RandomForestClassifier from sklearn.ensemble with information gain and max_features set to ‘sqrt’.
n_estimators = [10, 30, 60, 100]
max_leaf_nodes = [6, 12]

def bestRFClassifier(X, y):
    """
    Finds the best Random Forest classifier using grid search with 10-fold stratified cross-validation.
    
    Arguments:
        X: numpy array of shape (n_samples, n_features), feature matrix.
        y: numpy array of shape (n_samples,), target labels.
    
    Returns:
        best_n_estimators: Best value of n_estimators (number of trees).
        best_max_leaf_nodes: Best value of max_leaf_nodes (maximum number of leaf nodes).
        best_cv_accuracy: Best cross-validation accuracy.
        test_accuracy: Test set accuracy.
        test_macro_f1: Test set macro average F1 score.
        test_weighted_f1: Test set weighted average F1 score.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    
    # Define the parameter grid for Random Forest
    param_grid = {'n_estimators': n_estimators, 'max_leaf_nodes': max_leaf_nodes}
    
    # Initialize Random Forest classifier with information gain and max_features='sqrt'
    rf = RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=0)
    
    # Perform grid search with 10-fold stratified cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=cvKFold, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and cross-validation accuracy
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_leaf_nodes = grid_search.best_params_['max_leaf_nodes']
    best_cv_accuracy = grid_search.best_score_
    
    # Evaluate on the test set
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    test_weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    return best_n_estimators, best_max_leaf_nodes, best_cv_accuracy, test_accuracy, test_macro_f1, test_weighted_f1

# Perform Grid Search with 10-fold stratified cross-validation (GridSearchCV in sklearn). 
# The stratified folds from cvKFold should be provided to GridSearchV

# This should include using train_test_split from sklearn.model_selection with stratification and random_state=0
# Print results for each classifier here. All results should be printed to 4 decimal places except for
# "k", "p", n_estimators" and "max_leaf_nodes" which should be printed as integers.

# KNN
best_k, best_p, knn_cv_accuracy, knn_test_accuracy = bestKNNClassifier(X, y)
print("KNN best k:", best_k)
print("KNN best p:", best_p)
print("KNN cross-validation accuracy: {:.4f}".format(knn_cv_accuracy))
print("KNN test set accuracy: {:.4f}".format(knn_test_accuracy))
print()

# SVM
best_C, best_gamma, svm_cv_accuracy, svm_test_accuracy = bestSVMClassifier(X, y)
print("SVM best C:", best_C)
print("SVM best gamma:", best_gamma)
print("SVM cross-validation accuracy: {:.4f}".format(svm_cv_accuracy))
print("SVM test set accuracy: {:.4f}".format(svm_test_accuracy))
print()

# Random Forest
best_n_estimators, best_max_leaf_nodes, rf_cv_accuracy, rf_test_accuracy, rf_macro_f1, rf_weighted_f1 = bestRFClassifier(X, y)
print("RF best n_estimators:", best_n_estimators)
print("RF best max_leaf_nodes:", best_max_leaf_nodes)
print("RF cross-validation accuracy: {:.4f}".format(rf_cv_accuracy))
print("RF test set accuracy: {:.4f}".format(rf_test_accuracy))
print("RF test set macro average F1: {:.4f}".format(rf_macro_f1))
print("RF test set weighted average F1: {:.4f}".format(rf_weighted_f1))
