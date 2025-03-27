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



