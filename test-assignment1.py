import pandas as pd
import numpy as np
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# ----------------------------
# Expected Results (to 4 decimal places as provided)
# ----------------------------
expected_rows = [
    "0.0621,0.4999,0.5410,0.2079,0.2594,0.0613,0",
    "0.8073,0.7474,0.6721,0.2634,0.2038,0.0586,0",
    "0.3105,0.6030,0.4187,0.0000,0.0000,0.0900,0",
    "0.3105,0.5618,0.6148,0.3604,0.0000,0.0950,0",
    "0.1863,0.8144,0.6230,0.4990,0.4539,0.1597,1",
    "0.1863,0.6039,0.4754,0.1525,0.1000,0.0655,1",
    "0.6832,0.7114,0.6230,0.0000,0.0000,0.0877,1",
    "0.5589,0.5258,0.6230,0.5129,0.0000,0.0869,0",
    "0.1242,0.4639,0.5574,0.5822,0.0000,0.1009,1",
    "0.2484,0.5722,0.5902,0.6515,0.3835,0.0979,0"
]

expected_part1 = {
    "logreg_score": 0.6700,
    "nb_score": 0.6555,
    "dt_score": 0.7702,
    "bag_score": 0.7514,
    "ada_score": 0.7562,
    "gb_score": 0.7464
}

expected_part2 = {
    "knn": {
        "best_k": 1,
        "best_p": 1,
        "cv_accuracy": 0.7329,
        "test_accuracy": 0.6415
    },
    "svm": {
        "best_C": 5.0000,
        "best_gamma": 10.0000,
        "cv_accuracy": 0.6858,
        "test_accuracy": 0.5849
    },
    "rf": {
        "best_n_estimators": 60,
        "best_max_leaf_nodes": 12,
        "cv_accuracy": 0.7883,
        "test_accuracy": 0.6981,
        "macro_f1": 0.6845,
        "weighted_f1": 0.6956
    }
}

tol = 1e-4  # Tolerance for float comparisons

# ----------------------------
# Helper function to compare floats and print PASS/FAIL
# ----------------------------
def check_result(name, computed, expected, tol=tol):
    if isinstance(computed, float):
        result = "PASS" if np.isclose(computed, expected, atol=tol) else "FAIL"
    else:
        result = "PASS" if computed == expected else "FAIL"
    print(f"{name}: computed={computed}, expected={expected} --> {result}")

# ----------------------------
# Pre-processing Section using test-before.csv
# ----------------------------
file_path = "test-before.csv"  # File for testing

# Load dataset
df = pd.read_csv(file_path)

# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Convert all feature columns to float (assume last column is class label)
for col in df.columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with column mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Normalize the features using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Convert class labels: "class1" -> 0, "class2" -> 1 (if applicable)
df.iloc[:, -1] = df.iloc[:, -1].replace({"class1": 0, "class2": 1}).astype(int)

# Convert DataFrame to NumPy arrays
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

# ----------------------------
# Function to get printed rows as list of strings
# ----------------------------
def get_printed_rows(X, y, n_rows=10):
    rows = []
    for example_num in range(n_rows):
        row_str = ",".join("{:.4f}".format(feature) for feature in X[example_num]) + "," + str(y[example_num])
        rows.append(row_str)
    return rows

print("Testing Pre-processed Data (first 10 rows):")
printed_rows = get_printed_rows(X, y, n_rows=10)
all_rows_correct = True
for i, (computed_row, expected_row) in enumerate(zip(printed_rows, expected_rows)):
    if computed_row == expected_row:
        print(f"Row {i+1}: PASS")
    else:
        print(f"Row {i+1}: FAIL\n  Computed: {computed_row}\n  Expected: {expected_row}")
        all_rows_correct = False
print("Pre-processed Data Test:", "PASS" if all_rows_correct else "FAIL")
print()

# ----------------------------
# Part 1: Cross-validation without parameter tuning
# ----------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Ensure y is an integer array
y = np.array(y, dtype=int)

cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

def logregClassifier(X, y):
    model = LogisticRegression(random_state=0, max_iter=1000)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

def nbClassifier(X, y):
    model = GaussianNB()
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

def dtClassifier(X, y):
    model = DecisionTreeClassifier(criterion="entropy", random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

def bagDTClassifier(X, y, n_estimators=50, max_samples=1.0, max_depth=None):
    base_tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=0)
    model = BaggingClassifier(base_tree, n_estimators=n_estimators, max_samples=max_samples, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

def adaDTClassifier(X, y, n_estimators=50, learning_rate=1.0, max_depth=1):
    base_tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=0)
    model = AdaBoostClassifier(base_tree, n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

def gbClassifier(X, y, n_estimators=50, learning_rate=0.1):
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=0)
    scores = cross_val_score(model, X, y, cv=cvKFold, scoring='accuracy')
    return scores.mean()

# Parameters for Part 1:
bag_n_estimators = 50
bag_max_samples = 0.8  # using fraction of samples
bag_max_depth = 5

ada_n_estimators = 50
ada_learning_rate = 0.5
ada_max_depth = 5

gb_n_estimators = 50
gb_learning_rate = 0.5

# Run Part 1 classifiers
logreg_score = logregClassifier(X, y)
nb_score = nbClassifier(X, y)
dt_score = dtClassifier(X, y)
bag_score = bagDTClassifier(X, y, n_estimators=bag_n_estimators, max_samples=bag_max_samples, max_depth=bag_max_depth)
ada_score = adaDTClassifier(X, y, n_estimators=ada_n_estimators, learning_rate=ada_learning_rate, max_depth=ada_max_depth)
gb_score = gbClassifier(X, y, n_estimators=gb_n_estimators, learning_rate=gb_learning_rate)

print("Testing Part 1 (Cross-validation without parameter tuning):")
check_result("LogR average CV accuracy", round(logreg_score, 4), expected_part1["logreg_score"])
check_result("NB average CV accuracy", round(nb_score, 4), expected_part1["nb_score"])
check_result("DT average CV accuracy", round(dt_score, 4), expected_part1["dt_score"])
check_result("Bagging average CV accuracy", round(bag_score, 4), expected_part1["bag_score"])
check_result("AdaBoost average CV accuracy", round(ada_score, 4), expected_part1["ada_score"])
check_result("GB average CV accuracy", round(gb_score, 4), expected_part1["gb_score"])
print()

# ----------------------------
# Part 2: Cross-validation with parameter tuning
# ----------------------------
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

cvKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# KNN parameters
k = [1, 3, 5, 7]
p = [1, 2]

def bestKNNClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    param_grid = {'n_neighbors': k, 'p': p}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=cvKFold, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_k = grid_search.best_params_['n_neighbors']
    best_p = grid_search.best_params_['p']
    best_cv_accuracy = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    return best_k, best_p, best_cv_accuracy, test_accuracy

# SVM parameters
C = [0.01, 0.1, 1, 5]
gamma = [0.01, 0.1, 1, 10]

def bestSVMClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    param_grid = {'C': C, 'gamma': gamma}
    svm = SVC(kernel='rbf', random_state=0)
    grid_search = GridSearchCV(svm, param_grid, cv=cvKFold, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    best_cv_accuracy = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    return best_C, best_gamma, best_cv_accuracy, test_accuracy

# Random Forest parameters
n_estimators_grid = [10, 30, 60, 100]
max_leaf_nodes_grid = [6, 12]

def bestRFClassifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    param_grid = {'n_estimators': n_estimators_grid, 'max_leaf_nodes': max_leaf_nodes_grid}
    rf = RandomForestClassifier(criterion='entropy', max_features='sqrt', random_state=0)
    grid_search = GridSearchCV(rf, param_grid, cv=cvKFold, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_n_estimators = grid_search.best_params_['n_estimators']
    best_max_leaf_nodes = grid_search.best_params_['max_leaf_nodes']
    best_cv_accuracy = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    test_weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    return best_n_estimators, best_max_leaf_nodes, best_cv_accuracy, test_accuracy, test_macro_f1, test_weighted_f1

# Run parameter tuning classifiers (Part 2)
best_k, best_p, knn_cv_accuracy, knn_test_accuracy = bestKNNClassifier(X, y)
best_C, best_gamma, svm_cv_accuracy, svm_test_accuracy = bestSVMClassifier(X, y)
best_n_estimators, best_max_leaf_nodes, rf_cv_accuracy, rf_test_accuracy, rf_macro_f1, rf_weighted_f1 = bestRFClassifier(X, y)

print("Testing Part 2 (Cross-validation with parameter tuning):")
# KNN Results
check_result("KNN best k", best_k, expected_part2["knn"]["best_k"])
check_result("KNN best p", best_p, expected_part2["knn"]["best_p"])
check_result("KNN CV accuracy", round(knn_cv_accuracy, 4), expected_part2["knn"]["cv_accuracy"])
check_result("KNN test accuracy", round(knn_test_accuracy, 4), expected_part2["knn"]["test_accuracy"])
print()

# SVM Results
check_result("SVM best C", best_C, expected_part2["svm"]["best_C"])
check_result("SVM best gamma", best_gamma, expected_part2["svm"]["best_gamma"])
check_result("SVM CV accuracy", round(svm_cv_accuracy, 4), expected_part2["svm"]["cv_accuracy"])
check_result("SVM test accuracy", round(svm_test_accuracy, 4), expected_part2["svm"]["test_accuracy"])
print()

# Random Forest Results
check_result("RF best n_estimators", best_n_estimators, expected_part2["rf"]["best_n_estimators"])
check_result("RF best max_leaf_nodes", best_max_leaf_nodes, expected_part2["rf"]["best_max_leaf_nodes"])
check_result("RF CV accuracy", round(rf_cv_accuracy, 4), expected_part2["rf"]["cv_accuracy"])
check_result("RF test accuracy", round(rf_test_accuracy, 4), expected_part2["rf"]["test_accuracy"])
check_result("RF macro average F1", round(rf_macro_f1, 4), expected_part2["rf"]["macro_f1"])
check_result("RF weighted average F1", round(rf_weighted_f1, 4), expected_part2["rf"]["weighted_f1"])
