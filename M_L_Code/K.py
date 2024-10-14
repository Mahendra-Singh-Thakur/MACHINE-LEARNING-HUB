# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score


# Importing the dataset
dataset = pd.read_csv('MACHINE-LEARNING-HUB/Data_Sets/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classification models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy',random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=0),
    "Support Vector Machines (SVM)": SVC(kernel='rbf',random_state=0),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors (K-NN)": KNeighborsClassifier(),
    "Gradient Boosting Machines (GBM)": GradientBoostingClassifier(),
    "Adaboost" : AdaBoostClassifier(algorithm='SAMME'),
    "XGBoost": XGBClassifier(),
    "Gaussian Process": GaussianProcessClassifier(),
    "Quadratic Discriminant Analysis (QDA)": QuadraticDiscriminantAnalysis(),
    "Linear Discriminant Analysis (LDA)": LinearDiscriminantAnalysis()
}

# Train models and evaluate performance
best_model = None
best_accuracy = 0.0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = name
    print(name)
    print(accuracy)

# Output the best model and its accuracy
print(f'Best Model: {best_model}')
print(f'Accuracy: {best_accuracy * 100:.2f}%')
