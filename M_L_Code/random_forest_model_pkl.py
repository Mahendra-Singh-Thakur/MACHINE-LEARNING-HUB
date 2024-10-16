import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib  # or import pickle

# Importing the data
data = pd.read_csv('MACHINE-LEARNING-HUB\Data_Sets\wine_quality_white.csv')

# Outlier Removal using IQR method
Q1 = data.quantile(0.25) 
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Data after Outlier Removal:')
print(data.head())

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Oversampling the minority class using SMOTE
print('Original data shape:', X.shape)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print('Resampled data shape:', X.shape)

# Splitting the data into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate the Random Forest model with the best hyperparameters
model = RandomForestClassifier(n_estimators=144, max_depth=18, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f'Best value (accuracy from Optuna): {0.8004377510141625}')
print(f'Final model accuracy on test set: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Save the model using joblib
joblib.dump(model, 'random_forest_model.pkl')

# Or using pickle
# with open('random_forest_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
