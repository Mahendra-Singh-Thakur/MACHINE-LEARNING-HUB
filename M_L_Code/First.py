# # Data Preprocessing Tools

# # Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # Importing the dataset
# dataset = pd.read_csv('Data_Sets/Data.csv')
# x = dataset.iloc[:, 0:-1].values
# y = dataset.iloc[:, -1].values
# print(x)
# print(y)

# # Taking care of missing data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(x[:, 1:])
# x[:, 1:] = imputer.transform(x[:, 1:])
# print(x)

# # Encoding categorical data
# # Encoding the Independent Variable
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# x = np.array(ct.fit_transform(x))
# print(x)

# # Encoding the Dependent Variable
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)
# print(y)

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# FS = StandardScaler()
# x_train[:, 3:] = FS.fit_transform(x_train[:, 3:])
# x_test[:, 3:] = FS.transform(x_test[:, 3:])
# print(x_train)
# print(x_test)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split



DS=pd.read_csv("Data_Sets/Social_Network_Ads.csv")
x=DS.iloc[:,:-1].values
y=DS.iloc[:,2:].values

print(x.shape)
print(y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

FS=StandardScaler()
x_train = FS.fit_transform(x_train)
x_test = FS.transform(x_test)

y_train = y_train.flatten()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

# print(np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1,1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# y_pred=classifier.predict(x_train)
# cm = confusion_matrix(y_train, y_pred)
# print(cm)
# print(accuracy_score(y_train, y_pred))




# from matplotlib.colors import ListedColormap
# X_set, y_set = FS.inverse_transform(x_train), y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, classifier.predict(FS.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('path_to_dataset')
X = data.drop('target_column', axis=1)
y = data['target_column']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

from sklearn.svm import SVC

# Training the Support Vector Machine classifier on the Training set
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)