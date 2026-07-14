import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
data = pd.read_csv("diabetes.csv")

# Display first 5 records
print("First 5 Records")
print(data.head())

# Input and Output
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1)

# Summary of Training Dataset
print("\nTraining Dataset Summary")
print(X_train.describe())

# Train Naive Bayes Model
model = GaussianNB()
model.fit(X_train, Y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
print("\nAccuracy =", accuracy_score(Y_test, prediction))

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(Y_test, prediction))

# Actual vs Predicted
print("\nActual  Predicted")
for i in range(10):
    print(Y_test.iloc[i], "      ", prediction[i])
