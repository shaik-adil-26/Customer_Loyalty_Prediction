import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = {
    "CustomerID": range(1, 11),
    "MonthlySpend": [500, 800, 200, 1200, 1500, 300, 700, 1600, 400, 1000],
    "PurchaseFrequency": [2, 5, 1, 6, 7, 1, 4, 8, 2, 5],
    "LoyalCustomer": [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print("\n1) Customer Dataset:")
print(df)

X_cluster = df[["MonthlySpend", "PurchaseFrequency"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["CustomerSegment"] = kmeans.fit_predict(X_cluster)

print("\n2) Customer Segmentation using K-Means:")
print(df[["CustomerID", "MonthlySpend", "PurchaseFrequency", "CustomerSegment"]])

X = df[["MonthlySpend", "PurchaseFrequency"]]
y = df["LoyalCustomer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)

print("\n3) Logistic Regression Performance:")

print("Accuracy :", accuracy_score(y_test, log_predictions))
print("Precision:", precision_score(y_test, log_predictions, zero_division=0))
print("Recall   :", recall_score(y_test, log_predictions, zero_division=0))
print("F1 Score :", f1_score(y_test, log_predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_predictions))

log_model_tuned = LogisticRegression(C=0.5)
log_model_tuned.fit(X_train, y_train)

tuned_predictions = log_model_tuned.predict(X_test)

print("\n4) Tuned Logistic Regression Accuracy:")
print("Accuracy:", accuracy_score(y_test, tuned_predictions))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

print("\n5) KNN Model Accuracy:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))


tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)

print("\n6) Decision Tree Accuracy:")
print("Accuracy:", accuracy_score(y_test, tree_predictions))


joblib.dump(log_model_tuned, "customer_loyalty_model.pkl")
print("\n7) Model saved as customer_loyalty_model.pkl")

loaded_model = joblib.load("customer_loyalty_model.pkl")



new_customer = pd.DataFrame(
    [[100, 1]],
    columns=["MonthlySpend", "PurchaseFrequency"]
)

deployment_prediction = loaded_model.predict(new_customer)

print("\n8) Deployment Prediction:")
if deployment_prediction[0] == 1:
    print("Customer is likely to be a LOYAL customer")
else:
    print("Customer is NOT likely to be a loyal customer")

print("\n--- End of Industry ML Capstone Project ---\n")

