import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from myproject.path_utils import data_path


# ---- 1. Load the data ----
# train_path = "data/Q3/mnist_train.csv"
# test_path  = "data/Q3/mnist_test.csv"
train_path = data_path("Q3","mnist_train.csv")
test_path  = data_path("Q3","mnist_test.csv")

# Read CSVs without header, so pandas uses numeric column names
df_train = pd.read_csv(train_path, header=None)
df_test  = pd.read_csv(test_path,  header=None)

# ---- 2. Assign column names ----
# 1. Figure out how many columns are in the training table.
num_columns = df_train.shape[1]   # e.g. 785 for MNIST
# print(num_columns)
# 2. Start a new list of names with “label” for the first column. Column name is an array so i just kept on naming the columns.
column_names = ["label"]
# 3. For every remaining column (784 of them), name it “pixel0”, “pixel1”, … “pixel783”.
for i in range(num_columns - 1):
    column_names.append("pixel" + str(i))
# 4. Now assign that list of names back to both your train and test tables.
df_train.columns = column_names
df_test.columns  = column_names

# ---- 3. Split into features and target ----
# axis: 1 means drop 
X_train = df_train.drop('label', axis=1).values 
y_train = df_train['label'].values
X_test  = df_test.drop('label', axis=1).values
y_test  = df_test['label'].values

# # ---- 4. Scale features ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# # ---- 5A. Logistic Regression ----
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)*100:.2f}%")
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

# ---- 5B. k-Nearest Neighbors ----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print(f"KNN (k=5) Accuracy: {accuracy_score(y_test, y_pred_knn)*100:.2f}%")
print("Classification Report (KNN):")
print(classification_report(y_test, y_pred_knn))
