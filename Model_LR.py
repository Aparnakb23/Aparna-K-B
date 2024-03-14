import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the feature matrix with Sequence IDs
feature_matrix_path = r"C:\Users\hp\Desktop\CRABTREE\6mer\feature_matrix_6mer.csv"
feature_matrix = pd.read_csv(feature_matrix_path)

# Assuming the last column is the label column indicating Crabtree positivity/negativity
X = feature_matrix.iloc[:, :-1]  # Exclude the last column
y = feature_matrix.iloc[:, -1]  # Use the last column as labels

# Initialize StratifiedKFold for 10 folds
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define a dictionary to store metrics for each fold
metrics_per_fold = {'Accuracy': [], 'F1 Score': [], 'AUC': [],
                    'Sensitivity': [], 'Specificity': []}

# Perform cross validation
fold = 1  # Initialize fold
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a Logistic Regression classifier
    model = LogisticRegression()

    # Train the model on standardized data
    model.fit(X_train_scaled, y_train)

    # Make predictions on the standardized test set
    y_pred = model.predict(X_test_scaled)
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate and store metrics for each fold
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob_test)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    metrics_per_fold['Accuracy'].append(accuracy)
    metrics_per_fold['F1 Score'].append(f1)
    metrics_per_fold['AUC'].append(auc)
    metrics_per_fold['Sensitivity'].append(sensitivity)
    metrics_per_fold['Specificity'].append(specificity)

    print(f"Fold {fold} \nAccuracy: {accuracy} \nF1 Score: {f1} \nAUC: {auc} "
          f"\nSensitivity: {sensitivity} \nSpecificity: {specificity}\n")

    fold += 1

# Calculate and print mean and standard deviation of metrics across all folds using numpy
mean_metrics = {metric: np.mean(values) for metric, values in metrics_per_fold.items()}
std_metrics = {metric: np.std(values) for metric, values in metrics_per_fold.items()}

print("\nMean Metrics Across All Folds:\n")
for metric, value in mean_metrics.items():
    print(f"{metric}: {value}")

print("\nStandard Deviation Across All Folds:\n")
for metric, value in std_metrics.items():
    print(f"{metric}: {value}")
