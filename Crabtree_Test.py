import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Bio import SeqIO

# Function to read sequences from a folder
def read_sequences_from_folder(folder_path, label):
    sequences = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.fasta'):
            with open(file_path, "r") as fasta_file:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    sequences.append([record.id, str(record.seq), label])
    return sequences

# Specify the paths to the folders containing positive and negative samples
positive_folder = "/home/group_shyam01/Desktop/Aparna/crabtree_positive"
negative_folder = "/home/group_shyam01/Desktop/Aparna/crabtree_negative"

# Read sequences from both folders
positive_sequences = read_sequences_from_folder(positive_folder, label=1)
negative_sequences = read_sequences_from_folder(negative_folder, label=0)

# Concatenate the sequences from both folders
crabtree_data = pd.DataFrame(positive_sequences + negative_sequences, columns=['ID', 'Sequence', 'Class'])

# Function to generate k-mers
def Kmers_funct(seq, size=6):
    return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

# Apply Kmers_funct to create the 'words' column
crabtree_data['words'] = crabtree_data['Sequence'].apply(lambda x: Kmers_funct(x))

# Convert lists of k-mers into string sentences of words
crabtree_data['words'] = crabtree_data['words'].apply(lambda x: ' '.join(x))

# Separate features (X) and labels (y)
X = crabtree_data['words']
y = crabtree_data['Class']

# Creating the Bag of Words model using CountVectorizer()
cv = CountVectorizer(ngram_range=(6, 6))
X = cv.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X.toarray())

print("Shape of the document-term matrix:", X.shape)
# Perform cross-validation
metrics_per_fold = {'Accuracy': [], 'F1 Score': [], 'AUC': [], 'Sensitivity': [], 'Specificity': []}
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold = 1  # Initialize fold

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Logistic Regression Classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate and store metrics for each fold
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])

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
