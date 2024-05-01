import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Split the crabtree dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Logistic Regression Classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate standard deviation of model predictions
std_deviation = np.std(y_pred)
print(f"Standard Deviation of Model Predictions: {std_deviation:.4f}")

# Calculate and print evaluation metrics
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    sensitivity = recall_score(y_test, y_predicted)
    specificity = precision_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)

    return accuracy, sensitivity, specificity, f1

accuracy, sensitivity, specificity, f1 = get_metrics(y_test, y_pred)
print("Accuracy = %.3f \nSensitivity = %.3f \nSpecificity = %.3f \nF1 Score = %.3f" % (accuracy, sensitivity, specificity, f1))
