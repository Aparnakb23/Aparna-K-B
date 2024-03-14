import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def kmers_funct(seq, size):
    seq = seq.replace('\n', '')
    return ' '.join([seq[x:x + size].lower() for x in range(len(seq) - size + 1)])


def read_fasta_sequence(file_path):
    with open(file_path, "r") as file:
        # Skip header lines (lines starting with '>')
        lines = file.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])
        return sequence


# Directory containing the FASTA files
fasta_dir = r"C:\Users\hp\Desktop\CRABTREE\Crabtree_1"

# Output directory for saving k-mer sequences
output_dir = r"C:\Users\hp\Desktop\CRABTREE\6mer\kmer"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# List to store k-mers sentences for each file
sentences_list = []

# Iterate through each FASTA file in the directory
for file_name in os.listdir(fasta_dir):
    if file_name.endswith(".fasta"):
        file_path = os.path.join(fasta_dir, file_name)

        # Read the sequence from the file
        mySeq = read_fasta_sequence(file_path)

        # Generate k-mers for the sequence
        k_mers = kmers_funct(mySeq, size=6)

        # Append the k-mers sentence to the list
        sentences_list.append((file_name.replace('.fasta', ''), k_mers))

        # Write k-mers to a text file
        output_file_path = os.path.join(output_dir, f"{file_name.replace('.fasta', '_kmers.txt')}")
        with open(output_file_path, 'w') as output_file:
            output_file.write(k_mers + '\n')

# Initialize CountVectorizer
cv = CountVectorizer()

# Transform the k-mers sentences into a feature matrix
X = cv.fit_transform([sentence for _, sentence in sentences_list]).toarray()

# Print or use the feature matrix
print(X)

# Save the feature matrix into a CSV file with Sequence IDs
feature_matrix = pd.DataFrame(X, columns=cv.get_feature_names_out())
feature_matrix.insert(0, "Sequence ID", [seq_id for seq_id, _ in sentences_list])
print(feature_matrix)
feature_matrix.to_csv(r"C:\Users\hp\Desktop\CRABTREE\6mer\feature_matrix_6mer.csv", header=True, index=False)
