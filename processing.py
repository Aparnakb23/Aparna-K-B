import os


def read_annotation_file(annotation_file):
    annotations = {}
    try:
        with open(annotation_file, 'r') as file:
            for line in file:
                line = line.strip().split()
                seq_id = line[0]
                start_pos = int(line[1])
                end_pos = int(line[2])
                annotations[seq_id] = (start_pos, end_pos)
    except FileNotFoundError:
        print(f"Warning: Annotation file '{annotation_file}' not found.")
    return annotations


def preprocess_fasta(fasta_file, annotation_file, output_folder):
    annotation = read_annotation_file(annotation_file)
    seq_id = os.path.splitext(os.path.basename(fasta_file))[0]
    if seq_id not in annotation:
        print(f"Warning: Sequence ID {seq_id} not found in annotation file '{annotation_file}'.")
        return

    start_pos, end_pos = annotation[seq_id]

    with open(fasta_file, 'r') as f:
        header = f.readline()
        sequence = f.readline().strip()

    processed_sequence = sequence[:start_pos - 1] + sequence[end_pos:]

    output_file = os.path.join(output_folder, f"{seq_id}_processed.fasta")
    with open(output_file, 'w') as f:
        f.write(header)
        f.write(processed_sequence)


def preprocess_folder(input_folder, annotation_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fasta_file in os.listdir(input_folder):
        if fasta_file.endswith(".fasta"):
            fasta_path = os.path.join(input_folder, fasta_file)
            annotation_file = os.path.join(annotation_folder, f"{os.path.splitext(fasta_file)[0]}_annotation.txt")
            preprocess_fasta(fasta_path, annotation_file, output_folder)


def main():
    input_folder = r"C:\Users\hp\Desktop\CRABTREE\CRABTREE_1"  # Replace with the path to your folder containing original FASTA sequences
    annotation_folder = r"C:\Users\hp\Desktop\CRABTREE\annotations"  # Replace with the path to your folder containing annotation files
    output_folder = r"C:\Users\hp\Desktop\CRABTREE\processed_files"

    preprocess_folder(input_folder, annotation_folder, output_folder)


if __name__ == "__main__":
    main()
