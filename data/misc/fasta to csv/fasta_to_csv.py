from Bio import SeqIO
import pandas as pd

def fasta_to_csv(input_fasta, output_csv):
    # Initialize lists to store data
    ids = []
    descriptions = []
    sequences = []

    # Read the FASTA file
    for record in SeqIO.parse(input_fasta, "fasta"):
        ids.append(record.id)
        descriptions.append(record.description)
        sequences.append(str(record.seq))

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'Description': descriptions,
        'Sequence': sequences
    })

    # Write DataFrame to CSV
    df.to_csv(output_csv, index=False)

    print(f"Conversion from {input_fasta} to {output_csv} successful!")

if __name__ == "__main__":
    input_fasta = r"F:\AMR Predict\data\fasta to csv\nucleotide_fasta_protein_variant_model.fasta"
    output_csv = "genomic_data.csv"

    fasta_to_csv(input_fasta, output_csv)
