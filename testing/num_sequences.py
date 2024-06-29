import pandas as pd
import random
import string

# Parameters
num_sequences = 40000  # Total number of sequences

# Function to generate random DNA sequences
def generate_sequence(length):
    return ''.join(random.choice('ACGT') for _ in range(length))

# Generate sequences
sequences = []

for _ in range(num_sequences):
    sequence = generate_sequence(random.randint(50, 100))  # Random sequence length between 50 and 100
    sequences.append(sequence)

# Create a DataFrame
df = pd.DataFrame({
    'Sequence': sequences
})

# Save to CSV
csv_file = 'test_sequences03.csv'
df.to_csv(csv_file, index=False)

print(f"CSV file '{csv_file}' generated successfully with {num_sequences} sequences.")
