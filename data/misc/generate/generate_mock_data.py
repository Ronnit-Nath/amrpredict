import pandas as pd
import random

# Function to generate random DNA sequences
def generate_random_sequence(length):
    return ''.join(random.choice('ACGT') for _ in range(length))

# Generate mock genomic data
def generate_mock_genomic_data(num_samples):
    data = {
        'ID': ['Sample_' + str(i) for i in range(1, num_samples + 1)],
        'Species': ['Pseudomonas aeruginosa'] * (num_samples // 3) + ['Escherichia coli'] * (num_samples // 3) + ['Staphylococcus aureus'] * (num_samples - 2 * (num_samples // 3)),
        'Sequence': [generate_random_sequence(random.randint(1000, 5000)) for _ in range(num_samples)],
        'Resistance': [random.choice(['Sensitive', 'Resistant']) for _ in range(num_samples)]
    }
    return pd.DataFrame(data)

# Example: Generate 10 mock samples
num_samples = 10
mock_data = generate_mock_genomic_data(num_samples)

# Save mock data to CSV
mock_data.to_csv('mock_genomic_data.csv', index=False)