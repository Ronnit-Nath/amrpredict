import pandas as pd

aro_index_path = '../data/aro_index.tsv'
aro_index = pd.read_csv(aro_index_path, sep='\t')

print("Columns in aro_index.tsv:", aro_index.columns)
