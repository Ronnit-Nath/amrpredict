import pandas as pd

aro_categories_path = '../data/aro_categories.tsv'
aro_categories = pd.read_csv(aro_categories_path, sep='\t')

print("Columns in aro_categories.tsv:", aro_categories.columns)
