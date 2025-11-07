import pandas as pd

feat = pd.read_csv('data/Elliptic++ Dataset/txs_features.csv')
cls = pd.read_csv('data/Elliptic++ Dataset/txs_classes.csv')
df = feat.merge(cls, on='txId')

# Correct mapping: Class 1 = Illicit (fraud), Class 2 = Licit (legit), Class 3 = Unknown
# Target: 0 = Licit, 1 = Illicit for binary classification
df['label'] = df['class'].map({1: 1, 2: 0, 3: -1})  # 1=fraud, 0=legit, -1=unknown
labeled = df[df['label'] != -1]

print('='*60)
print('ELLIPTIC++ DATASET FRAUD DISTRIBUTION')
print('='*60)
print(f'\nTotal transactions: {len(df):,}')
print(f'Labeled transactions: {len(labeled):,}')

fraud = len(labeled[labeled['class']==1])  # Class 1 = Illicit/Fraud
legit = len(labeled[labeled['class']==2])  # Class 2 = Licit/Legit
unlabeled = len(df[df['class']==3])

print('\nClass distribution (labeled only):')
print(f'  Illicit/Fraud (class=1): {fraud:,} ({fraud/len(labeled)*100:.2f}%)')
print(f'  Licit/Legit (class=2): {legit:,} ({legit/len(labeled)*100:.2f}%)')
print(f'  Unlabeled (class=3): {unlabeled:,}')

print(f'\n✅ FRAUD PERCENTAGE: {fraud/len(labeled)*100:.2f}%')
print('='*60)
