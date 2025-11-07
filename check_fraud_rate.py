import pandas as pd

feat = pd.read_csv('data/Elliptic++ Dataset/txs_features.csv')
cls = pd.read_csv('data/Elliptic++ Dataset/txs_classes.csv')
df = feat.merge(cls, on='txId')

df['label'] = df['class'].map({1: 0, 2: 1, 3: -1})
labeled = df[df['label'] != -1]

print('='*60)
print('ELLIPTIC++ DATASET FRAUD DISTRIBUTION')
print('='*60)
print(f'\nTotal transactions: {len(df):,}')
print(f'Labeled transactions: {len(labeled):,}')

legit = len(labeled[labeled['class']==1])
fraud = len(labeled[labeled['class']==2])
unlabeled = len(df[df['class']==3])

print('\nFraud distribution (labeled only):')
print(f'  Legit (class=1): {legit:,} ({legit/len(labeled)*100:.2f}%)')
print(f'  Fraud (class=2): {fraud:,} ({fraud/len(labeled)*100:.2f}%)')
print(f'  Unlabeled (class=3): {unlabeled:,}')

print(f'\n✅ FRAUD PERCENTAGE: {fraud/len(labeled)*100:.2f}%')
print('='*60)
