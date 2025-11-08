import pandas as pd

print('='*70)
print('ELLIPTIC++ OFFICIAL CLASS ENCODING')
print('='*70)
print('class 1 = illicit (fraud)')
print('class 2 = licit (legitimate)')  
print('class 3 = unknown (unlabeled)')
print()

cls = pd.read_csv('data/Elliptic++ Dataset/txs_classes.csv')
print('Dataset distribution:')
print(cls['class'].value_counts().sort_index())
print()

pct = cls['class'].value_counts(normalize=True).sort_index() * 100
for c, p in pct.items():
    label_name = {1: 'illicit', 2: 'licit', 3: 'unknown'}[c]
    print(f'class {c} ({label_name:10s}): {p:6.2f}%')

print()
labeled = cls[cls['class'] != 3]
fraud_pct = len(labeled[labeled['class'] == 1]) / len(labeled) * 100
legit_pct = len(labeled[labeled['class'] == 2]) / len(labeled) * 100

print('='*70)
print('LABELED TRANSACTIONS ONLY (classes 1 & 2):')
print('='*70)
print(f'Legitimate (class 1): {legit_pct:6.2f}%')
print(f'Fraud (class 2):      {fraud_pct:6.2f}%')
print()
print('⚠️  WARNING: 90% fraud is NOT realistic for production fraud detection!')
print('='*70)
