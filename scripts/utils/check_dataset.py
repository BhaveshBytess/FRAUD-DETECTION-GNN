import pandas as pd

classes = pd.read_csv('data/Elliptic++ Dataset/txs_classes.csv')

print("=" * 60)
print("ELLIPTIC++ DATASET VERIFICATION")
print("=" * 60)

print("\nALL TRANSACTIONS:")
print(f"Total: {len(classes)}")
print(classes['class'].value_counts().sort_index())
print("\nDistribution:")
print(classes['class'].value_counts(normalize=True).sort_index() * 100)

labeled = classes[classes['class'] != 3]
print("\n" + "=" * 60)
print("LABELED TRANSACTIONS ONLY:")
print("=" * 60)
print(f"Total labeled: {len(labeled)}")
print(f"Class 1: {(labeled['class']==1).sum()} ({(labeled['class']==1).sum()/len(labeled)*100:.2f}%)")
print(f"Class 2: {(labeled['class']==2).sum()} ({(labeled['class']==2).sum()/len(labeled)*100:.2f}%)")

print("\n" + "=" * 60)
print("DATASET INTERPRETATION:")
print("=" * 60)
print("According to Elliptic++ paper:")
print("- Class 1 = ILLICIT (fraud)")
print("- Class 2 = LICIT (legitimate)")
print("- Class 3 = UNKNOWN (unlabeled)")
print("\nExpected fraud rate: ~8-10% among labeled transactions")
print(f"Actual fraud rate (Class 1): {(labeled['class']==1).sum()/len(labeled)*100:.2f}%")
print(f"Actual legit rate (Class 2): {(labeled['class']==2).sum()/len(labeled)*100:.2f}%")

if (labeled['class']==1).sum()/len(labeled) > 0.15:
    print("\n⚠️  WARNING: Fraud rate seems TOO HIGH (>15%)")
    print("This suggests labels might be FLIPPED!")
elif (labeled['class']==1).sum()/len(labeled) < 0.05:
    print("\n⚠️  WARNING: Fraud rate seems TOO LOW (<5%)")
else:
    print("\n✅ Fraud rate looks REALISTIC (5-15%)")
