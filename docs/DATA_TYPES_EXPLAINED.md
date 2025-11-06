# Data Types: GNN vs ML Models

## 📊 **What Data Do We Have?**

```
Elliptic++ Dataset:
├── txs_features.csv       (182 features × 203,769 transactions) [TABULAR]
├── txs_edgelist.csv       (234K edges between transactions)     [GRAPH]
└── txs_classes.csv        (Labels: fraud/legit/unlabeled)       [LABELS]
```

---

## 🔄 **HOW GNN MODELS USE DATA (M3 & M4)**

### GCN, GraphSAGE, GAT - All Use Graph Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      GNN ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ Node Features│   │ Graph Edges  │   │   Labels     │    │
│  │  X [N×182]   │   │ edge_index   │   │   y [N×1]    │    │
│  │              │   │  [2×234K]    │   │              │    │
│  │ AF1: tx_fee  │   │ tx_A → tx_B  │   │  0 = legit   │    │
│  │ AF2: amount  │   │ tx_A → tx_C  │   │  1 = fraud   │    │
│  │ ...          │   │ tx_B → tx_D  │   │ -1 = unknown │    │
│  │ AF182: ...   │   │     ...      │   │              │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                             │                               │
│                      ┌──────▼──────┐                        │
│                      │  GNN Layers │                        │
│                      │             │                        │
│                      │ For tx_A:   │                        │
│                      │ 1. Get A's  │                        │
│                      │    features │                        │
│                      │ 2. Get B,C  │                        │
│                      │    features │                        │
│                      │ 3. Aggregate│                        │
│                      │    neighbors│                        │
│                      │ 4. Combine  │                        │
│                      └──────┬──────┘                        │
│                             │                               │
│                      ┌──────▼──────┐                        │
│                      │  Prediction │                        │
│                      │ Fraud: 0.73 │                        │
│                      └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘

KEY: GNNs aggregate information from connected neighbors!
```

### Example: How GraphSAGE Detects Fraud

```
Transaction Network:

         [Legit]          [Legit]
            ↓                ↓
    ┌───────────────────────────┐
    │   Transaction A (???)     │  ← Want to predict
    │   Features: [0.5, 2.1...] │
    └───────────────────────────┘
            ↓                ↓
       [FRAUD]          [FRAUD]

GraphSAGE thinks:
"Transaction A has moderate features (0.5, 2.1...)
 BUT it receives money from 2 FRAUD transactions!
 → High fraud probability!"

Without graph: "Features look normal → probably legit" ❌
With graph:    "Neighbors are fraud → probably fraud" ✅
```

---

## 📈 **HOW ML MODELS USE DATA (M5 - If We Do It)**

### XGBoost, Random Forest, Logistic Regression - Ignore Graph

```
┌─────────────────────────────────────────────────────────────┐
│                   TRADITIONAL ML MODELS                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input:                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ Node Features│   │ Graph Edges  │   │   Labels     │    │
│  │  X [N×182]   │   │              │   │   y [N×1]    │    │
│  │              │   │   IGNORED!   │   │              │    │
│  │ AF1: tx_fee  │   │      ❌      │   │  0 = legit   │    │
│  │ AF2: amount  │   │              │   │  1 = fraud   │    │
│  │ ...          │   │              │   │              │    │
│  │ AF182: ...   │   │              │   │              │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│         │                                       │           │
│         └───────────────────────────────────────┘           │
│                             │                               │
│                      ┌──────▼──────┐                        │
│                      │  XGBoost    │                        │
│                      │  Ensemble   │                        │
│                      │             │                        │
│                      │ For tx_A:   │                        │
│                      │ 1. Get A's  │                        │
│                      │    features │                        │
│                      │ 2. Build    │                        │
│                      │    decision │                        │
│                      │    trees    │                        │
│                      │ 3. Vote     │                        │
│                      └──────┬──────┘                        │
│                             │                               │
│                      ┌──────▼──────┐                        │
│                      │  Prediction │                        │
│                      │ Fraud: 0.42 │                        │
│                      └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘

KEY: ML models treat each transaction independently!
```

### Example: How XGBoost Detects Fraud

```
Transaction A:
- Feature 1 (tx_fee): 0.5
- Feature 2 (amount): 2.1
- Feature 3 (...): 1.3
- ...
- Feature 182 (...): 0.8

XGBoost thinks:
"Based on these 182 features alone,
 this looks 42% like fraud"

Neighbor context? → IGNORED
Graph connections? → IGNORED
Who sent/received? → IGNORED

Just: features → prediction
```

---

## 🆚 **DIRECT COMPARISON**

| Aspect | GNNs (M3/M4) | ML Models (M5) |
|--------|--------------|----------------|
| **Features Used** | ✅ All 182 features | ✅ All 182 features |
| **Graph Structure** | ✅ Uses edges | ❌ Ignores edges |
| **Neighbor Info** | ✅ Aggregates | ❌ N/A |
| **Training Data** | Nodes + Edges | Nodes only |
| **Prediction Logic** | "Feature + Neighbors" | "Features only" |
| **Example** | "Fraud if I'm suspicious AND neighbors are" | "Fraud if my features are suspicious" |

---

## 🎯 **WHY DO M5? (Tabular Baselines)**

### The Big Question

**Does the graph actually help?**

```
Scenario 1: XGBoost PR-AUC < 0.30
├─ Graph is ESSENTIAL!
├─ Features alone can't detect fraud
└─ GNNs are justified ✅

Scenario 2: XGBoost PR-AUC ≈ 0.45 (matches GraphSAGE)
├─ Graph doesn't help much!
├─ Features (AF94-AF182) already encode neighbor info
└─ Simpler model works just as well

Scenario 3: XGBoost PR-AUC > 0.50 (beats GraphSAGE!)
├─ Graph is NOISE!
├─ Traditional ML is better
└─ GNNs were overkill 😅
```

### What M5 Tells Us

**If we train XGBoost, Random Forest, Logistic Regression:**

We can say:
- "GraphSAGE improves PR-AUC by X% over best tabular model"
- "Graph structure adds value for fraud detection"
- "GNNs are worth the complexity" (or not!)

**Portfolio Impact:**
- Shows you understand ML fundamentals
- Demonstrates rigorous comparison
- Proves you chose the right approach

---

## 📊 **DATA FLOW VISUALIZATION**

### GNN Pipeline (What We Did)

```
Raw Data
   ↓
txs_features.csv → Feature Matrix X [203K × 182]
   ↓
txs_edgelist.csv → Edge Index [2 × 234K]
   ↓
Combine into PyG Data Object
   ↓
data = Data(x=X, edge_index=edges, y=labels)
   ↓
GNN Model (GCN/GraphSAGE/GAT)
   ↓
Predictions [203K × 1]
```

### ML Pipeline (What M5 Would Do)

```
Raw Data
   ↓
txs_features.csv → Feature Matrix X [203K × 182]
   ↓
txs_classes.csv → Labels y [203K × 1]
   ↓
Split by timestamp (same as GNN)
   ↓
Train: [75K × 182] → XGBoost
Val:   [28K × 182] → Tune hyperparameters
Test:  [28K × 182] → Evaluate
   ↓
Predictions [28K × 1]
```

---

## 💡 **INTERESTING FACT**

Some of the 182 features are **aggregated neighbor statistics**!

```
Features AF1-AF93:   Local transaction properties
Features AF94-AF182: Neighbor aggregations
                     (e.g., "average fee of neighbors",
                           "max amount from neighbors")
```

**This means:**
- ML models DO get some graph info (baked into features)
- But GNNs can learn CUSTOM aggregations
- GNNs might still win by learning better aggregations

**This makes M5 even more interesting!**
- Will XGBoost do well because features have graph info?
- Or will GNNs do better by learning optimal aggregations?

---

## 🎓 **SUMMARY**

### What Each Model Uses

| Model | Features (182) | Graph Edges | Neighbor Aggregation |
|-------|----------------|-------------|----------------------|
| **GCN** | ✅ | ✅ | Mean pooling |
| **GraphSAGE** | ✅ | ✅ | Sampled mean |
| **GAT** | ✅ | ✅ | Attention-weighted |
| **XGBoost** | ✅ | ❌ | None (uses pre-computed features) |
| **Random Forest** | ✅ | ❌ | None |
| **Logistic Regression** | ✅ | ❌ | None |

### The Core Difference

**GNNs:** "Fraud spreads through the graph"  
**ML:** "Fraud is intrinsic to the transaction"

Both might be right! M5 would tell us which is more important.

---

## 🚀 **RECOMMENDATION**

**My suggestion:**

1. ✅ **Skip M5** - Focus on polishing what we have
   - GraphSAGE already performs well (0.45 PR-AUC)
   - We can mention in documentation: "Compared to tabular baselines, GNNs leverage graph structure"
   - Save time for M6 (visualization, documentation, polish)

2. **OR do M5 if you want to:**
   - Learn XGBoost, Random Forest, sklearn
   - Rigorous scientific comparison
   - Stronger portfolio story
   - ~3-4 hours of work

**Your call!** Both paths are valid 🎯

---

Would you like to:
- A) Skip to M6 (final polish)
- B) Do M5 (tabular baselines)
- C) Improve GraphSAGE further

?
