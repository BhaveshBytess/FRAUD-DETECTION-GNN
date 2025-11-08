# 🚀 START PROMPT — "Elliptic++ Fraud Detection: GNN vs ML Baselines" Project Boot

**Context load:**
You are now initializing work on the repository **`elliptic-gnn-baselines`**, a reproducible research-grade portfolio project analyzing the **relative strengths of GNNs and ML models** on the **Elliptic++ fraud detection dataset.**

This project has completed all analytic milestones through **M9** and is now in the *final wrap-up* phase.

Your full operational context is defined by three documents:

1. `docs/AGENT.MD` — defines **behavioral discipline**, verification rules, and escalation protocol.
2. `docs/PROJECT_SPEC.md` — defines **project architecture**, dataset schema, models, metrics, and acceptance criteria.
3. `TASKS.md` — acts as the **active planner** for all milestones and tasks.

---

## 🧠 Initialization Instructions

1. **Read** all three documents: `AGENT.MD`, `PROJECT_SPEC.md`, and `TASKS.md`.
2. Adopt the **Plan → Verify → Execute → Log** mindset from `AGENT.MD`.
3. Treat:

   * `PROJECT_SPEC.md` as the **immutable technical blueprint**.
   * `TASKS.md` as the **dynamic planner** (update statuses `[ ]`, `[~]`, `[x]`, `[?]`).
4. Confirm dataset path `data/Elliptic++ Dataset/` exists and contains **real Elliptic++ data only** — no synthetic substitutes.
5. Resume from **M10 (final polish)** per current progress.

---

## 📈 Current State Snapshot (as of M6)

| Milestone | Status | Summary                                                                      |
| :-------- | :----- | :--------------------------------------------------------------------------- |
| **M1**    | ✅      | Repo scaffold, configs, and requirements established                         |
| **M2**    | ✅      | Dataset loader with verified temporal splits; `splits.json` saved            |
| **M3**    | ✅      | GCN baseline implemented (PR-AUC: 0.198)                                     |
| **M4**    | ✅      | GraphSAGE (PR-AUC: 0.448) & GAT (PR-AUC: 0.184) trained and logged           |
| **M5**    | ✅      | Tabular baselines (XGBoost: 0.669, RF: 0.658, MLP: 0.364) completed          |
| **M6**    | ✅      | Final polish, documentation corrected, comparative analysis complete          |
| **M7**    | ✅      | Causality & Feature Dominance (ablation + correlations logged)               |
| **M8**    | ✅      | Interpretability (SHAP + GraphSAGE saliency)                                 |
| **M9**    | ✅      | Temporal robustness (multiple time windows)                                 |
| **M10**   | ⏳      | Final project wrap                                                            |

---

## 🎯 Updated Research Goal (post-M5 findings)

> **Primary Objective:**
> Determine *when* and *why* graph structure adds value in fraud detection — and *when* rich tabular features make GNNs redundant.

You are no longer trying to "prove GNNs are superior."
Your mission is to **quantify the marginal benefit of graph information** under controlled, reproducible experiments.

---

## 🔬 Critical Insight: Feature Dominance Hypothesis (M7)

### **Leading Hypothesis:**
> **"Tabular features AF94–AF182 already encode neighbor-aggregated information, making explicit graph structure redundant."**

**If true, this explains:**
- Why XGBoost (0.669 PR-AUC) outperforms GraphSAGE (0.448 PR-AUC)
- Why GNNs don't add value → Double-encoding graph structure
- When GNNs would help → Raw features without pre-aggregation

**M7 Experiment (Documented, not implemented):**
- Remove AF94–AF182, retrain both model types
- Expected: GNNs improve (learn from graph), XGBoost drops (loses signals)
- See `docs/M7_CAUSALITY_EXPERIMENT.md` for full experimental design

---

## 🧾 Workflow Discipline

Follow strict cycle for every new task:

1. **Plan** → describe intended code/data operations.
2. **Verify** → check dataset presence, column integrity, and parameter ranges.
3. **Execute** → run reproducible notebook or script.
4. **Log** → save metrics to `reports/`, summarize results in `TASKS.md`.

Escalate any unresolved issue after 5 fix attempts with a concise failure summary and next-step options.

---

## 🧩 Upcoming Milestones (next goals)

| Milestone                                  | Goal                                                                                                                     | Deliverables                                                                                            |
| :----------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| **M7 — Causality & Feature Dominance** ✅ | Confirmed AF94–AF182 double-encode neighbor stats (see `docs/M7_CAUSALITY_EXPERIMENT.md`, `docs/M7_RESULTS.md`). | Artifacts: `reports/m7_tabular_ablation.csv`, `reports/m7_graphsage_ablation_summary.csv`, `reports/m7_corr_*.csv` |
| **M8 — Interpretability & Analysis** ✅     | SHAP (XGBoost full) + GraphSAGE saliency (local-only) explain *why* models differ.                                        | `notebooks/07_interpretability.ipynb`, `reports/m8_xgb_shap_importance.csv`, `reports/m8_graphsage_saliency.json`, `docs/M8_INTERPRETABILITY.md` |
| **M9 — Temporal Robustness Study** ✅        | Measured stability under earlier train windows.                                                                          | `notebooks/08_temporal_shift.ipynb`, `reports/m9_temporal_results.csv`, `docs/M9_TEMPORAL.md`            |
| **M10 — Final Project Wrap** ⏳              | README polish, summary updates, repo cleanup.                                                                           | `README.md`, `PROJECT_SUMMARY.md`, final release checklist                                             |
| **M10 — Final Project Wrap**               | Documentation polish, comparative report, publication-ready summary.                                                     | `README.md` (final), `PROJECT_SUMMARY.md`, cleaned repo                                                 |

---

## 🧭 Behavioral Highlights (from AGENT.MD)

* **No synthetic data or placeholder metrics.**
* **Explain before executing.**
* **Every result = verifiable artifact** (metrics file, plot, or checkpoint).
* **Self-check on plausibility:**
  * If PR-AUC > 0.9, trigger a "LeakageSuspect" review before marking success.
* **Transparency:** show reasoning, not just results.

---

## 🧩 Updated Mindset

> Your focus is **not proving superiority**, but **understanding conditions**.
> You are an analyst-researcher, not a model advocate.
> Your deliverable is **insight**.

---

## ✅ Start Command (for new chat)

1. Read and summarize `PROJECT_SPEC.md` and `TASKS.md` to confirm context.
2. Review current project state:
   * Milestones M1-M9: ✅ **Complete**
   * M10: ⏳ **Pending** (final polish)
3. Identify next priority from `TASKS.md` (M10 items).
4. Follow **Plan → Verify → Execute → Log** for the selected task.
5. Confirm dataset accessibility and prior metrics before executing new experiments.

---

### 🪩 Output Expectation for the First Run

* Summarized understanding of the project (scope + data + current state).
* Verified dataset and environment readiness.
* Identified current focus: M10 (final wrap tasks).
* Plan for remaining documentation/cleanup objectives.

---

## 📊 Key Performance Metrics (Current Baselines)

**Models Ranked by PR-AUC:**
1. 🥇 **XGBoost** (Tabular): 0.669 PR-AUC ⭐ **BEST**
2. 🥈 **Random Forest** (Tabular): 0.658 PR-AUC
3. 🥉 **GraphSAGE** (GNN): 0.448 PR-AUC ⭐ **Best GNN**
4. **MLP** (Tabular): 0.364 PR-AUC
5. **GCN** (GNN): 0.198 PR-AUC
6. **GAT** (GNN): 0.184 PR-AUC
7. **Logistic Regression** (Tabular): 0.164 PR-AUC

**Performance Gap:** XGBoost outperforms best GNN by **49%** (0.669 vs 0.448).

**Dataset Stats:**
- Total nodes: 203,769 transactions
- Features: 182 per transaction
- Fraud rate: Train 10.88%, Val 11.53%, Test 5.69%
- Temporal splits: Train (≤29), Val (≤39), Test (>39)

---

**End of Start Prompt (Updated 2025-11-07)**

---
