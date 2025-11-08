# 🚀 START PROMPT — "Elliptic++ Fraud Detection: GNN vs ML Baselines" Project Boot

**Context load:**
You are now initializing work on the repository **`elliptic-gnn-baselines`**, a reproducible research-grade portfolio project analyzing the **relative strengths of GNNs and ML models** on the **Elliptic++ fraud detection dataset.**

This project has **COMPLETED** all milestones through **M10** and is now in **production-ready** state with published Zenodo DOI.

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
5. Project is **COMPLETE** - use for reference, extensions, or new experiments only.

---

## 📈 Current State Snapshot (as of Nov 8, 2025)

| Milestone | Status | Summary                                                                      |
| :-------- | :----- | :--------------------------------------------------------------------------- |
| **M1**    | ✅      | Repo scaffold, configs, and requirements established                         |
| **M2**    | ✅      | Dataset loader with verified temporal splits; `splits.json` saved            |
| **M3**    | ✅      | GCN baseline implemented (PR-AUC: 0.198)                                     |
| **M4**    | ✅      | GraphSAGE (PR-AUC: 0.448) & GAT (PR-AUC: 0.184) trained and logged           |
| **M5**    | ✅      | Tabular baselines (XGBoost: 0.669, RF: 0.658, MLP: 0.364) completed          |
| **M6**    | ✅      | Final polish, documentation corrected, comparative analysis complete          |
| **M7**    | ✅      | Causality & Feature Dominance - **EXECUTED** (ablation + correlations)      |
| **M8**    | ✅      | Interpretability - **EXECUTED** (SHAP + GraphSAGE saliency)                 |
| **M9**    | ✅      | Temporal robustness - **EXECUTED** (multiple time windows tested)           |
| **M10**   | ✅      | Final project wrap - **COMPLETE** (docs polished, Zenodo published)         |

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

**M7 Experiment (✅ EXECUTED - Nov 8, 2025):**
- ✅ Removed AF94–AF182, retrained both XGBoost and GraphSAGE
- ✅ **CONFIRMED:** GraphSAGE improved by 24% (0.448 → 0.556)
- ✅ **CONFIRMED:** XGBoost dropped only 3% (0.669 → 0.649)
- 📊 Results: `reports/m7_tabular_ablation.csv`, `reports/m7_graphsage_ablation_summary.csv`
- 📄 Full analysis: `docs/M7_RESULTS.md`, `docs/M7_CAUSALITY_EXPERIMENT.md`

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
| **M7 — Causality & Feature Dominance** ✅ | **EXECUTED & CONFIRMED** - AF94–AF182 double-encode neighbor stats. GraphSAGE +24% when removed. | `reports/m7_tabular_ablation.csv`, `reports/m7_graphsage_ablation_summary.csv`, `docs/M7_RESULTS.md` |
| **M8 — Interpretability & Analysis** ✅     | **EXECUTED** - SHAP (XGBoost) + gradient saliency (GraphSAGE) identify key features.             | `reports/m8_xgb_shap_importance.csv`, `reports/m8_graphsage_saliency.json`, `docs/M8_INTERPRETABILITY.md` |
| **M9 — Temporal Robustness Study** ✅        | **EXECUTED** - XGBoost stable across time; GraphSAGE benefits from earlier training windows.     | `notebooks/08_temporal_shift.ipynb`, `reports/m9_temporal_results.csv`, `docs/M9_TEMPORAL.md` |
| **M10 — Final Project Wrap** ✅              | **COMPLETE** - Documentation polished, Zenodo DOI published, repo production-ready.              | `README.md`, `CITATION.cff`, `PROJECT_REPORT.md`, DOI: 10.5281/zenodo.17560930 |

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
   * **All Milestones M1-M10: ✅ COMPLETE**
   * **Zenodo DOI: 10.5281/zenodo.17560930** ✅ Published
   * **Repository: Production-ready** ✅
3. For new work: Define new milestones beyond M10 or extensions.
4. For maintenance: Update documentation, fix bugs, or add features.
5. Dataset location: `data/Elliptic++ Dataset/` (download from Google Drive link in README).

---

### 🪩 Output Expectation for New Sessions

* Summarized understanding of the **completed** project (scope + data + findings).
* Confirmation that all milestones (M1-M10) are complete.
* Recognition that M7-M9 experiments were **executed** (not just documented).
* Understanding of the key finding: Feature dominance hypothesis **confirmed**.
* Awareness of Zenodo publication (DOI: 10.5281/zenodo.17560930).
* Ready to support: extensions, maintenance, or new research directions.

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

---

## 🏆 **Project Status: COMPLETE**

**Final Deliverables:**
- ✅ All models trained and evaluated (M1-M5)
- ✅ Feature dominance hypothesis **CONFIRMED** via ablation (M7)
- ✅ Interpretability analysis **EXECUTED** (M8)
- ✅ Temporal robustness **TESTED** (M9)
- ✅ Documentation **POLISHED** (M10)
- ✅ Zenodo DOI **PUBLISHED**: [10.5281/zenodo.17560930](https://doi.org/10.5281/zenodo.17560930)
- ✅ Repository **PRODUCTION-READY**

**Key Contributions:**
1. Demonstrated that pre-computed neighbor aggregates (AF94–AF182) make GNNs redundant
2. Showed GraphSAGE improves 24% when aggregate features are removed
3. Provided reproducible baselines for Elliptic++ fraud detection
4. Published findings with full artifact preservation

**Contact:** 10bhavesh7.11@gmail.com

---

**End of Start Prompt (Updated 2025-11-08 — Final Version)**

---
