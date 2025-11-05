# 🧭 AGENT.MD — Operational Discipline for Codex Agent

## 🎯 Project Context

**Project:** `elliptic-gnn-baselines`
**Goal:** Build, train, and evaluate clean baseline Graph Neural Networks (GCN, GraphSAGE, GAT) on the **Elliptic++** dataset.
**Purpose:** Demonstration & portfolio project for résumé / GitHub — clarity, reproducibility, and presentation over raw research.

---

## 🧠 Core Philosophy

**Rule:** *Think before you code.*
Every action follows this discipline:

> **Plan → Verify → Execute → Log**

1. **Plan**: Explain what you intend to do (in comments or markdown).
2. **Verify**: Check dataset availability, paths, imports, and prior outputs.
3. **Execute**: Run only when context and inputs are validated.
4. **Log**: Record metrics, plots, and notes; never just say “done”.

---

## 📝 To-Do List Discipline

Use an explicit, living TODO checklist to drive every action. Never start a new task until the current task’s checklist is ✅ complete.

Rules

Maintain a single project checklist in TASKS.md and a mini-checklist at the top of each notebook.

Every task has: ID, Goal, Steps, Done criteria (must include the “Verification Before Commit” items).

Update the checklist before and after each operation:

Before: mark planned steps as pending and state expected outputs.

After: mark completed steps, attach paths to artifacts (metrics JSON, plots, checkpoints), and note any warnings/errors.

If blocked > 5 fix attempts → stop, write an escalation note (what was tried, errors, hypotheses), and request guidance.

Allowed statuses

[ ] pending

[~] in progress

[?] blocked (requires user input)

[x] done (only after verification passes)

Project-level template (TASKS.md)

# TASKS (single source of truth)

## T-01 Bootstrap Repo
Goal: Scaffold folders, README, requirements, configs.
Steps:
- [ ] Create folder tree and empty notebooks
- [ ] Add requirements.txt and install
- [ ] Add configs (default/gcn/graphsage/gat)
Done when:
- [x] `pip install -r requirements.txt` succeeds
- [x] Tree matches scaffold; README renders

## T-02 Loader + Temporal Splits + --check
Goal: Implement `src/data/elliptic_loader.py` + `splits.json`.
Steps:
- [ ] Read `data/elliptic/{nodes.csv,edges.csv}`
- [ ] Build tx_id→index; filter/valid edges
- [ ] Create train/val/test by timestamp
- [ ] Save `splits.json` and `--check` prints stats
Done when:
- [x] `python -m src.data.elliptic_loader --check` prints counts/class balance
- [x] No future edges in train/val/test


Notebook-level header template (paste as first cell)

# Notebook TODO (auto-discipline)
- [ ] Load real Elliptic++ from `data/elliptic/` (no synthetic data)
- [ ] Set seeds + deterministic flags
- [ ] Train model (GCN/GraphSAGE/GAT) end-to-end
- [ ] Save: `reports/metrics.json`, `reports/plots/*.png`, append `reports/metrics_summary.csv`
- [ ] Verify metrics + artifacts paths printed in last cell
- [ ] Clear TODOs/placeholders before commit


Execution protocol with TODOs

Plan: Write/expand the relevant task block in TASKS.md and the notebook header checklist.

Verify: Check dataset presence, paths, and config (tick off items as you validate).

Execute: Implement steps in order; mark [~] while running.

Log: On success, change to [x] and paste artifact paths (e.g., reports/metrics.json, plots).

Blocked? Change to [?] and add an Escalation Note (what tried, error snippets, next hypotheses) before asking.

Non-negotiable

Do not mark a task [x] unless the “Verification Before Commit” section is satisfied for that task.

Do not create new TODO items that rely on synthetic/mocked data.

Keep tasks small (≤ 5–7 steps). Split large tasks into T-XX.a, T-XX.b, … rather than long checklists.

---

## ⚙️ Decision Chain Discipline

The agent must never assume. It must reason and confirm.

**Protocol before each operation:**

1. Describe intended change and expected outcome.
2. Validate environment (paths, packages, variables).
3. Run minimal, safe code to verify step correctness.
4. Summarize results and check for warnings/errors.
5. If uncertain, **pause and ask** before continuing.

**Forbidden behaviors:**

* Blindly continuing after an exception.
* Skipping error resolution.
* Generating “synthetic” or random dummy data to simulate results.

---

## 🧩 Data Handling Rules

**Dataset Identity:**
`Elliptic++` — a real transaction graph dataset.

**Data Policy:**

* 📁 Data lives in `data/elliptic/` with:

  * `nodes.csv` — transaction features & labels.
  * `edges.csv` — graph connections.
* 🛑 Never fabricate or sample fake data.
* 🧾 Always verify the existence of these files before import:

  * if missing, stop and request user confirmation for correct path.
* 💾 All metrics, plots, and outputs must reference the real dataset version in use.

---

## 📓 Notebook Workflow Discipline

**Main work happens in notebooks** under `/notebooks`.

### Notebook Rules:

1. Each experiment (EDA, baseline, GCN, GraphSAGE, GAT) must be a standalone `.ipynb`.
2. Use markdown cells to describe objectives, logic, and findings.
3. Maintain code cells lines, make sure not too much excessive long and should be readable.
4. Use `.py` files in `/src` **only** for reusable utilities (models, loaders, metrics).
5. Each notebook should:

   * Load data from `data/elliptic/`.
   * Run one clear experiment.
   * Produce:

     * `reports/metrics.json`
     * plots (`reports/plots/…`)
     * appended `metrics_summary.csv`.

### Notebook Flow Example

| Step | Notebook                        | Purpose                        |
| ---- | ------------------------------- | ------------------------------ |
| 0    | `00_baselines_tabular.ipynb`    | Logistic/XGBoost/MLP baselines |
| 1    | `01_eda.ipynb`                  | EDA + feature histograms       |
| 2    | `03_gcn_baseline.ipynb`         | Train GCN                      |
| 3    | `04_graphsage_gat.ipynb`        | Compare GNNs                   |
| 4    | `02_visualize_embeddings.ipynb` | Visualize learned embeddings   |

---

## 🧮 Verification Before Commit

Before declaring any task **complete**, the agent must verify:

✅ All scripts and notebooks run end-to-end on the **actual Elliptic++ dataset**.
✅ All metrics logged correctly (`metrics_summary.csv`, `metrics.json`).
✅ PR-AUC, ROC-AUC, and Recall@K plotted and saved.
✅ No TODOs or placeholder cells remain.
✅ All file paths are relative (`data/elliptic/...`).
✅ Seeds set (`torch.manual_seed`, NumPy, Python).
✅ No hardcoded absolute paths or environment leaks.

---

## 🧰 Error & Resolution Protocol

If an error occurs:

1. **Stop immediately.**
2. Attempt fix ≤ 5 times with context-aware reasoning.
3. For each attempt, log:

   * what was tried,
   * why it was tried,
   * result or stacktrace summary.
4. If unresolved after 5 tries:

   * summarize possible causes,
   * notify user,
   * request decision before continuing.

Never move forward “as if it worked.”

---

## 📊 Logging & Artifact Discipline

**Every notebook or run must output:**

* `/reports/metrics_summary.csv` — accumulates all experiment results.
* `/reports/plots/*.png` — PR/ROC curves, embedding visuals.
* `/checkpoints/model_best.pt` — if training occurred.
* `/data/elliptic/splits.json` — temporal split indices.

Each row in `metrics_summary.csv` must include:

| Field      | Example                |
| ---------- | ---------------------- |
| timestamp  | 1730843200             |
| experiment | elliptic-gnn-baselines |
| model      | GCN                    |
| split      | test                   |
| pr_auc     | 0.8473                 |
| roc_auc    | 0.9121                 |
| f1         | 0.621                  |
| recall@1%  | 0.447                  |

---

## 🧬 Reproducibility

Always call:

```python
from src.utils.seed import set_all_seeds
set_all_seeds(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

before training any model.

Save:

* Python / PyTorch / PyG versions in logs.
* Random seeds in JSON configs.

---

## 🧑‍💻 Communication Tone & Escalation

**Tone:** Analytical, cautious, transparent.
Always explain *why* before *doing*.

If progress stalls or data errors persist:

* Pause.
* Write a short structured note:

  ```
  ❗ Stopped execution
  Attempted fixes:
   1. …
   2. …
  Remaining issue: …
  Possible causes: …
  Awaiting your instruction.
  ```

Never hide or skip failed cells.

---

## ✅ Summary

| Aspect            | Policy                                     |
| ----------------- | ------------------------------------------ |
| Dataset           | Real Elliptic++ only                       |
| Code surface      | Primarily notebooks                        |
| Verification      | Strict, reproducible, logged               |
| Decision protocol | Plan → Verify → Execute → Log              |
| Errors            | Resolve or escalate                        |
| Communication     | Transparent, explain before act            |
| Goal              | A representable, readable ML baseline repo |

---

**End of AGENT.MD**

---

