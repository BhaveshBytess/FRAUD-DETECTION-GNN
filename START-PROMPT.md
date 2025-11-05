# 🚀 START PROMPT — “Elliptic GNN Baselines” Project Boot

**Context load:**
You are now initializing work on the repository **`elliptic-gnn-baselines`**, a reproducible portfolio project implementing GNN baselines on the **Elliptic++** dataset.
Your full operational context comes from these three documents:

1. `docs/AGENT.MD` — defines your **behavioral discipline**, error-handling, decision protocol, and communication rules.
2. `docs/PROJECT_SPEC.md` — defines **what** to build: scope, dataset schema, models, metrics, reproducibility, repo scaffold, and acceptance criteria.
3. `TASKS.md` — defines **what’s currently in progress** and serves as your live to-do list.

---

## 🧠 Initialization Instructions

1. **Read all three documents in full** (`AGENT.MD`, `PROJECT_SPEC.md`, `TASKS.md`).
2. **Adopt the mindset and policies** described in `AGENT.MD`:

   * Always follow: **Plan → Verify → Execute → Log**
   * Maintain and update **TODO lists** for every operation.
   * Never assume context or fabricate synthetic data.
   * Resolve or escalate any blocking errors before continuing.
3. Treat `PROJECT_SPEC.md` as immutable truth — every task, metric, and file structure must conform to it.
4. Treat `TASKS.md` as your personal project planner:

   * Create new tasks only when verified against the spec.
   * Update statuses (`[ ]`, `[~]`, `[x]`, `[?]`) as progress evolves.

---

## 🧾 Workflow Discipline

### Phase 1 — Planning

* Parse the repo scaffold and milestones from `PROJECT_SPEC.md`.
* Identify missing components (folders, notebooks, configs).
* Populate `TASKS.md` with initial checklist items (`M1`–`M6` milestones).

### Phase 2 — Verification

* Confirm `data/elliptic/` exists (real dataset only).
* Verify Python environment (≥3.10), and that `torch`, `torch-geometric`, `xgboost`, `sklearn`, etc. match the `requirements.txt` spec.
* Do **not** run or generate synthetic tests if files are missing — pause and request user confirmation for dataset paths.

### Phase 3 — Execution

* Follow one `TASKS.md` block at a time.
* Always describe what you will do before running code.
* After completion, validate with the “Verification Before Commit” criteria in `AGENT.MD`.

### Phase 4 — Logging & Communication

* For each task:

  * Append metrics to `reports/metrics_summary.csv` when applicable.
  * Save artifacts to the correct folders (`reports/`, `checkpoints/`, `plots/`).
  * Record results and verification notes inside `TASKS.md`.
* If a problem persists after 5 repair attempts:

  * Stop immediately.
  * Summarize all attempts and possible causes.
  * Ask for a decision before continuing.

---

## 🧩 Expected Outputs per Milestone

| Milestone | Deliverable                         | Verification                                                            |
| --------- | ----------------------------------- | ----------------------------------------------------------------------- |
| **M1**    | Repo scaffold, README, requirements | Folder tree matches spec, installs cleanly                              |
| **M2**    | Dataset loader & temporal splits    | `python -m src.data.elliptic_loader --check` works; `splits.json` saved |
| **M3**    | GCN baseline notebook               | Runs fully, saves metrics & plots, appends summary CSV                  |
| **M4**    | GraphSAGE & GAT notebooks           | Metrics logged; models train successfully                               |
| **M5**    | Tabular baselines                   | Notebook outputs comparable metrics in CSV                              |
| **M6**    | Final verification & readability    | All TODOs cleared; paths relative; seeds fixed                          |

---

## 🧭 Behavioral Reminders (from AGENT.MD)

* Be transparent: **explain before executing**.
* No synthetic or placeholder data.
* Always verify dataset presence and column names before import.
* Each notebook is self-contained and reproducible.
* Every metric, checkpoint, and plot must be **real and verifiable**.
* Never silently skip errors.
* If uncertain → **ask**, don’t assume.

---

## ✅ Start Command

**Your first command sequence should be:**

1. Read and summarize `PROJECT_SPEC.md` to ensure comprehension.
2. Initialize `TASKS.md` with TODOs for:

   * M1: Bootstrap repo
   * M2: Data loader & splits
   * M3: GCN notebook
   * M4: GraphSAGE/GAT
   * M5: Tabular baselines
   * M6: Readability checks
3. Plan → Verify → Execute → Log for **Task M1 (Bootstrap Repo)**.

---

### 🪩 Output Expectation for the First Run

* Print a **Plan summary** (what you’ll build and verify).
* Verify environment and dataset folder existence.
* Then begin **Task M1** (repo bootstrap) per `TASKS.md` checklist.

---

**End of Start Prompt**

---

