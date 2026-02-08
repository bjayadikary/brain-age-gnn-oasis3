
# GNN Coding Competition Template

This repository provides a **secure, reproducible template** for running a
Graph Neural Network (GNN) competition that supports **humans and LLMs**
competing on equal footing.

The design intentionally **does not execute participant code**. Instead,
participants submit **predictions only**, which are automatically evaluated
and ranked on a public leaderboard using GitHub Actions.

This makes the competition:
- Safe (no untrusted code execution)
- Fully reproducible
- Suitable for human-vs-LLM evaluation studies

---

## 1. Task Overview

**Task:** Node classification on a graph  
**Input:** Public graph structure and node features  
**Output:** Predictions for unseen test nodes  
**Metric:** ROC-AUC (binary classification)

Participants train any GNN or non-GNN model *offline* and submit predictions
for the test nodes.

---

## 2. Repository Structure

```
.
├── data/
│   ├── public/
│   │   ├── train_edges.csv
│   │   ├── train_labels.csv
│   │   ├── val_edges.csv
│   │   ├── val_labels.csv
│   │   ├── test_edges.csv
│   │   ├── test_nodes.csv
│   │   └── sample_submission.csv
│   └── private/
│       └── test_labels.csv   # never committed (used only in CI)
├── competition/
│   ├── config.yaml
│   ├── validate_submission.py
│   ├── evaluate.py
│   └── metrics.py
├── submissions/
│   ├── README.md
│   └── inbox/
├── leaderboard/
│   ├── leaderboard.csv
│   └── leaderboard.md
└── .github/workflows/
    ├── score_submission.yml
    └── publish_leaderboard.yml
```

---

## 3. Submission Format

Participants submit a **single CSV file**:

**predictions.csv**
```
id,y_pred
n0001,0.92
n0002,0.13
...
```

Rules:
- `id` must match exactly the IDs in `test_nodes.csv`
- One row per test node
- `y_pred` must be a float in [0,1]
- No missing or duplicate IDs

A sample is provided in:
```
data/public/sample_submission.csv
```

---

## 4. How to Submit

1. Fork this repository
2. Create a new folder:
```
submissions/inbox/<team_name>/<run_id>/
```
3. Add:
   - `predictions.csv`
   - `metadata.json`

Example `metadata.json`:
```json
{
  "team": "example_team",
  "model": "llm-only",
  "llm_name": "gpt-x",
  "notes": "Temporal GNN with class weighting"
}
```

4. Open a Pull Request to `main`

The PR will be **automatically scored** and the result posted as a comment.

---

## 5. Leaderboard

After a PR is merged, the submission is added to:
- `leaderboard/leaderboard.csv`
- `leaderboard/leaderboard.md`

Rankings are sorted by **descending score**.

---

## 6. Rules

- No external or private data
- No manual labeling of test data
- No modification of evaluation scripts
- Unlimited offline training is allowed
- Only predictions are submitted

Violations may result in disqualification.

---

## 7. Human vs LLM Studies

To use this competition for research:
- Fix a time budget (e.g., 2 hours)
- Fix a submission budget (e.g., 5 runs)
- Record metadata fields (`model`, `llm_name`)
- Compare:
  - validity rate
  - best score within K submissions
  - score vs submission index

---

## 8. Citation

If you use this template in academic work, please cite the repository.

---

## 9. License

MIT License.

## Interactive Leaderboard (GitHub Pages)

This template includes an interactive leaderboard page inspired by modern benchmark sites.

**Enable GitHub Pages** (Settings → Pages) and set the source to the `main` branch `/docs` folder.
Then open `https://<your-org>.github.io/<repo>/leaderboard.html`.
