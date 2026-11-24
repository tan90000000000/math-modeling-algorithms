"""
AHP (Analytic Hierarchy Process) for Personal Time Management

Reference: Saaty, T.L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fix CJK font issues (fallback-safe); chart labels are English anyway
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 0. Configuration
FILE = sys.argv[1] if len(sys.argv) > 1 else "个人时间安排.xlsx"
if not os.path.exists(FILE):
    raise FileNotFoundError(f"Input file not found: {FILE}")

# Sheet-name mapping (keep Chinese sheet names in the Excel file,
# but expose English aliases for the paper / judges)
SHEET_MAP = {
    "criteria":         "准则层",
    "urgency":          "时间紧迫性",
    "long_term_value":  "长期价值",
    "sunk_cost":        "沉没成本",
    "health":           "身体健康",
}

# English names of the alternatives (for charts / paper)
EN_NAMES = {
    "高数期中复习": "Calculus Review",
    "线代学习": "Linear Algebra",
    "准备数学竞赛": "Math Contest",
    "准备电赛": "Electronics Contest",
    "准备美赛": "MCM/ICM Prep",
    "运动": "Exercise",
}

# Saaty's Random Index (Saaty, 1980). RI_list[n-1] for matrix size n.
RI_list = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49,
           1.51, 1.52, 1.53, 1.54]


# ============================================================
# 1. Read the criteria-level matrix
# ============================================================
# skiprows=1 : skip the header row
# usecols    : read only the judgment-matrix columns
# header=None: do not treat the first row as column names
df = pd.read_excel(FILE, sheet_name=SHEET_MAP["criteria"],skiprows=1, usecols="B:E", header=None)
arr = df.to_numpy(dtype=float)
print("Criteria-level judgment matrix:")
print(arr)


# ============================================================
# 2. AHP core function
# ============================================================
def calculate_ahp(matrix, name="criteria", verbose=True):
    """Return weights from three methods + consistency check."""
    n = matrix.shape[0]

    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = np.argmax(eigvals.real)
    lambda_max = eigvals[max_idx].real

    # Consistency check
    RI = RI_list[n - 1]
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI

    if CR >= 0.1:
        # Warn (do not crash) so the full pipeline still produces results;
        # in the paper, explain how the matrix should be revised.
        if verbose:
            print(f"WARNING: {name} failed consistency test "
                  f"(CR = {CR:.4f} >= 0.1). Results are for reference only.")
        raise ValueError(f"{name} inconsistent (CR={CR:.4f})")
    if verbose:
        print(f"Consistency test passed (CR = {CR:.4f}).")

    # ---- Method 1: Arithmetic mean (column-normalized average) ----
    column_sum = np.sum(matrix, axis=0)
    stand_matrix = matrix / column_sum
    weights_1 = np.sum(stand_matrix, axis=1) / n

    # ---- Method 2: Geometric mean (row-product normalization) ----
    row_prod = np.prod(matrix, axis=1)
    geo = row_prod ** (1.0 / n)
    weights_2 = geo / np.sum(geo)

    # ---- Method 3: Eigenvector (principal eigenvector) ----
    weights_3 = eigvecs[:, max_idx].real
    weights_3 = weights_3 / np.sum(weights_3)

    return weights_1, weights_2, weights_3


# ============================================================
# 3. Criteria-level weights
# ============================================================
df_labels = pd.read_excel(FILE, sheet_name=SHEET_MAP["criteria"],
                          usecols="A", skiprows=1, header=None)
criteria_names = df_labels[0].str.strip().tolist()

print("\n===== Criteria-level weights (3 methods) =====")
w1, w2, w3 = calculate_ahp(arr, "Criteria")

print("-- Method 1: Arithmetic Mean --")
for name, w in zip(criteria_names, w1):
    print(f"{name}: {float(w) * 100:.2f}%")
print("-- Method 2: Geometric Mean --")
for name, w in zip(criteria_names, w2):
    print(f"{name}: {float(w) * 100:.2f}%")
print("-- Method 3: Eigenvector --")
for name, w in zip(criteria_names, w3):
    print(f"{name}: {float(w) * 100:.2f}%")


# ============================================================
# 4. Alternative-level matrices (one per criterion)
# ============================================================
criteria_sheets = [SHEET_MAP[k] for k in
                   ["urgency", "long_term_value", "sunk_cost", "health"]]

all_w1, all_w2, all_w3 = [], [], []
original_alt_mats = []          # keep original matrices for sensitivity
for sheet in criteria_sheets:
    print(f"\n===== Alternative weights under '{sheet}' =====")
    df_alt = pd.read_excel(FILE, sheet_name=sheet,
                           skiprows=1, usecols="B:G", header=None)
    arr_alt = df_alt.to_numpy(dtype=float)
    original_alt_mats.append(arr_alt)
    wa1, wa2, wa3 = calculate_ahp(arr_alt, sheet)
    all_w1.append(wa1)
    all_w2.append(wa2)
    all_w3.append(wa3)

# Final composite scores: W_alternatives (6 x 4) @ w_criteria (4,)
finalscore1 = np.column_stack(all_w1) @ w1
finalscore2 = np.column_stack(all_w2) @ w2
finalscore3 = np.column_stack(all_w3) @ w3

# Alternative names
df_alts = pd.read_excel(FILE, sheet_name=criteria_sheets[0],
                        usecols="A", skiprows=1, header=None)
alts_names = df_alts[0].str.strip().tolist()


# ============================================================
# 5. Ranking printer (generic, no hardcoded 6)
# ============================================================
def printrank(scores, method_name):
    print(f"\n{method_name}:")
    data = [[alts_names[i], float(scores[i])] for i in range(len(alts_names))]
    for rank in range(1, len(data) + 1):
        max_score, max_index = -999.0, -1
        for i in range(len(data)):
            if data[i][1] > max_score:
                max_score, max_index = data[i][1], i
        en = EN_NAMES.get(data[max_index][0], data[max_index][0])
        print(f"TOP {rank}: {data[max_index][0]} ({en}) —— {max_score * 100:.2f}%")
        data[max_index][1] = -999.0


printrank(finalscore1, "1. Arithmetic Mean Method")
printrank(finalscore2, "2. Geometric Mean Method")
printrank(finalscore3, "3. Eigenvector Method")


# ============================================================
# 6. Sensitivity analysis (strongly recommended for MCM/ICM)
#    Perturb each judgment by up to +/-20%, keep reciprocity,
#    recompute composite scores, count how often each
#    alternative ranks #1.
# ============================================================
def perturb_matrix(matrix, perturb=0.2, rng=None):
    """Randomly perturb upper triangle, restore reciprocity."""
    rng = rng or np.random.default_rng()
    m = matrix.copy()
    n = m.shape[0]
    iu = np.triu_indices(n, 1)
    noise = 1 + perturb * (2 * rng.random(len(iu[0])) - 1)  # [1-p, 1+p]
    m[iu] = m[iu] * noise
    m[iu[1], iu[0]] = 1.0 / m[iu]                            # reciprocity
    np.fill_diagonal(m, 1.0)
    return m


def sensitivity_analysis(n_sim=1000, perturb=0.2):
    rng = np.random.default_rng(42)   # fixed seed for reproducibility
    rank1_counts = {name: 0 for name in alts_names}
    skipped = 0
    for _ in range(n_sim):
        try:
            pw1, _, _ = calculate_ahp(
                perturb_matrix(arr, perturb, rng), verbose=False)
            alt_ws = []
            for i, sheet in enumerate(criteria_sheets):
                pw, _, _ = calculate_ahp(
                    perturb_matrix(original_alt_mats[i], perturb, rng),
                    verbose=False)
                alt_ws.append(pw)
            score = np.column_stack(alt_ws) @ pw1
            winner = alts_names[int(np.argmax(score))]
            rank1_counts[winner] += 1
        except ValueError:
            # Perturbed matrix failed the consistency check -> skipped
            skipped += 1
    valid = n_sim - skipped
    print(f"\n===== Sensitivity analysis "
          f"({valid}/{n_sim} valid simulations, +/-{int(perturb*100)}%) =====")
    print(f"({skipped} simulations skipped: perturbed matrices "
          f"failed the CR < 0.1 check, which is expected behavior.)")
    for name, cnt in rank1_counts.items():
        en = EN_NAMES.get(name, name)
        print(f"{name} ({en}): ranked #1 in {cnt / valid * 100:.1f}% of runs")
    return rank1_counts


rank1_counts = sensitivity_analysis(n_sim=1000, perturb=0.2)

# Sensitivity chart for the paper appendix (English labels, no font issues)
plt.figure(figsize=(8, 5))
names_en = [EN_NAMES.get(n, n) for n in rank1_counts.keys()]
counts = [rank1_counts[n] for n in rank1_counts.keys()]
plt.bar(range(len(names_en)), counts, color="steelblue")
plt.xticks(range(len(names_en)), names_en, rotation=30, ha="right")
plt.ylabel("Times ranked #1 (out of 1000 runs)")
plt.xlabel("Alternative")
plt.title("Sensitivity Analysis: #1 Ranking Frequency under +/-20% Perturbation")
plt.tight_layout()
plt.savefig("sensitivity.png", dpi=300)
print("\nSensitivity chart saved to sensitivity.png")


# ============================================================
# 7. Export results to CSV for direct citation in the paper
# ============================================================
out = pd.DataFrame({
    "Alternative": [EN_NAMES.get(n, n) for n in alts_names],
    "Arithmetic": finalscore1,
    "Geometric": finalscore2,
    "Eigenvector": finalscore3,
})
out.to_csv("ahp_results.csv", index=False, encoding="utf-8-sig")
print("Results saved to ahp_results.csv")
