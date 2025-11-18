import pandas as pd
import numpy as np

def quantize_fico_scores(fico_scores, default_flags, num_buckets, method='mse'):
    df = pd.DataFrame({'score': fico_scores, 'default': default_flags})
    grouped = df.groupby('score').agg(count=('default', 'size'), defaults=('default', 'sum')).reset_index()

    scores = grouped['score'].values
    counts = grouped['count'].values
    defaults = grouped['defaults'].values
    n = len(scores)

    cumulative_counts = np.concatenate(([0], np.cumsum(counts)))
    cumulative_defaults = np.concatenate(([0], np.cumsum(defaults)))

    cost_matrix = np.zeros((n, n))
    eps = 1e-9

    for i in range(n):
        for j in range(i, n):
            total = cumulative_counts[j + 1] - cumulative_counts[i]
            total_defaults = cumulative_defaults[j + 1] - cumulative_defaults[i]
            if total == 0:
                cost = 0
            else:
                p = total_defaults / total
                if method == 'likelihood':
                    cost = - (total_defaults * np.log(p + eps) + (total - total_defaults) * np.log(1 - p + eps))
                else:
                    cost = total_defaults * (1 - p) ** 2 + (total - total_defaults) * (p ** 2)
            cost_matrix[i, j] = cost

    dp = np.full((num_buckets, n), np.inf)
    prev = np.full((num_buckets, n), -1)

    for j in range(n):
        dp[0, j] = cost_matrix[0, j]

    for b in range(1, num_buckets):
        for j in range(b, n):
            candidates = dp[b - 1, b - 1:j] + cost_matrix[np.arange(b - 1, j) + 1, j]
            i = np.argmin(candidates) + (b - 1)
            dp[b, j] = candidates[i - (b - 1)]
            prev[b, j] = i

    cut_points = []
    j = n - 1
    for b in range(num_buckets - 1, 0, -1):
        i = prev[b, j]
        cut_points.append(i)
        j = i
    cut_points = sorted(cut_points)

    edges = [int(scores[0])] + [int(scores[i]) for i in cut_points] + [int(scores[-1])]
    labels = list(range(num_buckets, 0, -1))
    buckets = pd.cut(fico_scores, bins=edges, labels=labels, include_lowest=True)

    return edges, labels, buckets


# REPORT
def generate_report(bin_edges, labels):
    report = f"""
JP Morgan Chase Quantitative Research Report

The provided code performs a credit risk quantization analysis by categorizing FICO scores into discrete credit rating buckets.
The primary objective is to optimally segment the continuous range of FICO scores into a fixed number of buckets in a way
that captures meaningful differences in default risk across segments. The method used for this segmentation is based on
minimizing the mean squared error (MSE).

The code begins by loading FICO score and default data from a CSV file into a pandas DataFrame. It then groups the data by
unique FICO scores to compute the count of borrowers and the number of defaults associated with each score.

To compute segmentation cost, it calculates cumulative statistics—number of observations and defaults—and constructs a cost matrix
storing the loss (MSE or negative log-likelihood) for each score segment.

Dynamic programming is used to find the segmentation that minimizes total cost across all buckets. It tracks the minimum cost in
a DP table and reconstructs the best cut points to separate score ranges.

Final bucket boundaries:
{bin_edges}

Assigned ratings (lower is better):
{labels}

This quantization method is effective in converting continuous scores into discrete inputs for modeling, while enhancing interpretability
of credit risk across a portfolio.
"""
    print(report)
    with open("Quantization_Report.txt", "w") as f:
        f.write(report)


# TEST CASE
df = pd.read_csv("/Users/ramko/Downloads/Task 3 and 4_Loan_Data.csv")

edges, labels, buckets = quantize_fico_scores(
    fico_scores=df["fico_score"],
    default_flags=df["default"],
    num_buckets=5,
    method='mse'
)

print("Bucket Edges:", edges)
print("Labels:", labels)
print("Sample Mapping:")
print(pd.DataFrame({
    "fico_score": df["fico_score"],
    "credit_rating": buckets
}).head(10))

generate_report(edges, labels)
