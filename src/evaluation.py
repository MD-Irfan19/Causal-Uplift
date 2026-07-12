import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Naive effect on real data, from baseline_models.py (Phase 3).
# Used as the random-targeting reference line's slope.
NAIVE_REAL_EFFECT = 10.73

# The range where decile analysis confirmed the model reliably beats
# random targeting -- used to shade the plot.
RELIABLE_ZONE = (0.10, 0.45)


# ============================================================
# Core computation
# ============================================================

def cumulative_uplift_curve(df, n_points=20, min_control=100, naive_effect=NAIVE_REAL_EFFECT):
    """
    Ranks customers by predicted CATE (descending), then for each
    cutoff % computes the realized incremental gain among the top-n
    customers: (treated mean outcome - control mean outcome) * n.

    Points backed by fewer than `min_control` control customers are
    flagged unreliable rather than silently included -- a mean
    difference estimated from a handful of control customers is too
    noisy to trust (this is what caused the misleading dip at the 5%
    cutoff in the first version of this chart).

    Returns the per-cutoff results dataframe and the random-targeting
    reference line (same length, for plotting).
    """
    df_sorted = df.sort_values("predicted_cate", ascending=False).reset_index(drop=True)

    cutoffs = np.linspace(0.05, 1.0, n_points)
    results = []
    for cutoff in cutoffs:
        n = int(len(df_sorted) * cutoff)
        subset = df_sorted.iloc[:n]
        treated = subset.loc[subset["treatment"] == 1, "outcome"]
        control = subset.loc[subset["treatment"] == 0, "outcome"]
        reliable = len(control) >= min_control
        gain = (treated.mean() - control.mean()) * n if reliable else np.nan
        results.append({
            "pct_targeted": cutoff,
            "n_control": len(control),
            "cumulative_gain": gain,
            "reliable": reliable,
        })

    qini_df = pd.DataFrame(results)
    total_naive_gain = naive_effect * len(df_sorted)
    random_line = qini_df["pct_targeted"] * total_naive_gain

    return qini_df, random_line


# ============================================================
# Plotting
# ============================================================

def plot_qini_curve(qini_df, random_line, reliable_zone=RELIABLE_ZONE,
                     save_path="reports/figures/qini_curve.png"):
    reliable = qini_df[qini_df["reliable"]]
    unreliable = qini_df[~qini_df["reliable"]]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Shade the range where the model reliably beat random targeting
    ax.axvspan(reliable_zone[0], reliable_zone[1], color="green", alpha=0.08,
               label="Reliable zone (beats random)")

    ax.plot(reliable["pct_targeted"], reliable["cumulative_gain"],
            marker="o", color="steelblue", label="Model-ranked targeting")

    ax.plot(qini_df["pct_targeted"], random_line,
            linestyle="--", color="gray", label="Random targeting")

    # Explicitly annotate any excluded low-percentage point rather than
    # leaving a silent gap or a phantom legend entry
    if len(unreliable) > 0:
        ax.annotate(
            f"{unreliable['pct_targeted'].iloc[0]:.0%} point excluded\n"
            f"(n_control={int(unreliable['n_control'].iloc[0])}, too few to trust)",
            xy=(0.05, 8000),            # arrow tip: near the excluded point's location
            xytext=(0.62, 18000),       # text box: empty space, clear of both lines
            fontsize=8, color="firebrick", ha="left",
            arrowprops=dict(arrowstyle="->", color="firebrick", alpha=0.6,
                             connectionstyle="arc3,rad=0.15"),
        )

    ax.set_xlabel("% of Customers Targeted (ranked by predicted uplift)")
    ax.set_ylabel("Cumulative Incremental Sales Gain")

    fig.suptitle("Qini-style Curve: Model-Ranked vs Random Targeting", fontsize=13)
    ax.set_title("Reliably beats random only in the 10%\u201345% range; "
                  "mid-range noise erodes the edge beyond that",
                  fontsize=10, color="dimgray")

    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ============================================================
# Script entry point 
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_starbucks_with_cate.csv")

    qini_df, random_line = cumulative_uplift_curve(df)
    print(qini_df)

    plot_qini_curve(qini_df, random_line)
    print("\nSaved reports/figures/qini_curve.png")