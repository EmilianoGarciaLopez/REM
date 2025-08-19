#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#######################
# CONFIGURATION PARAMETERS
#######################

CSV_FILE = "src/all_questions_aggregated_with_text_new_prompt.csv"

# Process only non-number questions (e.g., "comparison_in_frame", "order_preserving", etc.)
QUESTION_TYPE = "comparison_in_frame"  # change as needed; must NOT be "number"

# Optionally, filter to a specific question text.
SELECTED_QUESTION = None  # e.g., "Are there more red cubes or blue spheres?" or None

# Toggle the independent variable (x‑axis):
# "duplicates" uses duplicate counts; "objects" uses number_of_objects_it_sees.
X_AXIS_VAR = "duplicates"  # choose either "duplicates" or "objects"

# When grouping by objects, these bin edges are used.
OBJ_BIN_EDGES = [0, 6, 12, 18, 24, 30]

# Allowed trajectory lengths (used if trajectory grouping is available)
ALLOWED_TRAJ = [2, 4, 8, 16, 32, 64]

# List of models; here we assume that for each model there is a column
# "answered_correctly_<model>" that contains either "yes" or another value.
MODEL_LIST = [
    "gpt-4o",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
   # "nova-lite-v1",
    "llama-3.2-11b-vision-instruct",
    "gemini-1.5-flash-latest-TEXT",
    "gemini-1.5-flash-latest-TEXT-NEW-PROMPT",
]

# For non-number questions, we now rely on the column answered_correctly_<model>
# to indicate correctness (a value of "yes" means correct).
# (No need to use ground_truth here.)
GROUND_TRUTH_COL = "ground_truth"  # not used now

# For grouping by duplicates, we use this column.
DUPLICATE_COL = "total_number_of_duplicates_seen"

# Smoothing: rolling window size (set to 1 for no smoothing)
SMOOTHING_WINDOW = 1

# ---------- ORIGINAL DUPLICATES BINNING STRUCTURE ----------
custom_order = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8-9": 8,
    "10-11": 10,
    "12-13": 12,
    "14-16": 14,
    "17-18": 17,
    "19-21": 19,
    "22-24": 22,
    "25-32": 25,
    "33-46": 33,
    "Other": 100
}

#######################
# UTILITY FUNCTIONS
#######################

def bin_duplicates(value):
    """
    Bin duplicate counts using the original structure:
      - Values 0–7 get their own bin,
      - Then ranges: "8-9", "10-11", etc.
    """
    try:
        value = int(value)
    except (ValueError, TypeError):
        return "Other"
    if value <= 7:
        return str(value)
    elif 8 <= value <= 9:
        return "8-9"
    elif 10 <= value <= 11:
        return "10-11"
    elif 12 <= value <= 13:
        return "12-13"
    elif 14 <= value <= 16:
        return "14-16"
    elif 17 <= value <= 18:
        return "17-18"
    elif 19 <= value <= 21:
        return "19-21"
    elif 22 <= value <= 24:
        return "22-24"
    elif 25 <= value <= 32:
        return "25-32"
    elif 33 <= value <= 46:
        return "33-46"
    else:
        return "Other"

def mid_of_interval(interval):
    """Return the midpoint of a pandas Interval."""
    return (interval.left + interval.right) / 2

def map_traj(val):
    """Map a trajectory length to the nearest allowed value."""
    try:
        v = float(val)
    except:
        return np.nan
    return min(ALLOWED_TRAJ, key=lambda x: abs(x - v))

#######################
# MAIN SCRIPT
#######################

def main():
    # 1) Read CSV and filter to non-number questions.
    df = pd.read_csv(CSV_FILE)
    if "question_type" in df.columns:
        df = df[df["question_type"] != "number"].copy()
    else:
        print("Warning: 'question_type' column not found; processing all rows.")

    if SELECTED_QUESTION is not None:
        df = df[df["question"] == SELECTED_QUESTION].copy()
        if df.empty:
            print(f"No rows found for question: {SELECTED_QUESTION}")
            return

    # 2) Convert duplicate counts to numeric.
    df[DUPLICATE_COL] = pd.to_numeric(df[DUPLICATE_COL], errors="coerce")
    
    # 3) Process the "number_of_objects_it_sees" column.
    if "number_of_objects_it_sees" in df.columns:
        df["number_of_objects_it_sees"] = pd.to_numeric(df["number_of_objects_it_sees"], errors="coerce")
        df = df.dropna(subset=["number_of_objects_it_sees"])
        df["number_of_objects_it_sees"] = df["number_of_objects_it_sees"].astype(int)
        df = df[(df["number_of_objects_it_sees"] >= OBJ_BIN_EDGES[0]) &
                (df["number_of_objects_it_sees"] < OBJ_BIN_EDGES[-1])]
        df["obj_bin"] = pd.cut(df["number_of_objects_it_sees"], bins=OBJ_BIN_EDGES, right=False)
    else:
        df["obj_bin"] = "All"

    # 4) Compute correctness for each model by simply checking if answered_correctly_<model> equals "yes".
    for model in MODEL_LIST:
        correct_col_name = "answered_correctly_" + model
        # Create a binary correctness column: 1 if "yes" (case-insensitive), 0 otherwise.
        df[model + "_correct"] = df[correct_col_name].astype(str).str.strip().str.lower().apply(
            lambda x: 1 if x == "yes" else 0
        )
    
    # 5) Process trajectory lengths if available.
    if "len_of_trajectory" in df.columns:
        df["len_of_trajectory"] = pd.to_numeric(df["len_of_trajectory"], errors="coerce")
        df["traj_bin"] = df["len_of_trajectory"].apply(map_traj)
    else:
        print("Column 'len_of_trajectory' not found.")
        df["traj_bin"] = np.nan

    # 6) Bin duplicate counts.
    df["dup_bin"] = df[DUPLICATE_COL].apply(bin_duplicates)

    # 7) Aggregate data for plotting.
    # If valid trajectory values exist, we group by trajectory; otherwise we aggregate over all rows.
    valid_traj = sorted(df["traj_bin"].dropna().unique())
    plot_rows = []
    if valid_traj:
        for model in MODEL_LIST:
            correct_col = model + "_correct"
            for traj in valid_traj:
                subdf = df[df["traj_bin"] == traj].copy()
                if subdf.empty:
                    continue
                if X_AXIS_VAR == "duplicates":
                    subdf["x_bin"] = subdf[DUPLICATE_COL].apply(bin_duplicates)
                    subdf["x_numeric"] = subdf["x_bin"].apply(lambda x: custom_order.get(x, np.nan))
                    x_label = "Total Number of Duplicates (binned)"
                else:  # X_AXIS_VAR == "objects"
                    subdf["x_bin"] = pd.cut(subdf["number_of_objects_it_sees"], bins=OBJ_BIN_EDGES, right=False)
                    subdf["x_numeric"] = subdf["x_bin"].apply(mid_of_interval)
                    x_label = "Number of Objects Seen (mid-bin)"
                grouped = subdf.groupby("x_bin", observed=False)[correct_col].mean().reset_index()
                if X_AXIS_VAR == "duplicates":
                    grouped["x_numeric"] = grouped["x_bin"].apply(lambda x: custom_order.get(x, np.nan))
                else:
                    grouped["x_numeric"] = grouped["x_bin"].apply(mid_of_interval)
                grouped = grouped.sort_values("x_numeric")
                if SMOOTHING_WINDOW > 1:
                    grouped["smoothed_metric"] = grouped[correct_col].rolling(
                        window=SMOOTHING_WINDOW, center=True, min_periods=1
                    ).mean()
                else:
                    grouped["smoothed_metric"] = grouped[correct_col]
                for _, row in grouped.iterrows():
                    plot_rows.append({
                        "model": model,
                        "traj_bin": traj,
                        "x_numeric": row["x_numeric"],
                        "fraction_correct": row["smoothed_metric"]
                    })
    else:
        for model in MODEL_LIST:
            correct_col = model + "_correct"
            subdf = df.copy()
            if X_AXIS_VAR == "duplicates":
                subdf["x_bin"] = subdf[DUPLICATE_COL].apply(bin_duplicates)
                subdf["x_numeric"] = subdf["x_bin"].apply(lambda x: custom_order.get(x, np.nan))
                x_label = "Total Number of Duplicates (binned)"
            else:
                subdf["x_bin"] = pd.cut(subdf["number_of_objects_it_sees"], bins=OBJ_BIN_EDGES, right=False)
                subdf["x_numeric"] = subdf["x_bin"].apply(mid_of_interval)
                x_label = "Number of Objects Seen (mid-bin)"
            grouped = subdf.groupby("x_bin", observed=False)[correct_col].mean().reset_index()
            if X_AXIS_VAR == "duplicates":
                grouped["x_numeric"] = grouped["x_bin"].apply(lambda x: custom_order.get(x, np.nan))
            else:
                grouped["x_numeric"] = grouped["x_bin"].apply(mid_of_interval)
            grouped = grouped.sort_values("x_numeric")
            if SMOOTHING_WINDOW > 1:
                grouped["smoothed_metric"] = grouped[correct_col].rolling(
                    window=SMOOTHING_WINDOW, center=True, min_periods=1
                ).mean()
            else:
                grouped["smoothed_metric"] = grouped[correct_col]
            for _, row in grouped.iterrows():
                plot_rows.append({
                    "model": model,
                    "x_numeric": row["x_numeric"],
                    "fraction_correct": row["smoothed_metric"]
                })

    combined_df = pd.DataFrame(plot_rows)

    # 8) (Optional) Print frequency counts for the independent variable bins.
    print("Frequency counts for independent variable bins:")
    if valid_traj:
        for traj in valid_traj:
            subdf = df[df["traj_bin"] == traj]
            if X_AXIS_VAR == "duplicates":
                counts = subdf["dup_bin"].value_counts().sort_index()
                print(f"Trajectory group {traj}:")
                print(counts, "\n")
            else:
                subdf["x_bin"] = pd.cut(subdf["number_of_objects_it_sees"], bins=OBJ_BIN_EDGES, right=False)
                counts = subdf["x_bin"].value_counts().sort_index()
                print(f"Trajectory group {traj}:")
                print(counts, "\n")
    else:
        if X_AXIS_VAR == "duplicates":
            counts = df["dup_bin"].value_counts().sort_index()
        else:
            df["x_bin"] = pd.cut(df["number_of_objects_it_sees"], bins=OBJ_BIN_EDGES, right=False)
            counts = df["x_bin"].value_counts().sort_index()
        print(counts, "\n")
    
    # 9) Create subplots in a 2x3 grid (one subplot per model).
    num_models = len(MODEL_LIST)
    num_rows, num_cols = 2, 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), sharey=True)
    ax_flat = axs.flat

    if valid_traj:
        unique_traj = sorted(combined_df["traj_bin"].dropna().unique())
        markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
        marker_map = { traj: markers[i % len(markers)] for i, traj in enumerate(unique_traj) }
        for i, model in enumerate(MODEL_LIST[:num_rows * num_cols]):
            ax = ax_flat[i]
            model_df = combined_df[combined_df["model"] == model]
            for traj in unique_traj:
                data = model_df[model_df["traj_bin"] == traj]
                if data.empty:
                    continue
                ax.plot(
                    data["x_numeric"],
                    data["fraction_correct"],
                    marker=marker_map[traj],
                    linestyle="-",
                    label=f"Traj={traj}"
                )
            ax.axhline(0.33, linestyle="--", color="red", linewidth=1.5, label="33% baseline")
            ax.set_title(model, fontsize=12)
            ax.set_xlabel(x_label)
            if i % num_cols == 0:
                ax.set_ylabel("Fraction Correct")
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", color="lightgray", linewidth=0.5)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper center",
                      bbox_to_anchor=(0.5, 1.15), ncol=len(unique_traj) + 1, prop={"size": 9})
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("none")
    else:
        for i, model in enumerate(MODEL_LIST[:num_rows * num_cols]):
            ax = ax_flat[i]
            model_df = combined_df[combined_df["model"] == model]
            ax.plot(model_df["x_numeric"], model_df["fraction_correct"],
                    marker="o", linestyle="-", label=model)
            ax.axhline(0.33, linestyle="--", color="red", linewidth=1.5, label="33% baseline")
            ax.set_title(model, fontsize=12)
            ax.set_xlabel(x_label)
            if i % num_cols == 0:
                ax.set_ylabel("Fraction Correct")
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", color="lightgray", linewidth=0.5)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, prop={"size": 9})
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("none")

    for j in range(num_models, num_rows * num_cols):
        ax_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
