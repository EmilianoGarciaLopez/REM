#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== CONFIGURATION PARAMETERS ==========
CSV_FILE = "src/all_questions_aggregated_with_text_new_prompt.csv"

# Choose any question type other than "number"
QUESTION_TYPE = "comparison_in_frame"  # Change as desired

# We'll restrict to rows with fewer than 30 objects
OBJ_BIN_EDGES = [0, 6, 12, 18, 24, 30]

# List of models; for non-number questions, we use the "answered_correctly_<model>" columns.
MODEL_LIST = [
    "gpt-4o", 
    "gemini-1.5-pro-latest", 
    "gemini-1.5-flash-latest", 
    #"nova-lite-v1", 
    "llama-3.2-11b-vision-instruct",
    "gemini-1.5-flash-latest-TEXT",
    "gemini-1.5-flash-latest-TEXT-NEW-PROMPT",
]

# (For non-number questions, we don’t use ground_truth to compute correctness.)
GROUND_TRUTH_COL = "ground_truth"
DUPLICATE_COL = "total_number_of_duplicates_seen"

SMOOTHING_WINDOW = 1  # Set to >1 to apply rolling smoothing; 1 for no smoothing

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

def bin_duplicates(value):
    """
    Bins the duplicate counts using the original structure:
      individual bins for values 0–7, then combined ranges.
    """
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
# --------------------------------------------------

def main():
    # 1) Read CSV and filter by chosen non-"number" question type
    df = pd.read_csv(CSV_FILE)
    if "question_type" in df.columns:
        df = df[df["question_type"] == QUESTION_TYPE].copy()
    else:
        print("Warning: 'question_type' column not found; processing all rows.")
    
    # Convert duplicate counts to numeric and then to int
    df[DUPLICATE_COL] = pd.to_numeric(df[DUPLICATE_COL], errors="coerce")
    df[DUPLICATE_COL] = df[DUPLICATE_COL].astype(int)
    
    # For each model, convert its answer column to numeric (if possible)
    # (These columns are not used for correctness in non-number questions.)
    for model in MODEL_LIST:
        df[model] = pd.to_numeric(df[model], errors="coerce")
    
    # We require the following columns: duplicate info, number_of_objects_it_sees,
    # and the answered_correctly columns for each model.
    required_cols = [DUPLICATE_COL, "number_of_objects_it_sees"] + [f"answered_correctly_{m}" for m in MODEL_LIST]
    df = df.dropna(subset=required_cols)
    
    # 2) Compute correctness for non-number questions using the "answered_correctly_<model>" columns.
    # Map "yes" (case-insensitive, stripped) to 1; everything else to 0.
    for model in MODEL_LIST:
        correct_col = "answered_correctly_" + model
        df[model + "_correct"] = df[correct_col].astype(str).str.strip().str.lower().apply(lambda x: 1 if x == "yes" else 0)
    
    # 3) Restrict to rows with number_of_objects_it_sees < 30 and bin these values.
    if "number_of_objects_it_sees" in df.columns:
        df["number_of_objects_it_sees"] = pd.to_numeric(df["number_of_objects_it_sees"], errors="coerce")
        df["number_of_objects_it_sees"] = df["number_of_objects_it_sees"].fillna(-1).astype(int)
        df = df[df["number_of_objects_it_sees"] < 30]
        df["obj_bin"] = pd.cut(
            df["number_of_objects_it_sees"],
            bins=OBJ_BIN_EDGES,  # Creates bins: [0,6), [6,12), [12,18), [18,24), [24,30)
            right=False
        )
    else:
        df["obj_bin"] = "All"
    
    if pd.api.types.is_categorical_dtype(df["obj_bin"]):
        obj_bin_categories = df["obj_bin"].cat.categories
    else:
        obj_bin_categories = sorted(df["obj_bin"].unique())
    
    # 4) Bin duplicates using the original function
    df["dup_bin"] = df[DUPLICATE_COL].apply(bin_duplicates)
    
    # 5) For each model and each object bin, compute the mean correctness by duplicate bin.
    plot_rows = []
    for model in MODEL_LIST:
        correct_col = model + "_correct"
        for obj_bin in obj_bin_categories:
            subdf = df[df["obj_bin"] == obj_bin]
            if subdf.empty:
                continue
            grouped = subdf.groupby("dup_bin")[correct_col].mean().reset_index()
            grouped["dup_numeric"] = grouped["dup_bin"].apply(lambda x: custom_order.get(x, np.nan))
            grouped = grouped.sort_values("dup_numeric")
            if SMOOTHING_WINDOW > 1:
                grouped["smoothed_metric"] = grouped[correct_col].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()
            else:
                grouped["smoothed_metric"] = grouped[correct_col]
            for _, row in grouped.iterrows():
                plot_rows.append({
                    "model": model,
                    "obj_bin": str(obj_bin),
                    "dup_bin": row["dup_bin"],
                    "dup_numeric": row["dup_numeric"],
                    "fraction_correct": row["smoothed_metric"]
                })
    combined_df = pd.DataFrame(plot_rows)
    
    # 6) Create subplots in a 2x3 grid (one subplot per model)
    num_models = len(MODEL_LIST)
    num_rows, num_cols = 2, 3  # 2 rows x 3 columns = up to 6 subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 5*num_rows), sharey=True)
    ax_flat = axs.flat
    
    # Use different markers for each object bin
    unique_obj_bins = combined_df["obj_bin"].unique()
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    marker_map = { bin_label: markers[i % len(markers)] for i, bin_label in enumerate(unique_obj_bins) }
    
    for i, model in enumerate(MODEL_LIST[:num_rows*num_cols]):
        ax = ax_flat[i]
        model_df = combined_df[combined_df["model"] == model]
        for bin_label in unique_obj_bins:
            data = model_df[model_df["obj_bin"] == bin_label]
            if data.empty:
                continue
            ax.plot(
                data["dup_numeric"],
                data["fraction_correct"],
                marker=marker_map[bin_label],
                linestyle="-",
                label=bin_label
            )
        # --- Add the 33% chance baseline ---
        ax.axhline(0.33, linestyle="--", color="red", linewidth=1.5, label="33% Baseline")
        ax.set_title(model, fontsize=12)
        ax.set_xlabel("Total Number of Duplicates")
        if i % num_cols == 0:
            ax.set_ylabel("Fraction Correct")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", color="lightgray", linewidth=0.5)
        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.15),
                  ncol=len(unique_obj_bins)+1, prop={"size": 9})
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_ticks_position("none")
    
    # Hide any extra subplots if fewer than 6 models are present
    for j in range(len(MODEL_LIST), num_rows*num_cols):
        ax_flat[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
